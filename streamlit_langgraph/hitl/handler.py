# HITL handler for orchestrating interrupt processing and UI.

import json
from typing import Any, Dict, List, Optional

import streamlit as st

from ..agent import AgentManager
from ..core.executor_registry import ExecutorRegistry
from ..state import WorkflowStateManager
from .utils import HITLUtils


class HITLHandler:
    """
    Orchestrator for Human-in-the-Loop (HITL) interrupt processing and UI.
    
    Uses HITLUtils for data transformation utilities.
    """
    
    def __init__(self, agent_manager, config, state_coordinator):
        """
        Initialize HITL handler with dependencies.
        
        Args:
            agent_manager: AgentManager instance for accessing agents
            config: UIConfig instance for UI settings
            state_coordinator: StateCoordinator instance for state management
        """
        self.agent_manager = agent_manager
        self.config = config
        self.state_coordinator = state_coordinator
    
    def handle_pending_interrupts(self, workflow_state: Dict[str, Any]) -> bool:
        """
        Display UI for pending human-in-the-loop interrupts and handle user decisions.
        
        Args:
            workflow_state: Current workflow state
            
        Returns:
            True if interrupts were found and handled (should block further processing)
            False if no interrupts (should continue normal processing)
        """
        if not workflow_state:
            return False
        
        valid_interrupts = HITLUtils.get_valid_interrupts(workflow_state)
        if not valid_interrupts:
            return False
        
        st.markdown("---")
        st.markdown("### ⚠️ **Human Approval Required**")
        st.info("The workflow has paused and is waiting for your approval.")
        
        # Process the first valid interrupt
        for executor_key, interrupt_data in valid_interrupts.items():
            if self.process_interrupt(workflow_state, executor_key, interrupt_data):
                return True
        
        return False
    
    def process_interrupt(self, workflow_state: Dict[str, Any], executor_key: str, 
                          interrupt_data: Dict[str, Any]) -> bool:
        """
        Process a single interrupt - returns True if handled.
        
        Args:
            workflow_state: Current workflow state
            executor_key: Key identifying the executor
            interrupt_data: Interrupt data dictionary
            
        Returns:
            True if interrupt was handled, False otherwise
        """
        agent_name = interrupt_data.get("agent", "Unknown")
        interrupt_raw = interrupt_data.get("__interrupt__", [])
        original_config = interrupt_data.get("config", {})
        thread_id = interrupt_data.get("thread_id")
        
        interrupt_info = HITLUtils.extract_action_requests_from_interrupt(interrupt_raw)
        if not interrupt_info:
            st.error("⚠️ Error: Could not extract action details from interrupt.")
            return False
        
        executor = self.get_or_create_executor(executor_key, agent_name, thread_id, workflow_state)
        if executor is None:
            return False
        
        decisions = HITLUtils.initialize_decisions(workflow_state, executor_key, len(interrupt_info))
        pending_action_index = HITLUtils.find_pending_action(decisions)
        
        if pending_action_index is None:
            return self.resume_with_decisions(workflow_state, executor_key, executor, agent_name, 
                                             decisions, original_config, thread_id)
        
        self.display_action_approval_ui(executor_key, executor, agent_name, interrupt_info, 
                                        pending_action_index, decisions, workflow_state)
        return True
    
    def get_or_create_executor(self, executor_key: str, agent_name: str, 
                               thread_id: str, workflow_state: Dict[str, Any]):
        """
        Get existing executor or create a new one.
        
        Supports both CreateAgentExecutor and ResponseAPIExecutor.
        
        Args:
            executor_key: Key identifying the executor
            agent_name: Name of the agent
            thread_id: Thread ID for the executor
            workflow_state: Current workflow state
            
        Returns:
            CreateAgentExecutor or ResponseAPIExecutor instance or None
        """
        executor = ExecutorRegistry.get(agent_name, executor_type="workflow")
        if executor is None:
            agent = self.agent_manager.agents.get(agent_name)
            if agent and thread_id:
                executor = ExecutorRegistry.create_for_hitl(agent, thread_id, executor_key)
        
        if executor is None:
            # Clear invalid interrupt
            clear_update = WorkflowStateManager.clear_pending_interrupt(workflow_state, executor_key)
            workflow_state["metadata"].update(clear_update["metadata"])
        
        return executor
    
    def resume_with_decisions(self, workflow_state: Dict[str, Any], executor_key: str,
                               executor, agent_name: str,
                               decisions: List[Dict[str, Any]], original_config: Dict[str, Any],
                               thread_id: str) -> bool:
        """
        Resume execution with all decisions made.
        
        Args:
            workflow_state: Current workflow state
            executor_key: Key identifying the executor
            executor: CreateAgentExecutor or ResponseAPIExecutor instance
            agent_name: Name of the agent
            decisions: List of user decisions
            original_config: Original execution config
            thread_id: Thread ID for the executor
            
        Returns:
            True if execution was resumed
        """
        # Clear interrupt using coordinator
        self.state_coordinator.clear_pending_interrupt(executor_key)
        
        formatted_decisions = HITLUtils.format_decisions(decisions)
        
        # Handle CreateAgentExecutor which needs agent_obj initialization
        if hasattr(executor, 'agent_obj') and executor.agent_obj is None:
            llm_client = AgentManager.get_llm_client(executor.agent)
            executor._build_agent(llm_client)
        
        resume_config = original_config or {"configurable": {"thread_id": thread_id}}
        
        # Get messages from workflow_state for resume
        conversation_messages = workflow_state.get("messages", [])
        
        with st.spinner("Processing your decision..."):
            resume_response = executor.resume(formatted_decisions, config=resume_config, messages=conversation_messages)
        
        # Handle additional interrupts
        if resume_response and resume_response.get("__interrupt__"):
            self.state_coordinator.set_pending_interrupt(agent_name, resume_response, executor_key)
            HITLUtils.clear_decisions(workflow_state, executor_key)
            st.rerun()
        
        # Add response using coordinator (automatic deduplication)
        if resume_response and resume_response.get("content"):
            self.state_coordinator.add_assistant_message(
                resume_response["content"],
                agent_name
            )
        
        HITLUtils.clear_interrupt_and_decisions(workflow_state, executor_key)
        
        # Persist workflow_state to session_state (single source of truth)
        st.session_state.workflow_state = workflow_state
        
        st.rerun()
    
    def display_action_approval_ui(self, executor_key: str, executor,
                                    agent_name: str, interrupt_info: List[Dict[str, Any]],
                                    action_index: int, decisions: List[Optional[Dict[str, Any]]],
                                    workflow_state: Dict[str, Any]):
        """
        Display UI for approving/rejecting/editing an action.
        
        Args:
            executor_key: Key identifying the executor
            executor: CreateAgentExecutor instance
            agent_name: Name of the agent
            interrupt_info: List of action information dictionaries
            action_index: Index of the current action to process
            decisions: List of user decisions
            workflow_state: Current workflow state
        """
        action = interrupt_info[action_index]
        tool_name, tool_input, action_id = HITLUtils.extract_action_info(action, action_index)
        
        agent_interrupt_on = getattr(executor.agent, 'interrupt_on', None)
        allow_edit = HITLUtils.check_edit_allowed(agent_interrupt_on, tool_name)
        
        def handle_approve():
            self.handle_decision(workflow_state, executor_key, decisions, action_index, {"type": "approve"})
        
        def handle_reject():
            self.handle_decision(workflow_state, executor_key, decisions, action_index, {"type": "reject"})
        
        def handle_edit(edit_text):
            parsed_input, error_msg = HITLUtils.parse_edit_input(edit_text, tool_input)
            if error_msg:
                st.error(error_msg)
            else:
                self.handle_decision(workflow_state, executor_key, decisions, action_index,
                                    {"type": "edit", "input": parsed_input})
                                    
        with st.container():
            st.markdown("---")
            st.markdown(f"**Agent:** {agent_name} is requesting approval to execute the following action:")
            st.write(f"**Tool:** `{tool_name}`")
            if tool_input:
                st.json(tool_input)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Approve", key=f"approve_{executor_key}_{action_id}"):
                    handle_approve()
            with col2:
                if st.button("❌ Reject", key=f"reject_{executor_key}_{action_id}"):
                    handle_reject()
            with col3:
                if allow_edit:
                    edit_key = f"edit_{executor_key}_{action_id}"
                    edit_btn_key = f"edit_btn_{executor_key}_{action_id}"
                    default_value = json.dumps(tool_input, indent=2) if tool_input else ""
                    
                    edit_text = st.text_area(
                        f"Edit {tool_name} input (optional)",
                        value=default_value, key=edit_key, height=100
                    )
                    
                    if st.button("✏️ Approve with Edit", key=edit_btn_key):
                        handle_edit(edit_text)
    
    def handle_decision(self, workflow_state: Dict[str, Any], executor_key: str,
                       decisions: List[Optional[Dict[str, Any]]], action_index: int,
                       decision: Dict[str, Any]):
        """
        Handle user decision and update workflow state.
        
        Args:
            workflow_state: Current workflow state
            executor_key: Key identifying the executor
            decisions: List of user decisions
            action_index: Index of the action being decided
            decision: Decision dictionary (type: approve/reject/edit)
        """        
        decisions[action_index] = decision
        decision_update = WorkflowStateManager.set_hitl_decision(workflow_state, executor_key, decisions)
        workflow_state["metadata"].update(decision_update["metadata"])
        st.rerun()

