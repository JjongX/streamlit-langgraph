# Human-in-the-Loop (HITL) middleware for interrupt management and handling.

import json
from typing import Any, Dict, List, Optional

import streamlit as st

from ...agent import AgentManager
from ..executor.registry import ExecutorRegistry
from ..state import WorkflowStateManager
from .interrupts import InterruptManager


class HITLUtils:
    """Utility functions for HITL approval/decision processing and data transformation."""
    
    @staticmethod
    def extract_action_requests_from_interrupt(interrupt_raw: Any) -> List[Dict[str, Any]]:
        """Extract action_requests from Interrupt objects."""
        if not interrupt_raw:
            return []
        
        if isinstance(interrupt_raw, list):
            return HITLUtils._extract_from_list(interrupt_raw)
        elif isinstance(interrupt_raw, dict):
            return HITLUtils._extract_from_dict(interrupt_raw)
        
        return []
    
    @staticmethod
    def _extract_from_list(interrupt_list: List[Any]) -> List[Dict[str, Any]]:
        """Extract action requests from a list of interrupt items."""
        result = []
        for item in interrupt_list:
            if hasattr(item, 'value'):
                result.extend(HITLUtils._extract_from_dict(item.value))
            elif isinstance(item, dict):
                result.extend(HITLUtils._extract_from_dict(item))
        return result
    
    @staticmethod
    def _extract_from_dict(interrupt_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract action requests from a dict."""
        if 'action_requests' in interrupt_dict:
            return interrupt_dict['action_requests']
        return [interrupt_dict]
    
    @staticmethod
    def format_decisions(decisions: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Format decisions for HumanInTheLoopMiddleware.
        
        The middleware expects decisions in format:
        [{"type": "approve|reject|edit", "edit": {...} if edit}]
        
        Args:
            decisions: List of decision dicts or None values
            
        Returns:
            List of formatted decision dictionaries
        """
        formatted_decisions = []
        for decision in decisions:
            if decision:
                formatted_decisions.append(decision)
            else:
                # If no decision, default to approve
                formatted_decisions.append({"type": "approve"})
        return formatted_decisions
    
    @staticmethod
    def check_edit_allowed(agent_interrupt_on: Optional[Dict[str, Any]], tool_name: str) -> bool:
        """
        Check if editing is allowed for a tool based on agent's interrupt_on configuration.
        
        Args:
            agent_interrupt_on: The agent's interrupt_on configuration dict
            tool_name: Name of the tool to check
            
        Returns:
            True if editing is allowed, False otherwise
        """
        if not agent_interrupt_on:
            return True  # Default to allowing edit if not configured
        
        tool_config = agent_interrupt_on.get(tool_name, {})
        if isinstance(tool_config, dict):
            allowed_decisions = tool_config.get("allowed_decisions", ["approve", "reject", "edit"])
            return "edit" in allowed_decisions
        
        return True  # Default to allowing edit
    
    @staticmethod
    def parse_edit_input(edit_text: str, default_input: Any) -> tuple:
        """
        Parse user edit input, attempting to parse as JSON if it looks like JSON.
        
        Args:
            edit_text: The text input from the user
            default_input: The default input value to use if parsing fails or text is empty
            
        Returns:
            Tuple of (parsed_input, error_message). error_message is None if successful.
        """
        if not edit_text.strip():
            return default_input, None
        
        # If it looks like JSON (starts with { or [), parse it
        if edit_text.strip().startswith('{') or edit_text.strip().startswith('['):
            parsed = json.loads(edit_text)
            return parsed, None
            
        # Try to parse as JSON, fallback to string if it fails
        parsed = json.loads(edit_text)
        return parsed, None
    
    @staticmethod
    def extract_action_info(action: Any, action_index: int) -> tuple:
        """Extract tool name, input, and ID from action."""
        if isinstance(action, dict):
            tool_name = action.get("name", action.get("tool", "Unknown"))
            tool_input = action.get("args", action.get("input", {}))
            action_id = action.get("id", f"action_{action_index}")
        else:
            tool_name = str(action)
            tool_input = {}
            action_id = f"action_{action_index}"
        return tool_name, tool_input, action_id
    
    @staticmethod
    def find_pending_action(decisions: List[Optional[Dict[str, Any]]]) -> Optional[int]:
        """Find the first action that needs a decision."""
        for i, decision in enumerate(decisions):
            if decision is None:
                return i
        return None
    
    @staticmethod
    def has_pending_interrupts(workflow_state: Optional[Dict[str, Any]]) -> bool:
        """
        Check if there are any pending interrupts in the workflow state.
        
        Delegates to InterruptManager for the actual check.
        
        Args:
            workflow_state: The workflow state dictionary (can be None)
            
        Returns:
            True if there are valid pending interrupts, False otherwise
        """
        if not workflow_state:
            return False
        return InterruptManager.has_pending_interrupts(workflow_state)
    
    @staticmethod
    def get_valid_interrupts(workflow_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract and filter valid interrupts from workflow state."""
        pending_interrupts = WorkflowStateManager.get_pending_interrupts(workflow_state)
        return {
            key: value for key, value in pending_interrupts.items()
            if isinstance(value, dict) and value.get("__interrupt__")
        }
    
    @staticmethod
    def initialize_decisions(workflow_state: Dict[str, Any], executor_key: str, 
                            num_actions: int) -> List[Optional[Dict[str, Any]]]:
        """Initialize decisions list from workflow state or create new one."""
        decisions = WorkflowStateManager.get_hitl_decision(workflow_state, executor_key)
        if decisions is None or len(decisions) != num_actions:
            decisions = [None] * num_actions
        return decisions
    
    @staticmethod
    def clear_interrupt_and_decisions(workflow_state: Dict[str, Any], executor_key: str):
        """Clear interrupt and decisions from workflow state."""
        if "pending_interrupts" in workflow_state.get("metadata", {}):
            workflow_state["metadata"]["pending_interrupts"].pop(executor_key, None)
        HITLUtils.clear_decisions(workflow_state, executor_key)
    
    @staticmethod
    def clear_decisions(workflow_state: Dict[str, Any], executor_key: str):
        """Clear decisions for an executor."""
        if "hitl_decisions" in workflow_state.get("metadata", {}):
            decisions_key = f"{executor_key}_decisions"
            workflow_state["metadata"]["hitl_decisions"].pop(decisions_key, None)


class HITLHandler:
    """
    Orchestrator for Human-in-the-Loop approval UI/UX and decision processing.
    
    Uses HITLUtils for data transformation utilities.
    """
    
    def __init__(self, agent_manager, config, state_manager, display_manager):
        """
        Initialize interrupt handler with dependencies.
        
        Args:
            agent_manager: AgentManager instance for accessing agents
            config: UIConfig instance for UI settings
            state_manager: StateSynchronizer instance for state management
            display_manager: DisplayManager instance for rendering messages
        """
        self.agent_manager = agent_manager
        self.config = config
        self.state_manager = state_manager
        self.display_manager = display_manager
    
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
        
        interrupt_info = HITLUtils.extract_action_requests_from_interrupt(interrupt_raw)
        if not interrupt_info:
            st.error("⚠️ Error: Could not extract action details from interrupt.")
            return False
        
        executor = self.get_or_create_executor(executor_key, agent_name, workflow_state)
        if executor is None:
            return False
        
        decisions = HITLUtils.initialize_decisions(workflow_state, executor_key, len(interrupt_info))
        pending_action_index = HITLUtils.find_pending_action(decisions)
        
        if pending_action_index is None:
            return self.resume_with_decisions(workflow_state, executor_key, executor, agent_name, 
                                             decisions, original_config)
        
        self.display_action_approval_ui(executor_key, executor, agent_name, interrupt_info, 
                                        pending_action_index, decisions, workflow_state)
        return True
    
    def get_or_create_executor(self, executor_key: str, agent_name: str, 
                               workflow_state: Dict[str, Any]):
        """
        Get existing executor or create a new one.
        
        Args:
            executor_key: Key identifying the executor
            agent_name: Name of the agent
            workflow_state: Current workflow state
            
        Returns:
            CreateAgentExecutor instance or None
        """
        registry = ExecutorRegistry()
        executor_key = f"workflow_executor_{agent_name}"
        executor = st.session_state.agent_executors.get(executor_key)
        if executor is None:
            agent = self.agent_manager.agents.get(agent_name)
            if agent:
                executor = registry.create_for_hitl(agent, executor_key)
        
        if executor is None:
            # Clear invalid interrupt
            clear_update = WorkflowStateManager.clear_pending_interrupt(workflow_state, executor_key)
            workflow_state["metadata"].update(clear_update["metadata"])
        
        return executor
    
    def resume_with_decisions(self, workflow_state: Dict[str, Any], executor_key: str,
                               executor, agent_name: str,
                               decisions: List[Dict[str, Any]], original_config: Dict[str, Any]) -> bool:
        """
        Resume execution with all decisions made.
        
        Args:
            workflow_state: Current workflow state
            executor_key: Key identifying the executor
            executor: CreateAgentExecutor instance
            agent_name: Name of the agent
            decisions: List of user decisions
            original_config: Original execution config (should contain thread_id)
            
        Returns:
            True if execution was resumed
        """
        # Clear interrupt using state manager
        self.state_manager.clear_pending_interrupt(executor_key)
        
        formatted_decisions = HITLUtils.format_decisions(decisions)
        
        # Handle CreateAgentExecutor which needs agent_obj initialization
        if hasattr(executor, 'agent_obj') and executor.agent_obj is None:
            llm_client = AgentManager.get_llm_client(executor.agent)
            executor._build_agent(llm_client)
        
        # Always use workflow_thread_id from workflow_state metadata (single source of truth)
        workflow_thread_id = workflow_state.get("metadata", {}).get("workflow_thread_id")
        if not workflow_thread_id:
            st.error("⚠️ Error: Could not find thread_id for resume.")
            return False
        
        # Build resume config
        resume_config = original_config.copy() if original_config else {}
        if "configurable" not in resume_config:
            resume_config["configurable"] = {}
        resume_config["configurable"]["thread_id"] = workflow_thread_id
        
        with st.spinner("Processing your decision..."):
            resume_response = executor.resume(formatted_decisions, config=resume_config)
        
        # Handle additional interrupts
        if resume_response and resume_response.get("__interrupt__"):
            self.state_manager.set_pending_interrupt(agent_name, resume_response, executor_key)
            HITLUtils.clear_decisions(workflow_state, executor_key)
            st.rerun()
        
        # Add response using state manager (automatic deduplication)
        if resume_response and resume_response.get("content"):
            self.state_manager.add_assistant_message(
                resume_response["content"],
                agent_name
            )
            
            # Render the message using the same method as workflow execution
            workflow_messages = workflow_state.get("messages", [])
            if workflow_messages:
                last_msg = workflow_messages[-1]
                self.display_manager.render_workflow_message(last_msg)
        
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
        
        # remove these three functions and just use the handle decision function
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

