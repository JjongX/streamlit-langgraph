# Workflow executor for compiled workflows with display callback support.

import copy
import uuid
from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from langgraph.graph import StateGraph

from ...agent import Agent
from ..state import WorkflowState, WorkflowStateManager


class WorkflowExecutor:
    """
    Unified workflow executor with display callback and single-agent execution support.
    
    Merged functionality from WorkflowOrchestrator for simpler architecture.
    """
    
    def execute_workflow(self, workflow: StateGraph, 
                        display_callback: Optional[Callable] = None,
                        initial_state: Optional[WorkflowState] = None) -> WorkflowState:
        """
        Execute a compiled workflow with optional real-time display updates.
        
        Args:
            workflow: Compiled LangGraph workflow
            display_callback: Optional callback(msg, msg_id) for displaying messages
            initial_state: Workflow state (user message should already be added)
            
        Returns:
            Final state after workflow execution (may contain pending_interrupts in metadata)
        """
        if initial_state is None:
            initial_state = st.session_state.workflow_state
        
        # Deep copy to avoid modifying the original workflow_state
        state = copy.deepcopy(initial_state)
        if "metadata" not in state:
            state["metadata"] = {}
        
        configurable = {}
        if "configurable" in state.get("metadata", {}):
            configurable.update(state["metadata"]["configurable"])
        if "thread_id" not in configurable:
            configurable["thread_id"] = str(uuid.uuid4())
        
        state["metadata"]["workflow_thread_id"] = configurable["thread_id"]
        workflow_config = {"configurable": configurable}
        
        # Setup display callback with deduplication if provided
        if display_callback:
            display_wrapper = self._create_display_wrapper(display_callback, initial_state)
            return self._execute_streaming(workflow, state, display_wrapper, workflow_config)
        
        return self._execute_invoke(workflow, state, workflow_config)
    
    def _create_display_wrapper(self, callback: Callable, initial_state: WorkflowState) -> Callable:
        """
        Create display callback wrapper with message deduplication.
        
        Prevents re-displaying old messages when a new question is asked.
        """
        last_user_msg_id = None
        workflow_messages = initial_state.get("messages", [])
        for msg in reversed(workflow_messages):
            if msg.get("role") == "user" and msg.get("id"):
                last_user_msg_id = msg.get("id")
                break
        
        # Track which messages have already been displayed to prevent duplicates
        display_sections = st.session_state.get("display_sections", [])
        displayed_message_ids = {s.get("message_id") for s in display_sections if s.get("message_id")}
        
        def wrapper(state: WorkflowState):
            """Display callback wrapper with deduplication logic."""
            if not state or "messages" not in state:
                return
            
            # Only process messages that come after the last user message
            found_last_user = last_user_msg_id is None
            for msg in state["messages"]:
                msg_id = msg.get("id")
                
                # Track when we've reached the last user message
                if last_user_msg_id and msg_id == last_user_msg_id:
                    found_last_user = True
                    continue
                
                # Only process messages after the last user message
                if not found_last_user:
                    continue
                
                # Skip if message has already been displayed
                if msg_id and msg_id in displayed_message_ids:
                    continue
                
                # Use display callback if provided
                callback(msg, msg_id)
                # Mark as displayed
                if msg_id:
                    displayed_message_ids.add(msg_id)
        
        return wrapper
    
    def execute_agent(self, agent: Agent, prompt: str,
                           llm_client: Any, config: Any,
                           file_messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute a single agent (non-workflow mode).
        
        Note: HITL is not supported for single agents.
        
        Args:
            agent: Agent configuration
            prompt: User's prompt
            llm_client: LLM client instance
            config: UI configuration (for stream setting)
            file_messages: Optional file messages
            
        Returns:
            Dict with 'role', 'content', 'agent', and optionally 'stream'
        """
        from .registry import ExecutorRegistry
        
        conversation_messages = st.session_state.workflow_state.get("messages", [])
        executor = ExecutorRegistry().get_or_create(agent, executor_type="single_agent")
        
        if agent.type == "response":
            response = executor.execute_agent(
                llm_client, prompt,
                stream=config.stream if config else True,
                file_messages=file_messages,
                messages=conversation_messages
            )
            return response
        else:
            response = executor.execute_agent(
                llm_client, prompt,
                messages=conversation_messages
            )
            return response
    
    def _execute_invoke(self, workflow: StateGraph, initial_state: WorkflowState, 
                       config: Dict[str, Any]) -> WorkflowState:
        """Execute workflow synchronously using invoke() method."""
        final_state = workflow.invoke(initial_state, config=config)
        WorkflowStateManager.preserve_hitl_metadata(initial_state, final_state)
        return final_state
    
    def _execute_streaming(self, workflow: StateGraph, initial_state: WorkflowState, 
                          display_callback: Callable, config: Dict[str, Any]) -> WorkflowState:
        """Execute workflow using stream() method with real-time display updates."""
        accumulated_state = initial_state.copy()
        
        for node_output in workflow.stream(initial_state, config=config):
            for node_name, state_update in node_output.items():
                # Apply state update FIRST to get latest metadata
                if isinstance(state_update, dict):
                    self._apply_state_update(accumulated_state, state_update)
                
                # Check if interrupt was detected AFTER applying update
                if self._has_interrupt(accumulated_state, node_name, state_update):
                    return accumulated_state
                
                # Display callback wrapper handles deduplication internally
                display_callback(accumulated_state)
        
        # Final check for interrupts before completing
        if "pending_interrupts" in accumulated_state.get("metadata", {}):
            return accumulated_state
        
        return accumulated_state
    
    def _apply_state_update(self, accumulated_state: WorkflowState, state_update: Dict[str, Any]) -> None:
        """Apply state update to accumulated state, handling reducers manually."""
        if "messages" in state_update:
            accumulated_state["messages"] = accumulated_state.get("messages", []) + state_update["messages"]
        
        if "metadata" in state_update:
            accumulated_state["metadata"] = WorkflowStateManager.merge_metadata(
                accumulated_state.get("metadata", {}),
                state_update["metadata"]
            )
        
        if "agent_outputs" in state_update:
            existing_outputs = accumulated_state.get("agent_outputs", {})
            accumulated_state["agent_outputs"] = {**existing_outputs, **state_update["agent_outputs"]}
        
        if "current_agent" in state_update and state_update["current_agent"] is not None:
            accumulated_state["current_agent"] = state_update["current_agent"]
    
    def _has_interrupt(self, accumulated_state: WorkflowState, node_name: str, state_update: Any) -> bool:
        """Check if state contains interrupt."""
        has_pending_interrupts = "pending_interrupts" in accumulated_state.get("metadata", {})
        is_interrupt_node = node_name == "__interrupt__"
        has_interrupt_in_update = isinstance(state_update, dict) and "__interrupt__" in state_update
        
        return (is_interrupt_node or has_interrupt_in_update) and has_pending_interrupts
    
