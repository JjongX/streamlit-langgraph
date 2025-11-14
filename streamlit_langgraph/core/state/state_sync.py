# Synchronizes between WorkflowState and Streamlit's session_state for UI rendering.

from typing import Any, Dict, List
import streamlit as st

from .state_schema import WorkflowStateManager
from ...workflow.agent_nodes.base import create_message_with_id


class StateSynchronizer:
    
    def update_workflow_state(self, updates: Dict[str, Any]) -> None:
        """
        Update workflow state with new data.
        
        Args:
            updates: Dictionary of state updates to apply
        """
        workflow_state = st.session_state.workflow_state
        
        # Apply updates using WorkflowState reducers
        if "messages" in updates:
            workflow_state["messages"].extend(updates["messages"])
        
        if "metadata" in updates:
            workflow_state["metadata"] = WorkflowStateManager.merge_metadata(
                workflow_state.get("metadata", {}),
                updates["metadata"]
            )
        
        if "agent_outputs" in updates:
            workflow_state["agent_outputs"].update(updates["agent_outputs"])
        
        if "current_agent" in updates and updates["current_agent"] is not None:
            workflow_state["current_agent"] = updates["current_agent"]
        
        if "files" in updates:
            workflow_state["files"].extend(updates["files"])
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to workflow state with unique ID."""
        self.update_workflow_state({
            "messages": [create_message_with_id("user", content, None)]
        })
    
    def add_assistant_message(self, content: str, agent_name: str) -> None:
        """Add an assistant message to workflow state with unique ID."""
        self.update_workflow_state({
            "messages": [create_message_with_id("assistant", content, agent_name)],
            "agent_outputs": {agent_name: content},
            "current_agent": agent_name
        })
    
    def set_pending_interrupt(
        self, 
        agent_name: str, 
        interrupt_data: Dict[str, Any], 
        executor_key: str
    ) -> None:
        """Store a pending interrupt in workflow state."""
        interrupt_update = WorkflowStateManager.set_pending_interrupt(
            st.session_state.workflow_state, agent_name, interrupt_data, executor_key
        )
        self.update_workflow_state(interrupt_update)
    
    def clear_pending_interrupt(self, executor_key: str) -> None:
        """Clear a pending interrupt from workflow state."""
        clear_update = WorkflowStateManager.clear_pending_interrupt(
            st.session_state.workflow_state, executor_key
        )
        self.update_workflow_state(clear_update)
    
    def set_hitl_decision(self, executor_key: str, decisions: List[Dict[str, Any]]) -> None:
        """Store HITL decisions in workflow state."""
        decision_update = WorkflowStateManager.set_hitl_decision(
            st.session_state.workflow_state, executor_key, decisions
        )
        self.update_workflow_state(decision_update)
    
    def clear_hitl_state(self) -> None:
        """Clear all HITL-related state."""
        workflow_state = st.session_state.workflow_state
        if "metadata" in workflow_state:
            if "pending_interrupts" in workflow_state["metadata"]:
                workflow_state["metadata"]["pending_interrupts"] = {}
            if "hitl_decisions" in workflow_state["metadata"]:
                workflow_state["metadata"]["hitl_decisions"] = {}
        st.session_state.agent_executors = {}

