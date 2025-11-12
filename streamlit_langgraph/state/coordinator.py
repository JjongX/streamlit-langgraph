# Centralized state coordinator for managing WorkflowState and session_state synchronization


import uuid
from typing import Any, Dict, List
import streamlit as st

from .workflow_state import WorkflowStateManager

class StateCoordinator:
    """
    Centralized coordinator for managing WorkflowState and session_state synchronization.
    
    This class ensures that WorkflowState remains the single source of truth while
    session_state is kept in sync for UI rendering purposes.
    
    Note: This class assumes session_state has been initialized (typically by LangGraphChat).
    """
    
    def __init__(self):
        """Initialize the coordinator."""
        pass
    
    def update_workflow_state(self, updates: Dict[str, Any], auto_sync: bool = True) -> None:
        """
        Update workflow state with new data.
        
        Args:
            updates: Dictionary of state updates to apply
            auto_sync: Whether to automatically sync to session_state (default: True)
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
        
        if auto_sync:
            self.sync_to_session_state()
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to workflow state with unique ID."""
        self.update_workflow_state({
            "messages": [{"id": str(uuid.uuid4()), "role": "user", "content": content, "agent": None}]
        })
    
    def add_assistant_message(self, content: str, agent_name: str) -> None:
        """Add an assistant message to workflow state with unique ID."""
        self.update_workflow_state({
            "messages": [{"id": str(uuid.uuid4()), "role": "assistant", "content": content, "agent": agent_name}],
            "agent_outputs": {agent_name: content},
            "current_agent": agent_name
        })
    
    def sync_to_session_state(self) -> None:
        """
        Sync workflow_state to session_state for UI rendering.
        
        This is the ONLY method that writes to session_state.messages.
        Uses ID-based duplicate detection for simple and reliable tracking.
        """
        workflow_messages = st.session_state.workflow_state.get("messages", [])
        session_messages = st.session_state.messages
        
        # Find messages in workflow_state that aren't in session_state yet
        new_messages = self._find_new_messages(workflow_messages, session_messages)
        st.session_state.messages.extend(new_messages)
    
    def _find_new_messages(
        self,
        workflow_messages: List[Dict[str, Any]],
        session_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find messages in workflow_messages that aren't in session_messages.
        
        Uses ID-based comparison for simple and reliable duplicate detection.
        Also checks content+role as fallback for messages without IDs.
        """
        # Create a set of message IDs for fast lookup
        session_ids = {msg.get("id") for msg in session_messages if msg.get("id")}
        
        # Also create a set of (role, content) tuples for messages without IDs (fallback)
        session_content_signatures = {
            (msg.get("role"), msg.get("content", ""))
            for msg in session_messages
            if not msg.get("id")  # Only for messages without IDs
        }
        
        new_messages = []
        for msg in workflow_messages:
            msg_id = msg.get("id")
            
            # Check by ID first (most reliable)
            if msg_id and msg_id in session_ids:
                continue
            
            # Fallback: check by content+role for messages without IDs
            if not msg_id:
                content_sig = (msg.get("role"), msg.get("content", ""))
                if content_sig in session_content_signatures:
                    continue
            
            # Convert to session format (copy message with ID)
            session_msg = {
                "id": msg_id if msg_id else str(uuid.uuid4()),  # Ensure all messages have IDs
                "role": msg.get("role"),
                "content": msg.get("content", "")
            }
            if "agent" in msg:
                session_msg["agent"] = msg["agent"]
            
            new_messages.append(session_msg)
        
        return new_messages
    
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
        self.update_workflow_state(interrupt_update, auto_sync=False)
    
    def clear_pending_interrupt(self, executor_key: str) -> None:
        """Clear a pending interrupt from workflow state."""
        clear_update = WorkflowStateManager.clear_pending_interrupt(
            st.session_state.workflow_state, executor_key
        )
        self.update_workflow_state(clear_update, auto_sync=False)
    
    def set_hitl_decision(self, executor_key: str, decisions: List[Dict[str, Any]]) -> None:
        """Store HITL decisions in workflow state."""
        decision_update = WorkflowStateManager.set_hitl_decision(
            st.session_state.workflow_state, executor_key, decisions
        )
        self.update_workflow_state(decision_update, auto_sync=False)
    
    def clear_hitl_state(self) -> None:
        """Clear all HITL-related state."""
        workflow_state = st.session_state.workflow_state
        if "metadata" in workflow_state:
            if "pending_interrupts" in workflow_state["metadata"]:
                workflow_state["metadata"]["pending_interrupts"] = {}
            if "hitl_decisions" in workflow_state["metadata"]:
                workflow_state["metadata"]["hitl_decisions"] = {}
        st.session_state.agent_executors = {}

