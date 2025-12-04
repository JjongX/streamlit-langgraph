# Single source of truth for all agents and workflows.

import operator
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from typing_extensions import Annotated


class WorkflowStateManager:
    """Manager class for workflow state operations and HITL state management."""
    
    @staticmethod
    def merge_metadata(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
        """Merge metadata dictionaries, preserving all keys. Used as a reducer in WorkflowState."""
        result = x.copy() if x else {}
        if y:
            for key, value in y.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = {**result[key], **value}
                else:
                    result[key] = value
        return result
    
    @staticmethod
    def create_initial_state(messages: Optional[List[Dict[str, Any]]] = None, current_agent: Optional[str] = None) -> "WorkflowState":
        """Create an initial WorkflowState with default values."""
        return WorkflowState(
            messages=messages or [],
            current_agent=current_agent,
            agent_outputs={},
            files=[],
            metadata={}
        )
    
    @staticmethod
    def set_pending_interrupt(state: "WorkflowState", agent_name: str, interrupt_data: Dict[str, Any], executor_key: str) -> Dict[str, Any]:
        """Store a pending interrupt in workflow state metadata."""
        if "pending_interrupts" not in state.get("metadata", {}):
            updated_metadata = state.get("metadata", {}).copy()
            updated_metadata["pending_interrupts"] = {}
        else:
            updated_metadata = state["metadata"].copy()
            updated_metadata["pending_interrupts"] = updated_metadata["pending_interrupts"].copy()
        
        updated_metadata["pending_interrupts"][executor_key] = {
            "agent": agent_name,
            "__interrupt__": interrupt_data.get("__interrupt__"),
            "thread_id": interrupt_data.get("thread_id"),
            "config": interrupt_data.get("config"),
            "executor_key": executor_key
        }
        return {"metadata": updated_metadata}
    
    @staticmethod
    def get_pending_interrupts(state: "WorkflowState") -> Dict[str, Dict[str, Any]]:
        """Get all pending interrupts from workflow state."""
        return state.get("metadata", {}).get("pending_interrupts", {})
    
    @staticmethod
    def clear_pending_interrupt(state: "WorkflowState", executor_key: str) -> Dict[str, Any]:
        """Clear a specific pending interrupt from workflow state."""
        if "pending_interrupts" not in state.get("metadata", {}):
            return {"metadata": state.get("metadata", {})}
        
        updated_metadata = state["metadata"].copy()
        updated_metadata["pending_interrupts"] = updated_metadata["pending_interrupts"].copy()
        updated_metadata["pending_interrupts"].pop(executor_key, None)
        return {"metadata": updated_metadata}
    
    @staticmethod
    def set_hitl_decision(state: "WorkflowState", executor_key: str, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store HITL decisions for an interrupt."""
        decisions_key = f"{executor_key}_decisions"
        if "hitl_decisions" not in state.get("metadata", {}):
            updated_metadata = state.get("metadata", {}).copy()
            updated_metadata["hitl_decisions"] = {}
        else:
            updated_metadata = state["metadata"].copy()
            updated_metadata["hitl_decisions"] = updated_metadata["hitl_decisions"].copy()
        
        updated_metadata["hitl_decisions"][decisions_key] = decisions
        return {"metadata": updated_metadata}
    
    @staticmethod
    def get_hitl_decision(state: "WorkflowState", executor_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get HITL decisions for an interrupt."""
        decisions_key = f"{executor_key}_decisions"
        return state.get("metadata", {}).get("hitl_decisions", {}).get(decisions_key)
    
    @staticmethod
    def preserve_hitl_metadata(initial_state: "WorkflowState", final_state: "WorkflowState") -> None:
        """
        Preserve HITL-related metadata from initial state to final state.
        
        This ensures that HITL state (pending_interrupts, executors, hitl_decisions)
        is maintained across workflow executions when using invoke() method.
        """
        HITL_METADATA_KEYS = ["pending_interrupts", "executors", "hitl_decisions"]
        initial_metadata = initial_state.get("metadata", {})
        if not initial_metadata:
            return
        
        if "metadata" not in final_state:
            final_state["metadata"] = {}
        
        final_metadata = final_state["metadata"]
        for key in HITL_METADATA_KEYS:
            if key not in initial_metadata:
                continue
            
            if key not in final_metadata: 
                final_metadata[key] = initial_metadata[key]
            elif isinstance(initial_metadata[key], dict) and isinstance(final_metadata[key], dict):
                final_metadata[key] = {**final_metadata[key], **initial_metadata[key]}
            else:
                final_metadata[key] = initial_metadata[key]
    
    @staticmethod
    def preserve_display_sections(initial_state: "WorkflowState", final_state: "WorkflowState") -> None:
        """Merge display_sections from initial_state into final_state to preserve UI state."""
        initial_metadata = initial_state.get("metadata", {})
        initial_display_sections = initial_metadata.get("display_sections", [])
        
        if not initial_display_sections:
            return
        
        if "metadata" not in final_state:
            final_state["metadata"] = {}
        
        final_metadata = final_state["metadata"]
        
        if "display_sections" not in final_metadata:
            final_metadata["display_sections"] = []
        
        existing_message_ids = {
            s.get("message_id") for s in final_metadata["display_sections"] 
            if s.get("message_id")
        }
        
        for section in initial_display_sections:
            msg_id = section.get("message_id")
            if msg_id and msg_id not in existing_message_ids:
                final_metadata["display_sections"].append(section)
            elif not msg_id:
                # Check for duplicates by comparing role and content
                if not WorkflowStateManager._section_exists(section, final_metadata["display_sections"]):
                    final_metadata["display_sections"].append(section)
    
    @staticmethod
    def _section_exists(section: Dict[str, Any], existing_sections: List[Dict[str, Any]]) -> bool:
        """Check if a section already exists by comparing role and content."""
        section_role = section.get("role")
        section_content = WorkflowStateManager._extract_text_content(section)
        
        for existing_section in existing_sections:
            if existing_section.get("role") != section_role:
                continue
            existing_content = WorkflowStateManager._extract_text_content(existing_section)
            if existing_content == section_content:
                return True
        return False
    
    @staticmethod
    def _extract_text_content(section: Dict[str, Any]) -> str:
        """Extract text content from the first text block in a section."""
        section_blocks = section.get("blocks", [])
        for block in section_blocks:
            if block.get("category") == "text":
                return block.get("content", "")
        return ""
    
    @staticmethod
    def get_or_create_workflow_config(state: "WorkflowState", executor_key: str) -> Tuple[Dict[str, Any], str]:
        """Get or create workflow thread_id and config."""
        if "metadata" not in state:
            state["metadata"] = {}
        if "executors" not in state["metadata"]:
            state["metadata"]["executors"] = {}
        
        workflow_thread_id = state.get("metadata", {}).get("workflow_thread_id")
        if not workflow_thread_id:
            workflow_thread_id = str(uuid.uuid4())
            state["metadata"]["workflow_thread_id"] = workflow_thread_id
        
        state["metadata"]["executors"][executor_key] = {"thread_id": workflow_thread_id}
        config = {"configurable": {"thread_id": workflow_thread_id}}
        return config, workflow_thread_id

class WorkflowState(TypedDict):
    """
    LangGraph-compatible state dictionary for workflow execution.
    
    This state maintains conversation history and workflow execution metadata
    while being compatible with LangGraph's state management requirements.
    
    Reducer functions handle concurrent updates during parallel execution:
    - messages: `operator.add` concatenates lists
    - current_agent: lambda takes latest non-None value
    - agent_outputs: `operator.or_` merges dictionaries
    - files: `operator.add` concatenates lists
    - metadata: WorkflowStateManager.merge_metadata merges dictionaries while preserving all keys
    """
    messages: Annotated[List[Dict[str, Any]], operator.add]
    current_agent: Annotated[Optional[str], lambda x, y: y if y is not None else x]
    agent_outputs: Annotated[Dict[str, Any], operator.or_]
    files: Annotated[List[Dict[str, Any]], operator.add]
    metadata: Annotated[Dict[str, Any], WorkflowStateManager.merge_metadata]

