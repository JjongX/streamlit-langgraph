from typing import Any, Dict, List, Optional, TypedDict
from typing_extensions import Annotated
import operator

# Define merge_metadata before WorkflowState since it's used as a reducer in the class definition
def merge_metadata(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge metadata dictionaries, preserving all keys from both.
    This ensures that pending_interrupts and other HITL state is not lost.
    """
    result = x.copy() if x else {}
    if y:
        for key, value in y.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Deep merge for nested dicts (e.g., pending_interrupts, executors)
                result[key] = {**result[key], **value}
            else:
                # Overwrite for non-dict values or new keys
                result[key] = value
    return result

class WorkflowState(TypedDict):
    """
    LangGraph-compatible state dictionary for workflow execution.
    
    This state maintains conversation history and workflow execution metadata
    while being compatible with LangGraph's state management requirements.
    
    Reducer functions handle concurrent updates during parallel execution:
    - messages: operator.add concatenates lists
    - current_agent: lambda takes latest non-None value
    - agent_outputs: operator.or_ merges dictionaries (Python 3.9+)
    - files: operator.add concatenates lists
    - metadata: merge_metadata merges dictionaries while preserving all keys
    """
    messages: Annotated[List[Dict[str, Any]], operator.add]
    current_agent: Annotated[Optional[str], lambda x, y: y if y is not None else x]
    agent_outputs: Annotated[Dict[str, Any], operator.or_]
    files: Annotated[List[Dict[str, Any]], operator.add]
    metadata: Annotated[Dict[str, Any], merge_metadata]

def create_initial_state(messages: Optional[List[Dict[str, Any]]] = None, current_agent: Optional[str] = None) -> WorkflowState:
    """Create an initial WorkflowState with default values."""
    return WorkflowState(
        messages=messages or [],
        current_agent=current_agent,
        agent_outputs={},
        files=[],
        metadata={}
    )


# Human-in-the-loop state management functions
def set_pending_interrupt(state: WorkflowState, agent_name: str, interrupt_data: Dict[str, Any], executor_key: str) -> Dict[str, Any]:
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

def get_pending_interrupts(state: WorkflowState) -> Dict[str, Dict[str, Any]]:
    """Get all pending interrupts from workflow state."""
    return state.get("metadata", {}).get("pending_interrupts", {})

def clear_pending_interrupt(state: WorkflowState, executor_key: str) -> Dict[str, Any]:
    """Clear a specific pending interrupt from workflow state."""
    if "pending_interrupts" not in state.get("metadata", {}):
        return {"metadata": state.get("metadata", {})}
    
    updated_metadata = state["metadata"].copy()
    updated_metadata["pending_interrupts"] = updated_metadata["pending_interrupts"].copy()
    updated_metadata["pending_interrupts"].pop(executor_key, None)
    return {"metadata": updated_metadata}

def set_hitl_decision(state: WorkflowState, executor_key: str, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
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

def get_hitl_decision(state: WorkflowState, executor_key: str) -> Optional[List[Dict[str, Any]]]:
    """Get HITL decisions for an interrupt."""
    decisions_key = f"{executor_key}_decisions"
    return state.get("metadata", {}).get("hitl_decisions", {}).get(decisions_key)
