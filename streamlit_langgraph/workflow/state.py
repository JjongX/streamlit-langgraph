from typing import Dict, List, Optional, Any, TypedDict

class WorkflowState(TypedDict):
    """
    LangGraph-compatible state dictionary for workflow execution.
    
    This state maintains conversation history and workflow execution metadata
    while being compatible with LangGraph's state management requirements.
    """
    messages: List[Dict[str, Any]]
    current_agent: Optional[str]
    agent_outputs: Dict[str, Any]
    files: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Helper functions to work with WorkflowState

def create_initial_state(
    messages: Optional[List[Dict[str, Any]]] = None,
    current_agent: Optional[str] = None
) -> WorkflowState:
    """Create an initial WorkflowState with default values."""
    return WorkflowState(
        messages=messages or [],
        current_agent=current_agent,
        agent_outputs={},
        files=[],
        metadata={}
    )

def get_current_agent(state: WorkflowState) -> Optional[str]:
    """Get the current agent from state."""
    return state.get("current_agent")

def set_current_agent(state: WorkflowState, agent_name: str) -> Dict[str, Any]:
    """Return state update to set current agent."""
    return {"current_agent": agent_name}

def add_message(state: WorkflowState, role: str, content: str, agent: Optional[str] = None) -> Dict[str, Any]:
    """Return state update to add a message."""
    new_message = {
        "role": role,
        "content": content,
        "agent": agent,
        "timestamp": None  # Can add timestamp if needed
    }
    return {"messages": state["messages"] + [new_message]}

def set_agent_output(state: WorkflowState, agent_name: str, output: Any) -> Dict[str, Any]:
    """Return state update to set agent output."""
    updated_outputs = state["agent_outputs"].copy()
    updated_outputs[agent_name] = output
    return {"agent_outputs": updated_outputs}

def get_agent_output(state: WorkflowState, agent_name: str) -> Any:
    """Get agent output from state."""
    return state["agent_outputs"].get(agent_name)

def set_metadata(state: WorkflowState, key: str, value: Any) -> Dict[str, Any]:
    """Return state update to set metadata."""
    updated_metadata = state["metadata"].copy()
    updated_metadata[key] = value
    return {"metadata": updated_metadata}

def get_metadata(state: WorkflowState, key: str, default: Any = None) -> Any:
    """Get metadata value from state."""
    return state["metadata"].get(key, default)

def set_delegated_agent(state: WorkflowState, agent_name: str) -> Dict[str, Any]:
    """Return state update to set delegated agent for supervisor workflows."""
    return set_metadata(state, "delegated_agent", agent_name)

def get_delegated_agent(state: WorkflowState) -> Optional[str]:
    """Get the delegated agent from state."""
    return get_metadata(state, "delegated_agent")

def is_workflow_complete(state: WorkflowState) -> bool:
    """Check if workflow is marked as complete."""
    metadata = state["metadata"]
    return (metadata.get("delegated_agent") == "COMPLETE" or
            metadata.get("task_complete") == True or
            metadata.get("workflow_complete") == True)

def mark_workflow_complete(state: WorkflowState) -> Dict[str, Any]:
    """Return state update to mark workflow as complete."""
    updated_metadata = state["metadata"].copy()
    updated_metadata["workflow_complete"] = True
    updated_metadata["delegated_agent"] = "COMPLETE"
    return {"metadata": updated_metadata}

def get_workflow_iteration_count(state: WorkflowState) -> int:
    """Get current workflow iteration count."""
    return get_metadata(state, "iteration_count", 0)

def increment_iteration_count(state: WorkflowState) -> Dict[str, Any]:
    """Return state update to increment iteration count."""
    count = get_workflow_iteration_count(state) + 1
    return set_metadata(state, "iteration_count", count)

def set_max_iterations(state: WorkflowState, max_iter: int) -> Dict[str, Any]:
    """Return state update to set maximum iterations."""
    return set_metadata(state, "max_iterations", max_iter)

def get_max_iterations(state: WorkflowState) -> int:
    """Get maximum allowed iterations."""
    return get_metadata(state, "max_iterations", 5)

def is_max_iterations_reached(state: WorkflowState) -> bool:
    """Check if maximum iterations have been reached."""
    return get_workflow_iteration_count(state) >= get_max_iterations(state)
