from typing import Any, Dict, List, Optional, TypedDict
from typing_extensions import Annotated
import operator

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
    - metadata: operator.or_ merges dictionaries
    """
    messages: Annotated[List[Dict[str, Any]], operator.add]
    current_agent: Annotated[Optional[str], lambda x, y: y if y is not None else x]
    agent_outputs: Annotated[Dict[str, Any], operator.or_]
    files: Annotated[List[Dict[str, Any]], operator.add]
    metadata: Annotated[Dict[str, Any], operator.or_]

# Helper functions to work with WorkflowState

def create_initial_state(messages: Optional[List[Dict[str, Any]]] = None, current_agent: Optional[str] = None) -> WorkflowState:
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
        "timestamp": None
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
