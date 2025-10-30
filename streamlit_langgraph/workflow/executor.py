from typing import Any, Callable, Dict, Optional

from langgraph.graph import StateGraph

from .state import WorkflowState, create_initial_state

class WorkflowExecutor:
    """
    Handles execution of compiled workflows with Streamlit integration.
    """
    
    def execute_workflow(self, workflow: StateGraph, user_input: str, 
                        display_callback: Optional[Callable] = None,
                        config: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """
        Execute a compiled workflow with the given user input.
        
        Args:
            workflow (StateGraph): Compiled workflow graph
            user_input (str): User's input/request
            display_callback (Callable): Optional callback for displaying messages
            config (Dict): Optional configuration for workflow execution
            
        Returns:
            WorkflowState: Final state after workflow execution
        """
        # Initialize workflow state
        initial_state = create_initial_state(
            messages=[{"role": "user", "content": user_input}]
        )    
        if config:
            initial_state["metadata"].update(config)
    
        if display_callback:
            return self._execute_with_display(workflow, initial_state, display_callback)
        else:
            return self._execute_basic(workflow, initial_state)

    def _execute_basic(self, workflow: StateGraph, initial_state: WorkflowState) -> WorkflowState:
        """Execute workflow without display callbacks."""
        final_state = workflow.invoke(initial_state)

        last_agent = final_state["messages"][-1].get("agent") if final_state["messages"] else None
        if last_agent in ["END", "__end__"]:
            final_state["messages"].append({
                "role": "system",
                "content": "Workflow completed successfully.",
                "agent": None,
                "timestamp": None
            })
        return final_state
    
    def _execute_with_display(self, workflow: StateGraph, initial_state: WorkflowState, 
                            display_callback: Callable) -> WorkflowState:
        """Execute workflow with real-time display updates."""
        accumulated_state = initial_state.copy()
        for node_output in workflow.stream(initial_state):
            for node_name, state_update in node_output.items():
                if isinstance(state_update, dict):
                    # Apply reducers manually when accumulating state for display
                    if "messages" in state_update:
                        accumulated_state["messages"] = accumulated_state.get("messages", []) + state_update["messages"]
                    if "metadata" in state_update:
                        accumulated_state["metadata"] = {**accumulated_state.get("metadata", {}), **state_update["metadata"]}
                    if "agent_outputs" in state_update:
                        accumulated_state["agent_outputs"] = {**accumulated_state.get("agent_outputs", {}), **state_update["agent_outputs"]}
                    if "current_agent" in state_update:
                        if state_update["current_agent"] is not None:
                            accumulated_state["current_agent"] = state_update["current_agent"]
                if display_callback:
                    display_callback(accumulated_state)

        last_agent = accumulated_state["messages"][-1].get("agent") if accumulated_state["messages"] else None
        if last_agent in ["END", "__end__"]:
            accumulated_state["messages"].append({
                "role": "system",
                "content": "Workflow completed successfully.",
                "agent": None,
                "timestamp": None
            })
        return accumulated_state
