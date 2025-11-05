from typing import Any, Callable, Dict, Optional
import uuid

from langgraph.graph import StateGraph

from .state import WorkflowState, create_initial_state, merge_metadata

class WorkflowExecutor:
    """
    Handles execution of compiled workflows with Streamlit integration.
    """
    
    # HITL-related metadata keys that need special preservation
    HITL_METADATA_KEYS = ["pending_interrupts", "executors", "hitl_decisions"]
    
    def execute_workflow(self, workflow: StateGraph, user_input: str, 
                        display_callback: Optional[Callable] = None,
                        config: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """
        Execute a compiled workflow with the given user input.
        
        Args:
            workflow: Compiled workflow graph
            user_input: User's input/request
            display_callback: Optional callback for displaying messages
            config: Optional configuration for workflow execution (may include thread_id for HITL)
            
        Returns:
            Final state after workflow execution (may contain pending_interrupts in metadata)
        """
        initial_state = self._create_initial_state(user_input, config)
        workflow_config = self._build_workflow_config(config)
        
        if display_callback:
            return self._execute_with_display(workflow, initial_state, display_callback, workflow_config)
        return self._execute_basic(workflow, initial_state, workflow_config)
    
    def _create_initial_state(self, user_input: str, config: Optional[Dict[str, Any]]) -> WorkflowState:
        """Create and initialize workflow state."""
        initial_state = create_initial_state(
            messages=[{"role": "user", "content": user_input}]
        )
        if config:
            initial_state["metadata"].update(config)
        return initial_state
    
    def _build_workflow_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build workflow configuration with thread_id for checkpointing."""
        configurable = {}
        if config and "configurable" in config:
            configurable.update(config["configurable"])
        
        if "thread_id" not in configurable:
            configurable["thread_id"] = str(uuid.uuid4())
        
        return {"configurable": configurable}
    
    def _execute_basic(self, workflow: StateGraph, initial_state: WorkflowState, 
                      config: Dict[str, Any]) -> WorkflowState:
        """Execute workflow without display callbacks."""
        final_state = workflow.invoke(initial_state, config=config)
        
        # Preserve HITL metadata from initial state
        self._preserve_hitl_metadata(initial_state, final_state)
        
        # Add completion message if workflow ended
        if final_state.get("messages"):
            last_agent = final_state["messages"][-1].get("agent")
            if last_agent in ["END", "__end__"]:
                final_state["messages"].append({
                    "role": "system",
                    "content": "Workflow completed successfully.",
                    "agent": None,
                    "timestamp": None
                })
        
        return final_state
    
    def _preserve_hitl_metadata(self, initial_state: WorkflowState, final_state: WorkflowState) -> None:
        """Preserve HITL-related metadata from initial state to final state."""
        initial_metadata = initial_state.get("metadata", {})
        if not initial_metadata:
            return
        
        if "metadata" not in final_state:
            final_state["metadata"] = {}
        
        final_metadata = final_state["metadata"]
        for key in self.HITL_METADATA_KEYS:
            if key not in initial_metadata:
                continue
            
            if key not in final_metadata: 
                final_metadata[key] = initial_metadata[key]
            elif isinstance(initial_metadata[key], dict) and isinstance(final_metadata[key], dict):
                final_metadata[key] = {**final_metadata[key], **initial_metadata[key]}
            else:
                final_metadata[key] = initial_metadata[key]
    
    def _execute_with_display(self, workflow: StateGraph, initial_state: WorkflowState, 
                             display_callback: Callable, config: Dict[str, Any]) -> WorkflowState:
        """Execute workflow with real-time display updates."""
        accumulated_state = initial_state.copy()
        
        for node_output in workflow.stream(initial_state, config=config):
            for node_name, state_update in node_output.items():
                # Apply state update FIRST to get latest metadata
                if isinstance(state_update, dict):
                    self._apply_state_update(accumulated_state, state_update)
                
                # Check if interrupt was detected AFTER applying update
                has_pending_interrupts = "pending_interrupts" in accumulated_state.get("metadata", {})
                is_interrupt_node = node_name == "__interrupt__"
                has_interrupt_in_update = isinstance(state_update, dict) and "__interrupt__" in state_update
                
                if (is_interrupt_node or has_interrupt_in_update) and has_pending_interrupts:
                    return accumulated_state
                
                display_callback(accumulated_state)
        
        # Final check for interrupts before completing
        if "pending_interrupts" in accumulated_state.get("metadata", {}):
            return accumulated_state
        
        # Add completion message if workflow ended
        if accumulated_state.get("messages"):
            last_agent = accumulated_state["messages"][-1].get("agent")
            if last_agent in ["END", "__end__"]:
                accumulated_state["messages"].append({
                    "role": "system",
                    "content": "Workflow completed successfully.",
                    "agent": None,
                    "timestamp": None
                })
        
        return accumulated_state
    
    def _apply_state_update(self, accumulated_state: WorkflowState, state_update: Dict[str, Any]) -> None:
        """Apply state update to accumulated state, handling reducers manually."""
        if "messages" in state_update:
            accumulated_state["messages"] = accumulated_state.get("messages", []) + state_update["messages"]
        
        if "metadata" in state_update:
            accumulated_state["metadata"] = merge_metadata(
                accumulated_state.get("metadata", {}),
                state_update["metadata"]
            )
        
        if "agent_outputs" in state_update:
            existing_outputs = accumulated_state.get("agent_outputs", {})
            accumulated_state["agent_outputs"] = {**existing_outputs, **state_update["agent_outputs"]}
        
        if "current_agent" in state_update and state_update["current_agent"] is not None:
            accumulated_state["current_agent"] = state_update["current_agent"]
