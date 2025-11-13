# Workflow executor for compiled workflows.

import copy
import uuid
from typing import Any, Callable, Dict, Optional

from langgraph.graph import StateGraph

from ..state import WorkflowState, WorkflowStateManager


class WorkflowExecutor:
    
    def execute_workflow(self, workflow: StateGraph, 
                        display_callback: Optional[Callable] = None,
                        config: Optional[Dict[str, Any]] = None,
                        initial_state: Optional[WorkflowState] = None) -> WorkflowState:
        """
        Execute a compiled workflow with the given initial state.
        
        Args:
            workflow: Compiled workflow graph
            display_callback: Optional callback for displaying messages
            config: Optional configuration for workflow execution (may include thread_id for HITL)
            initial_state: Required existing workflow state to use (user message should already be added)
            
        Returns:
            Final state after workflow execution (may contain pending_interrupts in metadata)
        """
        # Require initial_state to be provided (user message already added)
        if initial_state is None:
            raise ValueError("initial_state is required. User message should already be added to workflow_state.")
        
        # Deep copy to avoid modifying the original workflow_state
        initial_state = copy.deepcopy(initial_state)
        if "metadata" not in initial_state:
            initial_state["metadata"] = {}
        if config:
            initial_state["metadata"].update(config)

        # Build workflow configuration with thread_id for checkpointing
        configurable = {}
        if config and "configurable" in config:
            configurable.update(config["configurable"])
        if "thread_id" not in configurable:
            configurable["thread_id"] = str(uuid.uuid4())
        
        # Store workflow thread_id in state metadata so nodes can access it
        initial_state["metadata"]["workflow_thread_id"] = configurable["thread_id"]
        
        workflow_config = {"configurable": configurable}
        
        if display_callback:
            return self._execute_streaming(workflow, initial_state, display_callback, workflow_config)
        return self._execute_invoke(workflow, initial_state, workflow_config)
    
    def _execute_invoke(self, workflow: StateGraph, initial_state: WorkflowState, 
                       config: Dict[str, Any]) -> WorkflowState:
        """Execute workflow synchronously using invoke() method."""
        final_state = workflow.invoke(initial_state, config=config)
        
        # Preserve HITL metadata from initial state
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
    
