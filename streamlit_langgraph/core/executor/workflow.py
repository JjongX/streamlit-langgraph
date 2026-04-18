# Workflow executor for compiled workflows with display callback support.

import copy
import uuid
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import StateGraph

from ...agent import Agent
from .registry import ExecutorRegistry
from .response_api import ResponseAPIExecutor
from ..state import WorkflowState, WorkflowStateManager


class WorkflowExecutor:
    """
    Unified workflow executor with display callback and single-agent execution support.
    
    Handles execution of compiled LangGraph workflows with optional real-time
    display updates and single-agent execution for non-workflow scenarios.
    """

    def __init__(self, executor_registry: ExecutorRegistry):
        self.executor_registry = executor_registry

    def execute_workflow(
        self,
        workflow: StateGraph,
        display_callback: Optional[Callable] = None,
        initial_state: Optional[WorkflowState] = None,
        displayed_message_ids: Optional[set] = None,
    ) -> WorkflowState:
        """
        Execute a compiled workflow with optional real-time display updates.
        
        Args:
            workflow: Compiled LangGraph workflow
            display_callback: Optional callback(msg) for displaying messages
            initial_state: Workflow state (user message should already be added)
            
        Returns:
            Final state after workflow execution (may contain pending_interrupts in metadata)
        """
        if initial_state is None:
            raise ValueError("initial_state is required for workflow execution")
        
        state = copy.deepcopy(initial_state)
        if "metadata" not in state:
            state["metadata"] = {}
        WorkflowStateManager.ensure_stream_flag(state, default_stream=True)
        
        configurable = {}
        if "configurable" in state.get("metadata", {}):
            configurable.update(state["metadata"]["configurable"])
        if "thread_id" not in configurable:
            configurable["thread_id"] = str(uuid.uuid4())
        
        state["metadata"]["workflow_thread_id"] = configurable["thread_id"]
        workflow_config = {"configurable": configurable}
        
        if display_callback:
            display_wrapper = self._create_display_wrapper(
                display_callback,
                initial_state,
                displayed_message_ids or set(),
            )
            return self._execute_streaming(workflow, state, workflow_config, display_wrapper)
        
        return self._execute_invoke(workflow, state, workflow_config)
    
    def _create_display_wrapper(
        self,
        callback: Callable,
        initial_state: WorkflowState,
        displayed_message_ids: set,
    ) -> Callable:
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
        
        def wrapper(state: WorkflowState):
            """Display callback wrapper with deduplication logic."""
            if not state or "messages" not in state:
                return
            
            found_last_user = last_user_msg_id is None
            for msg in state["messages"]:
                msg_id = msg.get("id")
                if last_user_msg_id and msg_id == last_user_msg_id:
                    found_last_user = True
                    continue
                if not found_last_user:
                    continue
                if msg_id and msg_id in displayed_message_ids:
                    continue
                callback(msg)
                if msg_id:
                    displayed_message_ids.add(msg_id)
        
        return wrapper
    
    def execute_agent(
        self,
        agent: Agent,
        prompt: str,
        llm_client: Any,
        config: Any,
        file_messages: Optional[List[Dict[str, Any]]] = None,
        vector_store_ids: Optional[List[str]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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
        conversation_messages = messages or []
        executor = self.executor_registry.get_or_create(agent, executor_type="single_agent")
        if isinstance(executor, ResponseAPIExecutor):
            response = executor.execute_agent(
                prompt,
                stream=config.stream if config else True,
                file_messages=file_messages,
                messages=conversation_messages,
                vector_store_ids=vector_store_ids,
            )
        else:
            response = executor.execute_agent(
                llm_client,
                prompt,
                stream=config.stream if config else True,
                file_messages=file_messages,
                messages=conversation_messages,
            )
        return response
    
    def _execute_invoke(
        self, workflow: StateGraph,
        initial_state: WorkflowState, config: Dict[str, Any]
    ) -> WorkflowState:
        """Execute workflow synchronously using invoke() method."""
        final_state = workflow.invoke(initial_state, config=config)
        WorkflowStateManager.preserve_hitl_metadata(initial_state, final_state)
        return final_state
    
    def _execute_streaming(
        self, workflow: StateGraph,
        initial_state: WorkflowState, config: Dict[str, Any], 
        display_callback: Callable, 
    ) -> WorkflowState:
        """Execute workflow using stream() method with real-time display updates."""
        accumulated_state = initial_state.copy()
        
        for node_output in workflow.stream(initial_state, config=config):
            for node_name, state_update in node_output.items():
                if isinstance(state_update, dict):
                    # Apply state update to accumulated state, handling reducers manually
                    WorkflowStateManager.apply_reducer_updates(accumulated_state, state_update)
                # Check if state contains interrupt
                has_pending_interrupts = "pending_interrupts" in accumulated_state.get("metadata", {})
                is_interrupt_node = node_name == "__interrupt__"
                has_interrupt_in_update = isinstance(state_update, dict) and "__interrupt__" in state_update
                if (is_interrupt_node or has_interrupt_in_update) and has_pending_interrupts:
                    return accumulated_state
                display_callback(accumulated_state)
        
        if "pending_interrupts" in accumulated_state.get("metadata", {}):
            return accumulated_state
        
        # Ensure final state is displayed
        display_callback(accumulated_state)
        return accumulated_state
    
