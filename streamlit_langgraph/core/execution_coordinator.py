# Execution coordinator for workflow and single-agent execution.

from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from ..agent import Agent, AgentManager
from ..core.executor import ResponseAPIExecutor, CreateAgentExecutor, WorkflowExecutor
from ..core.executor_registry import ExecutorRegistry
from ..state import WorkflowState


class ExecutionCoordinator:
    """Coordinates workflow and single-agent execution."""
    
    def __init__(
        self,
        workflow_executor: Optional[WorkflowExecutor] = None,
        agent_manager: Optional[AgentManager] = None,
        llm_client: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize ExecutionCoordinator.
        
        Args:
            workflow_executor: Optional WorkflowExecutor for multiagent workflows
            agent_manager: Optional AgentManager for agent access
            llm_client: Optional LLM client for single-agent execution
            config: Optional UI configuration
        """
        self.workflow_executor = workflow_executor
        self.agent_manager = agent_manager
        self.llm_client = llm_client
        self.config = config
    
    def execute_workflow(
        self,
        workflow: Any,
        prompt: str,
        display_callback: Optional[Callable] = None,
        initial_state: Optional[WorkflowState] = None
    ) -> WorkflowState:
        """
        Execute a multiagent workflow.
        
        Args:
            workflow: Compiled LangGraph workflow
            prompt: User input/prompt
            display_callback: Optional callback for displaying agent responses
            initial_state: Optional existing workflow state to use
            
        Returns:
            Final workflow state after execution
        """
        if not self.workflow_executor:
            raise ValueError("WorkflowExecutor is required for workflow execution")
        
        workflow_state = st.session_state.workflow_state
        
        # Get the last user message ID to only display messages after this point
        # This prevents re-displaying old messages when a new question is asked
        last_user_msg_id = None
        workflow_messages = workflow_state.get("messages", [])
        for msg in reversed(workflow_messages):
            if msg.get("role") == "user" and msg.get("id"):
                last_user_msg_id = msg.get("id")
                break
        
        def display_agent_response(state):
            """Callback to display agent responses as they complete during workflow execution."""
            if not state or "messages" not in state:
                return
            
            # Only process messages that come after the last user message
            found_last_user = last_user_msg_id is None
            for msg in state["messages"]:
                # Track when we've reached the last user message
                if last_user_msg_id and msg.get("id") == last_user_msg_id:
                    found_last_user = True
                    continue
                
                # Only process messages after the last user message
                if not found_last_user:
                    continue
                
                # Use display callback if provided, otherwise skip
                if display_callback:
                    display_callback(msg, msg.get("id"))
            
        
        # Use existing workflow_state as initial state (user message already added)
        # This prevents duplicate user messages
        result_state = self.workflow_executor.execute_workflow(
            workflow, prompt, display_callback=display_agent_response,
            initial_state=workflow_state
        )
        
        return result_state
    
    def execute_single_agent(
        self,
        agent: Agent,
        prompt: str,
        file_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute a single agent (non-workflow mode).
        
        Args:
            agent: Agent to execute
            prompt: User input/prompt
            file_messages: Optional file messages for OpenAI Responses API
            
        Returns:
            Execution result dictionary
        """
        if agent.type == "response":
            # For HITL, persist executor in session_state
            if agent.human_in_loop and agent.interrupt_on:
                executor = ExecutorRegistry.get_or_create(agent, executor_type="single_agent")
                thread_id = executor.thread_id
                config = {"configurable": {"thread_id": thread_id}}
                conversation_messages = st.session_state.workflow_state.get("messages", [])
                response = executor.execute(
                    self.llm_client, prompt, stream=False, 
                    file_messages=file_messages, config=config, messages=conversation_messages
                )
            else:
                executor = ResponseAPIExecutor(agent)
                conversation_messages = st.session_state.workflow_state.get("messages", [])
                response = executor.execute(
                    self.llm_client, prompt, stream=self.config.stream if self.config else True,
                    file_messages=file_messages, messages=conversation_messages
                )
            
            # Handle interrupts from ResponseAPIExecutor
            if response.get("__interrupt__"):
                ExecutorRegistry._ensure_executors_dict()
                st.session_state.agent_executors["single_agent_executor"] = executor
                return response
            
            return response
        else:
            # Tools are loaded automatically by CreateAgentExecutor from CustomTool registry
            executor = CreateAgentExecutor(agent)
            response = executor.execute(self.llm_client, prompt, stream=False)
            
            if response.get("__interrupt__"):
                ExecutorRegistry._ensure_executors_dict()
                st.session_state.agent_executors["single_agent_executor"] = executor
            
            return response

