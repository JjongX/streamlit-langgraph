# Centralized executor registry for managing executor lifecycle.

from typing import Any, Optional

import streamlit as st

from ..agent import Agent, ExecutorFactory
from .executor import ResponseAPIExecutor, CreateAgentExecutor


class ExecutorRegistry:
    """
    Centralized registry for managing executor lifecycle.
    
    Provides consistent executor creation, retrieval, and key management
    across the application.
    """
    
    @staticmethod
    def _get_executor_key(agent_name: str, executor_type: str = "workflow") -> str:
        """
        Generate consistent executor key.
        
        Args:
            agent_name: Name of the agent
            executor_type: Type of executor ("workflow" or "single_agent")
            
        Returns:
            Executor key string
        """
        if executor_type == "single_agent":
            return "single_agent_executor"
        return f"workflow_executor_{agent_name}"
    
    @staticmethod
    def _ensure_executors_dict() -> None:
        """Ensure agent_executors exists in session state."""
        if "agent_executors" not in st.session_state:
            st.session_state.agent_executors = {}
    
    @staticmethod
    def get_or_create(
        agent: Agent,
        executor_type: str = "workflow",
        thread_id: Optional[str] = None,
        tools: Optional[list] = None
    ) -> Any:
        """
        Get existing executor or create a new one.
        
        Args:
            agent: Agent configuration
            executor_type: Type of executor ("workflow" or "single_agent")
            thread_id: Optional thread ID for the executor
            tools: Optional tools for CreateAgentExecutor
            
        Returns:
            Executor instance (ResponseAPIExecutor or CreateAgentExecutor)
        """
        ExecutorRegistry._ensure_executors_dict()
        
        executor_key = ExecutorRegistry._get_executor_key(agent.name, executor_type)
        
        # Get existing executor or create new one
        if executor_key not in st.session_state.agent_executors:
            executor = ExecutorFactory.create(agent, thread_id=thread_id, tools=tools)
            st.session_state.agent_executors[executor_key] = executor
        else:
            executor = st.session_state.agent_executors[executor_key]
            
            # Update tools for CreateAgentExecutor if needed
            if hasattr(executor, 'tools') and agent.tools:
                from ..utils import CustomTool # lazy import to avoid circular import
                executor.tools = CustomTool.get_langchain_tools(agent.tools)
        
        return executor
    
    @staticmethod
    def get(agent_name: str, executor_type: str = "workflow") -> Optional[Any]:
        """
        Get existing executor by key.
        
        Args:
            agent_name: Name of the agent
            executor_type: Type of executor ("workflow" or "single_agent")
            
        Returns:
            Executor instance or None if not found
        """
        ExecutorRegistry._ensure_executors_dict()
        
        executor_key = ExecutorRegistry._get_executor_key(agent_name, executor_type)
        return st.session_state.agent_executors.get(executor_key)
    
    @staticmethod
    def create_for_hitl(
        agent: Agent,
        thread_id: str,
        executor_key: Optional[str] = None
    ) -> Any:
        """
        Create executor for HITL scenarios.
        
        Args:
            agent: Agent configuration
            thread_id: Thread ID for the executor
            executor_key: Optional custom executor key
            
        Returns:
            Executor instance (ResponseAPIExecutor or CreateAgentExecutor)
        """
        ExecutorRegistry._ensure_executors_dict()
        
        if executor_key is None:
            executor_key = ExecutorRegistry._get_executor_key(agent.name, "workflow")
        
        # Determine which executor to use based on agent type
        if agent.provider.lower() == "openai" and agent.type == "response":
            executor = ResponseAPIExecutor(agent, thread_id=thread_id)
        else:
            executor = CreateAgentExecutor(agent, thread_id=thread_id)
        
        st.session_state.agent_executors[executor_key] = executor
        return executor

