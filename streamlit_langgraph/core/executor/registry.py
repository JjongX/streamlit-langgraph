# Executor registry for managing executor lifecycle.

from typing import Any, Optional

import streamlit as st

from ...agent import Agent


class ExecutorRegistry:
    """Registry for managing executor instances."""
    
    def get_or_create(
        self, agent: Agent, executor_type: str = "workflow",
        tools: Optional[list] = None
    ) -> Any:
        """
        Get existing executor or create a new one.
        
        Selection logic:
        - If HITL enabled → use CreateAgentExecutor (native tools automatically disabled)
        - If native tools enabled AND HITL disabled → use ResponseAPIExecutor
        - Otherwise → use CreateAgentExecutor
        
        Args:
            agent: Agent configuration
            executor_type: Type of executor ("workflow" or "single_agent")
            tools: Optional tools for CreateAgentExecutor (only used for CreateAgentExecutor)
            
        Returns:
            CreateAgentExecutor or ResponseAPIExecutor instance
        """
        from .create_agent import CreateAgentExecutor
        from .response_api import ResponseAPIExecutor

        executor_key = "single_agent_executor" if executor_type == "single_agent" else f"workflow_executor_{agent.name}"
        
        has_native = ExecutorRegistry.has_native_tools(agent)
        use_response_api = has_native and not agent.human_in_loop
        
        # Check if executor exists and is the correct type
        existing_executor = st.session_state.agent_executors.get(executor_key)
        executor_needs_recreation = False
        
        if existing_executor is not None:
            is_response_api = isinstance(existing_executor, ResponseAPIExecutor)
            is_create_agent = isinstance(existing_executor, CreateAgentExecutor)
            
            if use_response_api and not is_response_api:
                executor_needs_recreation = True
            elif not use_response_api and not is_create_agent:
                executor_needs_recreation = True
            elif hasattr(existing_executor, 'agent') and existing_executor.agent.name != agent.name:
                executor_needs_recreation = True
        
        if executor_key not in st.session_state.agent_executors or executor_needs_recreation:
            if use_response_api:
                executor = ResponseAPIExecutor(agent, tools=agent.get_tools())
            else:
                executor = CreateAgentExecutor(agent, tools=tools)
            st.session_state.agent_executors[executor_key] = executor
        else:
            executor = existing_executor
            if hasattr(executor, 'tools'):
                executor.tools = agent.get_tools()
        
        return executor
    
    def create_for_hitl(self, agent: Agent, executor_key: Optional[str] = None) -> Any:
        """Create executor for HITL scenarios."""
        from .create_agent import CreateAgentExecutor

        if executor_key is None:
            executor_key = f"workflow_executor_{agent.name}"
        executor = CreateAgentExecutor(agent)
        
        st.session_state.agent_executors[executor_key] = executor
        return executor
    
    @staticmethod
    def has_native_tools(agent: Agent) -> bool:
        """Check if agent has native OpenAI tools enabled."""
        return (
            agent.allow_file_search or
            agent.allow_code_interpreter or
            agent.allow_web_search or
            agent.allow_image_generation
        )
