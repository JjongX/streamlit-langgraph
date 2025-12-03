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
        from ...utils import CustomTool, MCPToolManager

        executor_key = "single_agent_executor" if executor_type == "single_agent" else f"workflow_executor_{agent.name}"
        
        has_native = ExecutorRegistry.has_native_tools(agent)
        has_hitl = agent.human_in_loop
        use_response_api = has_native and not has_hitl
        
        # Check if executor exists and is the correct type for this agent's configuration
        existing_executor = st.session_state.agent_executors.get(executor_key)
        executor_needs_recreation = False
        
        if existing_executor is not None:
            # Verify the existing executor is the correct type for current agent configuration
            is_response_api = isinstance(existing_executor, ResponseAPIExecutor)
            is_create_agent = isinstance(existing_executor, CreateAgentExecutor)
            
            # Recreate if executor type doesn't match current agent configuration
            if use_response_api and not is_response_api:
                executor_needs_recreation = True
            elif not use_response_api and not is_create_agent:
                executor_needs_recreation = True
            # Also recreate if the agent configuration changed (different agent object)
            elif hasattr(existing_executor, 'agent') and existing_executor.agent.name != agent.name:
                executor_needs_recreation = True
        
        if executor_key not in st.session_state.agent_executors or executor_needs_recreation:
            if use_response_api:
                executor = ResponseAPIExecutor(agent, tools=tools)
            else:
                executor = CreateAgentExecutor(agent, tools=tools)
            st.session_state.agent_executors[executor_key] = executor
        else:
            executor = existing_executor
            
            if isinstance(executor, CreateAgentExecutor) and hasattr(executor, 'tools'):
                custom_tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
                mcp_tools = []
                if agent.mcp_servers:
                    mcp_manager = MCPToolManager()
                    mcp_manager.add_servers(agent.mcp_servers)
                    mcp_tools = mcp_manager.get_tools()
                executor.tools = custom_tools + mcp_tools
        
        return executor
    
    def create_for_hitl(self, agent: Agent, executor_key: Optional[str] = None) -> Any:
        """
        Create executor for HITL scenarios.
        
        Args:
            agent: Agent configuration
            executor_key: Optional custom executor key
            
        Returns:
            CreateAgentExecutor instance
        """
        from .create_agent import CreateAgentExecutor

        if executor_key is None:
            executor_key = f"workflow_executor_{agent.name}"
        executor = CreateAgentExecutor(agent)
        
        st.session_state.agent_executors[executor_key] = executor
        return executor
    
    @staticmethod
    def has_native_tools(agent: Agent) -> bool:
        """
        Check if agent has native OpenAI tools enabled.
        
        Args:
            agent: Agent configuration
            
        Returns:
            True if any native tool is enabled
        """
        return (
            agent.allow_file_search or
            agent.allow_code_interpreter or
            agent.allow_web_search or
            agent.allow_image_generation
        )
