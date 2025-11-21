# Executor registry for managing executor lifecycle.

from typing import Any, Optional

import streamlit as st

from ...agent import Agent


class ExecutorRegistry:
    
    def get_or_create(self, agent: Agent, executor_type: str = "workflow",
        tools: Optional[list] = None) -> Any:
        """
        Get existing executor or create a new one.
        
        Args:
            agent: Agent configuration
            executor_type: Type of executor ("workflow" or "single_agent")
            tools: Optional tools for CreateAgentExecutor
            
        Returns:
            CreateAgentExecutor instance
        """
        from .create_agent import CreateAgentExecutor
        from ...utils import CustomTool, MCPToolManager

        executor_key = "single_agent_executor" if executor_type == "single_agent" else f"workflow_executor_{agent.name}"
        
        # Get existing executor or create new one
        if executor_key not in st.session_state.agent_executors:
            executor = CreateAgentExecutor(agent, tools=tools)
            st.session_state.agent_executors[executor_key] = executor
        else:
            executor = st.session_state.agent_executors[executor_key]
            
            if hasattr(executor, 'tools'):
                
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
