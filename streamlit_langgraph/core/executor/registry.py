# Base executor class and centralized executor registry for managing executor lifecycle.

from typing import Any, Dict, List, Optional

import streamlit as st

from ...agent import Agent

class BaseExecutor:
    """
    Base class for all agent executors.
    
    Provides common functionality for executing agents and managing execution state.
    Includes common HITL (Human-in-the-Loop) methods shared across executor implementations.
    Subclasses implement their specific execution logic.
    """
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.pending_tool_calls: List[Dict[str, Any]] = []
    
    def _prepare_workflow_config(self, config: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], str]:
        # Setting config 
        if config is None:
            raise ValueError(
                "config is required for workflow execution. "
                "It should contain thread_id from the workflow's checkpointer configuration."
            )
        if "configurable" not in config:
            config["configurable"] = {}
        
        # Setting thread_id using config
        if "thread_id" not in config["configurable"]:
            raise ValueError(
                "thread_id must be provided in config for workflow execution. "
                "It should come from the workflow's checkpointer configuration."
            )
        thread_id = config["configurable"]["thread_id"]

        return config, thread_id
    
    def _create_interrupt_response(self, interrupt_data: Any, thread_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create response dictionary for interrupt.
                
        Args:
            interrupt_data: The interrupt data (format varies by executor type)
            thread_id: Thread ID for the conversation
            config: Execution configuration
            
        Returns:
            Dictionary with interrupt information
        """
        return {
            "role": "assistant",
            "content": "",
            "agent": self.agent.name,
            "__interrupt__": interrupt_data,
            "thread_id": thread_id,
            "config": config
        }
    
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Execute a tool by name from the CustomTool registry.
        
        Common method used by executors that need to execute tools directly.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool function
            
        Returns:
            Tool execution result
        """
        from ...utils import CustomTool  # lazy import to avoid circular import
        tool = CustomTool._registry.get(tool_name)
        if tool and tool.function:
            return tool.function(**tool_args)
        return f"Tool {tool_name} not found"


class ExecutorRegistry:
    
    def _create_executor(self, agent: Agent, tools: Optional[list] = None) -> Any:
        """
        Create appropriate executor for the agent.
        
        Executor Selection Logic:
        - type="agent" → CreateAgentExecutor (supports HITL)
        - type="response" → ResponseAPIExecutor (no HITL support)
        - type="response" + HITL enabled → Error (must use type="agent" for HITL)
        """
        # Lazy imports to avoid circular import
        from .create_agent import CreateAgentExecutor
        from .response_api import ResponseAPIExecutor

        if agent.type == "response" and agent.human_in_loop and agent.interrupt_on:
            raise ValueError(
                f"Agent '{agent.name}' has type='response' but HITL is enabled. "
                "Responses API cannot intercept tool calls, so HITL requires LangChain. "
                "Please change the agent type to 'agent' in your configuration to enable HITL support."
            )
        
        if agent.type == "response" and agent.mcp_servers:
            stdio_servers = [
                name for name, config in agent.mcp_servers.items()
                if config.get("transport", "stdio") == "stdio"
            ]
            if stdio_servers:
                raise ValueError(
                    f"Agent '{agent.name}' has type='response' but uses stdio MCP servers: {', '.join(stdio_servers)}. "
                    "Responses API only supports HTTP/SSE transport. Use type='agent' for stdio MCP servers."
                )
        
        if agent.type == "agent":
            return CreateAgentExecutor(agent, tools=tools)
        if agent.type == "response":
            if agent.provider.lower() == "openai":
                return ResponseAPIExecutor(agent)
            else:
                raise ValueError(
                    f"Agent '{agent.name}' has type='response' but provider is '{agent.provider}'. "
                    "ResponseAPIExecutor only supports OpenAI. "
                    "Please change the agent type to 'agent' for other providers."
                )
        
        raise ValueError(
            f"Agent '{agent.name}' has invalid type='{agent.type}'. "
            "Type must be either 'agent' or 'response'."
        )
    
    def get_or_create(self, agent: Agent, executor_type: str = "workflow",
        tools: Optional[list] = None) -> Any:
        """
        Get existing executor or create a new one.
        
        Args:
            agent: Agent configuration
            executor_type: Type of executor ("workflow" or "single_agent")
            tools: Optional tools for CreateAgentExecutor
            
        Returns:
            Executor instance (ResponseAPIExecutor or CreateAgentExecutor)
        """
        executor_key = "single_agent_executor" if executor_type == "single_agent" else f"workflow_executor_{agent.name}"
        
        # Get existing executor or create new one
        if executor_key not in st.session_state.agent_executors:
            executor = self._create_executor(agent, tools=tools)
            st.session_state.agent_executors[executor_key] = executor
        else:
            executor = st.session_state.agent_executors[executor_key]
            
            if hasattr(executor, 'tools'):
                from ...utils import CustomTool, MCPToolManager
                
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
            Executor instance (ResponseAPIExecutor or CreateAgentExecutor)
        """
        if executor_key is None:
            executor_key = f"workflow_executor_{agent.name}"
        # Create executor - thread_id comes from config in workflows
        executor = self._create_executor(agent)
        
        st.session_state.agent_executors[executor_key] = executor
        return executor
