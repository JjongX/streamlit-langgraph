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
        """
        Initialize the executor.
        
        Args:
            agent: The agent configuration to execute
        """
        self.agent = agent
        self.pending_tool_calls: List[Dict[str, Any]] = []
    
    def _prepare_workflow_config(self, config: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], str]:
        """Prepare workflow execution configuration and extract thread_id."""
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

    def _get_executor_key(self, agent_name: str, executor_type: str = "workflow") -> str:
        """Generate consistent executor key."""
        if executor_type == "single_agent":
            return "single_agent_executor"
        return f"workflow_executor_{agent_name}"
    
    def _ensure_executors_dict(self) -> None:
        """Ensure agent_executors exists in session state."""
        if "agent_executors" not in st.session_state:
            st.session_state.agent_executors = {}
    
    def _create_executor(self, agent: Agent, tools: Optional[list] = None) -> Any:
        """Create appropriate executor for the agent."""
        # Lazy imports to avoid circular import
        from .create_agent import CreateAgentExecutor
        from .response_api import ResponseAPIExecutor

        if agent.provider.lower() == "openai" and agent.type == "response":
            return ResponseAPIExecutor(agent)
        return CreateAgentExecutor(agent, tools=tools)
    
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
        self._ensure_executors_dict()
        
        executor_key = self._get_executor_key(agent.name, executor_type)
        
        # Get existing executor or create new one
        if executor_key not in st.session_state.agent_executors:
            executor = self._create_executor(agent, tools=tools)
            st.session_state.agent_executors[executor_key] = executor
        else:
            executor = st.session_state.agent_executors[executor_key]
            
            # Update tools for CreateAgentExecutor if needed
            if hasattr(executor, 'tools') and agent.tools:
                from ...utils import CustomTool # lazy import to avoid circular import
                executor.tools = CustomTool.get_langchain_tools(agent.tools)
        
        return executor
    
    def get(self, agent_name: str, executor_type: str = "workflow") -> Optional[Any]:
        """Get existing executor by key."""
        self._ensure_executors_dict()
        
        executor_key = self._get_executor_key(agent_name, executor_type)
        return st.session_state.agent_executors.get(executor_key)
    
    def create_for_hitl(self, agent: Agent, executor_key: Optional[str] = None) -> Any:
        """
        Create executor for HITL scenarios.
        
        Args:
            agent: Agent configuration
            executor_key: Optional custom executor key
            
        Returns:
            Executor instance (ResponseAPIExecutor or CreateAgentExecutor)
        """
        self._ensure_executors_dict()
        
        if executor_key is None:
            executor_key = self._get_executor_key(agent.name, "workflow")
        
        # Create executor - thread_id comes from config in workflows
        executor = self._create_executor(agent)
        
        st.session_state.agent_executors[executor_key] = executor
        return executor
