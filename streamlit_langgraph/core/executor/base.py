"""Base executor class providing common interface for all agent executors."""

import uuid
from typing import Any, Dict, List, Optional

from ...agent import Agent


class BaseExecutor:
    """
    Base class for all agent executors.
    
    Provides common functionality for executing agents and managing execution state.
    Includes common HITL (Human-in-the-Loop) methods shared across executor implementations.
    Subclasses implement their specific execution logic.
    """
    
    def __init__(self, agent: Agent, thread_id: Optional[str] = None):
        """
        Initialize the executor.
        
        Args:
            agent: The agent configuration to execute
            thread_id: Optional thread ID for conversation tracking (generated if not provided)
        """
        self.agent = agent
        self.thread_id = thread_id or str(uuid.uuid4())
        self.pending_tool_calls: List[Dict[str, Any]] = []
    
    def _prepare_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare execution configuration with defaults.
        
        Args:
            config: Optional configuration dict
            
        Returns:
            Configuration dict with thread_id
        """
        if config is None:
            return {"configurable": {"thread_id": self.thread_id}}
        
        if "configurable" not in config:
            config["configurable"] = {}
        
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = self.thread_id
        
        return config
    
    def get_thread_id(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract thread ID from config or return default.
        
        Args:
            config: Optional execution config
            
        Returns:
            Thread ID string
        """
        if config and "configurable" in config:
            return config["configurable"].get("thread_id", self.thread_id)
        return self.thread_id
    
    # ========== Common HITL Methods ==========
    
    def _create_interrupt_response(self, interrupt_data: Any, thread_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create response dictionary for interrupt.
        
        Common method used by all executor types to format interrupt responses consistently.
        
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

