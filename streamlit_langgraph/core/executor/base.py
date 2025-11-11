"""Base executor class providing common interface for all agent executors."""

import uuid
from typing import Any, Dict, List, Optional

from ...agent import Agent


class BaseExecutor:
    """
    Base class for all agent executors.
    
    Provides common functionality for executing agents and managing execution state.
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

