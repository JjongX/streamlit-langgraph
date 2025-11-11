# Custom tool creation and management utilities.

import inspect
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional

from langchain_core.tools import StructuredTool


@dataclass
class CustomTool:
    """
    A custom tool that can be used by agents in the multiagent system.
    """
    # Class-level registry for storing all registered tools
    # Using ClassVar to indicate this is a class variable, not an instance field
    _registry: ClassVar[Dict[str, "CustomTool"]] = {}
    
    name: str
    description: str
    function: Callable
    parameters: Optional[Dict[str, Any]] = None
    return_direct: bool = False

    @classmethod
    def register_tool(cls, name: str, description: str, function: Callable, **kwargs) -> "CustomTool":
        """
        Register a custom tool and add it to the class-level registry.
        """
        tool = cls(
            name=name,
            description=description,
            function=function,
            **kwargs
        )
        cls._registry[name] = tool
        return tool
    
    @classmethod
    def tool(cls, name: str, description: str, **kwargs):
        """
        Decorator for registering functions as tools.
        
        Example:
            @CustomTool.tool("calculator", "Performs basic arithmetic")
            def calculate(expression: str) -> float:
                return eval(expression)
        """
        def decorator(func: Callable) -> Callable:
            cls.register_tool(name, description, func, **kwargs)
            return func
        return decorator
    
    def __post_init__(self):
        """Extract function parameters if not provided."""
        if self.parameters is None:
            self.parameters = self._extract_parameters()
    
    @classmethod
    def get_openai_tools(cls, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Convert CustomTool registry items to OpenAI function calling format."""
        tools = []
        registry = cls._registry
        if tool_names:
            registry = {name: registry[name] for name in tool_names if name in registry}

        for tool_name, custom_tool in registry.items():
            # Use parameters if available, otherwise extract from function
            parameters = custom_tool.parameters if custom_tool.parameters else custom_tool._extract_parameters()
            
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": custom_tool.description,
                    "parameters": parameters
                }
            }
            tools.append(tool_dict)
        
        return tools
    
    @classmethod
    def get_langchain_tools(cls, tool_names: Optional[List[str]] = None) -> List[Any]:
        """Convert CustomTool registry items to LangChain tools."""
        tools = []
        registry = cls._registry
        if tool_names:
            registry = {name: registry[name] for name in tool_names if name in registry}
            
        for tool_name, custom_tool in registry.items():
            tool = StructuredTool.from_function(
                func=custom_tool.function,
                name=tool_name,
                description=custom_tool.description,
            )
            tools.append(tool)
        
        return tools
    
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameters from function signature."""
        sig = inspect.signature(self.function)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            # Try to extract type from annotation
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is str:
                    param_info["type"] = "string"
                elif param.annotation is int:
                    param_info["type"] = "integer"
                elif param.annotation is float:
                    param_info["type"] = "number"
                elif param.annotation is bool:
                    param_info["type"] = "boolean"
                elif param.annotation is list:
                    param_info["type"] = "array"
                elif param.annotation is dict:
                    param_info["type"] = "object"
            
            parameters["properties"][param_name] = param_info
            
            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return parameters

