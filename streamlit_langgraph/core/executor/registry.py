# Executor registry for managing executor lifecycle.

from typing import Any, Dict, Optional

from ...agent import Agent
from .create_agent import CreateAgentExecutor
from .response_api import ResponseAPIExecutor


class ExecutorRegistry:
    """Registry for managing executor instances."""

    def __init__(self):
        self._executors: Dict[str, Any] = {}
    
    def get_or_create(
        self, agent: Agent, executor_type: str = "workflow",
        tools: Optional[list] = None
    ) -> Any:
        """
        Get existing executor or create a new one.
        
        Selection logic:
        - If HITL enabled → use CreateAgentExecutor (native tools automatically disabled)
        - If OpenAI native tools enabled AND HITL disabled → use ResponseAPIExecutor
        - If Gemini native tools enabled → use CreateAgentExecutor (Gemini tools handled via LangChain)
        - Otherwise → use CreateAgentExecutor
        
        Args:
            agent: Agent configuration
            executor_type: Type of executor ("workflow" or "single_agent")
            tools: Optional tools for CreateAgentExecutor (only used for CreateAgentExecutor)
            
        Returns:
            CreateAgentExecutor or ResponseAPIExecutor instance
        """
        executor_key = "single_agent_executor" if executor_type == "single_agent" else f"workflow_executor_{agent.name}"
        
        # Only use ResponseAPIExecutor for OpenAI provider with native tools
        has_openai_native = ExecutorRegistry.has_openai_native_tools(agent)
        use_response_api = has_openai_native and not agent.human_in_loop
        desired_executor_cls = ResponseAPIExecutor if use_response_api else CreateAgentExecutor
        desired_tools = tools if tools is not None else agent.get_tools()
        
        existing_executor = self._executors.get(executor_key)
        if not isinstance(existing_executor, desired_executor_cls):
            executor = desired_executor_cls(agent, tools=desired_tools)
            self._executors[executor_key] = executor
        else:
            executor = existing_executor
            if hasattr(executor, "tools"):
                executor.tools = desired_tools
        
        return executor

    def get(self, executor_key: str) -> Optional[Any]:
        """Get an executor by key if available."""
        return self._executors.get(executor_key)

    def clear(self) -> None:
        """Clear all cached executors."""
        self._executors.clear()
    
    @staticmethod
    def has_native_tools(agent: Agent) -> bool:
        """Check if agent has any native tools enabled (OpenAI or Gemini)."""
        return (
            agent.allow_file_search or
            agent.allow_code_interpreter or
            agent.allow_web_search or
            agent.allow_image_generation
        )
    
    @staticmethod
    def has_openai_native_tools(agent: Agent) -> bool:
        """
        Check if agent has native OpenAI tools enabled.
        
        Only returns True for OpenAI provider with native tools enabled.
        Gemini native tools are handled differently via LangChain's ChatGoogleGenerativeAI.
        """
        provider = agent.provider.lower()
        if provider != "openai":
            return False
        return (
            agent.allow_file_search or
            agent.allow_code_interpreter or
            agent.allow_web_search or
            agent.allow_image_generation
        )
    
    @staticmethod
    def has_gemini_native_tools(agent: Agent) -> bool:
        """
        Check if agent has native Gemini tools enabled.
        
        Gemini supports:
        - google_search (web search grounding)
        - code_execution (Python code execution)
        - file_search (RAG with file search stores)
        
        Note: image_generation is not directly supported as a Gemini tool.
        """
        provider = agent.provider.lower()
        if provider != "google":
            return False
        return (
            agent.allow_web_search or
            agent.allow_code_interpreter or
            agent.allow_file_search
        )
