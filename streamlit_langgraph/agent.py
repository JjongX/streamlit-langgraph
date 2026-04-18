# Main agent class.

import os
from typing import Any, List, Optional

import yaml
from langchain.chat_models import init_chat_model

from .utils import CustomTool, MCPToolManager


class Agent:
    """
    Agent configuration for multiagent workflows.
    
    This class represents a single agent in a multi-agent system, including its 
    configuration, capabilities, and behavior settings.
    
    **Configuration via YAML:**
        Load agents from YAML file:
        
        Single agent:
        ```python
        agent = Agent("configs/agent.yaml")
        ```
        
        Multiple agents:
        ```python
        agents = Agent("configs/agents.yaml")
        supervisor = agents[0]
        workers = agents[1:]
        ```
    
    Executor selection logic:
    - HITL enabled -> CreateAgentExecutor (HITL requires tool interception, 
    which is not available in the Response API)
    - Native OpenAI tools enabled + HITL disabled -> ResponseAPIExecutor 
    (to utilize native tools and access full Response API features such as partial image support, streaming, and more)
    - Default -> CreateAgentExecutor
    
    Attributes:
        name: Unique identifier for the agent
        role: Brief description of the agent's role
        instructions: Detailed instructions guiding agent behavior
        provider: LLM provider name (default: "openai")
        model: Model name to use (default: "gpt-4.1-mini")
        temperature: Sampling temperature for responses (None if not specified)
        reasoning_effort: Reasoning effort for OpenAI Responses API ("low", "medium", "high")
        allow_file_search: Enable file search capability
        allow_code_interpreter: Enable code interpreter capability
        container_id: Container ID for code interpreter (auto-created by FileHandler when code_interpreter is enabled, not loaded from YAML)
        allow_web_search: Enable web search capability
        allow_image_generation: Enable image generation capability
        tools: List of custom tool names available to the agent
        mcp_servers: MCP server configurations
        context: Context mode ("full", "summary", or "least")
        human_in_loop: Enable human-in-the-loop approval
        interrupt_on: HITL configuration per tool
        conversation_history_mode: Conversation history mode ("full", "filtered", or "disable")
    """
    
    def __new__(cls, file_path: str = None, **kwargs):
        """
        Create Agent instance(s) from YAML file.

        Returns:
            Agent instance if single agent, or List[Agent] if multiple agents
        """
        agents = cls._load_from_yaml(file_path)
        if len(agents) == 1:
            return agents[0]
        elif len(agents) > 1:
            return agents
        return agents

    def get_tools(self) -> List[Any]:
        """
        Get all tools for this agent (custom tools + MCP tools).
        """
        tools = []
        if self.tools:
            tools.extend(CustomTool.get_langchain_tools(self.tools))
        if self.mcp_servers:
            mcp_manager = MCPToolManager()
            mcp_manager.add_servers(self.mcp_servers)
            tools.extend(mcp_manager.get_tools())
        return tools
    
    @staticmethod
    def sync_container_ids(agents):
        """Share container_id across all code_interpreter agents."""
        code_interpreter_agents = [a for a in agents if a.allow_code_interpreter]
        if not code_interpreter_agents:
            return # No code_interpreter agents to sync
        
        # Find first agent with a container_id set
        shared_container_id = next(
            (a.container_id for a in code_interpreter_agents 
             if a.container_id and isinstance(a.container_id, str)), 
            None
        )
        # Apply the shared container_id to all code_interpreter agents
        if shared_container_id:
            for agent in code_interpreter_agents:
                agent.container_id = shared_container_id
    
    @classmethod
    def _load_from_yaml(cls, yaml_path: str) -> List["Agent"]:
        """
        Internal method to load Agent instances from a YAML configuration file.
        Creates agents directly without going through constructor to avoid double initialization.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            agent_configs = yaml.safe_load(f)
        if agent_configs is None:
            raise ValueError("YAML file is empty or contains no configuration.")
        if not isinstance(agent_configs, list):
            raise ValueError(f"YAML file must contain a list of agent configurations. Got: {type(agent_configs)}")

        return [cls._build_agent_from_config(cfg) for cfg in agent_configs]

    @classmethod
    def _build_agent_from_config(cls, cfg: dict) -> "Agent":
        """Create an Agent instance from a single configuration dict."""
        if not isinstance(cfg, dict):
            raise ValueError(f"Each agent configuration must be a dictionary. Got: {type(cfg)}")

        agent = object.__new__(cls)
        agent.name = cfg.get("name")
        agent.role = cfg.get("role")
        agent.instructions = cfg.get("instructions")
        agent.provider = cfg.get("provider", "openai")
        agent.model = cfg.get("model", "gpt-4.1-mini")
        agent.temperature = cfg.get("temperature")
        agent.reasoning_effort = cfg.get("reasoning_effort")
        agent.allow_file_search = cfg.get("allow_file_search", False)
        agent.allow_code_interpreter = cfg.get("allow_code_interpreter", False)
        agent.container_id = None
        agent.allow_web_search = cfg.get("allow_web_search", False)
        agent.allow_image_generation = cfg.get("allow_image_generation", False)
        agent.tools = cfg.get("tools", [])
        agent.mcp_servers = cfg.get("mcp_servers")
        agent.context = cfg.get("context", "least")
        agent.human_in_loop = cfg.get("human_in_loop", False)
        agent.interrupt_on = cfg.get("interrupt_on")
        agent.conversation_history_mode = cfg.get("conversation_history_mode", "filtered")
        cls._apply_agent_settings(agent)
        return agent

    @staticmethod
    def _apply_agent_settings(agent: "Agent") -> None:
        """Normalize tool flags and validate conversation history mode."""
        tools = agent.tools or []
        if "file_search" in tools:
            agent.allow_file_search = True
        if "code_interpreter" in tools:
            agent.allow_code_interpreter = True
        if "web_search" in tools:
            agent.allow_web_search = True
        if "image_generation" in tools:
            agent.allow_image_generation = True
        
        valid_modes = {"full", "filtered", "disable"}
        if agent.conversation_history_mode not in valid_modes:
            raise ValueError(
                f"conversation_history_mode must be one of {sorted(valid_modes)}, "
                f"got '{agent.conversation_history_mode}'"
            )

        valid_reasoning_efforts = {"low", "medium", "high"}
        if agent.reasoning_effort is not None and agent.reasoning_effort not in valid_reasoning_efforts:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(valid_reasoning_efforts)}, "
                f"got '{agent.reasoning_effort}'"
            )


def get_llm_client(
    agent: Agent,
    vector_store_ids: Optional[List[str]] = None,
    file_search_store_names: Optional[List[str]] = None
) -> Any:
    """Return a provider-appropriate LLM client (LangChain or minimal wrapper)."""
    provider = agent.provider.lower()
    from .core.executor.registry import ExecutorRegistry

    # If OpenAI native tools are enabled (and HITL is off), we use ResponseAPIExecutor.
    # In that case, return a minimal client that only carries vector_store_ids.
    use_response_api = (
        provider == "openai"
        and not agent.human_in_loop
        and ExecutorRegistry.has_openai_native_tools(agent)
    )
    if use_response_api:
        class MinimalClient:
            """Minimal client object for ResponseAPIExecutor to read vector_store_ids."""
            def __init__(self, vector_store_ids: Optional[List[str]] = None):
                if vector_store_ids:
                    self._vector_store_ids = vector_store_ids
                self._provider = provider
        return MinimalClient(vector_store_ids)

    # CreateAgentExecutor path (LangChain chat model). Use model_provider "google_genai" for Google (not "google").
    model_provider = "google_genai" if provider == "google" else provider
    init_kwargs = {
        "model": agent.model,
        "temperature": agent.temperature,
        "model_provider": model_provider,
    }

    if provider == "openai" and agent.reasoning_effort:
        init_kwargs["reasoning"] = {"effort": agent.reasoning_effort, "summary": "auto"}

    chat_model = init_chat_model(**init_kwargs)
    setattr(chat_model, "_provider", provider)

    if provider == "google" and ExecutorRegistry.has_gemini_native_tools(agent):
        tools = []
        if agent.allow_web_search:
            tools.append({"google_search": {}})
        if agent.allow_code_interpreter:
            tools.append({"code_execution": {}})
        if tools:
            chat_model = chat_model.bind_tools(tools)
        setattr(chat_model, "_has_gemini_native_tools", True)

    return chat_model
