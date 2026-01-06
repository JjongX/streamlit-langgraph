# Main agent class.

import os
from typing import Any, List, Optional

import yaml
from langchain.chat_models import init_chat_model


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
        temperature: Sampling temperature for responses (default: 0.0)
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
        hitl_description_prefix: Prefix for HITL approval messages
        conversation_history_mode: Conversation history mode ("full", "filtered", or "disable")
    """
    
    def __new__(cls, file_path: str = None, **kwargs):
        """
        Create Agent instance(s) from YAML file.

        Returns:
            Agent instance if single agent, or List[Agent] if multiple agents
        """
        if file_path is None:
            raise ValueError("file_path is required. Use Agent(file_path='config.yaml')")
        
        agents = cls._load_from_yaml(file_path)
        if len(agents) == 1:
            return agents[0]
        elif len(agents) > 1:
            return agents
        else:
            raise ValueError("YAML file contains no agents.")

    def __post_init__(self):
        """Initialize and validate settings."""
        if "file_search" in self.tools:
            self.allow_file_search = True
        if "code_interpreter" in self.tools:
            self.allow_code_interpreter = True
        if "web_search" in self.tools:
            self.allow_web_search = True
        if "image_generation" in self.tools:
            self.allow_image_generation = True
        
        valid_modes = ["full", "filtered", "disable"]
        if self.conversation_history_mode not in valid_modes:
            raise ValueError(
                f"conversation_history_mode must be one of {valid_modes}, "
                f"got '{self.conversation_history_mode}'"
            )
    
    @staticmethod
    def sync_container_ids(agents):
        """Share container_id across all code_interpreter agents."""
        code_interpreter_agents = [a for a in agents if a.allow_code_interpreter]
        if not code_interpreter_agents:
            return
        
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
    
    def get_tools(self) -> List[Any]:
        """
        Get all tools for this agent (custom tools + MCP tools).
        """
        from .utils import CustomTool, MCPToolManager
        
        tools = []
        if self.tools:
            tools.extend(CustomTool.get_langchain_tools(self.tools))
        if self.mcp_servers:
            mcp_manager = MCPToolManager()
            mcp_manager.add_servers(self.mcp_servers)
            tools.extend(mcp_manager.get_tools())
        return tools
    
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
            raise ValueError(
                f"YAML file must contain a list of agent configurations. Got: {type(agent_configs)}"
            )
        
        agents: List[Agent] = []
        for cfg in agent_configs:
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"Each agent configuration must be a dictionary. Got: {type(cfg)}"
                )
            # Create agent directly without going through constructor
            agent = object.__new__(cls)
            agent.name = cfg.get("name")
            agent.role = cfg.get("role")
            agent.instructions = cfg.get("instructions")
            agent.provider = cfg.get("provider", "openai")
            agent.model = cfg.get("model", "gpt-4.1-mini")
            agent.temperature = cfg.get("temperature", 0.0)
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
            agent.hitl_description_prefix = cfg.get("hitl_description_prefix", "Tool execution pending approval")
            agent.conversation_history_mode = cfg.get("conversation_history_mode", "filtered")
            agents.append(agent)
        
        return agents


def get_llm_client(agent: Agent, vector_store_ids: Optional[List[str]] = None) -> Any:
    """
    Get the appropriate LLM client for an agent based on its configuration.
    
    When HITL is enabled, native tools are automatically disabled to ensure
    CreateAgentExecutor is used (Response API does not support HITL).
    
    When native tools are enabled (and HITL is disabled) with OpenAI provider,
    ResponseAPIExecutor will be used. In this case, returns a minimal client
    object that only holds vector_store_ids, since ResponseAPIExecutor uses
    its own OpenAI client and doesn't need a LangChain client.
    
    Otherwise, returns a LangChain chat model via init_chat_model for CreateAgentExecutor.
    """
    if agent.human_in_loop:
        has_native_tools = False
    else:
        from .core.executor.registry import ExecutorRegistry
        has_native_tools = ExecutorRegistry.has_native_tools(agent)
    
    if agent.provider.lower() == "openai" and has_native_tools:
        class MinimalClient:
            """Minimal client object for ResponseAPIExecutor to read vector_store_ids."""
            def __init__(self, vector_store_ids: Optional[List[str]] = None):
                if vector_store_ids:
                    self._vector_store_ids = vector_store_ids
                self._provider = agent.provider.lower()
        return MinimalClient(vector_store_ids)
    else:
        chat_model = init_chat_model(
            model=agent.model,
            temperature=agent.temperature
        )
        setattr(chat_model, "_provider", agent.provider.lower())
        return chat_model
