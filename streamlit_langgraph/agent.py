# Main agent class.

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

@dataclass
class Agent:
    """
    Configuration class for defining individual agents in a multiagent system.
    Required fields: name, role, instructions.
    provider and model default to 'openai' and 'gpt-4.1-mini' if not specified.
    
    All agents use CreateAgentExecutor, which automatically uses Responses API
    when native OpenAI tools are enabled.
    """
    name: str
    role: str
    instructions: str
    provider: Optional[str] = "openai"
    model: Optional[str] = "gpt-4.1-mini"
    system_message: Optional[str] = None
    temperature: float = 0.0
    allow_file_search: bool = False
    allow_code_interpreter: bool = False
    container_id: Optional[str] = None  # For code_interpreter functionality
    allow_web_search: bool = False
    allow_image_generation: bool = False
    tools: List[str] = field(default_factory=list)
    mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None  # MCP server configurations
    context: Optional[str] = "least"  # Context mode: "full", "summary", or "least"
    human_in_loop: bool = False  # Enable human-in-the-loop approval (multiagent workflows only)
    interrupt_on: Optional[Dict[str, Union[bool, Dict[str, Any]]]] = None  # Tool names to interrupt on
    hitl_description_prefix: Optional[str] = "Tool execution pending approval"  # Prefix for interrupt messages

    def __post_init__(self):
        """Post-initialization processing and validation."""
        if self.system_message is None:
            self.system_message = f"You are a {self.role}. {self.instructions}"
        # Auto-enable OpenAI's native tools based on tools list configuration
        if "file_search" in self.tools:
            self.allow_file_search = True
        if "code_interpreter" in self.tools:
            self.allow_code_interpreter = True
        if "web_search" in self.tools:
            self.allow_web_search = True
        if "image_generation" in self.tools:
            self.allow_image_generation = True

    def to_dict(self) -> Dict:
        """Convert agent configuration to dictionary for serialization."""
        return {
            "name": self.name,
            "role": self.role,
            "instructions": self.instructions,
            "provider": self.provider,
            "model": self.model,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "allow_file_search": self.allow_file_search,
            "allow_code_interpreter": self.allow_code_interpreter,
            "container_id": self.container_id,
            "allow_web_search": self.allow_web_search,
            "allow_image_generation": self.allow_image_generation,
            "tools": self.tools,
            "mcp_servers": self.mcp_servers,
            "context": self.context,
            "human_in_loop": self.human_in_loop,
            "interrupt_on": self.interrupt_on,
            "hitl_description_prefix": self.hitl_description_prefix,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Agent":
        """Create an Agent instance from a dictionary configuration."""
        return cls(**data)


class AgentManager:
    """Manager class for handling multiple agents and their interactions."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.active_agent: Optional[str] = None
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the manager."""
        self.agents[agent.name] = agent
        if self.active_agent is None:
            self.active_agent = agent.name
    
    def remove_agent(self, name: str) -> None:
        """Remove an agent from the manager."""
        if name in self.agents:
            del self.agents[name]
            if self.active_agent == name:
                self.active_agent = next(iter(self.agents.keys())) if self.agents else None
    
    @staticmethod
    def load_from_yaml(yaml_path: str) -> List[Agent]:
        """
        Load multiple Agent instances from a YAML configuration file.
        
        This method is designed for multi-agent configurations. For single agents,
        use the Agent class directly: Agent(name="...", role="...", ...)
        
        Example:
            # Load agents from a config file
            agents = AgentManager.load_from_yaml("./configs/supervisor_sequential.yaml")
            supervisor = agents[0]
            workers = agents[1:]
            
            # Or use relative to current file
            config_path = os.path.join(os.path.dirname(__file__), "./configs/my_agents.yaml")
            agents = AgentManager.load_from_yaml(config_path)
        """
        # Resolve path - handle both absolute and relative paths
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.abspath(yaml_path)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            agent_configs = yaml.safe_load(f)
        if not isinstance(agent_configs, list):
            raise ValueError(f"YAML file must contain a list of agent configurations. Got: {type(agent_configs)}")
        agents = []
        for cfg in agent_configs:
            if not isinstance(cfg, dict):
                raise ValueError(f"Each agent configuration must be a dictionary. Got: {type(cfg)}")
            agents.append(Agent(**cfg))
        
        return agents
    
    @staticmethod
    def get_llm_client(agent: Agent) -> Any:
        """
        Get the appropriate LLM client for an agent based on its configuration.
        
        Uses ChatOpenAI with use_responses_api=True when native OpenAI tools are enabled
        (code_interpreter, web_search, file_search, image_generation) to leverage
        OpenAI's Responses API for agentic behavior.
        """
        has_native_tools = (
            agent.allow_code_interpreter or
            agent.allow_web_search or
            agent.allow_file_search or
            agent.allow_image_generation
        )        
        # For OpenAI provider with native tools, use Responses API
        if agent.provider.lower() == "openai" and has_native_tools:
            chat_model = ChatOpenAI(
                model=agent.model,
                temperature=agent.temperature,
                use_responses_api=True
            )
        else:
            # Use standard init_chat_model for other cases
            chat_model = init_chat_model(
                model=agent.model,
                temperature=agent.temperature
            )
        
        setattr(chat_model, "_provider", agent.provider.lower())
        return chat_model
