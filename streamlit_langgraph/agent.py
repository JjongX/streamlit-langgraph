import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import openai
import yaml
from langchain.chat_models import init_chat_model

@dataclass
class Agent:
    """
    Configuration class for defining individual agents in a multiagent system.
    Required fields: name, role, instructions, type ('response' or 'agent').
    provider and model default to 'openai' and 'gpt-4.1-mini' if not specified.
    """
    name: str
    role: str
    instructions: str
    type: str # Must be 'response' or 'agent'
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
    context: Optional[str] = "least"  # Context mode: "full", "summary", or "least"
    human_in_loop: bool = False  # Enable human-in-the-loop approval (multiagent workflows only)
    interrupt_on: Optional[Dict[str, Union[bool, Dict[str, Any]]]] = None  # Tool names to interrupt on
    hitl_description_prefix: Optional[str] = "Tool execution pending approval"  # Prefix for interrupt messages

    def __post_init__(self):
        """Post-initialization processing and validation."""
        if self.type not in ("response", "agent"):
            raise ValueError("Agent 'type' must be either 'response' or 'agent'.")
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
            "type": self.type,
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
            "human_in_loop": self.human_in_loop,
            "interrupt_on": self.interrupt_on,
            "hitl_description_prefix": self.hitl_description_prefix,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Agent":
        """Create an Agent instance from a dictionary configuration."""
        return cls(**data)

class AgentManager:
    """
    Manager class for handling multiple agents and their interactions.
    Provides utilities for loading agents and creating LLM clients.
    """
    
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
    def get_llm_client(agent: Agent) -> Union[openai.OpenAI, Any]:
        """Get the appropriate LLM client for an agent based on its configuration."""
        if agent.type == "response" and agent.provider.lower() == "openai":
            return openai.OpenAI()
        else:
            chat_model = init_chat_model(model=agent.model)
            setattr(chat_model, "_provider", agent.provider.lower())
            return chat_model

class ExecutorFactory:
    """Factory for creating appropriate executor based on agent configuration."""
    
    @staticmethod
    def create(agent: "Agent", thread_id: Optional[str] = None, tools: Optional[list] = None):
        """Create appropriate executor for the agent."""
        from .core.executor import ResponseAPIExecutor, CreateAgentExecutor
        
        if agent.provider.lower() == "openai" and agent.type == "response":
            return ResponseAPIExecutor(agent, thread_id=thread_id)
        return CreateAgentExecutor(agent, tools=tools, thread_id=thread_id)
