# Main agent class.

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from logging.handlers import RotatingFileHandler

import yaml
from langchain.chat_models import init_chat_model


@dataclass
class Agent:
    """
    Agent configuration for multiagent workflows.
    
    Executor selection logic:
    - HITL enabled → CreateAgentExecutor (native tools disabled)
    - Native tools + no HITL → ResponseAPIExecutor
    - Otherwise → CreateAgentExecutor
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
    container_id: Optional[str] = None # require for code_interpreter functionality
    allow_web_search: bool = False
    allow_image_generation: bool = False
    tools: List[str] = field(default_factory=list)
    mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None
    context: Optional[str] = "least"
    human_in_loop: bool = False
    interrupt_on: Optional[Dict[str, Union[bool, Dict[str, Any]]]] = None
    hitl_description_prefix: Optional[str] = "Tool execution pending approval"
    enable_logging: bool = False
    conversation_history_mode: str = "filtered"  # Options: "full", "filtered", "disable"

    def __post_init__(self):
        """Initialize system message and validate settings."""
        if self.system_message is None:
            self.system_message = f"You are a {self.role}. {self.instructions}"
            
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
        
        if self.enable_logging:
            self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup file logging for this agent."""
        log_file = os.path.join(os.getcwd(), "agent_logging.log")
        logger = logging.getLogger("streamlit_langgraph")
        
        if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(log_file) 
                   for h in logger.handlers):
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.WARNING)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

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
            "enable_logging": self.enable_logging,
            "conversation_history_mode": self.conversation_history_mode,
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
        
        if agent.enable_logging:
            agent._setup_logging()
    
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
