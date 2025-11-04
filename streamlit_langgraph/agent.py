import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import openai
import yaml
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

@dataclass
class Agent:
    """
    Configuration class for defining individual agents in a multiagent system.
    Required fields: name, role, instructions, type ('response' or 'agent').
    provider and model default to 'openai' and 'gpt-4.1' if not specified.
    """
    name: str
    role: str
    instructions: str
    type: str # Must be 'response' or 'agent'
    provider: Optional[str] = "openai"
    model: Optional[str] = "gpt-4.1"
    system_message: Optional[str] = None
    temperature: float = 0.0
    allow_file_search: bool = False
    allow_code_interpreter: bool = False
    container_id: Optional[str] = None  # For code_interpreter functionality
    allow_web_search: bool = False
    allow_image_generation: bool = False
    tools: List[str] = field(default_factory=list)
    context: Optional[str] = "least"  # Context mode: "full", "summary", or "least"

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
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Agent":
        """Create an Agent instance from a dictionary configuration."""
        return cls(**data)

def load_agents_from_yaml(yaml_path: str) -> List[Agent]:
    """
    Load multiple Agent instances from a YAML configuration file.
    
    This function is designed for multi-agent configurations. For single agents,
    use the Agent class directly: Agent(name="...", role="...", ...)
    
    Args:
        yaml_path: Path to the YAML file containing agent configurations.
                  Can be absolute or relative path. If relative, it's resolved
                  relative to the current working directory.
    
    Returns:
        List of Agent instances loaded from the YAML file.
    
    Example:
        # Load agents from a config file
        agents = load_agents_from_yaml("./configs/supervisor_sequential.yaml")
        supervisor = agents[0]
        workers = agents[1:]
        
        # Or use relative to current file
        config_path = os.path.join(os.path.dirname(__file__), "./configs/my_agents.yaml")
        agents = load_agents_from_yaml(config_path)
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

def get_llm_client(agent: Agent) -> Union[openai.OpenAI, Any]:
    """Get the appropriate LLM client for an agent based on its configuration."""
    if agent.type == "response" and agent.provider.lower() == "openai":
        return openai.OpenAI()
    else:
        chat_model = init_chat_model(model=agent.model)
        setattr(chat_model, "_provider", agent.provider.lower())
        return chat_model

class ResponseAPIExecutor:
    """Executor for OpenAI Responses API generation."""

    def __init__(self, agent: Agent):
        self.agent = agent

    def execute(self, llm_client, prompt: str, stream: bool = True, file_messages: Optional[List] = None):
        """
        Execute prompt using the Responses API client.
        
        Supports file messages, code interpreter, web search, and image generation tools.
        Note: file_search requires vector_store_ids and is handled by FileHandler.

        Args:
            llm_client: Client exposing `responses.create` method
            prompt: Full prompt or input to send
            stream: Whether to request a streaming response
            file_messages: Optional list of file-related input messages

        Returns:
            Dict with keys 'role', 'content', 'agent' and optionally 'stream'
        """
        input_messages = []
        if file_messages:
            input_messages.extend(file_messages)
        input_messages.append({"role": "user", "content": prompt})

        tools = self._build_tools_config(llm_client)
        api_params = {
            "model": self.agent.model,
            "input": input_messages,
            "temperature": self.agent.temperature,
            "stream": stream,
        }
        if tools:
            api_params["tools"] = tools

        try:
            if stream:
                stream_iter = llm_client.responses.create(**api_params)
                return {"role": "assistant", "content": "", "agent": self.agent.name, "stream": stream_iter}
            else:
                response = llm_client.responses.create(**api_params)
                response_content = self._extract_response_content(response)
                return {"role": "assistant", "content": response_content, "agent": self.agent.name}
        except Exception as e:
            return {"role": "assistant", "content": f"Responses API error: {str(e)}", "agent": self.agent.name}

    def _build_tools_config(self, llm_client) -> List[Dict[str, Any]]:
        """Build tools configuration based on agent capabilities."""
        tools = []
        if self.agent.allow_code_interpreter:
            container = llm_client.containers.create(name=f"streamlit-{self.agent.name}")
            self.agent.container_id = container.id
            tools.append({"type": "code_interpreter", "container": self.agent.container_id})
        if self.agent.allow_web_search:
            tools.append({"type": "web_search"})
        if self.agent.allow_image_generation:
            tools.append({"type": "image_generation", "partial_images": 3})
        return tools

    def _extract_response_content(self, response) -> str:
        """Extract text content from API response object."""
        if hasattr(response, 'output') and isinstance(response.output, list):
            content_parts = []
            for message in response.output:
                if hasattr(message, 'content') and isinstance(message.content, list):
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            content_parts.append(content_item.text)
            return "".join(content_parts)
        elif hasattr(response, 'content'):
            return str(response.content)
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return str(response.message.content)
        return ""

class CreateAgentExecutor:
    """
    Executor that builds LangChain agents using `create_agent`.
    
    Uses LangChain's standard `create_agent` function which supports multiple providers
    (OpenAI, Anthropic, Google, etc.) through LangChain's chat model interface.
    """

    def __init__(self, agent: Agent, tools: Optional[List] = None):
        self.agent = agent
        self.tools = tools or []

    def execute(self, llm_chat_model, prompt: str, stream: bool = False):
        """
        Execute prompt through a LangChain agent.

        Args:
            llm_chat_model: A LangChain chat model instance
            prompt: User's question/prompt
            stream: Streaming support (not currently implemented)

        Returns:
            Dict with keys 'role', 'content', 'agent'
        """
        try:
            agent_obj = create_agent(
                model=llm_chat_model,
                tools=self.tools,
                system_prompt=self.agent.system_message
            )
            out = agent_obj.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            result_text = self._extract_response_text(out)
            return {"role": "assistant", "content": result_text, "agent": self.agent.name}
        except Exception as e:
            return {"role": "assistant", "content": f"Agent error: {str(e)}", "agent": self.agent.name}

    def _extract_response_text(self, out: Any) -> str:
        """Extract text content from LangChain agent output."""
        if isinstance(out, dict):
            if 'output' in out:
                return str(out['output'])
            elif 'messages' in out and out['messages']:
                last_message = out['messages'][-1]
                return last_message.content if hasattr(last_message, 'content') else str(last_message)
        elif isinstance(out, str):
            return out
        elif hasattr(out, 'content'):
            return out.content
        return str(out)

class AgentManager:
    """
    Manager class for handling multiple agents and their interactions.
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
    
