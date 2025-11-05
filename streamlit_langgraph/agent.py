import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import uuid

import openai
import yaml
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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
    # Human-in-the-loop configuration (only available for multiagent workflows)
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
    
    Supports human-in-the-loop approval when enabled via agent configuration.
    """

    def __init__(self, agent: Agent, tools: Optional[List] = None, thread_id: Optional[str] = None):
        self.agent = agent
        self.thread_id = thread_id or str(uuid.uuid4())
        self.checkpointer = None
        self.agent_obj = None
        
        # Initialize checkpointer if human-in-the-loop is enabled
        if self.agent.human_in_loop:
            self.checkpointer = MemorySaver()
        
        # Build tools configuration from CustomTool registry if not explicitly provided
        if tools is not None:
            self.tools = tools
        else:
            from .utils import CustomTool # lazy import to avoid circular import
            self.tools = CustomTool.get_langchain_tools(self.agent.tools) if self.agent.tools else []
    
    def _build_agent(self, llm_chat_model):
        """Build the agent with optional human-in-the-loop middleware."""
        middleware = []
        if self.agent.human_in_loop and self.agent.interrupt_on:
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=self.agent.interrupt_on,
                    description_prefix=self.agent.hitl_description_prefix,
                )
            )
        
        agent_kwargs = {
            "model": llm_chat_model,
            "tools": self.tools,
            "system_prompt": self.agent.system_message,
        }
        
        if middleware:
            agent_kwargs["middleware"] = middleware
        
        if self.checkpointer:
            agent_kwargs["checkpointer"] = self.checkpointer
        
        self.agent_obj = create_agent(**agent_kwargs)
        return self.agent_obj

    def execute(self, llm_chat_model, prompt: str, stream: bool = False, config: Optional[Dict[str, Any]] = None):
        """
        Execute prompt through a LangChain agent.

        Args:
            llm_chat_model: A LangChain chat model instance
            prompt: User's question/prompt
            stream: Streaming support (not currently implemented)
            config: Optional execution config with thread_id and interrupt handling

        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if HITL is active
        """
        try:
            if self.agent_obj is None or (self.agent.human_in_loop and self.checkpointer is None):
                self._build_agent(llm_chat_model)
            
            execution_config = config or {"configurable": {"thread_id": self.thread_id}}
            thread_id = execution_config.get("configurable", {}).get("thread_id", self.thread_id)
            
            # Stream events to detect interrupts
            interrupt_data = self._detect_interrupt_in_stream(execution_config, prompt)
            if interrupt_data:
                return self._create_interrupt_response(interrupt_data, thread_id, execution_config)
            
            # Execute normally if no interrupt detected
            out = self.agent_obj.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=execution_config
            )
            
            # Check for interrupt in output
            if isinstance(out, dict) and "__interrupt__" in out:
                return self._create_interrupt_response(out["__interrupt__"], thread_id, execution_config)
            
            result_text = self._extract_response_text(out)
            return {"role": "assistant", "content": result_text, "agent": self.agent.name}
        except Exception as e:
            return {"role": "assistant", "content": f"Agent error: {str(e)}", "agent": self.agent.name}
    
    def _detect_interrupt_in_stream(self, execution_config: Dict[str, Any], prompt: str) -> Optional[Any]:
        """Detect interrupt from agent stream events."""
        for event in self.agent_obj.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=execution_config
        ):
            # Check for direct interrupt key
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"]
                return list(interrupt_data) if isinstance(interrupt_data, (tuple, list)) else interrupt_data
            
            # Check each node in the event
            for node_state in event.values():
                if isinstance(node_state, dict) and "__interrupt__" in node_state:
                    return node_state["__interrupt__"]
                elif isinstance(node_state, (tuple, list)) and node_state:
                    return list(node_state) if isinstance(node_state, tuple) else node_state
        
        return None
    
    def _create_interrupt_response(self, interrupt_data: Any, thread_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create response dict for interrupt."""
        return {
            "role": "assistant",
            "content": "",
            "agent": self.agent.name,
            "__interrupt__": interrupt_data,
            "thread_id": thread_id,
            "config": config
        }
    
    def resume(self, decisions: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None):
        """
        Resume agent execution after human approval/rejection.
        
        Args:
            decisions: List of decision dicts with 'type' ('approve', 'reject', 'edit') and optional 'edit' content
            config: Execution config with thread_id
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if more approvals needed
        """
        if not self.agent.human_in_loop or not self.agent_obj:
            raise ValueError("Cannot resume: human-in-the-loop not enabled or agent not initialized")
        
        execution_config = config or {"configurable": {"thread_id": self.thread_id}}
        thread_id = execution_config.get("configurable", {}).get("thread_id", self.thread_id)
        
        resume_command = Command(resume={"decisions": decisions})
        out = self.agent_obj.invoke(resume_command, config=execution_config)
        
        # Check for additional interrupts
        if isinstance(out, dict) and "__interrupt__" in out:
            return self._create_interrupt_response(out["__interrupt__"], thread_id, execution_config)
        
        result_text = self._extract_response_text(out)
        return {"role": "assistant", "content": result_text, "agent": self.agent.name}

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
    
