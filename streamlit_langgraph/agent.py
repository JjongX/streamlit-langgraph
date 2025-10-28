from typing import List, Optional, Dict
from dataclasses import dataclass, field

from langchain.agents import create_agent

@dataclass
class Agent:
    """
    Configuration class for defining individual agents in a multiagent system.
    Required fields: name, role, instructions, type ('response' or 'agent').
    provider and model default to 'openai' and 'gpt-4o' if not specified.
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

    def __post_init__(self):
        """Post-initialization processing and validation."""
        if self.type not in ("response", "agent"):
            raise ValueError("Agent 'type' must be either 'response' or 'agent'.")
        # Compose system_message if not provided
        if self.system_message is None:
            self.system_message = f"You are a {self.role}. {self.instructions}"
        # Auto-enable native tools based on tools list
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

class ResponseAPIExecutor:
    """Executor that uses the OpenAI Responses API (or compatible) for generation.

    This executor expects an LLM client instance exposing a `responses.create`
    method (for example `openai.OpenAI`). It returns a dict compatible with
    the rest of the codebase: {'role','content','agent'} and optionally
    a 'stream' generator under streaming mode.
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    def execute(self, llm_client, prompt: str, stream: bool = True, file_messages: Optional[List] = None):
        """Execute prompt using the Responses API client.

        Args:
            llm_client: Client exposing `responses.create`
            prompt: Full prompt or input to send
            stream: Whether to request a streaming response
            file_messages: Optional list of file-related input messages

        Returns:
            Dict with keys 'role','content','agent' and optionally 'stream'
        """
        # Build input messages including file messages
        input_messages = []
        if file_messages:
            input_messages.extend(file_messages)
        input_messages.append({"role": "user", "content": prompt})

        # Build tools configuration based on agent capabilities
        tools = []
        # Note: file_search requires vector_store_ids, so we skip it if none are configured
        # This will be handled by the FileHandler's dynamic vector store creation
        if self.agent.allow_code_interpreter:
            container = llm_client.containers.create(name=f"streamlit-{self.agent.name}")
            self.agent.container_id = container.id
            tools.append({"type": "code_interpreter", "container": self.agent.container_id})
        if self.agent.allow_web_search:
            tools.append({"type": "web_search"})
        if self.agent.allow_image_generation:
            tools.append({"type": "image_generation", "partial_images": 3})

        try:
            # Prepare API call parameters
            api_params = {
                "model": self.agent.model,
                "input": input_messages,
                "temperature": self.agent.temperature,
                "stream": stream,
            }
            if tools:
                api_params["tools"] = tools

            if stream: # Streaming
                stream_iter = llm_client.responses.create(**api_params)
                return {"role": "assistant", "content": "", "agent": self.agent.name, "stream": stream_iter}
            else:
                response = llm_client.responses.create(**api_params)
                response_content = ""
                if hasattr(response, 'output') and isinstance(response.output, list):
                    for message in response.output:
                        if hasattr(message, 'content') and isinstance(message.content, list):
                            for content_item in message.content:
                                if hasattr(content_item, 'text'):
                                    response_content += content_item.text
                elif hasattr(response, 'content'):
                    response_content = str(response.content)
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    response_content = str(response.message.content)

                return {"role": "assistant", "content": response_content, "agent": self.agent.name}

        except Exception as e:
            return {"role": "assistant", "content": f"Responses API error: {str(e)}", "agent": self.agent.name}

class CreateAgentExecutor:
    """Executor that builds a LangChain using `create_agent`.

    Uses LangChain's standard `create_agent` function which works with any provider
    (OpenAI, Anthropic, Google, etc.) via init_chat_model.
    """

    def __init__(self, agent: Agent, tools: Optional[List] = None):
        self.agent = agent
        self.tools = tools or []

    def execute(self, llm_chat_model, prompt: str, stream: bool = False):
        """Execute prompt through a LangChain v1 agent.

        Args:
            llm_chat_model: A LangChain chat model instance
            prompt: User's question/prompt
            stream: Streaming support (not currently implemented)

        Returns:
            Dict with keys 'role', 'content', 'agent'
        """
        try:
            # Create agent using LangChain's create_agent
            agent_obj = create_agent(
                model=llm_chat_model,
                tools=self.tools,
                system_prompt=self.agent.system_message
            )

            # Invoke agent with proper message format
            out = agent_obj.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            
            # Extract response text
            result_text = ""
            if isinstance(out, dict):
                if 'output' in out:
                    result_text = str(out['output'])
                elif 'messages' in out and out['messages']:
                    last_message = out['messages'][-1]
                    result_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            elif isinstance(out, str):
                result_text = out
            elif hasattr(out, 'content'):
                result_text = out.content
            else:
                result_text = str(out)

            return {"role": "assistant", "content": result_text, "agent": self.agent.name}

        except Exception as e:
            return {"role": "assistant", "content": f"Agent error: {str(e)}", "agent": self.agent.name}

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
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all agent names."""
        return list(self.agents.keys())
