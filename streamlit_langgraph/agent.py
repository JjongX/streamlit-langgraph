from typing import List, Optional, Dict
from dataclasses import dataclass, field

from langchain.agents import create_agent
import logging

logger = logging.getLogger(__name__)

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
    allow_web_search: bool = False
    allow_image_generation: bool = False
    tools: List[str] = field(default_factory=list)
    container_id: Optional[str] = None  # For code_interpreter functionality

    def __post_init__(self):
        """Post-initialization processing and validation."""
        if not self.provider:
            self.provider = "openai"
        if not self.model:
            self.model = "gpt-4o"
        if not self.type:
            raise ValueError("Agent must specify type ('response' or 'agent')")
        if self.type not in ("response", "agent"):
            raise ValueError("Agent 'type' must be either 'response' or 'agent'")
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
            "provider": self.provider,
            "model": self.model,
            "type": self.type,
            "tools": self.tools,
            "temperature": self.temperature,
            "allow_file_search": self.allow_file_search,
            "allow_code_interpreter": self.allow_code_interpreter,
            "allow_web_search": self.allow_web_search,
            "allow_image_generation": self.allow_image_generation,
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

    def _create_container_if_needed(self, llm_client):
        """Create container for code_interpreter if needed and not already created."""
        current_container_id = getattr(self.agent, 'container_id', None)
        if self.agent.allow_code_interpreter and not current_container_id:
            logger.info(f"Creating container for agent {self.agent.name}")
            try:
                container = llm_client.containers.create(name=f"streamlit-{self.agent.name}")
                self.agent.container_id = container.id
                logger.info(f"Successfully created container {container.id} for agent {self.agent.name}")
            except Exception as e:
                logger.error(f"Failed to create container for agent {self.agent.name}: {str(e)}")
                # Disable code_interpreter if container creation fails
                self.agent.allow_code_interpreter = False
        elif self.agent.allow_code_interpreter:
            logger.info(f"Agent {self.agent.name} already has container_id: {current_container_id}")
    
    def _check_container_status(self, llm_client):
        """Check container status and recreate if expired."""
        container_id = getattr(self.agent, 'container_id', None)
        if self.agent.allow_code_interpreter and container_id:
            logger.info(f"Checking container status for agent {self.agent.name}, container_id: {container_id}")
            
            if not container_id:
                logger.warning(f"Agent {self.agent.name} has no container_id, creating one")
                self._create_container_if_needed(llm_client)
                return
            
            try:
                result = llm_client.containers.retrieve(container_id=container_id)
                logger.info(f"Container {container_id} status: {result.status}")
                if result.status == "expired":
                    logger.info(f"Container {container_id} expired, creating new one")
                    container = llm_client.containers.create(name=f"streamlit-{self.agent.name}")
                    self.agent.container_id = container.id
                    logger.info(f"Created new container {container.id} to replace expired one")
            except Exception as e:
                logger.error(f"Failed to check/recreate container: {str(e)}")
                # Try to create a new container
                logger.info("Attempting to create new container after check failure")
                self._create_container_if_needed(llm_client)

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
        model = self.agent.model or "gpt-4o"
        temperature = self.agent.temperature

        # Build input messages including file messages
        input_messages = []
        if file_messages:
            input_messages.extend(file_messages)
        input_messages.append({"role": "user", "content": prompt})

        # Handle container creation and management for code_interpreter
        if self.agent.allow_code_interpreter:
            logger.info(f"Agent {self.agent.name} has code_interpreter enabled")
            self._create_container_if_needed(llm_client)
            self._check_container_status(llm_client)

        # Build tools configuration based on agent capabilities
        tools = []
        if self.agent.allow_code_interpreter:
            container_id_value = getattr(self.agent, 'container_id', None)
            logger.info(f"Agent {self.agent.name}: container_id={container_id_value}")
            
            if container_id_value:
                tools.append({
                    "type": "code_interpreter",
                    "container": container_id_value
                })
                logger.info(f"Added code_interpreter tool with container: {container_id_value}")
            else:
                logger.warning(f"Agent {self.agent.name}: code_interpreter enabled but no valid container_id")
        # Note: file_search requires vector_store_ids, so we skip it if none are configured
        # This will be handled by the FileHandler's dynamic vector store creation
        # if self.agent.allow_file_search:
        #     tools.append({"type": "file_search"})
        if self.agent.allow_web_search:
            tools.append({"type": "web_search"})
        if self.agent.allow_image_generation:
            tools.append({"type": "image_generation", "partial_images": 3})

        logger.info(f"Final tools configuration for agent {self.agent.name}: {tools}")

        try:
            # Prepare API call parameters
            api_params = {
                "model": model,
                "input": input_messages,
                "temperature": temperature,
                "stream": stream,
            }
            
            # Add tools if any are enabled
            if tools:
                api_params["tools"] = tools
                logger.info(f"API call will include tools: {tools}")

            if stream:
                stream_iter = llm_client.responses.create(**api_params)
                return {"role": "assistant", "content": "", "agent": self.agent.name, "stream": stream_iter}
            else:
                response = llm_client.responses.create(**api_params)

                response_content = ""
                # Responses API shape can vary; try a few extraction strategies
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
            logger.exception("Responses API call failed")
            return {"role": "assistant", "content": f"Responses API error: {str(e)}", "agent": self.agent.name}

class CreateAgentExecutor:
    """Executor that builds a LangChain agent using `create_agent`.

    This executor is intended for non-OpenAI providers (for example, Gemini via
    a LangChain chat model). It will initialize a chat model using
    `init_chat_model` and call `create_agent` to produce an agent instance.
    """

    def __init__(self, agent: Agent, tools: Optional[List] = None):
        self.agent = agent
        self.tools = tools or []

    def execute(self, llm_chat_model, prompt: str, stream: bool = False):
        """Run the prompt through a LangChain-created agent.

        Args:
            llm_chat_model: A LangChain chat model (returned by init_chat_model)
            prompt: Prompt string
            stream: Streaming support is implementation-dependent; default False

        Returns:
            Dict with keys 'role','content','agent'
        """
        try:
            # Build agent using LangChain helper
            agent_obj = create_agent(
                model=self.agent.model,
                tools=self.tools,
                system_prompt=self.agent.system_message
            )

            # Different LangChain agent implementations expose different run/invoke APIs.
            result_text = ""
            try:
                # `invoke` is a common method on new agent APIs
                if hasattr(agent_obj, 'invoke'):
                    out = agent_obj.invoke({"input": prompt})
                    # try to extract a string
                    if isinstance(out, dict) and 'output' in out:
                        result_text = str(out['output'])
                    else:
                        result_text = str(out)
                # fall back to run
                elif hasattr(agent_obj, 'run'):
                    result_text = str(agent_obj.run(prompt))
                else:
                    # last resort: try calling the object
                    result_text = str(agent_obj(prompt))

            except Exception:
                # If the agent returns a complex object, stringify gracefully
                logger.exception("Error calling LangChain agent; attempting best-effort stringify")
                try:
                    result_text = str(out)
                except Exception:
                    result_text = "<unable to extract agent output>"

            return {"role": "assistant", "content": result_text, "agent": self.agent.name}

        except Exception as e:
            logger.exception("create_agent execution failed")
            return {"role": "assistant", "content": f"create_agent error: {str(e)}", "agent": self.agent.name}

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
