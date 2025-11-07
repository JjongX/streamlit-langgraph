import json
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

def load_agents_from_yaml(yaml_path: str) -> List[Agent]:
    """
    Load multiple Agent instances from a YAML configuration file.
    
    This function is designed for multi-agent configurations. For single agents,
    use the Agent class directly: Agent(name="...", role="...", ...)
    
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
    """
    Executor for OpenAI Responses API generation.
    
    Supports human-in-the-loop approval when enabled via agent configuration.
    When HITL is enabled, uses Chat Completions API with function calling for tool interception.
    """

    def __init__(self, agent: Agent, thread_id: Optional[str] = None):
        self.agent = agent
        self.thread_id = thread_id or str(uuid.uuid4())
        # Store conversation history for HITL resume
        self.conversation_history: List[Dict[str, Any]] = []
        # Store pending tool calls for HITL
        self.pending_tool_calls: List[Dict[str, Any]] = []

    def execute(self, llm_client, prompt: str, stream: bool = True, file_messages: Optional[List] = None, config: Optional[Dict[str, Any]] = None):
        """
        Execute prompt using the Responses API client.
        
        Supports file messages, code interpreter, web search, and image generation tools.
        Note: file_search requires vector_store_ids and is handled by FileHandler.
        
        When human_in_loop is enabled, uses Chat Completions API to intercept tool calls.

        Args:
            llm_client: Client exposing `responses.create` method
            prompt: Full prompt or input to send
            stream: Whether to request a streaming response
            file_messages: Optional list of file-related input messages
            config: Optional execution config with thread_id

        Returns:
            Dict with keys 'role', 'content', 'agent', optionally 'stream', and '__interrupt__' if HITL is active
        """
        execution_config = config or {"configurable": {"thread_id": self.thread_id}}
        thread_id = execution_config.get("configurable", {}).get("thread_id", self.thread_id)
        
        # If HITL is enabled, use Chat Completions API for tool interception
        if self.agent.human_in_loop and self.agent.interrupt_on:
            return self._execute_with_hitl(llm_client, prompt, stream, file_messages, thread_id, execution_config)
        
        # Normal execution using Responses API
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
    
    def _execute_with_hitl(self, llm_client, prompt: str, stream: bool, file_messages: Optional[List], 
                           thread_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute with human-in-the-loop support using Chat Completions API.
        
        This allows us to intercept tool calls before execution.
        """
        # Build messages from conversation history
        messages = self.conversation_history.copy() if self.conversation_history else []
        
        # Add file messages if provided
        if file_messages:
            messages.extend(file_messages)
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Build function tools from CustomTool registry
        from .utils import CustomTool
        custom_tools = CustomTool.get_openai_tools(self.agent.tools) if self.agent.tools else []
        
        # Also include built-in tools as function definitions if needed
        # Note: Built-in tools (code_interpreter, web_search, image_generation) are handled differently
        # We'll focus on custom tools for HITL
        
        try:
            # Use Chat Completions API for tool interception
            response = llm_client.chat.completions.create(
                model=self.agent.model,
                messages=messages,
                temperature=self.agent.temperature,
                tools=custom_tools if custom_tools else None,
                tool_choice="auto" if custom_tools else None
            )
            
            message = response.choices[0].message
            
            # Check for tool calls that need approval
            if message.tool_calls:
                interrupt_data = self._check_tool_calls_for_interrupt(message.tool_calls)
                if interrupt_data:
                    # Store conversation state including the assistant message with tool_calls
                    # This is critical for resume() to work correctly
                    self.pending_tool_calls = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
                    assistant_message_with_tools = self._message_to_dict(message)
                    self.conversation_history = messages + [assistant_message_with_tools]
                    interrupt_response = self._create_interrupt_response(interrupt_data, thread_id, config)
                    return interrupt_response
            
            # No interrupt needed, continue execution
            # Add assistant message to history
            self.conversation_history = messages + [self._message_to_dict(message)]
            
            # If there are tool calls, execute them (for non-interrupted tools)
            if message.tool_calls:
                # Execute tool calls and continue conversation
                tool_results = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Execute tool (this should be handled by CustomTool)
                    tool_result = self._execute_tool(tool_name, tool_args)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(tool_result)
                    })
                
                # Continue conversation with tool results
                messages_with_tools = self.conversation_history + tool_results
                followup_response = llm_client.chat.completions.create(
                    model=self.agent.model,
                    messages=messages_with_tools,
                    temperature=self.agent.temperature,
                    tools=custom_tools if custom_tools else None,
                    tool_choice="auto" if custom_tools else None
                )
                
                final_message = followup_response.choices[0].message
                self.conversation_history = messages_with_tools + [self._message_to_dict(final_message)]
                
                content = final_message.content or ""
            else:
                content = message.content or ""
            
            return {"role": "assistant", "content": content, "agent": self.agent.name}
            
        except Exception as e:
            return {"role": "assistant", "content": f"Responses API error: {str(e)}", "agent": self.agent.name}
    
    def _check_tool_calls_for_interrupt(self, tool_calls: List[Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Check if any tool calls require human approval based on interrupt_on config.
        
        Returns interrupt data in the format expected by HITL system.
        """
        if not self.agent.interrupt_on:
            return None
        
        action_requests = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # Check if this tool requires interruption
            should_interrupt = False
            if tool_name in self.agent.interrupt_on:
                tool_config = self.agent.interrupt_on[tool_name]
                if isinstance(tool_config, dict):
                    should_interrupt = True
                elif tool_config is True:
                    should_interrupt = True
            
            if should_interrupt:
                # Create action request in format expected by HITL
                action_requests.append({
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_call.id,
                    "description": f"{self.agent.hitl_description_prefix or 'Tool execution pending approval'}: {tool_name}"
                })
        
        return action_requests if action_requests else None
    
    def _tool_call_to_dict(self, tool_call: Any) -> Dict[str, Any]:
        """Convert OpenAI tool call to dictionary."""
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
        }
    
    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        """Convert OpenAI message to dictionary."""
        msg_dict = {
            "role": message.role,
            "content": message.content or ""
        }
        if hasattr(message, 'tool_calls') and message.tool_calls:
            msg_dict["tool_calls"] = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
        return msg_dict
    
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        from .utils import CustomTool
        tool_func = CustomTool.get_tool_function(tool_name)
        if tool_func:
            return tool_func(**tool_args)
        return f"Tool {tool_name} not found"
    
    def _create_interrupt_response(self, interrupt_data: List[Dict[str, Any]], thread_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
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
        Resume execution after human approval/rejection.
        
        Args:
            decisions: List of decision dicts with 'type' ('approve', 'reject', 'edit') and optional 'edit' content
            config: Execution config with thread_id
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if more approvals needed
        """
        if not self.agent.human_in_loop:
            raise ValueError("Cannot resume: human-in-the-loop not enabled")
        
        execution_config = config or {"configurable": {"thread_id": self.thread_id}}
        thread_id = execution_config.get("configurable", {}).get("thread_id", self.thread_id)
        
        # Get LLM client
        llm_client = get_llm_client(self.agent)
        
        # Apply decisions to pending tool calls
        tool_results = []
        for i, decision in enumerate(decisions):
            if i >= len(self.pending_tool_calls):
                break
            
            tool_call_dict = self.pending_tool_calls[i]
            tool_name = tool_call_dict["function"]["name"]
            tool_args = json.loads(tool_call_dict["function"]["arguments"])
            
            decision_type = decision.get("type", "approve")
            
            if decision_type == "reject":
                # Skip this tool call
                continue
            elif decision_type == "edit":
                # Use edited arguments
                tool_args = decision.get("edit", tool_args)
            
            # Execute approved/edited tool
            tool_result = self._execute_tool(tool_name, tool_args)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_dict["id"],
                "name": tool_name,
                "content": str(tool_result)
            })
        
        # Conversation history should already have the assistant message with tool_calls
        # (stored when interrupt was detected). Just add tool results.
        messages_with_tools = self.conversation_history.copy()
        
        # Verify the last message is an assistant message with tool_calls
        if not messages_with_tools:
            raise ValueError("Cannot resume: conversation history is empty")
        
        last_message = messages_with_tools[-1]
        if last_message.get("role") != "assistant" or "tool_calls" not in last_message:
            # Fallback: reconstruct assistant message from pending_tool_calls
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": self.pending_tool_calls
            }
            messages_with_tools.append(assistant_message)
        
        # Add tool results (must come after assistant message with tool_calls)
        messages_with_tools.extend(tool_results)
        
        from .utils import CustomTool
        custom_tools = CustomTool.get_openai_tools(self.agent.tools) if self.agent.tools else []
        
        try:
            followup_response = llm_client.chat.completions.create(
                model=self.agent.model,
                messages=messages_with_tools,
                temperature=self.agent.temperature,
                tools=custom_tools if custom_tools else None,
                tool_choice="auto" if custom_tools else None
            )
            
            message = followup_response.choices[0].message
            self.conversation_history = messages_with_tools + [self._message_to_dict(message)]
            
            # Check for additional tool calls that need approval
            if message.tool_calls:
                interrupt_data = self._check_tool_calls_for_interrupt(message.tool_calls)
                if interrupt_data:
                    self.pending_tool_calls = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
                    return self._create_interrupt_response(interrupt_data, thread_id, execution_config)
            
            # Clear pending tool calls
            self.pending_tool_calls = []
            
            content = message.content or ""
            return {"role": "assistant", "content": content, "agent": self.agent.name}
            
        except Exception as e:
            return {"role": "assistant", "content": f"Error resuming execution: {str(e)}", "agent": self.agent.name}

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
    
