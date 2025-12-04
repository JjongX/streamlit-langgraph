"""CreateAgentExecutor for LangChain agents."""

import json
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command

from ...agent import Agent
from .conversation_history import ConversationHistoryMixin


class CreateAgentExecutor(ConversationHistoryMixin):
    """
    Executor that builds LangChain agents using `create_agent`.
    
    Uses LangChain's standard `create_agent` function which supports multiple providers
    (OpenAI, Anthropic, Google, etc.) through LangChain's chat model interface.
    
    This executor is used for:
    - create_agent: LangChain's create_agent function
    - HITL (Human-in-the-Loop) scenarios (even if native tools are enabled, HITL requires this executor)
    - Multi-provider support (Anthropic, Google, etc.)
    
    HITL Handling: Uses LangChain's built-in `HumanInTheLoopMiddleware` which is automatically
    integrated into the agent during construction (in `_build_agent`)
    
    Note: For native OpenAI tools (code_interpreter, file_search, etc.) without HITL,
    use ResponseAPIExecutor instead.
    """

    def __init__(self, agent: Agent, tools: Optional[List] = None):
        """
        Initialize CreateAgentExecutor.
        
        Args:
            agent: Agent configuration
            tools: Optional list of LangChain tools
        """
        self.agent = agent
        self.agent_obj = None
        self._last_vector_store_ids = None
        self._init_conversation_history(agent)
        
        if tools is not None:
            self.tools = tools
        else:
            from ...utils import CustomTool, MCPToolManager
            
            custom_tools = CustomTool.get_langchain_tools(self.agent.tools) if self.agent.tools else []
            mcp_tools = []
            if self.agent.mcp_servers:
                mcp_manager = MCPToolManager()
                mcp_manager.add_servers(self.agent.mcp_servers)
                mcp_tools = mcp_manager.get_tools()
            self.tools = custom_tools + mcp_tools
    
    def execute_agent(
        self, llm_client: Any, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Execute prompt for single-agent mode (non-workflow).
        
        Single-agent mode: no checkpointer, no HITL, no thread_id needed.
        
        Args:
            llm_client: A LangChain chat model instance
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)

        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally 'stream'
        """
        try:
            if stream:
                return self._stream_agent(llm_client, prompt, messages, file_messages, config={})
            else:
                out = self._invoke_agent(llm_client, prompt, messages, file_messages, config={})
                result_text = self._extract_response_text(out)
                blocks = self._convert_message_to_blocks(result_text)
                self._add_to_conversation_history("assistant", blocks)
                return {"role": "assistant", "content": result_text, "agent": self.agent.name}
                
        except Exception as e:
            return {"role": "assistant", "content": f"Agent error: {str(e)}", "agent": self.agent.name}
    
    def execute_workflow(
        self, llm_client: Any, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None, 
    ) -> Dict[str, Any]:
        """
        Execute prompt for workflow mode (requires config with thread_id).

        Args:
            llm_client: A LangChain chat model instance
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            config: Execution config with thread_id (required for workflows)

        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' or 'stream' if HITL is active
        """
        try:
            config, workflow_thread_id = self._prepare_workflow_config(config)
            
            if stream:
                return self._stream_agent(llm_client, prompt, messages, file_messages, config=config)
            else:
                out = self._invoke_agent(llm_client, prompt, messages, file_messages, config=config)
                if isinstance(out, dict) and "__interrupt__" in out:
                    return self._create_interrupt_response(out["__interrupt__"], workflow_thread_id, config)
                result_text = self._extract_response_text(out)
                blocks = self._convert_message_to_blocks(result_text)
                self._add_to_conversation_history("assistant", blocks)
                return {"role": "assistant", "content": result_text, "agent": self.agent.name}

        except Exception as e:
            return {"role": "assistant", "content": f"Agent error: {str(e)}", "agent": self.agent.name}
    
    def resume(
        self, 
        decisions: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Resume agent execution after human approval/rejection.
        
        Args:
            decisions: List of decision dicts with 'type' ('approve', 'reject', 'edit') and optional 'edit' content
            config: Execution config with thread_id (workflow_thread_id)
            messages: Conversation history from workflow_state (unified message-based approach)
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if more approvals needed
        """
        if not self.agent.human_in_loop or not self.agent_obj:
            raise ValueError("Cannot resume: human-in-the-loop not enabled or agent not initialized")
        
        config, workflow_thread_id = self._prepare_workflow_config(config)
        resume_command = Command(resume={"decisions": decisions})
        out = self.agent_obj.invoke(resume_command, config=config)
        
        if isinstance(out, dict) and "__interrupt__" in out:
            return self._create_interrupt_response(out["__interrupt__"], workflow_thread_id, config)
        
        result_text = self._extract_response_text(out)
        blocks = self._convert_message_to_blocks(result_text)
        self._add_to_conversation_history("assistant", blocks)
        return {"role": "assistant", "content": result_text, "agent": self.agent.name}
    
    def detect_interrupt_in_stream(
        self, 
        execution_config: Dict[str, Any], 
        messages: List[BaseMessage]
    ) -> Optional[Any]:
        """Detect interrupt from agent stream events."""
        for event in self.agent_obj.stream(
            {"messages": messages},
            config=execution_config
        ):
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"]
                return list(interrupt_data) if isinstance(interrupt_data, (tuple, list)) else interrupt_data
            for node_state in event.values():
                if isinstance(node_state, dict) and "__interrupt__" in node_state:
                    return node_state["__interrupt__"]
                elif isinstance(node_state, (tuple, list)) and node_state:
                    return list(node_state) if isinstance(node_state, tuple) else node_state
        
        return None

    def _invoke_agent(
        self, llm_client: Any, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Invoke the agent (non-streaming).

        Args:
            llm_client: A LangChain chat model instance
            prompt: User's question/prompt
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            config: Optional execution config (for workflows)
            
        Returns:
            Agent output (dict or other format)
        """
        self._check_and_update_vector_store_ids(llm_client)
        
        if self.agent_obj is None:
            self._build_agent(llm_client)
        
        langchain_messages = self._convert_to_langchain_messages(messages, prompt, file_messages)
        execution_config = config if config is not None else {}
        out = self.agent_obj.invoke(
            {"messages": langchain_messages}, 
            config=execution_config
        )
        return out
    
    def _stream_agent(self, llm_client: Any, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke the agent with streaming support.
        
        Args:
            llm_client: A LangChain chat model instance
            prompt: User's question/prompt
            file_messages: Optional file messages (OpenAI format)
            messages: Conversation history from workflow_state
            config: Optional execution config (for workflows)
            
        Returns:
            Dict with 'role', 'content', 'agent', and 'stream' key containing iterator
        """
        self._check_and_update_vector_store_ids(llm_client)
        
        if self.agent_obj is None:
            self._build_agent(llm_client)
        
        langchain_messages = self._convert_to_langchain_messages(messages, prompt, file_messages)
        execution_config = config if config is not None else {}
        stream_iter = self.agent_obj.stream(
            {"messages": langchain_messages}, 
            config=execution_config,
            stream_mode="messages"
        )
        return {"role": "assistant", "content": "", "agent": self.agent.name, "stream": stream_iter}
  
    def _build_agent(self, llm_chat_model):
        """
        Build the agent with optional human-in-the-loop middleware.
        
        This executor uses LangChain's create_agent which works with ChatCompletion API.
        For native OpenAI tools without HITL, ResponseAPIExecutor should be used instead.
        """
        middleware = []
        if self.agent.human_in_loop and self.agent.interrupt_on:
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=self.agent.interrupt_on,
                    description_prefix=self.agent.hitl_description_prefix,
                )
            )
        
        all_tools = list(self.tools) if self.tools else []
        
        # Use original system message (conversation history will be in input messages)
        agent_kwargs = {
            "model": llm_chat_model,
            "tools": all_tools,
            "system_prompt": self._original_system_message,
        }
        
        if middleware:
            agent_kwargs["middleware"] = middleware
        
        if self.agent.human_in_loop and self.agent.interrupt_on:
            agent_checkpointer = InMemorySaver()
            agent_kwargs["checkpointer"] = agent_checkpointer
        
        self.agent_obj = create_agent(**agent_kwargs)
        
        return self.agent_obj
        
    def _convert_to_langchain_messages(self, 
        messages: Optional[List[Dict[str, Any]]], 
        current_prompt: str,
        file_messages: Optional[List] = None) -> List[BaseMessage]:
        """
        Convert workflow_state messages to LangChain message format.
        Conversation history is added as system message in input, current prompt is also included.
        
        Args:
            messages: List of message dicts from workflow_state
            current_prompt: Current user prompt
            file_messages: Optional file messages (OpenAI format) to include
            
        Returns:
            List containing system message (conversation history) and current prompt as HumanMessage
        """
        # Update conversation history from messages
        self._update_conversation_history_from_messages(messages, file_messages)
        
        # Add current prompt to conversation history if not already there
        if not self._conversation_history or not (
            self._conversation_history[-1].role == "user" and
            any(block.content == current_prompt for block in self._conversation_history[-1].blocks)
        ):
            current_blocks = self._convert_message_to_blocks(current_prompt)
            if current_blocks:
                self._add_to_conversation_history("user", current_blocks)
        
        langchain_messages: List[BaseMessage] = []
        langchain_messages.append(HumanMessage(content=current_prompt))
        # Add conversation history as system message after current prompt
        sections_dict = self._get_conversation_history_sections_dict()
        if sections_dict:
            system_content = json.dumps(sections_dict, ensure_ascii=False)
            langchain_messages.append(SystemMessage(content=system_content))
        
        return langchain_messages
    
    
    def _extract_response_text(self, out: Any) -> str:
        """Extract text content from LangChain agent output."""
        if isinstance(out, dict):
            if 'output' in out:
                output = out['output']
                if output:
                    return str(output)
            
            if 'messages' in out and out['messages']:
                messages = out['messages']
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, 'content') and msg.content:
                            if isinstance(msg.content, str):
                                return msg.content
                            elif isinstance(msg.content, list):
                                text_parts = []
                                for block in msg.content:
                                    if isinstance(block, dict) and block.get('type') == 'text':
                                        text_parts.append(block.get('text', ''))
                                    elif isinstance(block, str):
                                        text_parts.append(block)
                                if text_parts:
                                    return ''.join(text_parts)
                        return str(msg) if msg else ""
                
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        return ''.join(str(c) for c in content if c)
                return str(last_message) if last_message else ""
        
        elif isinstance(out, str):
            return out
        
        elif hasattr(out, 'content'):
            content = out.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return ''.join(str(c) for c in content if c)
            return str(content) if content else ""
        
        result = str(out) if out else ""
        return result
    
    def _prepare_workflow_config(self, config: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], str]:
        """
        Prepare workflow configuration and extract thread_id.
        
        Args:
            config: Execution config dictionary
            
        Returns:
            Tuple of (config, thread_id)
        """
        if config is None:
            raise ValueError(
                "config is required for workflow execution. "
                "It should contain thread_id from the workflow's checkpointer configuration."
            )
        if "configurable" not in config:
            config["configurable"] = {}
        
        if "thread_id" not in config["configurable"]:
            raise ValueError(
                "thread_id must be provided in config for workflow execution. "
                "It should come from the workflow's checkpointer configuration."
            )
        thread_id = config["configurable"]["thread_id"]
        
        return config, thread_id
    
    def _create_interrupt_response(self, interrupt_data: Any, thread_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create response dictionary for interrupt.
        
        Args:
            interrupt_data: Interrupt data from agent
            thread_id: Workflow thread ID
            config: Execution config
            
        Returns:
            Response dictionary with interrupt information
        """
        return {
            "role": "assistant",
            "content": "",
            "agent": self.agent.name,
            "__interrupt__": interrupt_data,
            "thread_id": thread_id,
            "config": config
        }
    
    def _check_and_update_vector_store_ids(self, llm_client: Any) -> None:
        """
        Check if vector_store_ids have changed and invalidate agent if needed.
        
        Args:
            llm_client: LLM client instance to check for vector_store_ids
        """
        current_vector_ids = getattr(llm_client, '_vector_store_ids', None)
        if not hasattr(self, '_last_vector_store_ids'):
            self._last_vector_store_ids = None
        
        if self.agent_obj is not None and current_vector_ids != self._last_vector_store_ids:
            self.agent_obj = None
        
        self._last_vector_store_ids = current_vector_ids
