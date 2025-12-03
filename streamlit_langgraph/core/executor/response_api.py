# ResponseAPIExecutor for OpenAI Responses API

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ...agent import Agent
from .conversation_history import ConversationHistoryMixin


class ResponseAPIExecutor(ConversationHistoryMixin):
    """
    Executor that uses OpenAI's native Responses API directly.
    
    This executor is used when native OpenAI tools (code_interpreter, file_search,
    web_search, image_generation) are enabled and HITL is not enabled.
    
    The Response API does not support HITL because it cannot intercept tool calls.
    For HITL scenarios, use CreateAgentExecutor instead.
    """
    
    def __init__(self, agent: Agent, tools: Optional[List] = None):
        """
        Initialize ResponseAPIExecutor.
        
        Args:
            agent: Agent configuration
            tools: Optional list of tools (stored for interface compatibility, though Response API uses native tools)
        """
        self.agent = agent
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._vector_store_ids = None
        self._tools_config = None
        self.tools = tools if tools is not None else []
        self._init_conversation_history(agent)
    
    def execute_agent(
        self, llm_client: Any, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Execute prompt for single-agent mode (non-workflow).
        
        Args:
            llm_client: Used to get vector_store_ids (kept for interface compatibility)
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally 'stream'
        """
        return self._execute(llm_client, prompt, stream, messages, file_messages)
    
    def execute_workflow(
        self, llm_client: Any, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute prompt for workflow mode.
        
        Note: Response API does not support HITL, so this method does not handle interrupts.
        For HITL scenarios, use CreateAgentExecutor instead.
        
        Args:
            llm_client: Used to get vector_store_ids (kept for interface compatibility)
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            config: Execution config (not used for Response API, but kept for interface compatibility)
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally 'stream'
        """
        return self._execute(llm_client, prompt, stream, messages, file_messages)
    
    def _execute(
        self, llm_client: Any, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Common execution logic for both agent and workflow modes.
        
        Args:
            llm_client: Used to get vector_store_ids (kept for interface compatibility)
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally 'stream'
        """
        self._update_vector_store_ids(llm_client)
        
        if stream:
            return self._stream_response_api(prompt, messages, file_messages)
        else:
            return self._invoke_response_api(prompt, messages, file_messages)
    
    def _invoke_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None
    ) -> Dict[str, Any]:
        """Invoke the Response API."""
        return self._call_response_api(prompt, stream=False, messages=messages, file_messages=file_messages)
    
    def _stream_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None
    ) -> Dict[str, Any]:
        """Stream the Response API."""
        return self._call_response_api(prompt, stream=True, messages=messages, file_messages=file_messages)
    
    def _call_response_api(
        self, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Call the Response API (streaming or non-streaming).
        
        Args:
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            
        Returns:
            Dict with 'role', 'content', 'agent', and optionally 'stream' key
        """
        input_text = self._convert_messages_to_input(messages, prompt, file_messages)
        tools_config = self._build_tools_config(self._vector_store_ids, stream=stream)
        
        response = self.openai_client.responses.create(
            model=self.agent.model,
            input=input_text,
            instructions=self._original_system_message,
            temperature=self.agent.temperature,
            tools=tools_config if tools_config else [],
            stream=stream,
            reasoning={"summary": "auto"},
        )
        
        if stream:
            # For streaming, we'll add the response to history after streaming completes
            return {
                "role": "assistant",
                "content": "",
                "agent": self.agent.name,
                "stream": response
            }
        else:
            content = self._extract_response_content(response)
            blocks = self._convert_message_to_blocks(content)
            self._add_to_conversation_history("assistant", blocks)
            return {
                "role": "assistant",
                "content": content,
                "agent": self.agent.name
            }
    
    def _build_tools_config(self, vector_store_ids: Optional[List[str]] = None, stream: bool = True) -> List[Dict[str, Any]]:
        """Build tools configuration for OpenAI Response API."""
        tools = []
        vs_ids = vector_store_ids or self._vector_store_ids
        
        if self.agent.allow_file_search and vs_ids:
            tools.append({"type": "file_search", "vector_store_ids": vs_ids if isinstance(vs_ids, list) else [vs_ids]})
        if self.agent.allow_code_interpreter:
            tools.append({"type": "code_interpreter", "container": self.agent.container_id if self.agent.container_id else {"type": "auto"}})
        if self.agent.allow_web_search:
            tools.append({"type": "web_search"})
        if self.agent.allow_image_generation:
            tools.append({"type": "image_generation", "partial_images": 3} if stream else {"type": "image_generation"})
        
        if self.agent.mcp_servers:
            from ...utils import MCPToolManager
            mcp_manager = MCPToolManager()
            mcp_manager.add_servers(self.agent.mcp_servers)
            mcp_tools = mcp_manager.get_openai_tools()
            tools.extend(mcp_tools)
        
        if self.tools:
            for tool in self.tools:
                openai_tool = self._convert_langchain_tool_to_openai(tool)
                if openai_tool:
                    tools.append(openai_tool)
        
        return tools
    
    def _convert_langchain_tool_to_openai(self, tool: Any) -> Dict[str, Any]:
        """Convert a LangChain StructuredTool to OpenAI function format."""
        from langchain_core.tools import StructuredTool
        
        if not isinstance(tool, StructuredTool):
            if isinstance(tool, dict) and "type" in tool:
                return tool
            return None
        
        args_schema = tool.args_schema
        properties = {}
        required = []
        
        if args_schema:
            schema_dict = args_schema.schema() if hasattr(args_schema, 'schema') else {}
            properties = schema_dict.get("properties", {})
            required = schema_dict.get("required", [])
        
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def _convert_messages_to_input(
        self,
        messages: Optional[List[Dict[str, Any]]],
        current_prompt: str,
        file_messages: Optional[List] = None
    ) -> str:
        """
        Convert workflow_state messages to input string.
        Conversation history is added as system message in input, current prompt is also included.
        
        Args:
            messages: List of message dicts from workflow_state
            current_prompt: Current user prompt
            file_messages: Optional file messages (OpenAI format) to include
            
        Returns:
            Input string with system message (conversation history) and current prompt
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
        
        input_parts = []
        input_parts.append(current_prompt)
        # Add conversation history as system message after current prompt
        sections_dict = self._get_conversation_history_sections_dict()
        if sections_dict:
            system_content = json.dumps(sections_dict, ensure_ascii=False)
            system_msg = json.dumps({"role": "system", "content": system_content}, ensure_ascii=False)
            input_parts.append(system_msg)
        
        return "\n\n".join(input_parts) if input_parts else current_prompt
    
    
    def _extract_response_content(self, response: Any) -> str:
        """
        Extract text content from OpenAI Response API response.
        
        Args:
            response: OpenAI Response API response object
            
        Returns:
            Extracted text content
        """
        if not response:
            return ""
        
        text_parts = []
        
        if hasattr(response, 'items') and response.items:
            for item in response.items:
                if hasattr(item, 'type') and item.type == 'output_text':
                    if hasattr(item, 'text'):
                        text_parts.append(item.text)
                    elif hasattr(item, 'content'):
                        text_parts.append(str(item.content))
        
        if hasattr(response, 'output_text'):
            if isinstance(response.output_text, str):
                text_parts.append(response.output_text)
            elif hasattr(response.output_text, 'text'):
                text_parts.append(response.output_text.text)
        
        if not text_parts:
            if hasattr(response, 'text'):
                text_parts.append(response.text)
            elif hasattr(response, 'content'):
                text_parts.append(str(response.content))
        
        result = ''.join(text_parts) if text_parts else str(response) if response else ""
        return result
    
    def _update_vector_store_ids(self, llm_client: Any) -> None:
        """Update vector_store_ids from llm_client and invalidate tools config if changed."""
        current_vector_ids = getattr(llm_client, '_vector_store_ids', None)
        if current_vector_ids != self._vector_store_ids:
            self._vector_store_ids = current_vector_ids
            self._tools_config = None
