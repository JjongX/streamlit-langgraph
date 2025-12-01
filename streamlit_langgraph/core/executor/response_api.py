# ResponseAPIExecutor for OpenAI Responses API

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ...agent import Agent


class ResponseAPIExecutor:
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
            tools: Optional list of tools (not used for Response API, but kept for interface compatibility)
        """
        self.agent = agent
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._vector_store_ids = None
        self._tools_config = None
    
    def _build_tools_config(self, vector_store_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Build tools configuration for OpenAI Response API."""
        tools = []
        
        # Use provided vector_store_ids or stored ones
        vs_ids = vector_store_ids or self._vector_store_ids
        
        if self.agent.allow_file_search and vs_ids:
            tools.append({
                "type": "file_search",
                "vector_store_ids": vs_ids if isinstance(vs_ids, list) else [vs_ids]
            })
        
        if self.agent.allow_code_interpreter:
            tools.append({
                "type": "code_interpreter",
                "container": self.agent.container_id if self.agent.container_id else {"type": "auto"}
            })
        
        if self.agent.allow_web_search:
            tools.append({"type": "web_search_preview"})
        
        if self.agent.allow_image_generation:
            tools.append({"type": "image_generation", "partial_images": 3})
        
        # Add MCP tools if configured
        if self.agent.mcp_servers:
            from ...utils import MCPToolManager
            mcp_manager = MCPToolManager()
            mcp_manager.add_servers(self.agent.mcp_servers)
            mcp_tools = mcp_manager.get_openai_tools()
            tools.extend(mcp_tools)
        
        return tools
    
    def _update_vector_store_ids(self, llm_client: Any) -> None:
        """Update vector_store_ids from llm_client and invalidate tools config if changed."""
        current_vector_ids = getattr(llm_client, '_vector_store_ids', None)
        if current_vector_ids != self._vector_store_ids:
            self._vector_store_ids = current_vector_ids
            self._tools_config = None  # Invalidate cached tools config
    
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
    
    def _prepare_request_params(self, prompt: str, messages: Optional[List[Dict[str, Any]]] = None,
                                file_messages: Optional[List] = None, stream: bool = False) -> Dict[str, Any]:
        """
        Prepare request parameters for Responses API.
        
        Args:
            prompt: User's question/prompt
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            stream: Whether to enable streaming
            
        Returns:
            Request parameters dictionary
        """
        # Convert messages to input string for Responses API
        input_text = self._convert_messages_to_input(messages, prompt, file_messages)
        
        # Get or build tools configuration
        if self._tools_config is None:
            self._tools_config = self._build_tools_config(self._vector_store_ids)
        
        # Prepare request parameters for Responses API
        request_params = {
            "model": self.agent.model,
            "input": input_text,
        }
        
        if stream:
            request_params["stream"] = True
        
        # Add tools if configured
        if self._tools_config:
            request_params["tools"] = self._tools_config
        
        return request_params
    
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
        request_params = self._prepare_request_params(prompt, messages, file_messages, stream=stream)
        
        # Create response using Responses API
        response = self.openai_client.responses.create(**request_params)
        
        if stream:
            return {
                "role": "assistant",
                "content": "",
                "agent": self.agent.name,
                "stream": response
            }
        else:
            # Extract content from response
            content = self._extract_response_content(response)
            return {
                "role": "assistant",
                "content": content,
                "agent": self.agent.name
            }
    
    def _invoke_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Invoke the Response API (non-streaming).
        
        Args:
            prompt: User's question/prompt
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            
        Returns:
            Dict with 'role', 'content', and 'agent' keys
        """
        return self._call_response_api(prompt, stream=False, messages=messages, file_messages=file_messages)
    
    def _stream_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Invoke the Response API with streaming.
        
        Args:
            prompt: User's question/prompt
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            
        Returns:
            Dict with 'role', 'content', 'agent', and 'stream' key containing iterator
        """
        return self._call_response_api(prompt, stream=True, messages=messages, file_messages=file_messages)
    
    def _convert_messages_to_input(
        self,
        messages: Optional[List[Dict[str, Any]]],
        current_prompt: str,
        file_messages: Optional[List] = None
    ) -> str:
        """
        Convert workflow_state messages to a single input string for Responses API.
        
        The Responses API uses 'input' as a string parameter instead of 'messages' array.
        
        Args:
            messages: List of message dicts from workflow_state
            current_prompt: Current user prompt
            file_messages: Optional file messages (OpenAI format) to include
            
        Returns:
            Single input string combining all messages
        """
        input_parts = []
        
        # Add system message if available
        if self.agent.system_message:
            input_parts.append(self.agent.system_message)
        
        # Convert conversation history
        if messages:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if not content:
                    continue  # Skip empty messages
                
                if role == "user":
                    if isinstance(content, str):
                        input_parts.append(f"User: {content}")
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "input_text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "input_file":
                                    text_parts.append(f"[File: {block.get('file_id', 'unknown')}]")
                        if text_parts:
                            input_parts.append(f"User: {' '.join(text_parts)}")
                    else:
                        input_parts.append(f"User: {str(content)}")
                elif role == "assistant":
                    if isinstance(content, str):
                        input_parts.append(f"Assistant: {content}")
                    elif isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "output_text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            input_parts.append(f"Assistant: {' '.join(text_parts)}")
                    else:
                        input_parts.append(f"Assistant: {str(content)}")
        
        # Add file messages if provided
        if file_messages:
            for file_msg in file_messages:
                if isinstance(file_msg, dict):
                    role = file_msg.get("role", "user")
                    content = file_msg.get("content", [])
                    if role == "user" and content:
                        file_refs = []
                        for block in content if isinstance(content, list) else [content]:
                            if isinstance(block, dict):
                                if block.get("type") == "input_file":
                                    file_refs.append(f"[File: {block.get('file_id', 'unknown')}]")
                                elif block.get("type") == "input_text":
                                    file_refs.append(block.get("text", ""))
                        if file_refs:
                            input_parts.append(f"User: {' '.join(file_refs)}")
        
        # Add current prompt
        input_parts.append(f"User: {current_prompt}")
        
        # Combine all parts with newlines
        return "\n\n".join(input_parts)
    
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
        
        # Response API returns response with items
        text_parts = []
        
        # Try to get items from response
        if hasattr(response, 'items') and response.items:
            for item in response.items:
                if hasattr(item, 'type'):
                    if item.type == 'output_text':
                        if hasattr(item, 'text'):
                            text_parts.append(item.text)
                        elif hasattr(item, 'content'):
                            text_parts.append(str(item.content))
                    elif item.type == 'tool_call':
                        # Tool calls are handled by the API
                        pass
        
        # Try alternative structure (response.output_text or similar)
        if hasattr(response, 'output_text'):
            if isinstance(response.output_text, str):
                text_parts.append(response.output_text)
            elif hasattr(response.output_text, 'text'):
                text_parts.append(response.output_text.text)
        
        # Fallback: try to get text from response directly
        if not text_parts:
            if hasattr(response, 'text'):
                text_parts.append(response.text)
            elif hasattr(response, 'content'):
                text_parts.append(str(response.content))
        
        result = ''.join(text_parts) if text_parts else str(response) if response else ""
        return result

