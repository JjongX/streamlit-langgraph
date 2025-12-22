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
        self.update_vector_store_ids(llm_client)
        
        if stream:
            return self._stream_response_api(prompt, messages, file_messages)
        else:
            return self.invoke_response_api(prompt, messages, file_messages)
    
    def invoke_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        delegation_tool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Invoke the Response API."""
        return self._call_response_api(
            prompt, stream=False, messages=messages, file_messages=file_messages,
            delegation_tool=delegation_tool
        )
    
    def _stream_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        delegation_tool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Stream the Response API."""
        return self._call_response_api(
            prompt, stream=True, messages=messages, file_messages=file_messages,
            delegation_tool=delegation_tool
        )
    
    def _call_response_api(
        self, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        delegation_tool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Call the Response API (streaming or non-streaming) with custom function execution loop.
        
        Args:
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            delegation_tool: Optional delegation tool for supervisor routing
            
        Returns:
            Dict with 'role', 'content', 'agent', and optionally 'stream' or 'output' key
        """
        api_input = self._convert_messages_to_input(messages, prompt, file_messages)
        
        # Build tools config
        if delegation_tool:
            tools_config = self._build_tools_config_for_delegation(delegation_tool)
        else:
            tools_config = self._build_base_tools_config(self._vector_store_ids, stream=stream)
        
        response = self.openai_client.responses.create(
            model=self.agent.model,
            input=api_input,
            instructions=self._original_system_message,
            temperature=self.agent.temperature,
            tools=tools_config if tools_config else [],
            stream=stream,
            reasoning={"summary": "auto"},
        )
        
        if stream:
            return {
                "role": "assistant",
                "content": "",
                "agent": self.agent.name,
                "stream": response
            }
        else:            
            # For delegation scenarios (when delegation_tool is provided), return output items directly
            if delegation_tool:
                return {"output": response.output if hasattr(response, 'output') else []}
            
            # Check if there are function calls that need to be executed
            response_with_tool_results = self._handle_function_calls(response, api_input, tools_config, stream)
            
            # For regular execution, extract content and update history
            content = self._extract_response_content(response_with_tool_results)
            blocks = self._convert_message_to_blocks(content)
            self._add_to_conversation_history("assistant", blocks)
            return {
                "role": "assistant",
                "content": content,
                "agent": self.agent.name
            }
    
    def _build_base_tools_config(
        self, 
        vector_store_ids: Optional[List[str]] = None, 
        stream: bool = True
    ) -> List[Dict[str, Any]]:
        """Build base tools configuration (native OpenAI tools, MCP servers, and custom tools)."""
        from ...utils import MCPToolManager

        tools = []
        vs_ids = vector_store_ids or self._vector_store_ids
        
        if self.agent.allow_file_search and vs_ids:
            tools.append({"type": "file_search", "vector_store_ids": vs_ids if isinstance(vs_ids, list) else [vs_ids]})
        if self.agent.allow_code_interpreter:
            container_config = self.agent.container_id if self.agent.container_id else {"type": "auto"}
            tools.append({"type": "code_interpreter", "container": container_config})
        if self.agent.allow_web_search:
            tools.append({"type": "web_search"})
        if self.agent.allow_image_generation:
            tools.append({"type": "image_generation", "partial_images": 3} if stream else {"type": "image_generation"})
        
        if self.agent.mcp_servers:
            mcp_manager = MCPToolManager()
            mcp_manager.add_servers(self.agent.mcp_servers)
            mcp_tools = mcp_manager.get_openai_tools()
            tools.extend(mcp_tools)
        
        # Add custom tools - Response API supports them via function calling
        if self.tools:
            for tool in self.tools:
                openai_tool = self._convert_langchain_tool_to_openai(tool)
                if openai_tool:
                    tools.append(openai_tool)
        
        return tools
    
    def _handle_function_calls(
        self, response: Any, api_input: List[Dict[str, Any]], 
        tools_config: List[Dict[str, Any]], stream: bool = False,
        max_iterations: int = 10
    ) -> Any:
        """
        Handle function calls from Response API by executing custom functions and continuing the conversation.
        
        Args:
            response: Initial Response API response
            api_input: Original API input messages
            tools_config: Tools configuration
            stream: Whether streaming is enabled
            max_iterations: Maximum number of function call iterations
            
        Returns:
            Final response after all function calls are executed
        """
        function_map = self._build_function_map()
        iteration = 0
        current_response = response
        accumulated_input = list(api_input)
        
        while iteration < max_iterations:
            function_results = self._extract_and_execute_function_calls(
                current_response, function_map, iteration
            )
            
            if not function_results:
                return current_response
            
            accumulated_input = self._accumulate_function_results(
                accumulated_input, current_response, function_results
            )
            current_response = self._call_api_with_results(
                accumulated_input, tools_config
            )
            iteration += 1
        
        return current_response
    
    def _extract_and_execute_function_calls(
        self, response: Any, function_map: Dict[str, Any], iteration: int
    ) -> List[Dict[str, Any]]:
        """Extract function calls from response and execute them."""
        function_results = []
        output_items = response.output if hasattr(response, 'output') else []
        
        for item in output_items:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            
            if item_type == "function_call":
                result = self._execute_single_function_call(item, function_map, iteration)
                if result:
                    function_results.append(result)
        
        return function_results
    
    def _execute_single_function_call(
        self, item: Any, function_map: Dict[str, Any], iteration: int
    ) -> Optional[Dict[str, Any]]:
        """Execute a single function call and return result."""
        # Extract function call details
        if isinstance(item, dict):
            function_name = item.get("name")
            arguments = item.get("arguments", "{}")
            call_id = item.get("call_id", f"call_{iteration}")
        else:
            function_name = getattr(item, "name", None)
            arguments = getattr(item, "arguments", "{}")
            call_id = getattr(item, "call_id", f"call_{iteration}")
        
        if function_name not in function_map:
            return {
                "call_id": call_id,
                "name": function_name,
                "result": f"Error: Function {function_name} not found"
            }
        
        try:
            # Parse arguments
            if isinstance(arguments, str):
                args_dict = json.loads(arguments)
            else:
                args_dict = arguments if isinstance(arguments, dict) else {}
            
            # Execute the custom function
            function_impl = function_map[function_name]
            result = function_impl(**args_dict)
            result_str = str(result)
            
            return {
                "call_id": call_id,
                "name": function_name,
                "result": result_str
            }
        except Exception as e:
            return {
                "call_id": call_id,
                "name": function_name,
                "result": f"Error: {str(e)}"
            }
    
    def _accumulate_function_results(
        self, accumulated_input: List[Dict[str, Any]],
        current_response: Any, function_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Accumulate conversation history with function results."""
        # Add response output to build full history
        new_input = list(accumulated_input)
        new_input.extend(current_response.output)
        
        # Add function results in Response API format
        for func_result in function_results:
            new_input.append({
                "type": "function_call_output",
                "call_id": func_result["call_id"],
                "output": json.dumps({"result": func_result["result"]})
            })
        
        return new_input
    
    def _call_api_with_results(
        self, accumulated_input: List[Dict[str, Any]], tools_config: List[Dict[str, Any]]
    ) -> Any:
        """Call Response API with accumulated input and function results."""
        return self.openai_client.responses.create(
            model=self.agent.model,
            input=accumulated_input,
            instructions=self._original_system_message,
            temperature=self.agent.temperature,
            tools=tools_config if tools_config else [],
            stream=False,  # Don't stream during function call loop
            reasoning={"summary": "auto"},
        )
    
    def _build_function_map(self) -> Dict[str, Any]:
        """
        Build a map of function names to their implementations from custom tools.
        
        Returns:
            Dict mapping function names to callable implementations
        """
        function_map = {}
        
        if not self.tools:
            return function_map
        
        from langchain_core.tools import StructuredTool
        
        for tool in self.tools:
            if isinstance(tool, StructuredTool):
                function_map[tool.name] = tool.func
            elif isinstance(tool, dict) and "function" in tool:
                # If tool is a dict with function info, we need access to the actual function
                # This shouldn't happen with our current setup, but handle it gracefully
                pass
        
        return function_map
        
    def _convert_langchain_tool_to_openai(self, tool: Any) -> Dict[str, Any]:
        """Convert a LangChain StructuredTool to Response API format."""
        from langchain_core.tools import StructuredTool
        
        if not isinstance(tool, StructuredTool):
            if isinstance(tool, dict) and "type" in tool:
                # If already in Response API format, return as-is
                if "name" in tool and tool.get("type") == "function":
                    return tool
                # If in ChatCompletion format, convert to Response API format
                if "function" in tool and tool.get("type") == "function":
                    function_def = tool["function"]
                    return {
                        "type": "function",
                        "name": function_def.get("name"),
                        "description": function_def.get("description", ""),
                        "parameters": function_def.get("parameters", {})
                    }
                return tool
            return None
        
        args_schema = tool.args_schema
        properties = {}
        required = []
        
        if args_schema:
            schema_dict = args_schema.schema() if hasattr(args_schema, 'schema') else {}
            properties = schema_dict.get("properties", {})
            required = schema_dict.get("required", [])
        
        # Return Response API format directly (flattened, not nested)
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def _convert_messages_to_input(
        self,
        messages: Optional[List[Dict[str, Any]]],
        current_prompt: str,
        file_messages: Optional[List] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert workflow_state messages to Response API input format.
        Response API uses a list of message dicts with 'role' and 'content' keys.
        
        Like the reference code, conversation history (including file context) is sent as a system message.
        
        Args:
            messages: List of message dicts from workflow_state
            current_prompt: Current user prompt
            file_messages: Optional file messages (OpenAI format) to include
            
        Returns:
            List of messages in Response API input format
        """
        input_list = []
        
        # Update conversation history from messages (this includes file messages)
        self._update_conversation_history_from_messages(messages, file_messages)
        
        # Add current prompt as user message (like reference code does)
        input_list.append({"role": "user", "content": current_prompt})

        # Add conversation history as system message (like reference code does)
        # This includes file information from previous turns
        sections_dict = self._get_conversation_history_sections_dict()
        if sections_dict:
            system_content = json.dumps(sections_dict, ensure_ascii=False)
            input_list.append({"role": "system", "content": system_content})

        return input_list
    
    
    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from OpenAI Response API response."""
        from .conversation_history import extract_text_from_content

        if not response:
            return ""

        text_parts = []
        # Check response.items first (Response API format)
        if hasattr(response, 'items') and response.items:
            for item in response.items:
                if hasattr(item, 'type') and item.type == 'output_text':
                    if hasattr(item, 'text'):
                        text_parts.append(str(item.text))
                    elif hasattr(item, 'content'):
                        text_parts.append(extract_text_from_content(item.content))
        
        # Check response.output for text items
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                if item_type == 'output_text':
                    if isinstance(item, dict):
                        text_content = item.get('text', '')
                    else:
                        text_content = getattr(item, 'text', '')
                    if text_content:
                        text_parts.append(str(text_content))
                elif item_type == 'message':
                    # Message items contain a 'content' field with ResponseOutputText objects
                    if isinstance(item, dict):
                        content_blocks = item.get('content', [])
                    else:
                        content_blocks = getattr(item, 'content', [])
                    for block in content_blocks:
                        if hasattr(block, 'text'):
                            text_parts.append(str(block.text))
                        elif isinstance(block, dict) and 'text' in block:
                            text_parts.append(str(block.get('text', '')))
        
        # Check response.output_text
        if hasattr(response, 'output_text'):
            text_parts.append(extract_text_from_content(response.output_text))
        
        # Fallback to direct attributes
        if not text_parts:
            if hasattr(response, 'text'):
                text_parts.append(str(response.text))
            elif hasattr(response, 'content'):
                text_parts.append(extract_text_from_content(response.content))
        
        result = ''.join(text_parts) if text_parts else str(response) if response else ""
        return result
    
    def update_vector_store_ids(self, llm_client):
        """Update vector_store_ids from llm_client and invalidate tools config if changed."""
        current_vector_ids = getattr(llm_client, '_vector_store_ids', None)
        if current_vector_ids != self._vector_store_ids:
            self._vector_store_ids = current_vector_ids
            self._tools_config = None
    
    def _build_tools_config_for_delegation(
        self, additional_tools: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build tools configuration for Response API with delegation support.
        
        Response API expects tools in format:
        {
            "type": "function",
            "name": "function_name",
            "description": "...",
            "parameters": {...}
        }
        
        Args:
            additional_tools: Additional tools to include (e.g., delegation tool in Response API format)
            
        Returns:
            List of tools in Response API format
        """
        tools = []
        
        if additional_tools:
            tools_to_process = additional_tools if isinstance(additional_tools, list) else [additional_tools]
            for tool in tools_to_process:
                if isinstance(tool, dict):
                    # If already in Response API format, use as-is
                    if "name" in tool and tool.get("type") == "function":
                        tools.append(tool)
        
        # Add base tools (native OpenAI tools, MCP tools, custom tools)
        base_tools = self._build_base_tools_config(
            vector_store_ids=None,
            stream=False  # Delegation doesn't use streaming
        )
        tools.extend(base_tools)
        
        return tools