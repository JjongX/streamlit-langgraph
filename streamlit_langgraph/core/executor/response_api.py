# ResponseAPIExecutor for OpenAI Responses API

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from openai import OpenAI

from ...agent import Agent
from ...ui.display_manager import Block
from ...utils import MCPToolManager
from .conversation_history import ConversationHistoryMixin
from ...ui.nonstream_processor import NonStreamProcessor


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
        self.tools = tools if tools is not None else []
        self._init_conversation_history(agent)
    
    def execute_agent(
        self, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        vector_store_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute prompt for single-agent mode (non-workflow).
        """
        return self._execute(prompt, stream, messages, file_messages, vector_store_ids)
    
    def execute_workflow(
        self, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        vector_store_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute prompt for workflow mode.
        
        Note: Response API does not support HITL, so this method does not handle interrupts.
        For HITL scenarios, use CreateAgentExecutor instead.
        """
        return self._execute(prompt, stream, messages, file_messages, vector_store_ids)
    
    def invoke_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        delegation_tool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the Response API (non-streaming).
        """
        return self._call_response_api(prompt, stream=False, messages=messages, 
                                       file_messages=file_messages, delegation_tool=delegation_tool)
    
    def set_vector_store_ids(self, vector_store_ids: Optional[List[str]]) -> None:
        """Directly set vector store IDs supplied by callers."""
        self._vector_store_ids = vector_store_ids
    
    def _execute(
        self, prompt: str, stream: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        vector_store_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Common execution logic for both agent and workflow modes.
        
        Args:
            prompt: User's question/prompt
            stream: Whether to stream the response
            messages: Conversation history from workflow_state
            file_messages: Optional file messages (OpenAI format)
            vector_store_ids: Optional vector store IDs supplied by FileHandler
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally 'stream'
        """
        self.set_vector_store_ids(vector_store_ids)
        
        if stream:
            return self._stream_response_api(prompt, messages, file_messages)
        else:
            return self.invoke_response_api(prompt, messages, file_messages)
    
    def _stream_response_api(
        self, prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        file_messages: Optional[List] = None,
        delegation_tool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Stream the Response API."""
        return self._call_response_api(prompt, stream=True, messages=messages,
                                      file_messages=file_messages, delegation_tool=delegation_tool)
    
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
        tools_config = (
            self._build_tools_config_for_delegation(delegation_tool)
            if delegation_tool else
            self._build_base_tools_config(self._vector_store_ids, stream=stream)
        )
        response = self.openai_client.responses.create(
            model=self.agent.model,
            input=api_input,
            instructions=self._original_system_message,
            temperature=self.agent.temperature,
            tools=tools_config if tools_config else [],
            stream=stream,
            reasoning=self._build_reasoning_config(),
        )
        
        if stream:
            return self._create_response_dict(stream=response)
        
        # For delegation scenarios, return output items directly
        if delegation_tool:
            return {"output": getattr(response, 'output', [])}
        
        # Check if there are function calls that need to be executed
        response_with_tool_results = self._handle_function_calls(response, api_input, tools_config, stream)
        
        # For regular execution, extract content and reasoning, then update history
        content = NonStreamProcessor.extract_response_api_text(response_with_tool_results)
        blocks = self._convert_message_to_blocks(content)
        
        # Extract reasoning blocks if present (for non-streaming responses)
        reasoning_blocks = self._extract_reasoning_blocks(response_with_tool_results)
        blocks.extend(reasoning_blocks)
        
        self._add_to_conversation_history("assistant", blocks)
        
        # Include reasoning blocks in response so they can be displayed
        response_dict = self._create_response_dict(content=content)
        if reasoning_blocks:
            # Convert blocks to dict format for response
            blocks_data = []
            for block in reasoning_blocks:
                block_dict = {
                    "category": block.category,
                    "content": block.content,
                    "filename": block.filename,
                    "file_id": block.file_id
                }
                blocks_data.append(block_dict)
            response_dict["blocks"] = blocks_data
        return response_dict
    
    def _create_response_dict(self, content: str = "", stream: Any = None, output: Any = None) -> Dict[str, Any]:
        """Create a standardized response dictionary."""
        response = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": content,
            "agent": self.agent.name
        }
        if stream is not None:
            response["stream"] = stream
        if output is not None:
            response["output"] = output
        return response
    
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
        
        # Add file messages directly to input (Response API needs them in the input array)
        # Only include messages with actual file references (input_file, input_image), not text messages
        file_blocks = self._collect_file_blocks(file_messages)
        input_list.extend(file_blocks)
        
        # Add current prompt as user message (like reference code does)
        input_list.append({"role": "user", "content": current_prompt})

        # Add conversation history as system message (like reference code does)
        # This includes file information from previous turns
        sections_dict = self._get_conversation_history_sections_dict()
        if sections_dict:
            system_content = json.dumps(sections_dict, ensure_ascii=False)
            input_list.append({"role": "system", "content": system_content})

        return input_list

    def _collect_file_blocks(self, file_messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Collect file blocks from file messages for the Response API input."""
        if not file_messages:
            return []
        
        collected = []
        for file_msg in file_messages:
            if not (isinstance(file_msg, dict) and file_msg.get("role") == "user"):
                continue
            content = file_msg.get("content", [])
            if not isinstance(content, list):
                continue
            
            file_blocks = [
                block for block in content
                if isinstance(block, dict) and block.get("type") in ("input_file", "input_image")
            ]
            if file_blocks:
                collected.append({"role": "user", "content": file_blocks})
        return collected
    
    def _build_base_tools_config(
        self, 
        vector_store_ids: Optional[List[str]] = None, 
        stream: bool = True
    ) -> List[Dict[str, Any]]:
        """Build base tools configuration (native OpenAI tools, MCP servers, and custom tools)."""
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
    
    def _build_tools_config_for_delegation(
        self, additional_tools: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build tools configuration for Response API with delegation support.
        
        Args:
            additional_tools: Additional tools to include (e.g., delegation tool in Response API format)
            
        Returns:
            List of tools in Response API format
        """
        tools = []
        
        if additional_tools:
            tools_to_process = additional_tools if isinstance(additional_tools, list) else [additional_tools]
            tools.extend(t for t in tools_to_process 
                        if isinstance(t, dict) and "name" in t and t.get("type") == "function")
        
        # Add base tools (native OpenAI tools, MCP tools, custom tools)
        base_tools = self._build_base_tools_config(vector_store_ids=None, stream=False)
        tools.extend(base_tools)
        
        return tools
    
    def _convert_langchain_tool_to_openai(self, tool: Any) -> Dict[str, Any]:
        """Convert a LangChain StructuredTool to Response API format."""
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
    
    def _build_function_map(self) -> Dict[str, Any]:
        """
        Build a map of function names to their implementations from custom tools.
        
        Returns:
            Dict mapping function names to callable implementations
        """
        function_map = {}
        
        if not self.tools:
            return function_map
        
        for tool in self.tools:
            if isinstance(tool, StructuredTool):
                function_map[tool.name] = tool.func
            elif isinstance(tool, dict) and "function" in tool:
                # If tool is a dict with function info, we need access to the actual function
                # This shouldn't happen with our current setup, but handle it gracefully
                pass
        
        return function_map
    
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
            
            self._accumulate_function_results(accumulated_input, current_response, function_results)
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
        output_items = getattr(response, 'output', [])
        
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
        function_name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        arguments = item.get("arguments", "{}") if isinstance(item, dict) else getattr(item, "arguments", "{}")
        call_id = item.get("call_id", f"call_{iteration}") if isinstance(item, dict) else getattr(item, "call_id", f"call_{iteration}")
        
        if function_name not in function_map:
            return {"call_id": call_id, "name": function_name, "result": f"Error: Function {function_name} not found"}
        
        try:
            # Parse arguments
            args_dict = json.loads(arguments) if isinstance(arguments, str) else (arguments if isinstance(arguments, dict) else {})
            
            # Execute the custom function
            result = function_map[function_name](**args_dict)
            
            return {"call_id": call_id, "name": function_name, "result": str(result)}
        except Exception as e:
            return {"call_id": call_id, "name": function_name, "result": f"Error: {str(e)}"}
    
    def _accumulate_function_results(
        self,
        accumulated_input: List[Dict[str, Any]],
        current_response: Any,
        function_results: List[Dict[str, Any]]
    ) -> None:
        """Accumulate conversation history with function results in-place."""
        accumulated_input.extend(current_response.output)
        
        for func_result in function_results:
            accumulated_input.append({
                "type": "function_call_output",
                "call_id": func_result["call_id"],
                "output": json.dumps({"result": func_result["result"]})
            })
    
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
            reasoning=self._build_reasoning_config(),
        )

    def _extract_reasoning_blocks(self, response: Any) -> List[Block]:
        """Extract reasoning blocks from Response API response (for non-streaming)."""
        if not response:
            return []

        def val(obj, key, default=None):
            if obj is None:
                return default
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        # For stream=True we only show reasoning *summary* deltas; keep stream=False consistent.
        summary_texts: List[str] = []

        reasoning = val(response, "reasoning")
        summary = val(reasoning, "summary") if reasoning else None
        if summary and not isinstance(summary, str):
            text = val(summary, "text")
            if text:
                summary_texts.append(str(text).strip())

        output_items = val(response, "output") or val(response, "items") or []
        for item in output_items:
            item_type = val(item, "type")
            if item_type == "reasoning":
                summary = val(item, "summary")
                if isinstance(summary, list):
                    texts = [str(val(s, "text")).strip() for s in summary if val(s, "text")]
                    if texts:
                        summary_texts.append("\n\n".join(t for t in texts if t))
                elif summary and not isinstance(summary, str):
                    text = val(summary, "text")
                    if text:
                        summary_texts.append(str(text).strip())
            elif item_type == "reasoning_summary_text":
                text = val(item, "text")
                if text:
                    summary_texts.append(str(text).strip())

        # Deduplicate while preserving order
        seen = set()
        blocks: List[Block] = []
        for text in summary_texts:
            if text and text not in seen:
                seen.add(text)
                blocks.append(self._history_display_manager.create_block("reasoning", content=text))

        return blocks
    
    def _build_reasoning_config(self) -> Dict[str, Any]:
        """Build the Response API reasoning configuration."""
        reasoning_config = {"summary": "auto"}
        effort = getattr(self.agent, "reasoning_effort", None)
        if effort:
            reasoning_config["effort"] = effort
        return reasoning_config
