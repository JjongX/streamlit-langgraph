# ResponseAPIExecutor for OpenAI Responses API.

import json
from typing import Any, Dict, List, Optional

from ...agent import AgentManager
from .registry import BaseExecutor


class ResponseAPIExecutor(BaseExecutor):
    """
    Executor for OpenAI Responses API.
    
    Uses OpenAI's Responses API which automatically executes tools and returns final results.
    Supports file messages, code interpreter, web search, and image generation tools.
    HITL Handling: Uses a custom `_execute_with_hitl` method that calls `_call_responses_api`
    """

    def execute_single_agent(self, llm_client: Any, prompt: str, stream: bool = False,
        file_messages: Optional[List] = None,
        messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute prompt for single-agent mode (non-workflow).
        
        Single-agent mode: no checkpointer, no HITL, no thread_id needed.
        
        Args:
            llm_client: OpenAI client
            prompt: User's prompt
            stream: Whether to stream the response
            file_messages: Optional file messages
            messages: Conversation history from workflow_state
            
        Returns:
            Dict with 'role', 'content', 'agent', and optionally 'stream'
        """
        input_messages = self._build_input_messages(prompt, file_messages, messages)
        return self._call_responses_api(llm_client, input_messages, stream)
    
    def execute_workflow(self, llm_client: Any, prompt: str, stream: bool = False,
        file_messages: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute prompt for workflow mode (requires config with thread_id).
                
        Args:
            llm_client: OpenAI client
            prompt: User's prompt
            stream: Whether to stream the response
            file_messages: Optional file messages
            config: Execution config with thread_id (required for workflows)
            messages: Conversation history from workflow_state
            
        Returns:
            Dict with 'role', 'content', 'agent', and optionally '__interrupt__' or 'stream'
        """
        config, thread_id = self._prepare_workflow_config(config)
        # If HITL is enabled, use Chat Completions API for tool interception
        if self.agent.human_in_loop and self.agent.interrupt_on:
            return self._execute_with_hitl(llm_client, prompt, stream, file_messages, thread_id, config, messages)
        input_messages = self._build_input_messages(prompt, file_messages, messages)
        return self._call_responses_api(llm_client, input_messages, stream)
    
    def resume(self,
        decisions: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Resume execution after human approval/rejection.
        
        Uses Chat Completions API (not Responses API) because we need to handle
        tool results and continue the conversation with proper tool message format.
        
        Args:
            decisions: List of decision dicts with 'type' ('approve', 'reject', 'edit') and optional 'edit' content
            config: Execution config with thread_id
            messages: Conversation history from workflow_state
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if more approvals needed
        """
        if not self.agent.human_in_loop:
            raise ValueError("Cannot resume: human-in-the-loop not enabled")
        config, thread_id = self._prepare_workflow_config(config)
        
        llm_client = AgentManager.get_llm_client(self.agent)
        
        # Apply decisions to pending tool calls
        # OpenAI requires a tool response for EVERY tool_call_id, even if rejected
        tool_results = []
        # Process all pending tool calls, using decisions if available
        for i, tool_call_dict in enumerate(self.pending_tool_calls):
            tool_name = tool_call_dict["function"]["name"]
            tool_call_id = tool_call_dict["id"]
            # Get decision for this tool call (default to approve if not provided)
            decision = decisions[i] if i < len(decisions) else {"type": "approve"}
            decision_type = decision.get("type", "approve") if decision else "approve"
            
            if decision_type == "reject":
                # Still need to provide a tool response for rejected calls
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": "Tool call was rejected by user"})
                })
                continue
            elif decision_type == "edit":
                # Use edited arguments
                tool_args = decision.get("input", decision.get("edit", {}))
            else:
                # Approve - use original arguments
                tool_args = json.loads(tool_call_dict["function"]["arguments"])
            
            # Execute approved/edited tool
            tool_result = self._execute_tool(tool_name, tool_args)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": str(tool_result)
            })
        
        # Get conversation history from workflow_state and clean for Chat Completions API
        messages_with_tools = []
        if messages:
            for msg in messages:
                # Clean message for Chat Completions API format
                clean_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content")
                }
                # Include tool_calls if present
                if "tool_calls" in msg:
                    clean_msg["tool_calls"] = msg["tool_calls"]
                # Include tool_call_id and name for tool messages
                if msg.get("role") == "tool":
                    if "tool_call_id" in msg:
                        clean_msg["tool_call_id"] = msg["tool_call_id"]
                    if "name" in msg:
                        clean_msg["name"] = msg["name"]
                messages_with_tools.append(clean_msg)
        
        # Verify we have messages and the last message is an assistant message with tool_calls
        if not messages_with_tools:
            raise ValueError("Cannot resume: conversation history is empty")
        
        last_message = messages_with_tools[-1]
        if last_message.get("role") != "assistant" or "tool_calls" not in last_message:
            # Fallback: reconstruct assistant message from pending_tool_calls if missing
            if self.pending_tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": self.pending_tool_calls
                }
                messages_with_tools.append(assistant_message)
        
        # Add tool results (must come after assistant message with tool_calls)
        messages_with_tools.extend(tool_results)
        
        # Use Chat Completions API for resume (required for HITL with tool results)
        # Cannot use Responses API because it doesn't support tool messages format
        followup_response = self._call_chat_api(llm_client, messages_with_tools)
        
        message = followup_response.choices[0].message
        
        # Check for additional tool calls that need approval
        if message.tool_calls:
            interrupt_data = self._check_tool_calls_for_interrupt(message.tool_calls)
            if interrupt_data:
                self.pending_tool_calls = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
                return self._create_interrupt_response(
                    {"action_requests": interrupt_data}, thread_id, config
                )
        
        # Clear pending tool calls
        self.pending_tool_calls = []
        
        content = message.content or ""
        return {"role": "assistant", "content": content, "agent": self.agent.name}
    
    def _execute_with_hitl(self, llm_client,
        prompt: str, stream: bool, file_messages: Optional[List],
        thread_id: str, config: Dict[str, Any],
        conversation_messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute with human-in-the-loop support using Chat Completions API.
        
        IMPORTANT: This method uses Chat Completions API (not Responses API) because:
        - Responses API executes tools automatically and cannot intercept them
        - Chat Completions API allows us to intercept tool calls before execution
        - We can check which tools need approval and pause for human decision
        """
        # Build messages from passed conversation history (from workflow_state)
        messages = conversation_messages.copy() if conversation_messages else []
        
        # Add file messages if provided
        if file_messages:
            messages.extend(file_messages)
        
        # Add user prompt
        if not messages or messages[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})
        
        try:
            # Use Chat Completions API for tool interception (required for HITL)
            # Cannot use Responses API because it executes tools automatically
            response = self._call_chat_api(llm_client, messages)
            
            message = response.choices[0].message
            
            # Check for tool calls that need approval
            if message.tool_calls:
                interrupt_data = self._check_tool_calls_for_interrupt(message.tool_calls)
                if interrupt_data:
                    # Store pending tool calls for resume()
                    self.pending_tool_calls = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
                    interrupt_response = self._create_interrupt_response(
                        {"action_requests": interrupt_data}, thread_id, config
                    )
                    # Include the assistant message with tool_calls for caller to add to workflow_state
                    interrupt_response["assistant_message"] = self._message_to_dict(message)
                    return interrupt_response
            
            # No interrupt needed, continue execution
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
                
                # Continue conversation with tool results using Chat Completions API
                messages_with_assistant = messages + [self._message_to_dict(message)]
                messages_with_tools = messages_with_assistant + tool_results
                followup_response = self._call_chat_api(llm_client, messages_with_tools)
                
                final_message = followup_response.choices[0].message
                content = final_message.content or ""
            else:
                content = message.content or ""
            
            return {"role": "assistant", "content": content, "agent": self.agent.name}
            
        except Exception as e:
            return {"role": "assistant", "content": f"Responses API error: {str(e)}", "agent": self.agent.name}
    
    def _call_responses_api(self, llm_client: Any, input_messages: List[Dict[str, Any]], 
                           stream: bool = False) -> Dict[str, Any]:
        """Call OpenAI Responses API with prepared input messages."""
        tools = self._build_tools_config(llm_client)
        api_params = {
            "model": self.agent.model,
            "input": input_messages,
            "temperature": self.agent.temperature,
            "stream": stream,
        }
        # Add agent instructions/system message as context
        if self.agent.system_message:
            api_params["instructions"] = self.agent.system_message
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
    
    def _call_chat_api(self, llm_client: Any, messages: List[Dict[str, Any]]) -> Any:
        """Call OpenAI Chat Completions API with prepared messages."""
        from ...utils import CustomTool  # lazy import to avoid circular import
        custom_tools = CustomTool.get_openai_tools(self.agent.tools) if self.agent.tools else []
        
        # Prepend system message if it exists and not already in messages
        if self.agent.system_message:
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": self.agent.system_message}] + messages
        
        return llm_client.chat.completions.create(
            model=self.agent.model,
            messages=messages,
            temperature=self.agent.temperature,
            tools=custom_tools if custom_tools else None,
            tool_choice="auto" if custom_tools else None
        )
    
    def _build_input_messages(self, prompt: str, file_messages: Optional[List] = None,
                              messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Build input messages for Responses API from conversation history."""
        input_messages = []
        
        # Add conversation history from workflow_state
        if messages:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                # Skip empty messages and system messages
                if content and role in ("user", "assistant"):
                    input_messages.append({"role": role,"content": content})

        # Add file messages (FileHandler.get_openai_input_messages())
        if file_messages:
            input_messages.extend(file_messages)
        
        # Add current user prompt
        if not input_messages or input_messages[-1].get("content") != prompt:
            input_messages.append({"role": "user", "content": prompt})
        
        return input_messages
    
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

    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        """Convert OpenAI message to dictionary."""
        msg_dict = {
            "role": message.role,
            "content": message.content or ""
        }
        if hasattr(message, 'tool_calls') and message.tool_calls:
            msg_dict["tool_calls"] = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
        return msg_dict

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
