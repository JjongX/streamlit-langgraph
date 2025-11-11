"""ResponseAPIExecutor for OpenAI Responses API with HITL support."""

import json
from typing import Any, Dict, List, Optional

from ...agent import AgentManager
from .base import BaseExecutor


class ResponseAPIExecutor(BaseExecutor):
    """
    Executor for OpenAI Responses API generation.
    
    Supports human-in-the-loop approval when enabled via agent configuration.
    When HITL is enabled, uses Chat Completions API with function calling for tool interception.
    """
    
    def execute(
        self,
        llm_client: Any,
        prompt: str,
        stream: bool = False,
        file_messages: Optional[List] = None,
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute prompt using the Responses API client.
        
        Supports file messages, code interpreter, web search, and image generation tools.
        When human_in_loop is enabled, uses Chat Completions API to intercept tool calls.
        """
        config = self._prepare_config(config)
        thread_id = self.get_thread_id(config)
        
        # If HITL is enabled, use Chat Completions API for tool interception
        if self.agent.human_in_loop and self.agent.interrupt_on:
            return self._execute_with_hitl(llm_client, prompt, stream, file_messages, thread_id, config, messages)
        
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
    
    def resume(
        self,
        decisions: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Resume execution after human approval/rejection.
        
        Args:
            decisions: List of decision dicts with 'type' ('approve', 'reject', 'edit') and optional 'edit' content
            config: Execution config with thread_id
            messages: Conversation history from workflow_state
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if more approvals needed
        """
        if not self.agent.human_in_loop:
            raise ValueError("Cannot resume: human-in-the-loop not enabled")
        
        config = self._prepare_config(config)
        thread_id = self.get_thread_id(config)
        
        # Get LLM client
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
                # OpenAI requires responses for all tool_call_ids
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
        
        # Get conversation history from workflow_state and clean for OpenAI API
        # OpenAI API expects messages with only: role, content, tool_calls, tool_call_id, name
        # Remove extra fields like "id" and "agent" that are used for workflow_state tracking
        messages_with_tools = []
        if messages:
            for msg in messages:
                # Clean message for OpenAI API format
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
            # Fallback: reconstruct assistant message from pending_tool_calls
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": self.pending_tool_calls
            }
            messages_with_tools.append(assistant_message)
        
        # Add tool results (must come after assistant message with tool_calls)
        messages_with_tools.extend(tool_results)
        
        from ...utils import CustomTool  # lazy import to avoid circular import
        custom_tools = CustomTool.get_openai_tools(self.agent.tools) if self.agent.tools else []
        
        followup_response = llm_client.chat.completions.create(
            model=self.agent.model,
            messages=messages_with_tools,
            temperature=self.agent.temperature,
            tools=custom_tools if custom_tools else None,
            tool_choice="auto" if custom_tools else None
        )
        
        message = followup_response.choices[0].message
        
        # Check for additional tool calls that need approval
        if message.tool_calls:
            interrupt_data = self._check_tool_calls_for_interrupt(message.tool_calls)
            if interrupt_data:
                self.pending_tool_calls = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
                return self._create_interrupt_response(interrupt_data, thread_id, config)
        
        # Clear pending tool calls
        self.pending_tool_calls = []
        
        content = message.content or ""
        return {"role": "assistant", "content": content, "agent": self.agent.name}
    
    # ========== HITL Helper Methods ==========
    
    def _execute_with_hitl(
        self,
        llm_client,
        prompt: str,
        stream: bool,
        file_messages: Optional[List],
        thread_id: str,
        config: Dict[str, Any],
        conversation_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute with human-in-the-loop support using Chat Completions API.
        
        This allows us to intercept tool calls before execution.
        """
        # Build messages from passed conversation history (from workflow_state)
        messages = conversation_messages.copy() if conversation_messages else []
        
        # Add file messages if provided
        if file_messages:
            messages.extend(file_messages)
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        from ...utils import CustomTool  # lazy import to avoid circular import
        custom_tools = CustomTool.get_openai_tools(self.agent.tools) if self.agent.tools else []
        
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
                    # Store pending tool calls for resume()
                    self.pending_tool_calls = [self._tool_call_to_dict(tc) for tc in message.tool_calls]
                    interrupt_response = self._create_interrupt_response(interrupt_data, thread_id, config)
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
                
                # Continue conversation with tool results
                messages_with_assistant = messages + [self._message_to_dict(message)]
                messages_with_tools = messages_with_assistant + tool_results
                followup_response = llm_client.chat.completions.create(
                    model=self.agent.model,
                    messages=messages_with_tools,
                    temperature=self.agent.temperature,
                    tools=custom_tools if custom_tools else None,
                    tool_choice="auto" if custom_tools else None
                )
                
                final_message = followup_response.choices[0].message
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

