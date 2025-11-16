# ResponseAPIExecutor for OpenAI Responses API.

from typing import Any, Dict, List, Optional

from .registry import BaseExecutor


class ResponseAPIExecutor(BaseExecutor):
    """
    Executor for OpenAI Responses API.
    
    Uses OpenAI's Responses API which automatically executes tools and returns final results.
    Supports file messages, code interpreter, web search, and image generation tools.
    HITL Handling: Delegates to CreateAgentExecutor for clean HITL support with custom tools.
    """

    def execute_agent(self, llm_client: Any, prompt: str, stream: bool = False,
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
        
        # If HITL is enabled, delegate to CreateAgentExecutor for clean HITL support
        if self.agent.human_in_loop and self.agent.interrupt_on:
            from .create_agent import CreateAgentExecutor
            from ...utils import CustomTool
            
            # Create CreateAgentExecutor with custom tools for HITL
            hitl_executor = CreateAgentExecutor(
                self.agent,
                tools=CustomTool.get_langchain_tools(self.agent.tools) if self.agent.tools else []
            )
            return hitl_executor.execute_workflow(llm_client, prompt, config, messages)
        
        # Normal execution: use Responses API for native OpenAI features
        input_messages = self._build_input_messages(prompt, file_messages, messages)
        return self._call_responses_api(llm_client, input_messages, stream)
    
    def resume(self, decisions: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Resume execution after human approval/rejection.
        
        Delegates to CreateAgentExecutor for HITL support.
        
        Args:
            decisions: List of decision dicts with 'type' ('approve', 'reject', 'edit') and optional 'edit' content
            config: Execution config with thread_id
            messages: Conversation history from workflow_state
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if more approvals needed
        """
        if not self.agent.human_in_loop:
            raise ValueError("Cannot resume: human-in-the-loop not enabled")
        
        from .create_agent import CreateAgentExecutor
        from ...utils import CustomTool
        # Create CreateAgentExecutor with custom tools for HITL
        hitl_executor = CreateAgentExecutor(
            self.agent,
            tools=CustomTool.get_langchain_tools(self.agent.tools) if self.agent.tools else []
        )
        return hitl_executor.resume(decisions, config, messages)
    
    def _call_responses_api(self, llm_client: Any, input_messages: List[Dict[str, Any]], 
                           stream: bool = False) -> Dict[str, Any]:
        """
        Call OpenAI Responses API with prepared input messages.
        
        It automatically executes tools and returns final results. 
        This is its core behavior.
        """
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
    
    def _build_input_messages(self, prompt: str, file_messages: Optional[List] = None,
                              messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Build input messages for Responses API from conversation history.
        """
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
