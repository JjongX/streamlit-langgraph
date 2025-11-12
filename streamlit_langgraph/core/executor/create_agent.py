"""CreateAgentExecutor for LangChain agents with HITL support."""

from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command

from ...agent import Agent
from .base import BaseExecutor


class CreateAgentExecutor(BaseExecutor):
    """
    Executor that builds LangChain agents using `create_agent`.
    
    Uses LangChain's standard `create_agent` function which supports multiple providers
    (OpenAI, Anthropic, Google, etc.) through LangChain's chat model interface.
    
    Supports human-in-the-loop approval when enabled via agent configuration.
    """

    def __init__(self, agent: Agent, tools: Optional[List] = None, thread_id: Optional[str] = None):
        """
        Initialize CreateAgentExecutor.
        
        Args:
            agent: Agent configuration
            tools: Optional list of LangChain tools
            thread_id: Optional thread ID for conversation tracking
        """
        super().__init__(agent, thread_id)
        self.agent_obj = None
        
        # Build tools configuration from CustomTool registry if not explicitly provided
        if tools is not None:
            self.tools = tools
        else:
            from ...utils import CustomTool  # lazy import to avoid circular import
            self.tools = CustomTool.get_langchain_tools(self.agent.tools) if self.agent.tools else []
    
    def execute(
        self,
        llm_client: Any,
        prompt: str,
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute prompt through a LangChain agent.

        Args:
            llm_client: A LangChain chat model instance
            prompt: User's question/prompt
            config: Execution config with thread_id and interrupt handling
            messages: Conversation history from workflow_state (unified message-based approach)

        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if HITL is active
        """
        try:
            if self.agent_obj is None:
                self._build_agent(llm_client)
            
            config = self._prepare_config(config)
            thread_id = self.get_thread_id(config)
            
            # Convert workflow_state messages to LangChain message format
            langchain_messages = self._convert_to_langchain_messages(messages, prompt)
            
            # Stream events to detect interrupts
            interrupt_data = self._detect_interrupt_in_stream(config, langchain_messages)
            if interrupt_data:
                return self._create_interrupt_response(interrupt_data, thread_id, config)
            
            # Execute normally if no interrupt detected
            out = self.agent_obj.invoke(
                {"messages": langchain_messages},
                config=config
            )
            
            # Check for interrupt in output
            if isinstance(out, dict) and "__interrupt__" in out:
                return self._create_interrupt_response(out["__interrupt__"], thread_id, config)
            
            result_text = self._extract_response_text(out)
            return {"role": "assistant", "content": result_text, "agent": self.agent.name}
        except Exception as e:
            return {"role": "assistant", "content": f"Agent error: {str(e)}", "agent": self.agent.name}
    
    def resume(
        self,
        decisions: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Resume agent execution after human approval/rejection.
        
        Args:
            decisions: List of decision dicts with 'type' ('approve', 'reject', 'edit') and optional 'edit' content
            config: Execution config with thread_id
            messages: Conversation history from workflow_state (unified message-based approach)
            
        Returns:
            Dict with keys 'role', 'content', 'agent', and optionally '__interrupt__' if more approvals needed
        """
        if not self.agent.human_in_loop or not self.agent_obj:
            raise ValueError("Cannot resume: human-in-the-loop not enabled or agent not initialized")
        
        config = self._prepare_config(config)
        thread_id = self.get_thread_id(config)
        
        # The checkpointer should have the history, but we ensure consistency
        resume_command = Command(resume={"decisions": decisions})
        out = self.agent_obj.invoke(resume_command, config=config)
        
        # Check for additional interrupts
        if isinstance(out, dict) and "__interrupt__" in out:
            return self._create_interrupt_response(out["__interrupt__"], thread_id, config)
        
        result_text = self._extract_response_text(out)
        return {"role": "assistant", "content": result_text, "agent": self.agent.name}
        
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
        
        # Note: Checkpointer is now handled at workflow level, not executor level
        # The workflow checkpointer persists the entire workflow_state automatically
        
        self.agent_obj = create_agent(**agent_kwargs)
        return self.agent_obj
        
    def _detect_interrupt_in_stream(self, execution_config: Dict[str, Any], messages: List[BaseMessage]) -> Optional[Any]:
        """Detect interrupt from agent stream events."""
        for event in self.agent_obj.stream(
            {"messages": messages},
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
    
    
    def _convert_to_langchain_messages(
        self, 
        messages: Optional[List[Dict[str, Any]]], 
        current_prompt: str
    ) -> List[BaseMessage]:
        """
        Convert workflow_state messages to LangChain message format.
        
        Args:
            messages: List of message dicts from workflow_state
            current_prompt: Current user prompt (will be added if not already in messages)
            
        Returns:
            List of LangChain BaseMessage objects
        """
        langchain_messages: List[BaseMessage] = []
        
        # Convert existing messages from workflow_state
        if messages:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Skip empty messages
                if not content and role != "system":
                    continue
                
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                elif role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                # Note: tool messages are handled differently in LangChain
                # They're typically added during tool execution
        
        # Add current prompt if it's not already the last message
        # (This handles the case where prompt is new and not yet in workflow_state)
        if not langchain_messages or not isinstance(langchain_messages[-1], HumanMessage) or langchain_messages[-1].content != current_prompt:
            langchain_messages.append(HumanMessage(content=current_prompt))
        
        return langchain_messages
    
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

