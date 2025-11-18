# Base classes and common utilities for agent node creation.

import uuid
from typing import Any, Dict

from ...agent import Agent, AgentManager
from ...core.executor.registry import ExecutorRegistry
from ...core.middleware.interrupts import InterruptManager
from ...core.state import WorkflowState


def create_message_with_id(role: str, content: str, agent: str) -> Dict[str, Any]:
    """Helper to create a message with a unique ID."""
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "agent": agent
    }


class AgentNodeBase:
    """Base class providing common functionality for agent node operations."""
    
    @staticmethod
    def execute_agent(agent: Agent, state: WorkflowState, input_message: str) -> str:
        """
        Execute an agent and return the response.
                
        Args:
            agent: Agent to execute
            state: Current workflow state
            input_message: Input message/prompt for the agent
            
        Returns:
            Agent response content as string (empty string if interrupted)
        """
        from ...utils import CustomTool, MCPToolManager  # lazy import to avoid circular import
        
        executor = ExecutorRegistry().get_or_create(agent, executor_type="workflow")
        
        if hasattr(executor, 'tools'):
            custom_tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
            mcp_tools = []
            if agent.mcp_servers:
                mcp_manager = MCPToolManager()
                mcp_manager.add_servers(agent.mcp_servers)
                mcp_tools = mcp_manager.get_tools()
            executor.tools = custom_tools + mcp_tools
        
        if "metadata" not in state:
            state["metadata"] = {}
        if "executors" not in state["metadata"]:
            state["metadata"]["executors"] = {}
        
        # Use workflow's thread_id (set by WorkflowExecutor) - this matches the checkpointer
        workflow_thread_id = state.get("metadata", {}).get("workflow_thread_id")
        if not workflow_thread_id:
            workflow_thread_id = str(uuid.uuid4())
            state["metadata"]["workflow_thread_id"] = workflow_thread_id
        
        executor_key = f"workflow_executor_{executor.agent.name}"
        state["metadata"]["executors"][executor_key] = {"thread_id": workflow_thread_id}
        
        # Build execution config with workflow's thread_id (matches checkpointer)
        config = {"configurable": {"thread_id": workflow_thread_id}}
        
        # Execute agent
        llm_client = AgentManager.get_llm_client(agent)
        conversation_messages = state.get("messages", [])
        
        result = executor.execute_workflow(
            llm_client=llm_client,
            prompt=input_message,
            config=config,
            messages=conversation_messages
        )
        
        # Handle interrupt
        if InterruptManager.should_interrupt(result):
            interrupt_data = InterruptManager.extract_interrupt_data(result)
            
            # Add assistant_message to workflow_state for resume() to have complete conversation history
            if "assistant_message" in result:
                assistant_msg = result["assistant_message"]
                if "id" not in assistant_msg:
                    assistant_msg["id"] = str(uuid.uuid4())
                if "agent" not in assistant_msg:
                    assistant_msg["agent"] = agent.name
                # Add to workflow_state messages
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(assistant_msg)
            
            interrupt_update = InterruptManager.store_interrupt(
                state=state,
                agent_name=agent.name,
                interrupt_data=interrupt_data,
                executor_key=executor_key
            )
            state["metadata"].update(interrupt_update["metadata"])
            return ""  # Empty response for interrupt
        
        # Return normal response
        return result.get("content", "")

    @staticmethod
    def extract_user_query(state: WorkflowState) -> str:
        """Extract user query from state messages."""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                return msg["content"]
        return ""