# Base classes and common utilities for agent node creation.

import uuid
from typing import Any, Dict

from ...agent import Agent, AgentManager
from ...core.executor.registry import ExecutorRegistry
from ...middleware.interrupts import InterruptManager
from ...state import WorkflowState


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
    def extract_user_query(state: WorkflowState) -> str:
        """Extract user query from state messages."""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                return msg["content"]
        return ""
    
    @staticmethod
    def execute_agent(agent: Agent, state: WorkflowState, input_message: str) -> str:
        """
        Execute an agent and return the response.
        
        This method orchestrates agent execution using the ExecutorRegistry and InterruptManager.
        Broken down into smaller, testable methods for better maintainability.
        
        Args:
            agent: Agent to execute
            state: Current workflow state
            input_message: Input message/prompt for the agent
            
        Returns:
            Agent response content as string (empty string if interrupted)
        """
        # Get or create executor
        executor = AgentNodeBase._get_or_create_executor(agent, state)
        
        # Prepare execution config
        config = AgentNodeBase._prepare_execution_config(executor, state)
        
        # Execute agent
        result = AgentNodeBase._invoke_executor(executor, agent, input_message, config, state)
        
        # Handle result (including interrupts)
        return AgentNodeBase._handle_execution_result(result, agent, executor, state)
    
    @staticmethod
    def _get_or_create_executor(agent: Agent, state: WorkflowState):
        """
        Get existing executor from session state or create new one.
        
        Args:
            agent: Agent configuration
            state: Current workflow state
            
        Returns:
            Executor instance (ResponseAPIExecutor or CreateAgentExecutor)
        """
        from ...utils import CustomTool  # lazy import to avoid circular import
        
        # Get or create executor using registry
        executor = ExecutorRegistry().get_or_create(agent, executor_type="workflow")
        
        # Update tools for CreateAgentExecutor if needed
        if hasattr(executor, 'tools') and agent.tools:
            executor.tools = CustomTool.get_langchain_tools(agent.tools)
        
        return executor
    
    @staticmethod
    def _prepare_execution_config(executor, state: WorkflowState) -> Dict[str, Any]:
        """
        Prepare execution configuration with thread ID and metadata.
        
        Args:
            executor: Executor instance
            state: Current workflow state
            
        Returns:
            Configuration dictionary
        """
        # Ensure metadata structure exists
        if "metadata" not in state:
            state["metadata"] = {}
        if "executors" not in state["metadata"]:
            state["metadata"]["executors"] = {}
        
        # Store executor metadata
        executor_key = f"workflow_executor_{executor.agent.name}"
        state["metadata"]["executors"][executor_key] = {"thread_id": executor.thread_id}
        
        # Return execution config
        return {"configurable": {"thread_id": executor.thread_id}}
    
    @staticmethod
    def _invoke_executor(executor, agent: Agent, input_message: str, 
                        config: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """
        Invoke the executor with appropriate parameters.
        
        Args:
            executor: Executor instance
            agent: Agent configuration
            input_message: Input message/prompt
            config: Execution configuration
            state: Current workflow state
            
        Returns:
            Execution result dictionary
        """
        llm_client = AgentManager.get_llm_client(agent)
        conversation_messages = state.get("messages", [])
        
        # For ResponseAPIExecutor
        if hasattr(executor, '_execute_with_hitl'):
            enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
            return executor.execute(
                llm_client=llm_client,
                prompt=enhanced_instructions,
                stream=False,
                config=config,
                messages=conversation_messages
            )
        else:  # CreateAgentExecutor
            return executor.execute(
                llm_client=llm_client,
                prompt=input_message,
                config=config,
                messages=conversation_messages
            )
    
    @staticmethod
    def _handle_execution_result(result: Dict[str, Any], agent: Agent, 
                                 executor, state: WorkflowState) -> str:
        """
        Handle execution result, including interrupt detection and storage.
        
        Args:
            result: Execution result from executor
            agent: Agent configuration
            executor: Executor instance
            state: Current workflow state
            
        Returns:
            Response content string (empty if interrupted)
        """
        # Check for interrupt
        if InterruptManager.should_interrupt(result):
            executor_key = ExecutorRegistry()._get_executor_key(agent.name, "workflow")
            interrupt_data = InterruptManager.extract_interrupt_data(result)
            
            # If assistant_message is present (from ResponseAPIExecutor), add it to workflow_state
            # This is needed for resume() to have the complete conversation history
            if "assistant_message" in result:
                assistant_msg = result["assistant_message"]
                # Ensure the message has an ID
                if "id" not in assistant_msg:
                    assistant_msg["id"] = str(uuid.uuid4())
                # Ensure agent name is set
                if "agent" not in assistant_msg:
                    assistant_msg["agent"] = agent.name
                # Add to workflow_state messages
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(assistant_msg)
            
            # Store interrupt in state
            interrupt_update = InterruptManager.store_interrupt(
                state=state,
                agent_name=agent.name,
                interrupt_data=interrupt_data,
                executor_key=executor_key
            )
            state["metadata"].update(interrupt_update["metadata"])
            return ""
        
        # Return normal response
        return result.get("content", "")

