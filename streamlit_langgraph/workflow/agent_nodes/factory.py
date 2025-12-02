# Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes.

import uuid
from typing import Any, Callable, Dict, List

from ...agent import Agent, AgentManager
from ...core.executor.registry import ExecutorRegistry
from ...core.middleware.interrupts import InterruptManager
from ...core.state import WorkflowState, WorkflowStateManager
from ...utils import create_message_with_id
from ..prompts import SupervisorPromptBuilder


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
        
        workflow_thread_id = state.get("metadata", {}).get("workflow_thread_id")
        if not workflow_thread_id:
            workflow_thread_id = str(uuid.uuid4())
            state["metadata"]["workflow_thread_id"] = workflow_thread_id
        
        executor_key = f"workflow_executor_{executor.agent.name}"
        state["metadata"]["executors"][executor_key] = {"thread_id": workflow_thread_id}
        
        config = {"configurable": {"thread_id": workflow_thread_id}}
        
        llm_client = AgentManager.get_llm_client(agent)
        conversation_messages = state.get("messages", [])
        stream = False 
        
        result = executor.execute_workflow(
            llm_client=llm_client,
            prompt=input_message,
            stream=stream,
            config=config,
            messages=conversation_messages
        )
        
        if InterruptManager.should_interrupt(result):
            interrupt_data = InterruptManager.extract_interrupt_data(result)
            
            if "assistant_message" in result:
                assistant_msg = result["assistant_message"]
                if "id" not in assistant_msg:
                    assistant_msg["id"] = str(uuid.uuid4())
                if "agent" not in assistant_msg:
                    assistant_msg["agent"] = agent.name
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
            return ""
        
        return result.get("content", "")

    @staticmethod
    def extract_user_query(state: WorkflowState) -> str:
        """Extract user query from state messages."""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                return msg["content"]
        return ""


class AgentNodeFactory:
    """Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes."""

    @staticmethod
    def create_supervisor_agent_node(supervisor: Agent, workers: List[Agent], 
                                     allow_parallel: bool = False,
                                     delegation_mode: str = "handoff") -> Callable:
        """Create a supervisor agent node with structured routing."""
        if delegation_mode == "handoff":
            from .handoff_delegation import HandoffDelegation  # lazy import to avoid circular import
            
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
                if pending_interrupts:
                    return {"current_agent": supervisor.name, "metadata": state.get("metadata", {})}
                
                worker_outputs = HandoffDelegation.build_worker_outputs_summary(state, workers)
                user_query = AgentNodeBase.extract_user_query(state)
                supervisor_instructions = SupervisorPromptBuilder.get_supervisor_instructions(
                    role=supervisor.role,
                    instructions=supervisor.instructions,
                    user_query=user_query,
                    worker_list=", ".join([f"{w.name} ({w.role})" for w in workers]),
                    worker_outputs=worker_outputs
                )
                response, routing_decision = HandoffDelegation.execute_supervisor_with_routing(
                    supervisor, state, supervisor_instructions, workers, allow_parallel
                )
                return {
                    "current_agent": supervisor.name,
                    "messages": [create_message_with_id("assistant", response, supervisor.name)],
                    "agent_outputs": {supervisor.name: response},
                    "metadata": WorkflowStateManager.merge_metadata(state.get("metadata", {}), {"routing_decision": routing_decision})
                }
            return supervisor_agent_node
        else:  # tool calling delegation mode
            from .tool_calling_delegation import ToolCallingDelegation  # lazy import to avoid circular import
            
            tool_agents_map = {agent.name: agent for agent in workers}
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                user_query = AgentNodeBase.extract_user_query(state)
                agent_tools = ToolCallingDelegation.create_agent_tools(workers)
                response = ToolCallingDelegation.execute_agent_with_tools(
                    supervisor, state, user_query, agent_tools, tool_agents_map
                )
                return {
                    "current_agent": supervisor.name,
                    "messages": [create_message_with_id("assistant", response, supervisor.name)],
                    "agent_outputs": {supervisor.name: response}
                }
            return supervisor_agent_node
    
    @staticmethod
    def create_worker_agent_node(worker: Agent, supervisor: Agent) -> Callable:
        """Create a worker agent node for supervisor workflows."""
        from .handoff_delegation import HandoffDelegation  # lazy import to avoid circular import
        
        def worker_agent_node(state: WorkflowState) -> Dict[str, Any]:
            user_query = AgentNodeBase.extract_user_query(state)
            context_data, previous_worker_outputs = HandoffDelegation.build_worker_context(
                state, worker, supervisor
            )
            worker_instructions = SupervisorPromptBuilder.get_worker_agent_instructions(
                role=worker.role, instructions=worker.instructions, user_query=user_query,
                supervisor_output=context_data, previous_worker_outputs=previous_worker_outputs
            )
            response = AgentNodeBase.execute_agent(worker, state, worker_instructions)
            
            executor_key = f"workflow_executor_{worker.name}"
            pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
            if executor_key in pending_interrupts:
                return {
                    "current_agent": worker.name,
                    "metadata": state.get("metadata", {}),
                }
            return {
                "current_agent": worker.name,
                "messages": [create_message_with_id("assistant", response, worker.name)],
                "agent_outputs": {worker.name: response}
            }
        return worker_agent_node
