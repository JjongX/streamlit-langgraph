# Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes.

from typing import Any, Callable, Dict, List

from ...agent import Agent
from ...state import WorkflowState, WorkflowStateManager
from .base import AgentNodeBase, create_message_with_id
from .handoff_delegation import HandoffDelegation
from .tool_calling_delegation import ToolCallingDelegation
from ..prompts import SupervisorPromptBuilder


class AgentNodeFactory:
    """Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes."""

    @staticmethod
    def create_supervisor_agent_node(supervisor: Agent, workers: List[Agent], 
                                     allow_parallel: bool = False,
                                     delegation_mode: str = "handoff") -> Callable:
        """Create a supervisor agent node with structured routing."""
        if delegation_mode == "handoff":
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
                if pending_interrupts:
                    return {"current_agent": supervisor.name, "metadata": state.get("metadata", {})}
                
                worker_outputs = HandoffDelegation._build_worker_outputs_summary(state, workers)
                user_query = AgentNodeBase.extract_user_query(state)
                supervisor_instructions = SupervisorPromptBuilder.get_supervisor_instructions(
                    role=supervisor.role,
                    instructions=supervisor.instructions,
                    user_query=user_query,
                    worker_list=", ".join([f"{w.name} ({w.role})" for w in workers]),
                    worker_outputs=worker_outputs
                )
                response, routing_decision = HandoffDelegation._execute_supervisor_with_routing(
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
            tool_agents_map = {agent.name: agent for agent in workers}
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                user_query = AgentNodeBase.extract_user_query(state)
                agent_tools = ToolCallingDelegation._create_agent_tools(workers, state)
                response = ToolCallingDelegation._execute_agent_with_tools(
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
        """Create a worker agent node for workflow patterns (handoff delegation mode only)."""
        return HandoffDelegation.create_worker_agent_node(worker, supervisor)

