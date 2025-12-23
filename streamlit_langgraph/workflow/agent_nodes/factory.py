# Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes.

import uuid
from ...agent import AgentManager
from ...core.executor.registry import ExecutorRegistry
from ...core.middleware import InterruptManager
from ...core.state import WorkflowStateManager, create_message_with_id
from ..prompts import SupervisorPromptBuilder


class AgentNodeBase:
    """Base class providing common functionality for agent node operations."""
    
    @staticmethod
    def execute_agent(agent, state, input_message):
        """Execute an agent and return the response."""
        executor = ExecutorRegistry().get_or_create(agent, executor_type="workflow")
        
        if hasattr(executor, 'tools'):
            executor.tools = agent.get_tools()
        
        executor_key = f"workflow_executor_{executor.agent.name}"
        config, workflow_thread_id = WorkflowStateManager.get_or_create_workflow_config(state, executor_key)
        
        llm_client = AgentManager.get_llm_client(agent)
        conversation_messages = state.get("messages", [])
        stream = False
        
        file_messages = state.get("metadata", {}).get("file_messages")
        vector_store_ids = state.get("metadata", {}).get("vector_store_ids")
        # Update LLM client with vector_store_ids if file_search is enabled
        if agent.allow_file_search and vector_store_ids:
            current_vector_ids = getattr(llm_client, '_vector_store_ids', None)
            if current_vector_ids != vector_store_ids:
                llm_client = AgentManager.get_llm_client(agent, vector_store_ids=vector_store_ids)
        
        result = executor.execute_workflow(
            llm_client=llm_client,
            prompt=input_message,
            stream=stream,
            config=config,
            messages=conversation_messages,
            file_messages=file_messages
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
            
            interrupt_update = WorkflowStateManager.set_pending_interrupt(
                state, agent.name, interrupt_data, executor_key
            )
            state["metadata"].update(interrupt_update["metadata"])
            return ""
        
        return result.get("content", "")

    @staticmethod
    def extract_user_query(state) -> str:
        """Extract user query from state messages."""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                return msg["content"]
        return ""


class AgentNodeFactory:
    """Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes."""

    @staticmethod
    def create_supervisor_agent_node(supervisor, workers, allow_parallel=False, delegation_mode="handoff"):
        """Create a supervisor agent node with structured routing."""
        from .tool_calling_delegation import ToolCallingDelegation  # lazy import to avoid circular import

        if delegation_mode == "handoff":
            from .handoff_delegation import HandoffDelegation  # lazy import to avoid circular import
            
            def supervisor_agent_node(state):
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
                # Always create message
                messages_update = [create_message_with_id("assistant", response, supervisor.name)]
                return {
                    "current_agent": supervisor.name,
                    "messages": messages_update,
                    "agent_outputs": {supervisor.name: response},
                    "metadata": WorkflowStateManager.merge_metadata(state.get("metadata", {}), {"routing_decision": routing_decision})
                }
            return supervisor_agent_node
        else:  # tool calling delegation mode
            
            tool_agents_map = {agent.name: agent for agent in workers}
            def supervisor_agent_node(state):
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
    def create_worker_agent_node(worker, supervisor):
        """Create a worker agent node for supervisor workflows."""
        from .handoff_delegation import HandoffDelegation  # lazy import to avoid circular import
        
        def worker_agent_node(state):
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
    
    @staticmethod
    def create_network_agent_node(agent, peer_agents):
        """Create a network agent node that can hand off to any peer."""
        from .handoff_delegation import HandoffDelegation  # lazy import to avoid circular import
        from ..prompts import NetworkPromptBuilder
        
        def network_agent_node(state):
            pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
            if pending_interrupts:
                return {"current_agent": agent.name, "metadata": state.get("metadata", {})}
            
            # Build context from peer outputs
            peer_outputs = []
            for peer in peer_agents:
                if peer.name in state.get("agent_outputs", {}):
                    output = state["agent_outputs"][peer.name]
                    peer_outputs.append(f"**{peer.name}**: {output}")
            
            user_query = AgentNodeBase.extract_user_query(state)
            network_instructions = NetworkPromptBuilder.get_network_agent_instructions(
                role=agent.role,
                instructions=agent.instructions,
                user_query=user_query,
                peer_list=", ".join([f"{p.name} ({p.role})" for p in peer_agents]),
                peer_outputs=peer_outputs
            )
            
            response, routing_decision = HandoffDelegation.execute_supervisor_with_routing(
                agent, state, network_instructions, peer_agents, allow_parallel=False
            )
            
            messages_update = [create_message_with_id("assistant", response, agent.name)]
            return {
                "current_agent": agent.name,
                "messages": messages_update,
                "agent_outputs": {agent.name: response},
                "metadata": WorkflowStateManager.merge_metadata(
                    state.get("metadata", {}), {"routing_decision": routing_decision}
                )
            }
        
        return network_agent_node