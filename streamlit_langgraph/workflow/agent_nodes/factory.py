"""Factory for creating LangGraph agent nodes with delegation modes."""

import uuid

from ...agent import get_llm_client
from ...core.middleware import InterruptManager
from ...core.runtime import RuntimeHooks
from ...core.state import WorkflowStateManager
from ..prompts import SupervisorPromptBuilder
from ...utils.file_handler import FileHandler


class AgentNodeBase:
    """Base class providing common functionality for agent node operations."""

    def __init__(self, runtime: RuntimeHooks):
        self.runtime = runtime

    def execute_agent(self, agent, state, input_message, allow_stream=True):
        """Execute an agent and return the response."""
        executor = self.runtime.executor_registry.get_or_create(agent, executor_type="workflow")
        
        if hasattr(executor, 'tools'):
            executor.tools = agent.get_tools()
        
        executor_key = f"workflow_executor_{executor.agent.name}"
        config, workflow_thread_id = WorkflowStateManager.get_or_create_workflow_config(state, executor_key)
        
        llm_client = get_llm_client(agent)
        conversation_messages = AgentNodeBase.get_visible_conversation_messages(state)
        stream = AgentNodeBase._should_stream(agent, state, allow_stream)
        
        file_messages = state.get("metadata", {}).get("file_messages")
        vector_store_ids = state.get("metadata", {}).get("vector_store_ids")
        
        llm_client = FileHandler.ensure_llm_vector_ids(agent, llm_client, vector_store_ids)
        
        from ...core.executor.response_api import ResponseAPIExecutor
        if isinstance(executor, ResponseAPIExecutor):
            result = executor.execute_workflow(
                prompt=input_message,
                stream=stream,
                messages=conversation_messages,
                file_messages=file_messages,
                vector_store_ids=vector_store_ids,
            )
        else:
            result = executor.execute_workflow(
                llm_client=llm_client,
                prompt=input_message,
                stream=stream,
                config=config,
                messages=conversation_messages,
                file_messages=file_messages,
            )
        
        result.setdefault("id", str(uuid.uuid4()))

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
            return {"id": result["id"], "content": "", "agent": agent.name}
        
        if stream and self.runtime.stream_renderer:
            result["content"] = self.runtime.stream_renderer.render(
                agent,
                result["stream"],
                result["id"],
            )
            result.pop("stream", None)
        return {"id": result["id"], "content": result["content"], "agent": agent.name}

    @staticmethod
    def get_visible_conversation_messages(state):
        """Return workflow messages excluding runtime status events."""
        messages = state.get("messages", [])
        return [
            msg for msg in messages
            if isinstance(msg, dict) and not msg.get("is_status_event")
        ]

    @staticmethod
    def _should_stream(agent, state, allow_stream):
        """Decide whether to stream agent output in workflows."""
        metadata = state.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "Workflow state is missing metadata. Initialize state via WorkflowStateManager "
                "or LangGraphChat so metadata contracts are present."
            )
        stream_flag = WorkflowStateManager.require_stream_flag(state)
        routing = metadata.get("routing_decision", {})
        if routing.get("target_worker") == "PARALLEL":
            return False
        if getattr(agent, "human_in_loop", False):
            return False
        return allow_stream and stream_flag
    
    @staticmethod
    def extract_user_query(state) -> str:
        """Extract user query from state messages."""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                return msg["content"]
        return ""


class AgentNodeFactory:
    """Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes."""

    def __init__(self, runtime: RuntimeHooks):
        self.runtime = runtime
        self.base = AgentNodeBase(runtime)

    def create_supervisor_agent_node(self, supervisor, workers, allow_parallel=False, delegation_mode="handoff"):
        """Create a supervisor agent node with structured routing."""
        from .tool_calling_delegation import ToolCallingDelegation  # lazy import to avoid circular import

        if delegation_mode == "handoff":
            from .handoff_delegation import HandoffDelegation  # lazy import to avoid circular import
            handoff = HandoffDelegation(self.runtime, self.base)
            
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
                response, routing_decision = handoff.execute_supervisor_with_routing(
                    supervisor, state, supervisor_instructions, workers, allow_parallel
                )
                # Always create message
                messages_update = [{
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": response,
                    "agent": supervisor.name,
                }]
                return {
                    "current_agent": supervisor.name,
                    "messages": messages_update,
                    "agent_outputs": {supervisor.name: response},
                    "metadata": WorkflowStateManager.merge_metadata(state.get("metadata", {}), {"routing_decision": routing_decision})
                }
            return supervisor_agent_node
        else:  # tool calling delegation mode
            tool_agents_map = {agent.name: agent for agent in workers}
            tool_caller = ToolCallingDelegation(self.base)
            def supervisor_agent_node(state):
                user_query = AgentNodeBase.extract_user_query(state)
                agent_tools = ToolCallingDelegation.create_agent_tools(workers)
                response = tool_caller.execute_agent_with_tools(
                    supervisor, state, user_query, agent_tools, tool_agents_map
                )
                return {
                    "current_agent": supervisor.name,
                    "messages": [{
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": response,
                        "agent": supervisor.name,
                    }],
                    "agent_outputs": {supervisor.name: response}
                }
            return supervisor_agent_node
    
    def create_worker_agent_node(self, worker, supervisor):
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
            response = self.base.execute_agent(worker, state, worker_instructions)
            
            executor_key = f"workflow_executor_{worker.name}"
            pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
            if executor_key in pending_interrupts:
                return {
                    "current_agent": worker.name,
                    "metadata": state.get("metadata", {}),
                }
            completed_status = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "[status] Completed",
                "agent": worker.name,
                "is_status_event": True,
            }
            return {
                "current_agent": worker.name,
                "messages": [
                    completed_status,
                    {
                        "id": response["id"],
                        "role": "assistant",
                        "content": response["content"],
                        "agent": worker.name,
                    },
                ],
                "agent_outputs": {worker.name: response["content"]},
            }
        return worker_agent_node
    
    def create_network_agent_node(self, agent, peer_agents):
        """Create a network agent node that can hand off to any peer."""
        from .handoff_delegation import HandoffDelegation  # lazy import to avoid circular import
        from ..prompts import NetworkPromptBuilder
        handoff = HandoffDelegation(self.runtime, self.base)
        
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
            
            response, routing_decision = handoff.execute_supervisor_with_routing(
                agent, state, network_instructions, peer_agents, allow_parallel=False
            )
            
            messages_update = [{
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": response,
                "agent": agent.name,
            }]
            return {
                "current_agent": agent.name,
                "messages": messages_update,
                "agent_outputs": {agent.name: response},
                "metadata": WorkflowStateManager.merge_metadata(
                    state.get("metadata", {}), {"routing_decision": routing_decision}
                )
            }
        
        return network_agent_node
