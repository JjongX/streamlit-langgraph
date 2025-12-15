# Network workflow pattern - agents can communicate directly with any peer.

from typing import List, Optional, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from ...agent import Agent
from ..agent_nodes.factory import AgentNodeFactory
from ..agent_nodes.routing import RoutingHelper
from ...core.state import WorkflowState


class NetworkPattern:
    """
    Network workflow pattern - peer-to-peer mesh topology.
    
    Any agent can hand off to any other agent. No central supervisor.
    First agent in list is the entry point.
    """
    
    @staticmethod
    def _sync_container_ids(agents: List[Agent]) -> None:
        """Share container_id across all code_interpreter agents."""
        code_interpreter_agents = [a for a in agents if a.allow_code_interpreter]
        if not code_interpreter_agents:
            return
        
        # Find first agent with a container_id set, or None
        shared_container_id = None
        for agent in code_interpreter_agents:
            if agent.container_id and isinstance(agent.container_id, str):
                shared_container_id = agent.container_id
                break
        
        # Apply the shared container_id to all code_interpreter agents
        if shared_container_id:
            for agent in code_interpreter_agents:
                agent.container_id = shared_container_id
    
    @staticmethod
    def create_network_workflow(
        agents: List[Agent],
        checkpointer: Optional[Any] = None
    ) -> StateGraph:
        """
        Create a network workflow where agents can communicate peer-to-peer.
        
        The first agent in the list serves as the entry point. Each agent can
        delegate to any other agent in the network, or finish the workflow.
        
        Args:
            agents: List of agents that form the network. First agent is the entry point.
            checkpointer: Optional checkpointer for workflow state persistence.
            
        Returns:
            StateGraph: Compiled network workflow graph
        """
        if not agents or len(agents) < 2:
            raise ValueError("At least two agents are required for a network workflow")
        
        if checkpointer is None:
            workflow_checkpointer = InMemorySaver()
        else:
            workflow_checkpointer = checkpointer
        
        # Ensure all agents with code_interpreter share the same container_id
        NetworkPattern._sync_container_ids(agents)
        
        graph = StateGraph(WorkflowState)
        
        # Add all agents as nodes
        for agent in agents:
            # Get peer agents (all agents except the current one)
            peer_agents = [a for a in agents if a.name != agent.name]
            node = AgentNodeFactory.create_network_agent_node(agent, peer_agents)
            graph.add_node(agent.name, node)
        
        # Connect first agent to START
        entry_agent = agents[0]
        graph.add_edge(START, entry_agent.name)
        
        # Add conditional edges from each agent to all peers and END
        for agent in agents:
            peer_names = [a.name for a in agents if a.name != agent.name]
            route_fn = RoutingHelper.create_network_route(peer_names)
            
            # Build route mapping: peer names -> peer names, __end__ -> END
            routes = {name: name for name in peer_names}
            routes["__end__"] = END
            graph.add_conditional_edges(agent.name, route_fn, routes)
        
        return graph.compile(checkpointer=workflow_checkpointer)

