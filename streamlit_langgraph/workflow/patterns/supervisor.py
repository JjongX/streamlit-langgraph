from typing import List

from langgraph.graph import StateGraph, START, END

from ...agent import Agent
from ..nodes import AgentNodeFactory
from ..state import WorkflowState

class SupervisorPattern:
    """
    Supervisor workflow pattern supporting multiple delegation modes.
    
    Delegation modes:
    - "handoff": Agents transfer control between nodes, full context transfer
    - "tool_calling": Calling agent stays in control, agents called as tools
    """
    
    @staticmethod
    def create_supervisor_workflow(supervisor_agent: Agent, worker_agents: List[Agent], 
                                 execution_mode: str = "sequential",
                                 delegation_mode: str = "handoff") -> StateGraph:
        """
        Create a supervisor workflow where a supervisor agent coordinates and delegates tasks 
        to worker agents using specified execution and delegation modes.
        
        Args:
            supervisor_agent (Agent): The supervisor agent that coordinates tasks
            worker_agents (List[Agent]): List of worker agents with specialized capabilities
            execution_mode (str): "sequential" or "parallel" execution of workers
            delegation_mode (str): "handoff" or "tool_calling" delegation mode
            
        Returns:
            StateGraph: Compiled workflow graph
        """
        if not supervisor_agent or not worker_agents:
            raise ValueError("At least one supervisor agent and one worker agent are required")
        
        if delegation_mode not in ("handoff", "tool_calling"):
            raise ValueError(f"delegation_mode must be 'handoff' or 'tool_calling', got '{delegation_mode}'")
        
        # Tool calling mode - single node, agents as tools
        if delegation_mode == "tool_calling":
            return SupervisorPattern._create_tool_calling_workflow(
                supervisor_agent, worker_agents
            )
        
        # Handoff mode - multiple nodes, structural handoff
        graph = StateGraph(WorkflowState)
        
        # Create supervisor node with parallel support if needed
        allow_parallel = (execution_mode == "parallel")
        supervisor_node = AgentNodeFactory.create_supervisor_agent_node(
            supervisor_agent, worker_agents, allow_parallel=allow_parallel
        )
        graph.add_node(supervisor_agent.name, supervisor_node)
        graph.add_edge(START, supervisor_agent.name)

        if execution_mode == "sequential":
            return SupervisorPattern._create_sequential_supervisor_workflow(
                graph, supervisor_agent, worker_agents)
        else: # parallel
            return SupervisorPattern._create_parallel_supervisor_workflow(
                graph, supervisor_agent, worker_agents)
    
    @staticmethod
    def _create_sequential_supervisor_workflow(graph: StateGraph, supervisor_agent: Agent, 
                                             worker_agents: List[Agent]) -> StateGraph:
        """Create sequential supervisor workflow."""

        # Nodes
        for worker in worker_agents:
            graph.add_node(worker.name, AgentNodeFactory.create_worker_agent_node(worker, supervisor_agent))

        # Routes
        def supervisor_sequential_route(state: WorkflowState) -> str:
            """Route based on structured routing decision from supervisor."""
            routing_decision = state["metadata"].get("routing_decision", {})
            worker_names = [worker.name for worker in worker_agents]
            
            action = routing_decision.get("action", "finish")
            
            if action == "delegate":
                target_worker = routing_decision.get("target_worker", "")
                if target_worker in worker_names:
                    return target_worker
                else:
                    return "__end__"
            else:
                # action == "finish" or any other value
                return "__end__"
        
        supervisor_routes = {worker.name: worker.name for worker in worker_agents}
        supervisor_routes["__end__"] = END

        # Edges
        graph.add_conditional_edges(supervisor_agent.name, supervisor_sequential_route, supervisor_routes)
        # Workers always route back to supervisor (no conditional needed!)
        for worker in worker_agents:
            graph.add_edge(worker.name, supervisor_agent.name)
        
        return graph.compile()

    @staticmethod
    def _create_parallel_supervisor_workflow(graph: StateGraph, supervisor_agent: Agent, 
                                           worker_agents: List[Agent]) -> StateGraph:
        """
        Create parallel supervisor workflow.
        Supervisor delegates to all workers in parallel, then reviews their outputs.
        """
        
        # Add pass-through node for parallel fan-out (needed for conditional routing)
        graph.add_node("parallel_fanout", lambda state: state)
        
        # Add worker nodes
        for worker in worker_agents:
            graph.add_node(worker.name, AgentNodeFactory.create_worker_agent_node(worker, supervisor_agent))
        
        def supervisor_parallel_route(state: WorkflowState) -> str:
            """Route from supervisor to parallel execution or end."""
            routing_decision = state["metadata"].get("routing_decision", {})
            action = routing_decision.get("action", "finish")
            
            target_worker = routing_decision.get("target_worker", "")
            if action == "delegate" and target_worker == "PARALLEL":
                return "parallel_fanout"
            else:
                return "__end__"
        
        # Supervisor routes to parallel_fanout or END
        supervisor_routes = {"parallel_fanout": "parallel_fanout", "__end__": END}
        graph.add_conditional_edges(supervisor_agent.name, supervisor_parallel_route, supervisor_routes)
        
        # Fan-out from parallel_fanout to all workers
        for worker in worker_agents:
            graph.add_edge("parallel_fanout", worker.name)
        
        # Fan-in: all workers route back to supervisor
        for worker in worker_agents:
            graph.add_edge(worker.name, supervisor_agent.name)
        
        return graph.compile()
    
    @staticmethod
    def _create_tool_calling_workflow(calling_agent: Agent, tool_agents: List[Agent]) -> StateGraph:
        """
        Create a tool calling workflow where an agent can call other agents as tools.
        
        This implements the TOOL CALLING delegation mode where:
        - The calling agent stays in control (single node)
        - Tool agents are invoked synchronously and return results
        - Simple task descriptions are passed (not full context)
        - Results are returned directly to the calling agent
        
        Args:
            calling_agent (Agent): The agent that can call other agents as tools
            tool_agents (List[Agent]): List of agents that will be exposed as tools
        
        Returns:
            StateGraph: Compiled workflow graph
        """
        graph = StateGraph(WorkflowState)
        
        # Create calling agent node with tool agents exposed as tools
        calling_node = AgentNodeFactory.create_tool_calling_agent_node(
            calling_agent, tool_agents
        )
        graph.add_node(calling_agent.name, calling_node)
        graph.add_edge(START, calling_agent.name)
        
        # Tool calling agent always routes to END (no handoff, stays in control)
        graph.add_edge(calling_agent.name, END)
        
        return graph.compile()
    
