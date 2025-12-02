# Supervisor workflow pattern.

from typing import List, Optional, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from ...agent import Agent
from ..agent_nodes.factory import AgentNodeFactory
from ..agent_nodes.routing import RoutingHelper
from ...core.state import WorkflowState


class SupervisorPattern:
    """
    Supervisor workflow pattern supporting multiple delegation modes.
    
    Delegation modes:
    - "handoff": Agents transfer control between nodes
    - "tool_calling": Calling agent stays in control, agents called as tools
    """
    
    @staticmethod
    def create_supervisor_workflow(supervisor_agent: Agent, worker_agents: List[Agent], 
                                 execution_mode: str = "sequential", delegation_mode: str = "handoff",
                                 checkpointer: Optional[Any] = None) -> StateGraph:
        """
        Create a supervisor workflow where a supervisor agent coordinates and delegates tasks 
        to worker agents using specified execution and delegation modes.
        
        Args:
            supervisor_agent (Agent): The supervisor agent that coordinates tasks
            worker_agents (List[Agent]): List of worker agents with specialized capabilities
            execution_mode (str): "sequential" or "parallel" execution of workers
            delegation_mode (str): "handoff" or "tool_calling" delegation mode
            checkpointer: Optional checkpointer for workflow state persistence (enables memory, HITL, time travel).
            
        Returns:
            StateGraph: Compiled workflow graph
        """
        if not supervisor_agent or not worker_agents:
            raise ValueError("At least one supervisor agent and one worker agent are required")
        if delegation_mode not in ("handoff", "tool_calling"):
            raise ValueError(f"delegation_mode must be 'handoff' or 'tool_calling', got '{delegation_mode}'")
        if checkpointer is None:
            workflow_checkpointer = InMemorySaver()
        else:
            workflow_checkpointer = checkpointer
    
        # Tool calling mode - single node, agents as tools
        if delegation_mode == "tool_calling":
            graph = StateGraph(WorkflowState)
            calling_node = AgentNodeFactory.create_supervisor_agent_node(
                supervisor_agent, worker_agents, delegation_mode="tool_calling"
            )
            graph.add_node(supervisor_agent.name, calling_node)
            graph.add_edge(START, supervisor_agent.name)
            graph.add_edge(supervisor_agent.name, END)
            return graph.compile(checkpointer=workflow_checkpointer)
        
        graph = StateGraph(WorkflowState)
        allow_parallel = (execution_mode == "parallel")
        supervisor_node = AgentNodeFactory.create_supervisor_agent_node(
            supervisor_agent, worker_agents, allow_parallel=allow_parallel, delegation_mode="handoff"
        )
        graph.add_node(supervisor_agent.name, supervisor_node)
        graph.add_edge(START, supervisor_agent.name)

        if execution_mode == "sequential":
            return SupervisorPattern._create_sequential_supervisor_workflow(
                graph, supervisor_agent, worker_agents, workflow_checkpointer)
        else: # parallel
            return SupervisorPattern._create_parallel_supervisor_workflow(
                graph, supervisor_agent, worker_agents, workflow_checkpointer)
    
    @staticmethod
    def _create_sequential_supervisor_workflow(graph: StateGraph, supervisor_agent: Agent, 
                                             worker_agents: List[Agent], workflow_checkpointer: Optional[Any] = None) -> StateGraph:
        """
        Create sequential supervisor workflow.
        
        LangGraph pattern: Supervisor delegates to workers one at a time. Workers
        route back to supervisor, creating a loop until supervisor decides to finish.
        """
        for worker in worker_agents:
            graph.add_node(worker.name, AgentNodeFactory.create_worker_agent_node(worker, supervisor_agent))

        worker_names = [worker.name for worker in worker_agents]
        supervisor_sequential_route = RoutingHelper.create_sequential_route(worker_names)
        
        supervisor_routes = {worker.name: worker.name for worker in worker_agents}
        supervisor_routes["__end__"] = END
        graph.add_conditional_edges(supervisor_agent.name, supervisor_sequential_route, supervisor_routes)
        
        for worker in worker_agents:
            graph.add_edge(worker.name, supervisor_agent.name)
        
        return graph.compile(checkpointer=workflow_checkpointer)

    @staticmethod
    def _create_parallel_supervisor_workflow(graph: StateGraph, supervisor_agent: Agent, 
                                           worker_agents: List[Agent], workflow_checkpointer: Optional[Any] = None) -> StateGraph:
        """
        Create parallel supervisor workflow.
        
        LangGraph pattern: Supervisor delegates to all workers simultaneously using
        a fan-out node. All workers execute in parallel, then route back to supervisor.
        """
        graph.add_node("parallel_fanout", lambda state: state)
        
        for worker in worker_agents:
            graph.add_node(worker.name, AgentNodeFactory.create_worker_agent_node(worker, supervisor_agent))
        
        supervisor_parallel_route = RoutingHelper.create_parallel_route()
        
        supervisor_routes = {"parallel_fanout": "parallel_fanout", "__end__": END}
        graph.add_conditional_edges(supervisor_agent.name, supervisor_parallel_route, supervisor_routes)
        
        for worker in worker_agents:
            graph.add_edge("parallel_fanout", worker.name)
            graph.add_edge(worker.name, supervisor_agent.name)
        
        return graph.compile(checkpointer=workflow_checkpointer)
    