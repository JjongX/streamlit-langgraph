from typing import List

from langgraph.graph import StateGraph, START, END

from ...agent import Agent
from ..agent_nodes import AgentNodeFactory
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
                                 execution_mode: str = "sequential", delegation_mode: str = "handoff") -> StateGraph:
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
            return SupervisorPattern._create_tool_calling_workflow(supervisor_agent, worker_agents)
        # Handoff mode - multiple nodes, structural handoff
        graph = StateGraph(WorkflowState)
        
        # Create supervisor node
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
        """
        Create sequential supervisor workflow.
        
        LangGraph pattern: Supervisor delegates to workers one at a time. Workers
        route back to supervisor, creating a loop until supervisor decides to finish.
        """
        for worker in worker_agents:
            graph.add_node(worker.name, AgentNodeFactory.create_worker_agent_node(worker, supervisor_agent))

        def supervisor_sequential_route(state: WorkflowState) -> str:
            """
            Route based on supervisor's structured routing decision.
            
            LangGraph conditional edge function that routes to worker nodes or END
            based on routing_decision metadata set by supervisor node.
            """
            routing_decision = state["metadata"].get("routing_decision", {})
            worker_names = [worker.name for worker in worker_agents]
            action = routing_decision.get("action", "finish")
            
            if action == "delegate":
                target_worker = routing_decision.get("target_worker", "")
                return target_worker if target_worker in worker_names else "__end__"
            return "__end__"
        
        supervisor_routes = {worker.name: worker.name for worker in worker_agents}
        supervisor_routes["__end__"] = END

        graph.add_conditional_edges(supervisor_agent.name, supervisor_sequential_route, supervisor_routes)
        for worker in worker_agents:
            graph.add_edge(worker.name, supervisor_agent.name)
        
        return graph.compile()

    @staticmethod
    def _create_parallel_supervisor_workflow(graph: StateGraph, supervisor_agent: Agent, 
                                           worker_agents: List[Agent]) -> StateGraph:
        """
        Create parallel supervisor workflow.
        
        LangGraph pattern: Supervisor delegates to all workers simultaneously using
        a fan-out node. All workers execute in parallel, then route back to supervisor.
        """
        graph.add_node("parallel_fanout", lambda state: state)
        
        for worker in worker_agents:
            graph.add_node(worker.name, AgentNodeFactory.create_worker_agent_node(worker, supervisor_agent))
        
        def supervisor_parallel_route(state: WorkflowState) -> str:
            """
            Route from supervisor to parallel execution or end.
            
            Checks if supervisor delegated with "PARALLEL" target, routes to fan-out node.
            """
            routing_decision = state["metadata"].get("routing_decision", {})
            action = routing_decision.get("action", "finish")
            target_worker = routing_decision.get("target_worker", "")
            
            if action == "delegate" and target_worker == "PARALLEL":
                return "parallel_fanout"
            return "__end__"
        
        supervisor_routes = {"parallel_fanout": "parallel_fanout", "__end__": END}
        graph.add_conditional_edges(supervisor_agent.name, supervisor_parallel_route, supervisor_routes)
        
        for worker in worker_agents:
            graph.add_edge("parallel_fanout", worker.name)
            graph.add_edge(worker.name, supervisor_agent.name)
        
        return graph.compile()
    
    @staticmethod
    def _create_tool_calling_workflow(calling_agent: Agent, tool_agents: List[Agent]) -> StateGraph:
        """
        Create a tool calling workflow implementing the "agent-as-tools" pattern.
        
        LangGraph pattern: Single node workflow where the calling agent uses OpenAI
        function calling to invoke tool agents synchronously. Unlike supervisor pattern,
        control stays with the calling agent (no handoff between nodes).
        
        Args:
            calling_agent: Agent that can call other agents as tools
            tool_agents: List of agents exposed as callable tools
        
        Returns:
            Compiled LangGraph workflow with single node
        """
        graph = StateGraph(WorkflowState)
        calling_node = AgentNodeFactory.create_tool_calling_agent_node(
            calling_agent, tool_agents
        )
        graph.add_node(calling_agent.name, calling_node)
        graph.add_edge(START, calling_agent.name)
        graph.add_edge(calling_agent.name, END)
        
        return graph.compile()
    