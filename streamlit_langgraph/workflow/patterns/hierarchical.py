from dataclasses import dataclass
from typing import List

from langgraph.graph import StateGraph, START, END

from ...agent import Agent
from ..agent_nodes.factory import AgentNodeFactory
from ...state import WorkflowState


@dataclass
class SupervisorTeam:
    """
    Represents a sub-supervisor and their team of workers.
    Internal implementation for HierarchicalPattern.
    """
    supervisor: Agent
    workers: List[Agent]
    team_name: str = ""
    
    def __post_init__(self):
        if not self.team_name:
            self.team_name = f"{self.supervisor.name}_team"


class HierarchicalPattern:
    """
    Hierarchical workflow pattern - a top supervisor delegates to sub-supervisors,
    each managing their own team of workers.
    - Top level: supervisor pattern (top supervisor -> sub-supervisors)
    - Team level: supervisor pattern (sub-supervisor -> workers)
    """
    
    @staticmethod
    def create_hierarchical_workflow(
        top_supervisor: Agent,
        supervisor_teams: List[SupervisorTeam],
        execution_mode: str = "sequential"
    ) -> StateGraph:
        """
        Create a hierarchical workflow with a top supervisor coordinating multiple
        supervisor teams.
        
        Args:
            top_supervisor: The top-level supervisor that coordinates sub-supervisors
            supervisor_teams: List of SupervisorTeam objects, each with a supervisor and workers
            execution_mode: "sequential" execution (parallel not yet supported for hierarchical)
            
        Returns:
            StateGraph: Compiled hierarchical workflow graph
        """
        if not top_supervisor or not supervisor_teams:
            raise ValueError("Top supervisor and at least one supervisor team are required")
        
        if execution_mode != "sequential":
            raise NotImplementedError("Only sequential mode is currently supported for hierarchical workflows")
        
        graph = StateGraph(WorkflowState)
        
        # Extract all sub-supervisors for top-level delegation
        sub_supervisors = [team.supervisor for team in supervisor_teams]
        
        # Top supervisor treats sub-supervisors as its "workers"
        # Hierarchical pattern doesn't support parallel at this level
        top_supervisor_node = AgentNodeFactory.create_supervisor_agent_node(
            top_supervisor, sub_supervisors, allow_parallel=False
        )
        graph.add_node(top_supervisor.name, top_supervisor_node)
        graph.add_edge(START, top_supervisor.name)
        
        for team in supervisor_teams:
            # Sub-supervisor is just a supervisor for their team
            # Hierarchical pattern doesn't support parallel at team level
            sub_supervisor_node = AgentNodeFactory.create_supervisor_agent_node(
                team.supervisor, team.workers, allow_parallel=False
            )
            graph.add_node(team.supervisor.name, sub_supervisor_node)
            # Add worker nodes using standard worker node factory
            for worker in team.workers:
                worker_node = AgentNodeFactory.create_worker_agent_node(
                    worker, team.supervisor
                )
                graph.add_node(worker.name, worker_node)
        
        # Add routing using supervisor pattern routing logic
        graph = HierarchicalPattern._add_hierarchical_routing(
            graph, top_supervisor, supervisor_teams
        )
        
        return graph.compile()
    
    @staticmethod
    def _add_hierarchical_routing(graph: StateGraph, top_supervisor: Agent, 
                                  supervisor_teams: List[SupervisorTeam]) -> StateGraph:
        """
        Add routing edges for hierarchical workflow.
        
        REUSES supervisor pattern routing logic:
        - Top supervisor -> sub-supervisors (same as supervisor -> workers)
        - Sub-supervisors -> workers (same as supervisor -> workers)
        - Workers -> sub-supervisor (same as workers -> supervisor)
        - Sub-supervisors -> top supervisor (same as workers -> supervisor)
        """
        
        # Collect all sub-supervisor names
        sub_supervisor_names = [team.supervisor.name for team in supervisor_teams]
        
        # Top supervisor routing (same pattern as supervisor -> workers)
        def top_supervisor_route(state: WorkflowState) -> str:
            """Top supervisor routes to sub-supervisors or END."""
            routing_decision = state["metadata"].get("routing_decision", {})
            action = routing_decision.get("action", "finish")
            
            if action == "delegate":
                target = routing_decision.get("target_worker", "")
                if target in sub_supervisor_names:
                    return target
            return "__end__"
        
        top_routes = {name: name for name in sub_supervisor_names}
        top_routes["__end__"] = END
        graph.add_conditional_edges(top_supervisor.name, top_supervisor_route, top_routes)
        
        # For each team, apply standard supervisor pattern routing
        for team in supervisor_teams:
            worker_names = [w.name for w in team.workers]
            # Sub-supervisor routes to workers or back to top supervisor
            # Same logic as supervisor routing, but returns to top supervisor instead of END
            def make_subsupervisor_route(team_worker_names, top_sup_name):
                def subsupervisor_route(state: WorkflowState) -> str:
                    """Sub-supervisor routes to workers or back to top supervisor."""
                    routing_decision = state["metadata"].get("routing_decision", {})
                    action = routing_decision.get("action", "finish")
                    
                    if action == "delegate":
                        target = routing_decision.get("target_worker", "")
                        if target in team_worker_names:
                            return target
                    # When done, return to top supervisor (not END)
                    return top_sup_name
                return subsupervisor_route
            
            sub_routes = {name: name for name in worker_names}
            sub_routes[top_supervisor.name] = top_supervisor.name
            
            graph.add_conditional_edges(
                team.supervisor.name,
                make_subsupervisor_route(worker_names, top_supervisor.name),
                sub_routes
            )
            
            # Workers route back to their sub-supervisor
            for worker in team.workers:
                graph.add_edge(worker.name, team.supervisor.name)
        
        return graph

