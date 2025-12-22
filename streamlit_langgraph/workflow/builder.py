# High level workflow builder for creating different types of workflows.

from typing import List, Optional, Any

from langgraph.graph import StateGraph

from ..agent import Agent
from .patterns import SupervisorPattern, HierarchicalPattern, SupervisorTeam, NetworkPattern


class WorkflowBuilder:
    """High-level builder for creating workflow patterns."""
    
    SupervisorTeam = SupervisorTeam

    def create_supervisor_workflow(self, supervisor: Agent, workers: List[Agent], 
                                execution_mode: str = "sequential", 
                                delegation_mode: str = "handoff",
                                checkpointer: Optional[Any] = None) -> StateGraph:
        """
        Create a supervisor workflow with a coordinating supervisor and worker agents.
        
        Delegation modes:
        - "handoff" (default): Agents transfer control between nodes, full context transfer
        - "tool_calling": Calling agent stays in control, workers called as tools
        
        Args:
            supervisor (Agent): Supervisor agent that coordinates the workflow
            workers (List[Agent]): Worker agents that execute tasks
            execution_mode (str): "sequential" or "parallel" execution of workers (only for handoff mode)
            delegation_mode (str): "handoff" or "tool_calling" delegation mode
            checkpointer: Optional checkpointer for workflow state persistence (enables memory, HITL, time travel).
            
        Returns:
            StateGraph: Compiled workflow graph
        """
        return SupervisorPattern.create_supervisor_workflow(
            supervisor, workers, execution_mode, delegation_mode, checkpointer)
    
    def create_hierarchical_workflow(self, top_supervisor: Agent, 
                                   supervisor_teams: List[SupervisorTeam],
                                   execution_mode: str = "sequential",
                                   checkpointer: Optional[Any] = None) -> StateGraph:
        """
        Create a hierarchical workflow with a top supervisor coordinating multiple
        supervisor teams (sub-supervisors with their workers).
        
        This uses HANDOFF delegation mode at each level.
        
        Args:
            top_supervisor (Agent): Top-level supervisor that coordinates sub-supervisors
            supervisor_teams (List[SupervisorTeam]): List of supervisor teams, each containing
                                                     a supervisor and their workers
            execution_mode (str): "sequential" execution (default and only supported mode)
            checkpointer: Optional checkpointer for workflow state persistence (enables memory, HITL, time travel).
            
        Returns:
            StateGraph: Compiled hierarchical workflow graph
        """
        return HierarchicalPattern.create_hierarchical_workflow(
            top_supervisor, supervisor_teams, execution_mode, checkpointer)
    
    def create_network_workflow(self, agents: List[Agent],
                               checkpointer: Optional[Any] = None) -> StateGraph:
        """
        Create a network workflow where agents can communicate peer-to-peer.
        
        In the network pattern, agents form a mesh topology where any agent can
        hand off to any other agent. There is no central supervisor - all agents
        are peers. The first agent in the list serves as the entry point.
        
        Args:
            agents (List[Agent]): List of peer agents. First agent is the entry point.
            checkpointer: Optional checkpointer for workflow state persistence.
            
        Returns:
            StateGraph: Compiled network workflow graph
        """
        return NetworkPattern.create_network_workflow(agents, checkpointer)

