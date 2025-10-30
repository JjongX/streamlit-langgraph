from typing import List

from langgraph.graph import StateGraph

from ..agent import Agent
from .patterns import SupervisorPattern, HierarchicalPattern, SupervisorTeam

class WorkflowBuilder:
    """
    Workflow builder that delegates to specialized pattern classes.
    
    Provides a clean interface for creating supervised and hierarchical workflows.
    """
    
    def create_supervisor_workflow(self, supervisor: Agent, workers: List[Agent], 
                                execution_mode: str = "sequential", 
                                max_iterations: int = 5) -> StateGraph:
        """
        Create a supervisor workflow with a coordinating supervisor and worker agents.
        
        Args:
            supervisor (Agent): Supervisor agent that coordinates the workflow
            workers (List[Agent]): Worker agents that execute tasks
            execution_mode (str): "sequential" or "parallel" execution of workers
            max_iterations (int): Maximum number of supervisor iterations
            
        Returns:
            StateGraph: Compiled workflow graph
        """
        return SupervisorPattern.create_supervisor_workflow(
            supervisor, workers, execution_mode)
    
    def create_hierarchical_workflow(self, top_supervisor: Agent, 
                                   supervisor_teams: List[SupervisorTeam],
                                   execution_mode: str = "sequential") -> StateGraph:
        """
        Create a hierarchical workflow with a top supervisor coordinating multiple
        supervisor teams (sub-supervisors with their workers).
        
        Args:
            top_supervisor (Agent): Top-level supervisor that coordinates sub-supervisors
            supervisor_teams (List[SupervisorTeam]): List of supervisor teams, each containing
                                                     a supervisor and their workers
            execution_mode (str): "sequential" execution (default and only supported mode)
            
        Returns:
            StateGraph: Compiled hierarchical workflow graph
        """
        return HierarchicalPattern.create_hierarchical_workflow(
            top_supervisor, supervisor_teams, execution_mode)

