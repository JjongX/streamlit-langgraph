from typing import List
from langgraph.graph import StateGraph

from ..agent import Agent
from .executor import SequentialExecution, ParallelExecution
from .patterns import SupervisorPattern

class WorkflowBuilder:
    """
    Simplified workflow builder that delegates to specialized pattern classes.
    
    This replaces the monolithic WorkflowBuilder with a cleaner interface
    that delegates to pattern-specific implementations.
    """
    
    def __init__(self):
        """Initialize the workflow builder."""
        pass
    
    def create_sequential_workflow(self, agents: List[Agent]) -> StateGraph:
        """
        Create a sequential workflow where agents execute one after another.
        
        Args:
            agents (List[Agent]): List of agents to execute sequentially
            
        Returns:
            StateGraph: Compiled workflow graph
        """
        return SequentialExecution.create_sequential_workflow(agents)
    
    def create_parallel_workflow(self, agents: List[Agent], 
                               aggregation_strategy: str = "concatenate") -> StateGraph:
        """
        Create a parallel workflow where agents execute simultaneously.
        
        Args:
            agents (List[Agent]): List of agents to execute in parallel
            aggregation_strategy (str): How to combine results ("concatenate", "summarize", "vote")
            
        Returns:
            StateGraph: Compiled workflow graph
        """
        return ParallelExecution.create_parallel_workflow(agents, aggregation_strategy)
    
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
    
    # Convenience methods for supervisor workflows
    def create_supervisor_with_sequential_workers(self, supervisor: Agent, workers: List[Agent], 
                                                max_iterations: int = 5) -> StateGraph:
        """Create a supervisor workflow with sequential worker execution."""
        return self.create_supervisor_workflow(supervisor, workers, "sequential", max_iterations)
    
    def create_supervisor_with_parallel_workers(self, supervisor: Agent, workers: List[Agent], 
                                              max_iterations: int = 5) -> StateGraph:
        """Create a supervisor workflow with parallel worker execution."""
        return self.create_supervisor_workflow(supervisor, workers, "parallel", max_iterations)
