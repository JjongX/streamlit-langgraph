# Routing helper utilities for workflow patterns.

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ...core.state import WorkflowState


@dataclass
class RoutingDecision:
    """Type-safe representation of a routing decision."""
    action: str  # "delegate" or "finish"
    target_worker: Optional[str] = None
    task_description: Optional[str] = None
    priority: Optional[str] = None
    
    @classmethod
    def from_dict(cls, decision_dict: Dict[str, Any]) -> "RoutingDecision":
        """Create RoutingDecision from dictionary."""
        return cls(
            action=decision_dict.get("action", "finish"),
            target_worker=decision_dict.get("target_worker"),
            task_description=decision_dict.get("task_description"),
            priority=decision_dict.get("priority")
        )


class RoutingHelper:
    """Helper class for routing logic in workflow patterns."""
    
    @staticmethod
    def extract_routing_decision(state: WorkflowState) -> RoutingDecision:
        """Extract routing decision from workflow state."""
        routing_dict = state.get("metadata", {}).get("routing_decision", {})
        return RoutingDecision.from_dict(routing_dict)
    
    @staticmethod
    def create_sequential_route(worker_names: List[str], end_node: str = "__end__") -> callable:
        """Create a sequential routing function for supervisor workflows."""
        def route(state: WorkflowState) -> str:
            """Route based on supervisor's structured routing decision."""
            if state.get("metadata", {}).get("pending_interrupts", {}):
                return end_node
            
            decision = RoutingHelper.extract_routing_decision(state)
            if decision.action == "delegate" and decision.target_worker:
                if decision.target_worker in worker_names:
                    return decision.target_worker
            
            return end_node
        
        return route
    
    @staticmethod
    def create_parallel_route(fanout_node: str = "parallel_fanout", end_node: str = "__end__") -> callable:
        """Create a parallel routing function for supervisor workflows."""
        def route(state: WorkflowState) -> str:
            """Route from supervisor to parallel execution or end."""
            decision = RoutingHelper.extract_routing_decision(state)
            if decision.action == "delegate" and decision.target_worker == "PARALLEL":
                return fanout_node
            
            return end_node
        
        return route
    
    @staticmethod
    def create_hierarchical_top_route(sub_supervisor_names: List[str], end_node: str = "__end__") -> callable:
        """
        Create routing function for top supervisor in hierarchical workflows.
        """
        def route(state: WorkflowState) -> str:
            """Top supervisor routes to sub-supervisors or END."""
            decision = RoutingHelper.extract_routing_decision(state)
            if decision.action == "delegate" and decision.target_worker:
                if decision.target_worker in sub_supervisor_names:
                    return decision.target_worker
            
            return end_node
        
        return route
    
    @staticmethod
    def create_hierarchical_sub_route(
        worker_names: List[str], 
        top_supervisor_name: str
    ) -> callable:
        """Create routing function for sub-supervisor in hierarchical workflows."""
        def route(state: WorkflowState) -> str:
            """Sub-supervisor routes to workers or back to top supervisor."""
            decision = RoutingHelper.extract_routing_decision(state)
            if decision.action == "delegate" and decision.target_worker:
                if decision.target_worker in worker_names:
                    return decision.target_worker
            
            # When done, return to top supervisor
            return top_supervisor_name
        
        return route

