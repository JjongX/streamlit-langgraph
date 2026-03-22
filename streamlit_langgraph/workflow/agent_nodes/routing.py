# Routing helper utilities for workflow patterns.

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ...core.state import WorkflowState


@dataclass
class RoutingDecision:
    """Routing decision extracted from agent output."""
    action: str
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
    """Routing functions for workflow patterns."""

    @staticmethod
    def _normalize_target_worker_name(
        target_worker: Optional[str],
        worker_name_lookup: Dict[str, str],
    ) -> Optional[str]:
        """Normalize target worker name (trim + case-insensitive match)."""
        if not target_worker:
            return None
        normalized_key = str(target_worker).strip().lower()
        return worker_name_lookup.get(normalized_key)
    
    @staticmethod
    def extract_routing_decision(state: WorkflowState) -> RoutingDecision:
        """Extract routing decision from workflow state metadata."""
        routing_dict = state.get("metadata", {}).get("routing_decision", {})
        return RoutingDecision.from_dict(routing_dict)
    
    @staticmethod
    def create_sequential_route(worker_names: List[str], end_node: str = "__end__") -> callable:
        """Route to worker or end based on delegation decision."""
        worker_name_lookup = {name.lower(): name for name in worker_names}

        def route(state: WorkflowState) -> str:
            if state.get("metadata", {}).get("pending_interrupts", {}):
                return end_node
            
            decision = RoutingHelper.extract_routing_decision(state)
            target_worker = RoutingHelper._normalize_target_worker_name(
                decision.target_worker,
                worker_name_lookup,
            )
            if decision.action == "delegate" and target_worker:
                return target_worker
            return end_node
        return route
    
    @staticmethod
    def create_parallel_route(
        worker_names: Optional[List[str]] = None,
        fanout_node: str = "parallel_fanout",
        end_node: str = "__end__",
    ) -> callable:
        """Route to parallel fanout or end."""
        worker_name_lookup = {name.lower(): name for name in (worker_names or [])}

        def route(state: WorkflowState) -> str:
            decision = RoutingHelper.extract_routing_decision(state)
            target_worker = str(decision.target_worker).strip() if decision.target_worker else ""
            target_worker_upper = target_worker.upper()
            if decision.action == "delegate" and (
                target_worker_upper == "PARALLEL" or
                target_worker.lower() in worker_name_lookup
            ):
                return fanout_node
            return end_node
        return route
    
    @staticmethod
    def create_hierarchical_top_route(sub_supervisor_names: List[str], end_node: str = "__end__") -> callable:
        """Route top supervisor to sub-supervisor or end."""
        worker_name_lookup = {name.lower(): name for name in sub_supervisor_names}

        def route(state: WorkflowState) -> str:
            decision = RoutingHelper.extract_routing_decision(state)
            target_worker = RoutingHelper._normalize_target_worker_name(
                decision.target_worker,
                worker_name_lookup,
            )
            if decision.action == "delegate" and target_worker:
                return target_worker
            return end_node
        return route
    
    @staticmethod
    def create_hierarchical_sub_route(worker_names: List[str], top_supervisor_name: str) -> callable:
        """Route sub-supervisor to worker or back to top supervisor."""
        worker_name_lookup = {name.lower(): name for name in worker_names}

        def route(state: WorkflowState) -> str:
            decision = RoutingHelper.extract_routing_decision(state)
            target_worker = RoutingHelper._normalize_target_worker_name(
                decision.target_worker,
                worker_name_lookup,
            )
            if decision.action == "delegate" and target_worker:
                return target_worker
            return top_supervisor_name
        return route
    
    @staticmethod
    def create_network_route(peer_names: List[str], end_node: str = "__end__") -> callable:
        """Route network agent to peer or end."""
        worker_name_lookup = {name.lower(): name for name in peer_names}

        def route(state: WorkflowState) -> str:
            if state.get("metadata", {}).get("pending_interrupts", {}):
                return end_node
            
            decision = RoutingHelper.extract_routing_decision(state)
            target_worker = RoutingHelper._normalize_target_worker_name(
                decision.target_worker,
                worker_name_lookup,
            )
            if decision.action == "delegate" and target_worker:
                return target_worker
            return end_node
        return route
