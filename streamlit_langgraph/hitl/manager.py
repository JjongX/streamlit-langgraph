# Centralized interrupt management for HITL workflows.

from typing import Any, Dict, Optional


class InterruptManager:
    """Centralized interrupt management for HITL workflows."""
    
    @staticmethod
    def should_interrupt(result: Dict[str, Any]) -> bool:
        """Check if execution result contains an interrupt."""
        return "__interrupt__" in result and result["__interrupt__"] is not None
    
    @staticmethod
    def extract_interrupt_data(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract interrupt data from execution result."""
        if not InterruptManager.should_interrupt(result):
            return None
        
        return {
            "__interrupt__": result.get("__interrupt__"),
            "thread_id": result.get("thread_id"),
            "config": result.get("config"),
            "agent": result.get("agent"),
        }
    
    @staticmethod
    def store_interrupt(state: Dict[str, Any], agent_name: str, 
                       interrupt_data: Dict[str, Any], executor_key: str) -> Dict[str, Any]:
        """Store interrupt in workflow state metadata."""
        if "metadata" not in state:
            state["metadata"] = {}
        
        if "pending_interrupts" not in state["metadata"]:
            state["metadata"]["pending_interrupts"] = {}
        
        state["metadata"]["pending_interrupts"][executor_key] = {
            "agent": agent_name,
            "__interrupt__": interrupt_data.get("__interrupt__"),
            "thread_id": interrupt_data.get("thread_id"),
            "config": interrupt_data.get("config"),
            "executor_key": executor_key
        }
        
        return {"metadata": state["metadata"]}
    
    @staticmethod
    def has_pending_interrupts(state: Dict[str, Any]) -> bool:
        """Check if workflow state has any pending interrupts."""
        if not state or "metadata" not in state:
            return False
        
        pending_interrupts = state["metadata"].get("pending_interrupts", {})
        for value in pending_interrupts.values():
            if isinstance(value, dict) and value.get("__interrupt__"):
                return True
        return False

