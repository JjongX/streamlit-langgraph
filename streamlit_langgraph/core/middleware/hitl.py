# Human-in-the-Loop (HITL) middleware for interrupt management and handling.

import json
from ..state import WorkflowStateManager
from .interrupts import InterruptManager


class HITLUtils:
    """Utility functions for HITL approval/decision processing and data transformation."""
    
    @staticmethod
    def has_pending_interrupts(workflow_state):
        """Check if there are any pending interrupts in the workflow state."""
        if not workflow_state:
            return False
        return InterruptManager.has_pending_interrupts(workflow_state)
        
    @staticmethod
    def extract_action_requests_from_interrupt(interrupt_raw):
        """Extract action_requests from Interrupt objects."""
        if not interrupt_raw:
            return []
        
        if isinstance(interrupt_raw, list):
            result = []
            for item in interrupt_raw:
                if hasattr(item, 'value'):
                    action_requests = item.value.get('action_requests', [item.value]) if isinstance(item.value, dict) else [item.value]
                    result.extend(action_requests)
                elif isinstance(item, dict):
                    result.extend(item.get('action_requests', [item]))
            return result
        elif isinstance(interrupt_raw, dict):
            return interrupt_raw.get('action_requests', [interrupt_raw])
        
        return []
    
    @staticmethod
    def check_edit_allowed(agent_interrupt_on, tool_name):
        """Check if editing is allowed for a tool based on agent's interrupt_on configuration."""
        if not agent_interrupt_on:
            return True
        
        tool_config = agent_interrupt_on.get(tool_name, {})
        if isinstance(tool_config, dict):
            allowed_decisions = tool_config.get("allowed_decisions", ["approve", "reject", "edit"])
            return "edit" in allowed_decisions
        
        return True
    
    @staticmethod
    def parse_edit_input(edit_text, default_input):
        """Parse user edit input, attempting JSON parsing if applicable."""
        if not edit_text.strip():
            return default_input, None

        parsed = json.loads(edit_text)
        return parsed, None
    
    @staticmethod
    def get_valid_interrupts(workflow_state):
        """Extract and filter valid interrupts from workflow state."""
        pending_interrupts = WorkflowStateManager.get_pending_interrupts(workflow_state)
        return {
            key: value for key, value in pending_interrupts.items()
            if isinstance(value, dict) and value.get("__interrupt__")
        }
    
    @staticmethod
    def clear_interrupt_and_decisions(workflow_state, executor_key):
        """Clear interrupt and decisions from workflow state."""
        if "pending_interrupts" in workflow_state.get("metadata", {}):
            workflow_state["metadata"]["pending_interrupts"].pop(executor_key, None)
        if "hitl_decisions" in workflow_state.get("metadata", {}):
            workflow_state["metadata"]["hitl_decisions"].pop(f"{executor_key}_decisions", None)
