# HITL utility functions for data transformation and processing.

import json
from typing import Any, Dict, List, Optional

from ..state import WorkflowStateManager


class HITLUtils:
    """
    Static utility operations for Human-in-the-Loop (HITL) functionality.
    """
    
    @staticmethod
    def extract_action_requests_from_interrupt(interrupt_raw: Any) -> List[Dict[str, Any]]:
        """
        Extract action_requests from Interrupt objects.
        
        Args:
            interrupt_raw: Can be a list of Interrupt objects, a dict, or other formats
            
        Returns:
            List of action request dictionaries with keys like 'name', 'args', 'description', 'id'
        """
        if not interrupt_raw:
            return []
        
        if isinstance(interrupt_raw, list):
            return HITLUtils._extract_from_list(interrupt_raw)
        elif isinstance(interrupt_raw, dict):
            return HITLUtils._extract_from_dict(interrupt_raw)
        
        return []
    
    @staticmethod
    def _extract_from_list(interrupt_list: List[Any]) -> List[Dict[str, Any]]:
        """Extract action requests from a list of interrupt items."""
        result = []
        for item in interrupt_list:
            if hasattr(item, 'value'):
                result.extend(HITLUtils._extract_from_dict(item.value))
            elif isinstance(item, dict):
                result.extend(HITLUtils._extract_from_dict(item))
        return result
    
    @staticmethod
    def _extract_from_dict(interrupt_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract action requests from a dict."""
        if 'action_requests' in interrupt_dict:
            return interrupt_dict['action_requests']
        return [interrupt_dict]
    
    @staticmethod
    def format_decisions(decisions: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Format decisions for HumanInTheLoopMiddleware.
        
        The middleware expects decisions in format:
        [{"type": "approve|reject|edit", "edit": {...} if edit}]
        
        Args:
            decisions: List of decision dicts or None values
            
        Returns:
            List of formatted decision dictionaries
        """
        formatted_decisions = []
        for decision in decisions:
            if decision:
                formatted_decisions.append(decision)
            else:
                # If no decision, default to approve
                formatted_decisions.append({"type": "approve"})
        return formatted_decisions
    
    @staticmethod
    def check_edit_allowed(agent_interrupt_on: Optional[Dict[str, Any]], tool_name: str) -> bool:
        """
        Check if editing is allowed for a tool based on agent's interrupt_on configuration.
        
        Args:
            agent_interrupt_on: The agent's interrupt_on configuration dict
            tool_name: Name of the tool to check
            
        Returns:
            True if editing is allowed, False otherwise
        """
        if not agent_interrupt_on:
            return True  # Default to allowing edit if not configured
        
        tool_config = agent_interrupt_on.get(tool_name, {})
        if isinstance(tool_config, dict):
            allowed_decisions = tool_config.get("allowed_decisions", ["approve", "reject", "edit"])
            return "edit" in allowed_decisions
        
        return True  # Default to allowing edit
    
    @staticmethod
    def parse_edit_input(edit_text: str, default_input: Any) -> tuple:
        """
        Parse user edit input, attempting to parse as JSON if it looks like JSON.
        
        Args:
            edit_text: The text input from the user
            default_input: The default input value to use if parsing fails or text is empty
            
        Returns:
            Tuple of (parsed_input, error_message). error_message is None if successful.
        """
        if not edit_text.strip():
            return default_input, None
        
        # If it looks like JSON (starts with { or [), try to parse it
        if edit_text.strip().startswith('{') or edit_text.strip().startswith('['):
            try:
                parsed = json.loads(edit_text)
                return parsed, None
            except (json.JSONDecodeError, ValueError):
                return None, "Invalid JSON. Please fix the input format."
        
        # Try to parse as JSON anyway, but fallback to string if it fails
        try:
            parsed = json.loads(edit_text)
            return parsed, None
        except (json.JSONDecodeError, ValueError):
            return edit_text, None
    
    @staticmethod
    def extract_action_info(action: Any, action_index: int) -> tuple:
        """Extract tool name, input, and ID from action."""
        if isinstance(action, dict):
            tool_name = action.get("name", action.get("tool", "Unknown"))
            tool_input = action.get("args", action.get("input", {}))
            action_id = action.get("id", f"action_{action_index}")
        else:
            tool_name = str(action)
            tool_input = {}
            action_id = f"action_{action_index}"
        return tool_name, tool_input, action_id
    
    @staticmethod
    def find_pending_action(decisions: List[Optional[Dict[str, Any]]]) -> Optional[int]:
        """Find the first action that needs a decision."""
        for i, decision in enumerate(decisions):
            if decision is None:
                return i
        return None
    
    @staticmethod
    def has_pending_interrupts(workflow_state: Optional[Dict[str, Any]]) -> bool:
        """
        Check if there are any pending interrupts in the workflow state.
        
        Args:
            workflow_state: The workflow state dictionary (can be None)
            
        Returns:
            True if there are valid pending interrupts, False otherwise
        """
        if not workflow_state:
            return False
        
        pending_interrupts = WorkflowStateManager.get_pending_interrupts(workflow_state)
        
        # Check if any interrupt has valid __interrupt__ data
        for value in pending_interrupts.values():
            if isinstance(value, dict) and value.get("__interrupt__"):
                return True
        
        return False
    
    @staticmethod
    def get_valid_interrupts(workflow_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract and filter valid interrupts from workflow state."""
        pending_interrupts = WorkflowStateManager.get_pending_interrupts(workflow_state)
        return {
            key: value for key, value in pending_interrupts.items()
            if isinstance(value, dict) and value.get("__interrupt__")
        }
    
    @staticmethod
    def initialize_decisions(workflow_state: Dict[str, Any], executor_key: str, 
                            num_actions: int) -> List[Optional[Dict[str, Any]]]:
        """Initialize decisions list from workflow state or create new one."""
        decisions = WorkflowStateManager.get_hitl_decision(workflow_state, executor_key)
        if decisions is None or len(decisions) != num_actions:
            decisions = [None] * num_actions
        return decisions
    
    @staticmethod
    def clear_interrupt_and_decisions(workflow_state: Dict[str, Any], executor_key: str):
        """Clear interrupt and decisions from workflow state."""
        if "pending_interrupts" in workflow_state.get("metadata", {}):
            workflow_state["metadata"]["pending_interrupts"].pop(executor_key, None)
        HITLUtils.clear_decisions(workflow_state, executor_key)
    
    @staticmethod
    def clear_decisions(workflow_state: Dict[str, Any], executor_key: str):
        """Clear decisions for an executor."""
        if "hitl_decisions" in workflow_state.get("metadata", {}):
            decisions_key = f"{executor_key}_decisions"
            workflow_state["metadata"]["hitl_decisions"].pop(decisions_key, None)

