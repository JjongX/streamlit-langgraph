# Streamlit Human-in-the-Loop (HITL) handler.

import json
from typing import Any, Dict, List, Optional

import streamlit as st

from ..agent import get_llm_client
from ..core.middleware import HITLUtils
from ..core.state import WorkflowStateManager


class HITLHandler:
    """
    Orchestrator for Human-in-the-Loop approval UI/UX and decision processing.

    Uses HITLUtils for data transformation utilities.
    """

    def __init__(self, agents, config, state_manager, display_manager, executor_registry):
        """
        Initialize interrupt handler with dependencies.

        Args:
            agents: Map agent names to Agent instances
            config: UIConfig instance for UI settings
            state_manager: Streamlit-backed state manager for state management
            display_manager: DisplayManager instance for rendering messages
            executor_registry: ExecutorRegistry instance for executor lookup
        """
        self.agents = agents
        self.config = config
        self.state_manager = state_manager
        self.display_manager = display_manager
        self.executor_registry = executor_registry

    # Public API
    def handle_pending_interrupts(self, workflow_state):
        """Display UI for pending human-in-the-loop interrupts and handle user decisions."""
        if not workflow_state:
            return False

        valid_interrupts = HITLUtils.get_valid_interrupts(workflow_state)
        if not valid_interrupts:
            return False

        st.markdown("---")
        st.markdown("### ⚠️ **Human Approval Required**")
        st.info("The workflow has paused and is waiting for your approval.")

        # Process the first valid interrupt
        for executor_key, interrupt_data in valid_interrupts.items():
            if self.process_interrupt(workflow_state, executor_key, interrupt_data):
                return True

        return False

    def process_interrupt(self, workflow_state, executor_key, interrupt_data):
        """Process a single interrupt - returns True if handled."""
        agent_name = interrupt_data.get("agent", "Unknown")
        interrupt_raw = interrupt_data.get("__interrupt__", [])
        original_config = interrupt_data.get("config", {})

        interrupt_info = HITLUtils.extract_action_requests_from_interrupt(interrupt_raw)
        if not interrupt_info:
            st.error("⚠️ Error: Could not extract action details from interrupt.")
            return False

        executor = self.executor_registry.get(executor_key)
        if executor is None:
            raise ValueError(f"⚠️ Error: Executor not found for {executor_key}. This indicates a state inconsistency.")

        # Get or initialize decisions
        decisions = WorkflowStateManager.get_hitl_decision(workflow_state, executor_key)
        if decisions is None or len(decisions) != len(interrupt_info):
            decisions = [None] * len(interrupt_info)
            decision_update = WorkflowStateManager.set_hitl_decision(workflow_state, executor_key, decisions)
            workflow_state["metadata"].update(decision_update["metadata"])

        # If all decisions made, resume; otherwise show UI for first pending action
        if all(d is not None for d in decisions):
            return self._resume_workflow(workflow_state, executor_key, executor, agent_name, decisions, original_config)

        # Show UI for first pending action (sequential)
        pending_idx = next(i for i, d in enumerate(decisions) if d is None)
        self._display_action_ui(
            executor_key,
            executor,
            agent_name,
            interrupt_info[pending_idx],
            pending_idx,
            len(interrupt_info),
            decisions,
            workflow_state,
        )
        return True

    # Private helpers
    def _resume_workflow(self, workflow_state, executor_key, executor, agent_name, decisions, original_config):
        """Resume workflow execution with all decisions made."""
        formatted_decisions = [d if d else {"type": "approve"} for d in decisions]

        if hasattr(executor, "agent_obj") and executor.agent_obj is None:
            llm_client = get_llm_client(executor.agent)
            executor.build_agent(llm_client)

        workflow_thread_id = workflow_state.get("metadata", {}).get("workflow_thread_id")
        if not workflow_thread_id:
            st.error("⚠️ Error: Could not find thread_id for resume.")
            return False

        resume_config = original_config.copy() if original_config else {}
        resume_config.setdefault("configurable", {})
        resume_config["configurable"]["thread_id"] = workflow_thread_id

        with st.spinner("Processing your decision..."):
            resume_response = executor.resume(formatted_decisions, config=resume_config)

        # Handle response
        if resume_response and resume_response.get("__interrupt__"):
            self.state_manager.set_pending_interrupt(agent_name, resume_response, executor_key)
            hitl_decisions = workflow_state.get("metadata", {}).get("hitl_decisions", {})
            hitl_decisions.pop(f"{executor_key}_decisions", None)
        elif resume_response and resume_response.get("content"):
            self.state_manager.add_assistant_message(resume_response["content"], agent_name)
            self.state_manager.clear_pending_interrupt(executor_key)
            HITLUtils.clear_interrupt_and_decisions(workflow_state, executor_key)
            if workflow_state.get("messages"):
                self.display_manager.render_workflow_message(workflow_state["messages"][-1])
        else:
            self.state_manager.clear_pending_interrupt(executor_key)
            HITLUtils.clear_interrupt_and_decisions(workflow_state, executor_key)

        st.session_state.workflow_state = workflow_state
        st.rerun()

    def _display_action_ui(
        self,
        executor_key: str,
        executor,
        agent_name: str,
        action: Dict[str, Any],
        action_idx: int,
        total_actions: int,
        decisions: List[Optional[Dict[str, Any]]],
        workflow_state: Dict[str, Any],
    ):
        """Display approval UI for a single action (sequential flow)."""
        tool_name = action.get("name", action.get("tool", "Unknown")) if isinstance(action, dict) else str(action)
        tool_input = action.get("args", action.get("input", {})) if isinstance(action, dict) else {}
        action_id = action.get("id", f"action_{action_idx}") if isinstance(action, dict) else f"action_{action_idx}"

        st.markdown(f"Agent `{agent_name}` is requesting approval to execute the following tool `{tool_name}`")
        if tool_input:
            st.json(tool_input)

        agent_interrupt_on = getattr(executor.agent, "interrupt_on", None)
        allow_edit = HITLUtils.check_edit_allowed(agent_interrupt_on, tool_name)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve", key=f"approve_{executor_key}_{action_id}"):
                self._save_decision(workflow_state, executor_key, decisions, action_idx, {"type": "approve"})
        with col2:
            if st.button("❌ Reject", key=f"reject_{executor_key}_{action_id}"):
                self._save_decision(workflow_state, executor_key, decisions, action_idx, {"type": "reject"})
        with col3:
            if allow_edit:
                edit_key = f"edit_{executor_key}_{action_id}"
                default_value = json.dumps(tool_input, indent=2) if tool_input else ""
                edit_text = st.text_area("Edit input (optional)", value=default_value, key=edit_key, height=80)
                if st.button("✏️ Approve with Edit", key=f"edit_btn_{executor_key}_{action_id}"):
                    parsed_input, error_msg = HITLUtils.parse_edit_input(edit_text, tool_input)
                    if error_msg:
                        st.error(error_msg)
                    else:
                        self._save_decision(
                            workflow_state,
                            executor_key,
                            decisions,
                            action_idx,
                            {"type": "edit", "input": parsed_input},
                        )

    def _save_decision(
        self,
        workflow_state: Dict[str, Any],
        executor_key: str,
        decisions: List[Optional[Dict[str, Any]]],
        action_index: int,
        decision: Dict[str, Any],
    ):
        """Save a decision and trigger rerun."""
        decisions[action_index] = decision
        decision_update = WorkflowStateManager.set_hitl_decision(workflow_state, executor_key, decisions)
        workflow_state["metadata"].update(decision_update["metadata"])
        st.rerun()
