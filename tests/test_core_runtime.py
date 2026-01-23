# Headless/runtime tests for streamlit-langgraph core.

import unittest
from dataclasses import dataclass
from typing import List

from langgraph.graph import StateGraph

from streamlit_langgraph.core.executor import WorkflowExecutor
from streamlit_langgraph.core.executor.registry import ExecutorRegistry
from streamlit_langgraph.core.runtime import RuntimeHooks
from streamlit_langgraph.core.state import WorkflowState
from streamlit_langgraph.workflow import WorkflowBuilder


@dataclass
class FakeAgent:
    name: str
    role: str
    instructions: str
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    reasoning_effort: str = "low"
    allow_file_search: bool = False
    allow_code_interpreter: bool = False
    container_id: str = None
    allow_web_search: bool = False
    allow_image_generation: bool = False
    tools: List[str] = None
    mcp_servers: dict = None
    context: str = "least"
    human_in_loop: bool = False
    interrupt_on: dict = None
    conversation_history_mode: str = "filtered"

    def get_tools(self):
        return []


class RuntimeHeadlessTests(unittest.TestCase):
    def test_runtime_hooks_attached_to_workflow(self):
        runtime = RuntimeHooks(executor_registry=ExecutorRegistry())
        supervisor = FakeAgent(name="supervisor", role="Supervisor", instructions="Coordinate tasks.")
        workers = [
            FakeAgent(name="worker_a", role="Worker", instructions="Do task A."),
            FakeAgent(name="worker_b", role="Worker", instructions="Do task B."),
        ]

        workflow = WorkflowBuilder().create_supervisor_workflow(
            supervisor=supervisor,
            workers=workers,
            runtime=runtime,
        )

        self.assertIs(getattr(workflow, "runtime_hooks", None), runtime)

    def test_workflow_executor_runs_headless(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("noop", lambda state: state)
        graph.add_edge("__start__", "noop")
        graph.add_edge("noop", "__end__")
        workflow = graph.compile()

        executor = WorkflowExecutor(ExecutorRegistry())
        initial_state: WorkflowState = {
            "messages": [],
            "current_agent": None,
            "agent_outputs": {},
            "files": [],
            "metadata": {},
        }

        result = executor.execute_workflow(workflow, initial_state=initial_state)
        self.assertIn("messages", result)


if __name__ == "__main__":
    unittest.main()
