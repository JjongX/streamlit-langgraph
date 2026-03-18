import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from langchain_core.messages import AIMessage
    from streamlit_langgraph.workflow.agent_nodes.tool_calling_delegation import ToolCallingDelegation
except Exception as exc:
    raise unittest.SkipTest(f"Tool-calling delegation tests unavailable in this environment: {exc}")


class FakeAgentExecutor:
    def __init__(self):
        self.calls = []

    def execute_agent(self, agent, state, input_message, allow_stream=True):
        self.calls.append((agent.name, input_message, allow_stream))
        return {"content": f"{agent.name} completed"}


class FakeBoundClient:
    def __init__(self, responses):
        self._responses = iter(responses)

    def invoke(self, messages):
        return next(self._responses)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses

    def bind_tools(self, tools):
        return FakeBoundClient(self.responses)


class ToolCallingDelegationTests(unittest.TestCase):
    def setUp(self):
        self.supervisor = SimpleNamespace(
            name="Supervisor",
            role="Supervisor",
            instructions="Coordinate work",
            provider="openai",
            model="gpt-4.1-mini",
            temperature=0.0,
            reasoning_effort=None,
            allow_web_search=False,
            allow_code_interpreter=False,
        )
        self.worker = SimpleNamespace(
            name="Researcher",
            role="Researcher",
            instructions="Research topics",
        )

    def test_execute_agent_with_tools_runs_tool_loop(self):
        first = AIMessage(
            content="",
            tool_calls=[
                {"name": "Researcher", "args": {"task": "Find facts"}, "id": "call_1"}
            ],
        )
        second = AIMessage(content="Final supervisor answer", tool_calls=[])
        fake_client = FakeClient([first, second])
        fake_executor = FakeAgentExecutor()

        delegation = ToolCallingDelegation(fake_executor)
        tools = ToolCallingDelegation.create_agent_tools([self.worker])

        with patch(
            "streamlit_langgraph.workflow.agent_nodes.tool_calling_delegation.get_llm_client",
            return_value=fake_client,
        ):
            result = delegation.execute_agent_with_tools(
                agent=self.supervisor,
                state={"messages": [], "metadata": {}},
                input_message="Do the work",
                tools=tools,
                tool_agents_map={"Researcher": self.worker},
            )

        self.assertEqual(result, "Final supervisor answer")
        self.assertEqual(len(fake_executor.calls), 1)
        self.assertEqual(fake_executor.calls[0][0], "Researcher")
        self.assertFalse(fake_executor.calls[0][2])

    def test_execute_agent_with_tools_fails_for_incompatible_client(self):
        fake_executor = FakeAgentExecutor()
        delegation = ToolCallingDelegation(fake_executor)

        with patch(
            "streamlit_langgraph.workflow.agent_nodes.tool_calling_delegation.get_llm_client",
            return_value=object(),
        ):
            with self.assertRaises(RuntimeError):
                delegation.execute_agent_with_tools(
                    agent=self.supervisor,
                    state={"messages": [], "metadata": {}},
                    input_message="Do the work",
                    tools=[],
                    tool_agents_map={},
                )


if __name__ == "__main__":
    unittest.main()
