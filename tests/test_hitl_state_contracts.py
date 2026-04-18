import json
import unittest

try:
    from streamlit_langgraph.core.middleware.hitl import HITLUtils
    from streamlit_langgraph.core.state.state_schema import WorkflowStateManager
    from streamlit_langgraph.workflow.agent_nodes.factory import AgentNodeBase
except Exception as exc:
    raise unittest.SkipTest(f"Core contract tests unavailable in this environment: {exc}")


class StreamContractsTests(unittest.TestCase):
    def test_parse_edit_input_valid_json(self):
        parsed = HITLUtils.parse_edit_input('{"x": 1}', default_input={})
        self.assertEqual(parsed, {"x": 1})

    def test_parse_edit_input_invalid_json_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            HITLUtils.parse_edit_input('{invalid', default_input={})

    def test_ensure_stream_flag_sets_default(self):
        state = {"metadata": {}}
        value = WorkflowStateManager.ensure_stream_flag(state, default_stream=True)
        self.assertTrue(value)
        self.assertTrue(state["metadata"]["stream"])

    def test_require_stream_flag_missing_raises(self):
        state = {"metadata": {}}
        with self.assertRaises(KeyError):
            WorkflowStateManager.require_stream_flag(state)

    def test_should_stream_raises_when_stream_flag_missing(self):
        class FakeAgent:
            human_in_loop = False

        state = {"metadata": {}, "messages": []}
        with self.assertRaises(KeyError):
            AgentNodeBase._should_stream(FakeAgent(), state, allow_stream=True)


if __name__ == "__main__":
    unittest.main()
