import unittest
from types import SimpleNamespace

IMPORT_ERROR = None

try:
    from streamlit_langgraph.workflow.agent_nodes.handoff_delegation import HandoffDelegation
    from streamlit_langgraph.workflow.agent_nodes.routing import RoutingHelper
except Exception as exc:
    IMPORT_ERROR = exc


if IMPORT_ERROR is not None:
    @unittest.skip(f"Parallel routing tests unavailable in this environment: {IMPORT_ERROR}")
    class ParallelRoutingBehaviorTests(unittest.TestCase):
        def test_skipped(self):
            self.fail("This test should be skipped when imports are unavailable.")
else:
    class ParallelRoutingBehaviorTests(unittest.TestCase):
        def setUp(self):
            self.workers = [
                SimpleNamespace(name="Market_Analyst"),
                SimpleNamespace(name="Technical_Analyst"),
            ]

        def test_response_api_routing_accepts_target_worker_alias(self):
            output = {
                "output": [
                    {
                        "type": "function_call",
                        "name": "delegate_task",
                        "arguments": (
                            '{"target_worker":"PARALLEL","task_description":"Analyze all angles","priority":"high"}'
                        ),
                    }
                ]
            }

            _, routing = HandoffDelegation._extract_response_api_routing_decision(output, "prompt")

            self.assertEqual(routing["action"], "delegate")
            self.assertEqual(routing["target_worker"], "PARALLEL")
            self.assertEqual(routing["task_description"], "Analyze all angles")

        def test_normalization_coerces_single_worker_to_parallel_in_parallel_mode(self):
            decision = {
                "action": "delegate",
                "target_worker": " market_analyst ",
                "task_description": "Market-only view",
            }

            normalized = HandoffDelegation._normalize_routing_decision(
                decision,
                self.workers,
                allow_parallel=True,
            )

            self.assertEqual(normalized["action"], "delegate")
            self.assertEqual(normalized["target_worker"], "PARALLEL")

        def test_extract_state_context_omits_status_messages(self):
            state = {
                "messages": [
                    {
                        "id": "m-status",
                        "role": "assistant",
                        "content": "[status] Started",
                        "agent": "Market_Analyst",
                        "is_status_event": True,
                    },
                    {
                        "id": "m-user",
                        "role": "user",
                        "content": "Analyze iPhone",
                        "agent": None,
                    },
                ],
                "metadata": {},
            }

            messages, _, _ = HandoffDelegation._extract_state_context(state)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["id"], "m-user")

        def test_parallel_route_accepts_parallel_or_worker_name(self):
            route = RoutingHelper.create_parallel_route(worker_names=[w.name for w in self.workers])

            via_parallel = route(
                {"metadata": {"routing_decision": {"action": "delegate", "target_worker": "PARALLEL"}}}
            )
            via_worker = route(
                {"metadata": {"routing_decision": {"action": "delegate", "target_worker": " market_analyst "}}}
            )
            finish = route({"metadata": {"routing_decision": {"action": "finish"}}})

            self.assertEqual(via_parallel, "parallel_fanout")
            self.assertEqual(via_worker, "parallel_fanout")
            self.assertEqual(finish, "__end__")


if __name__ == "__main__":
    unittest.main()
