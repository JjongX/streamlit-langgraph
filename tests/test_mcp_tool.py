import asyncio
import unittest
from unittest.mock import AsyncMock

try:
    from streamlit_langgraph.utils.mcp_tool import MCPToolManager
except Exception as exc:
    raise unittest.SkipTest(f"MCP tool tests unavailable in this environment: {exc}")


class FakeAsyncTool:
    name = "test_tool"
    description = "test"
    args_schema = None

    async def ainvoke(self, kwargs):
        return kwargs.get("value", 0) + 1


class MCPToolTests(unittest.TestCase):
    def test_mcp_get_tools_raises_when_loop_running(self):
        manager = MCPToolManager()
        manager._server_configs = {"math": {"transport": "stdio"}}

        async def run_check():
            with self.assertRaises(RuntimeError):
                manager.get_tools()

        asyncio.run(run_check())

    def test_mcp_get_tools_runs_without_running_loop(self):
        manager = MCPToolManager()
        manager._server_configs = {"math": {"transport": "stdio"}}
        manager.get_tools_async = AsyncMock(return_value=[FakeAsyncTool()])

        tools = manager.get_tools()
        self.assertEqual(len(tools), 1)
        result = tools[0].func(value=4)
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
