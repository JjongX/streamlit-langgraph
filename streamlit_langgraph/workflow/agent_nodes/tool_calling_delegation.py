# Tool calling delegation pattern implementation for agent nodes.

import json
from typing import Any, Dict, List, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ...agent import Agent, get_llm_client
from ...core.executor.registry import ExecutorRegistry
from ...core.state.state_schema import WorkflowState
from ...utils.text_extraction import extract_text_from_content
from ..prompts import ToolCallingPromptBuilder
from .factory import AgentNodeBase


class ToolCallingDelegation:
    """Tool calling delegation pattern for supervisor-worker workflows."""

    MAX_TOOL_CALL_ITERATIONS = 10

    def __init__(self, agent_executor: AgentNodeBase):
        self.agent_executor = agent_executor

    def execute_agent_with_tools(
        self,
        agent: Agent,
        state: WorkflowState,
        input_message: str,
        tools: List[StructuredTool],
        tool_agents_map: Dict[str, Agent],
    ) -> str:
        """Execute an agent with access to tools (other agents wrapped as tools)."""
        client = self._get_tool_calling_client(agent)
        if not hasattr(client, "bind_tools"):
            raise RuntimeError(
                "Tool calling delegation requires a LangChain chat model with "
                "'bind_tools'."
            )

        bound_client = client.bind_tools(tools) if tools else client
        if not hasattr(bound_client, "invoke"):
            raise RuntimeError(
                "Tool calling delegation requires a bound chat model with 'invoke'."
            )
        messages = self._build_initial_messages(agent, input_message)

        for _ in range(self.MAX_TOOL_CALL_ITERATIONS):
            response = bound_client.invoke(messages)
            if not isinstance(response, AIMessage):
                raise RuntimeError(
                    f"Tool-calling supervisor '{agent.name}' returned unsupported response "
                    f"type '{type(response).__name__}'"
                )

            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return extract_text_from_content(response.content)

            for tool_call in tool_calls:
                tool_name, tool_args, tool_call_id = self._parse_tool_call(tool_call)
                tool_result = self._execute_tool_call(tool_name, tool_args, tool_agents_map, state)
                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

        raise RuntimeError(
            f"Tool-calling delegation exceeded {self.MAX_TOOL_CALL_ITERATIONS} iterations "
            f"for supervisor '{agent.name}'."
        )

    @staticmethod
    def _get_tool_calling_client(agent: Agent) -> Any:
        """Get a LangChain chat model for tool-calling delegation."""
        client = get_llm_client(agent)
        if hasattr(client, "bind_tools"):
            return client

        provider = agent.provider.lower()
        model_provider = "google_genai" if provider == "google" else provider
        init_kwargs = {
            "model": agent.model,
            "temperature": agent.temperature,
            "model_provider": model_provider,
        }
        if provider == "openai" and agent.reasoning_effort:
            init_kwargs["reasoning"] = {"effort": agent.reasoning_effort, "summary": "auto"}
        try:
            chat_model = init_chat_model(**init_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize fallback LangChain chat model for tool-calling delegation"
            ) from exc

        if provider == "google" and ExecutorRegistry.has_gemini_native_tools(agent):
            tools = []
            if agent.allow_web_search:
                tools.append({"google_search": {}})
            if agent.allow_code_interpreter:
                tools.append({"code_execution": {}})
            if tools:
                chat_model = chat_model.bind_tools(tools)
        return chat_model

    @staticmethod
    def create_agent_tools(tool_agents: List[Agent]) -> List[StructuredTool]:
        """Create LangChain tool definitions for each worker agent."""
        tools: List[StructuredTool] = []
        for worker in tool_agents:
            tools.append(
                StructuredTool.from_function(
                    func=ToolCallingDelegation._build_noop_worker_delegate(worker.name),
                    name=worker.name,
                    description=f"{worker.role}. {worker.instructions}",
                )
            )
        return tools

    @staticmethod
    def _build_noop_worker_delegate(worker_name: str):
        """Build a no-op function used only for tool schema binding."""

        def _delegate(task: str) -> str:
            return f"Delegated to {worker_name}: {task}"

        return _delegate

    @staticmethod
    def _build_initial_messages(agent: Agent, input_message: str) -> List[Any]:
        """Build initial system/user message list for tool-calling supervisor."""
        messages: List[Any] = []
        system_message = f"You are a {agent.role}. {agent.instructions}".strip()
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=input_message))
        return messages

    @staticmethod
    def _parse_tool_call(tool_call: Any) -> Tuple[str, Dict[str, Any], str]:
        """Parse a LangChain tool call into name/args/id."""
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_args = getattr(tool_call, "args", {})
            tool_call_id = getattr(tool_call, "id", None)

        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        if not isinstance(tool_args, dict):
            raise RuntimeError(
                f"Tool call arguments for '{tool_name}' must be a JSON object"
            )
        if not tool_name:
            raise RuntimeError("Tool call is missing required 'name'")
        if not tool_call_id:
            tool_call_id = f"call_{tool_name}"
        return tool_name, tool_args, tool_call_id

    def _execute_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_agents_map: Dict[str, Agent],
        state: WorkflowState,
    ) -> str:
        """Execute a tool call by invoking the corresponding worker agent."""
        tool_agent = tool_agents_map.get(tool_name)
        if not tool_agent:
            raise RuntimeError(f"Delegated worker '{tool_name}' was not found")

        tool_instructions = ToolCallingPromptBuilder.get_worker_tool_instructions(
            role=tool_agent.role,
            instructions=tool_agent.instructions,
            task=tool_args.get("task", ""),
        )
        response = self.agent_executor.execute_agent(
            tool_agent,
            state,
            tool_instructions,
            allow_stream=False,
        )
        return response.get("content", "")
