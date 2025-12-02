# Tool calling delegation pattern implementation for agent nodes.

import json
from typing import Any, Dict, List

import streamlit as st

from ...agent import Agent, AgentManager
from .factory import AgentNodeBase
from ...core.state.state_schema import WorkflowState
from ..prompts import ToolCallingPromptBuilder


class ToolCallingDelegation:
    """Tool calling delegation pattern for supervisor-worker workflows."""

    @staticmethod
    def execute_agent_with_tools(agent: Agent, state: WorkflowState, 
                                  input_message: str, tools: List[Dict[str, Any]],
                                  tool_agents_map: Dict[str, Agent]) -> str:
        """Execute an agent with access to tools (other agents wrapped as tools)."""
        if agent.provider.lower() != "openai":
            raise ValueError(
                f"Tool calling delegation requires OpenAI provider. "
                f"Agent '{agent.name}' uses provider '{agent.provider}'."
            )

        client = AgentManager.get_llm_client(agent)
        messages = []
        if agent.system_message:
            messages.append({"role": "system", "content": agent.system_message})
        messages.append({"role": "user", "content": input_message})
        
        for iteration in range(10):
            with st.spinner(f"🤖 {agent.name} is working..."):
                response = client.chat.completions.create(
                    model=agent.model, messages=messages, temperature=agent.temperature,
                    tools=tools if tools else None, tool_choice="auto" if tools else None
                )
            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                return message.content or ""
            
            for tool_call in message.tool_calls:
                tool_result = ToolCallingDelegation._execute_tool_call(
                    tool_call, tool_agents_map, state
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result
                })
            iteration += 1
        
        return message.content or "Maximum iterations reached"
    
    @staticmethod
    def create_agent_tools(tool_agents: List[Agent]) -> List[Dict[str, Any]]:
        """Create OpenAI function tool definitions for each agent."""
        return [{
            "type": "function",
            "function": {
                "name": agent.name,
                "description": f"{agent.role}. {agent.instructions}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": f"Clear description of the task for {agent.name} to perform. Be specific about what you need."}
                    },
                    "required": ["task"]
                }
            }
        } for agent in tool_agents]
    
    @staticmethod
    def _execute_tool_call(tool_call, tool_agents_map: Dict[str, Agent], state: WorkflowState) -> str:
        """Execute a tool call by invoking the corresponding agent."""
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        tool_agent = tool_agents_map.get(tool_name)
        if not tool_agent:
            return f"Error: Agent {tool_name} not found"
        tool_instructions = ToolCallingPromptBuilder.get_worker_tool_instructions(
            role=tool_agent.role,
            instructions=tool_agent.instructions,
            task=args.get("task", "")
        )
        return AgentNodeBase.execute_agent(tool_agent, state, tool_instructions)
