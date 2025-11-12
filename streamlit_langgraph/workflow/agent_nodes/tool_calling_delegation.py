# Tool calling delegation pattern implementation for agent nodes.

import json
from typing import Any, Dict, List

import streamlit as st

from ...agent import Agent, AgentManager
from ...state import WorkflowState
from .base import AgentNodeBase
from ..prompts import ToolCallingPromptBuilder


class ToolCallingDelegation:
    """Tool calling delegation pattern where agents are exposed as tools."""
    
    @staticmethod
    def _create_agent_tools(tool_agents: List[Agent], state: WorkflowState) -> List[Dict[str, Any]]:
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
    def _execute_agent_with_tools(agent: Agent, state: WorkflowState, 
                                  input_message: str, tools: List[Dict[str, Any]],
                                  tool_agents_map: Dict[str, Agent]) -> str:
        """Execute an agent with access to tools (other agents wrapped as tools)."""
        if agent.provider.lower() != "openai":
            return AgentNodeBase.execute_agent(agent, state, input_message, [], 0)

        client = AgentManager.get_llm_client(agent)
        enhanced_instructions = ToolCallingPromptBuilder.get_orchestrator_tool_instructions(
            role=agent.role,
            instructions=agent.instructions
        )
        messages = [{"role": "user", "content": f"{enhanced_instructions}\n\nUser request: {input_message}"}]
        
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
    def _execute_tool_call(tool_call, tool_agents_map: Dict[str, Agent], state: WorkflowState) -> str:
        """Execute a tool call by invoking the corresponding agent."""
        tool_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            tool_agent = tool_agents_map.get(tool_name)
            if not tool_agent:
                return f"Error: Agent {tool_name} not found"
            tool_instructions = ToolCallingPromptBuilder.get_worker_tool_instructions(
                role=tool_agent.role,
                instructions=tool_agent.instructions,
                task=args.get("task", "")
            )
            return AgentNodeBase.execute_agent(tool_agent, state, tool_instructions, [], 0)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

