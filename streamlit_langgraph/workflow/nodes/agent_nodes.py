from typing import Any, Callable, Dict, List

import openai
import streamlit as st
from langchain.chat_models import init_chat_model

from ...agent import Agent, ResponseAPIExecutor, CreateAgentExecutor
from ...prompts import (
    get_basic_agent_instructions,
    get_worker_agent_instructions,
    get_supervisor_action_guidance,
    get_supervisor_instructions,
    get_tool_calling_agent_instructions,
    get_tool_agent_instructions,
)
from ..state import WorkflowState

class AgentNodeFactory:
    """Simplified factory for creating core agent nodes."""
    
    @staticmethod
    def create_basic_agent_node(agent: Agent) -> Callable:
        """Create a basic agent node for sequential/parallel workflows."""
        def agent_node(state: WorkflowState) -> Dict[str, Any]:
            input_message = None
            context_messages = []
            
            for msg in state["messages"][-5:]:
                context_messages.append(f"{msg['role'].title()}: {msg['content'][:200]}")
            
            assistant_messages = [msg for msg in state["messages"] if msg["role"] == "assistant" and msg.get("agent")]
            agent_responses_count = len(assistant_messages)
            
            if agent_responses_count == 0:
                for msg in reversed(state["messages"]):
                    if msg["role"] == "user":
                        input_message = msg["content"]
                        break
            else:
                most_recent_agent_msg = assistant_messages[-1]
                input_message = most_recent_agent_msg["content"]
            
            state_updates = {"current_agent": agent.name}
            
            if input_message:
                response = AgentNodeFactory._execute_agent(agent, state, input_message, context_messages, agent_responses_count)
                # Return only the delta (new message), not the full state
                # Reducers will merge with existing state
                state_updates["messages"] = [{
                    "role": "assistant",
                    "content": response,
                    "agent": agent.name,
                    "timestamp": None
                }]
                
            return state_updates
        
        return agent_node
    
    @staticmethod
    def create_supervisor_agent_node(supervisor: Agent, workers: List[Agent], 
                                     allow_parallel: bool = False) -> Callable:
        """Create a supervisor agent node that delegates tasks to workers using structured routing."""
        def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
            worker_names = [w.name for w in workers]
            workers_used = set()
            
            for msg in state["messages"]:
                if msg["role"] == "assistant" and msg.get("agent") in worker_names:
                    workers_used.add(msg.get("agent"))
            
            worker_outputs = []
            for worker_name in worker_names:
                if worker_name in state["agent_outputs"]:
                    output_preview = state['agent_outputs'][worker_name][:800]
                    if len(state['agent_outputs'][worker_name]) > 800:
                        output_preview += "..."
                    worker_outputs.append(f"**{worker_name}**: {output_preview}")
            
            user_query = ""
            for msg in reversed(state["messages"]):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break
            
            worker_list = ", ".join([f"{w.name} ({w.role})" for w in workers])
            workers_used_list = ", ".join(workers_used) if workers_used else "None"
            unused_workers = [w.name for w in workers if w.name not in workers_used]
            
            action_guidance = get_supervisor_action_guidance(workers_used, unused_workers)
            supervisor_instructions = get_supervisor_instructions(
                role=supervisor.role,
                instructions=supervisor.instructions,
                user_query=user_query,
                worker_list=worker_list,
                workers_used_list=workers_used_list,
                worker_outputs=worker_outputs,
                action_guidance=action_guidance
            )

            response, routing_decision = AgentNodeFactory._execute_supervisor_with_routing(
                supervisor, state, supervisor_instructions, workers, workers_used, allow_parallel
            )
            
            # Return only the delta (new values), not the full state
            # Reducers will merge with existing state
            return {
                "current_agent": supervisor.name,
                "messages": [{
                    "role": "assistant",
                    "content": response,
                    "agent": supervisor.name,
                    "timestamp": None
                }],
                "agent_outputs": {supervisor.name: response},
                "metadata": {"routing_decision": routing_decision}
            }
        
        return supervisor_agent_node
    
    @staticmethod
    def create_worker_agent_node(worker: Agent, supervisor: Agent) -> Callable:
        """Create a worker agent node for supervisor workflows."""
        def worker_agent_node(state: WorkflowState) -> Dict[str, Any]:
            supervisor_output = state["agent_outputs"].get(supervisor.name, "")
            
            user_query = ""
            for msg in reversed(state["messages"]):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break
            
            worker_instructions = get_worker_agent_instructions(
                role=worker.role,
                instructions=worker.instructions,
                user_query=user_query,
                supervisor_output=supervisor_output
            )
        
            response = AgentNodeFactory._execute_agent(worker, state, worker_instructions, [], 0)
            
            # Return only the delta (new values), not the full state
            # Reducers will merge these with existing state
            return {
                "current_agent": worker.name,
                "messages": [{
                    "role": "assistant",
                    "content": response,
                    "agent": worker.name,
                    "timestamp": None
                }],
                "agent_outputs": {worker.name: response}
            }
        
        return worker_agent_node
    
    @staticmethod
    def _execute_agent(agent: Agent, state: WorkflowState, input_message: str, 
                      context_messages: List[str], agent_responses_count: int) -> str:
        """Execute an agent with the given input and return the response."""
        
        # Get basic agent instructions from prompts module
        enhanced_instructions = get_basic_agent_instructions(
            role=agent.role,
            instructions=agent.instructions,
            user_query=input_message,
            context_messages=context_messages
        )
        
        # Use appropriate executor based on agent type and provider
        if agent.provider.lower() == "openai" and agent.type == "response":
            client = openai.OpenAI()
            executor = ResponseAPIExecutor(agent)
            result = executor.execute(client, enhanced_instructions, stream=False)
        else:
            # Use LangChain agent (supports multiple providers)
            llm = init_chat_model(model=agent.model)
            executor = CreateAgentExecutor(agent, tools=[])
            result = executor.execute(llm, input_message, stream=False)
        
        return result.get("content", "")
                
    
    @staticmethod
    def _execute_supervisor_with_routing(agent: Agent, state: WorkflowState, 
                                        input_message: str, workers: List[Agent],
                                        workers_used: set, allow_parallel: bool = False) -> tuple[str, Dict[str, Any]]:
        """Execute supervisor agent with structured routing via function calling.
        
        Note: Function calling routing only supported for OpenAI. Other providers
        will fallback to text-based routing.
        
        Args:
            allow_parallel: If True, adds "PARALLEL" as an option to delegate to all workers simultaneously
        """
        
        try:
            # Only OpenAI supports function calling for routing currently
            if agent.provider.lower() != "openai":
                content = AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
                return content, {"action": "finish"}
            
            client = openai.OpenAI()
            
            # Define delegation function for structured routing
            available_workers = [w for w in workers if w.name not in workers_used]
            
            # Build worker name enum - include PARALLEL option if allowed
            worker_name_options = [w.name for w in available_workers] if available_workers else []
            if allow_parallel and len(workers) > 1:
                worker_name_options.append("PARALLEL")
            
            if not worker_name_options:
                worker_name_options = ["none"]
            
            # Build description
            worker_desc_parts = [f'{w.name} ({w.role})' for w in available_workers]
            if allow_parallel and len(workers) > 1:
                worker_desc_parts.append("PARALLEL (delegate to ALL workers simultaneously)")
            worker_description = f"The name of the worker to delegate to. Available: {', '.join(worker_desc_parts)}"
            
            tools = [{
                "type": "function",
                "function": {
                    "name": "delegate_task",
                    "description": "Delegate a task to a specialist worker agent. Use this when you need a specialist to handle specific work.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "worker_name": {
                                "type": "string",
                                "enum": worker_name_options,
                                "description": worker_description
                            },
                            "task_description": {
                                "type": "string",
                                "description": "Clear description of what the worker should do"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Priority level of this task"
                            }
                        },
                        "required": ["worker_name", "task_description"]
                    }
                }
            }] if available_workers else []
            
            enhanced_instructions = get_basic_agent_instructions(
                role=agent.role,
                instructions=agent.instructions,
                user_query=input_message,
                context_messages=None
            )
            
            messages = [{"role": "user", "content": enhanced_instructions}]
            
            with st.spinner(f"🤖 {agent.name} is working..."):
                if tools:
                    response = client.chat.completions.create(
                        model=agent.model,
                        messages=messages,
                        temperature=agent.temperature,
                        tools=tools,
                        tool_choice="auto"
                    )
                else:
                    # No workers available, just get response
                    response = client.chat.completions.create(
                        model=agent.model,
                        messages=messages,
                        temperature=agent.temperature
                    )
            
            message = response.choices[0].message
            content = message.content or ""
            
            # Check for function call (structured routing decision)
            routing_decision = {"action": "finish"}
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.function.name == "delegate_task":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    
                    routing_decision = {
                        "action": "delegate",
                        "target_worker": args.get("worker_name"),
                        "task_description": args.get("task_description"),
                        "priority": args.get("priority", "medium")
                    }
                    
                    # Append delegation info to content for user visibility
                    if content:
                        content += f"\n\n**🔄 Delegating to {args['worker_name']}**: {args['task_description']}"
                    else:
                        content = f"**🔄 Delegating to {args['worker_name']}**: {args['task_description']}"
            
            return content, routing_decision
                
        except Exception as e:
            error_message = f"Error executing supervisor {agent.name}: {str(e)}"
            st.error(error_message)
            return error_message, {"action": "finish"}
            
    @staticmethod
    def create_tool_calling_agent_node(calling_agent: Agent, tool_agents: List[Agent]) -> Callable:
        """
        Create a tool calling agent node where the agent can call other agents as tools.
        
        This implements the "agent-as-tools" pattern where:
        - The calling agent stays in control (single node)
        - Tool agents are invoked synchronously and return results
        - Simple task descriptions are passed (not full context)
        - Results are returned directly to the calling agent
        """
        # Create a mapping of agent names to agent objects for quick lookup
        tool_agents_map = {agent.name: agent for agent in tool_agents}
        
        def tool_calling_agent_node(state: WorkflowState) -> Dict[str, Any]:
            # Get user input
            user_query = ""
            for msg in reversed(state["messages"]):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break
            
            # Create agent tools - wrap each tool agent as a function tool
            agent_tools = AgentNodeFactory._create_agent_tools(tool_agents, state)
            
            # Execute the calling agent with agent tools available
            response = AgentNodeFactory._execute_agent_with_tools(
                calling_agent, state, user_query, agent_tools, tool_agents_map
            )
            
            return {
                "current_agent": calling_agent.name,
                "messages": [{
                    "role": "assistant",
                    "content": response,
                    "agent": calling_agent.name,
                    "timestamp": None
                }],
                "agent_outputs": {calling_agent.name: response}
            }
        
        return tool_calling_agent_node
    
    @staticmethod
    def _create_agent_tools(tool_agents: List[Agent], state: WorkflowState) -> List[Dict[str, Any]]:
        """
        Create OpenAI function tool definitions for each agent.
        
        Each agent becomes a callable tool that the calling agent can invoke.
        """
        tools = []
        
        for agent in tool_agents:
            tool_description = f"{agent.role}. {agent.instructions}"
            
            tools.append({
                "type": "function",
                "function": {
                    "name": agent.name,
                    "description": tool_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": f"Clear description of the task for {agent.name} to perform. Be specific about what you need."
                            }
                        },
                        "required": ["task"]
                    }
                }
            })
        
        return tools
    
    @staticmethod
    def _execute_agent_with_tools(agent: Agent, state: WorkflowState, 
                                  input_message: str, tools: List[Dict[str, Any]],
                                  tool_agents_map: Dict[str, Agent]) -> str:
        """
        Execute an agent with access to tools (other agents wrapped as tools).
        
        When the agent calls a tool, we execute the corresponding agent synchronously
        and return the result back to the calling agent.
        """
        if agent.provider.lower() != "openai":
            # Fallback to basic execution without tools
            return AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
        
        client = openai.OpenAI()
        
        enhanced_instructions = get_tool_calling_agent_instructions(
            role=agent.role,
            instructions=agent.instructions
        )

        messages = [{"role": "user", "content": f"{enhanced_instructions}\n\nUser request: {input_message}"}]
        
        max_iterations = 10
        iteration = 0        
        while iteration < max_iterations:
            with st.spinner(f"🤖 {agent.name} is working..."):
                response = client.chat.completions.create(
                    model=agent.model,
                    messages=messages,
                    temperature=agent.temperature,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None
                )
            
            message = response.choices[0].message
            messages.append(message)
            
            # If no tool calls, return the content
            if not message.tool_calls:
                return message.content or ""
            
            # Execute tool calls (agent invocations)
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                import json
                try:
                    args = json.loads(tool_call.function.arguments)
                    task = args.get("task", "")
                    
                    # Find the corresponding agent from the map
                    tool_agent = tool_agents_map.get(tool_name)
                    
                    if not tool_agent:
                        tool_result = f"Error: Agent {tool_name} not found"
                    else:
                        # Execute the tool agent synchronously with simple task description
                        tool_result = AgentNodeFactory._execute_tool_agent(
                            tool_agent, task, state
                        )
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result
                    })
                    
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": f"Error executing {tool_name}: {str(e)}"
                    })
            iteration += 1
        
        return message.content or "Maximum iterations reached"
    
    @staticmethod
    def _execute_tool_agent(agent: Agent, task: str, state: WorkflowState) -> str:
        """
        Execute a tool agent with a simple task description.
        
        Tool agents receive only the task description, not full context.
        They execute and return results synchronously.
        """
        tool_instructions = get_tool_agent_instructions(
            role=agent.role,
            instructions=agent.instructions,
            task=task
        )
        
        return AgentNodeFactory._execute_agent(agent, state, tool_instructions, [], 0)