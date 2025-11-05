import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from ..agent import Agent, ResponseAPIExecutor, CreateAgentExecutor, get_llm_client
from ..prompts import (
    get_supervisor_instructions,
    get_worker_agent_instructions,
    get_tool_calling_agent_instructions,
    get_tool_agent_instructions,
)
from .state import WorkflowState, set_pending_interrupt, merge_metadata

class AgentNodeFactory:
    """
    Factory for creating LangGraph agent nodes.
    
    LangGraph nodes are functions that take WorkflowState and return state updates.
    This factory creates node functions for different agent patterns used in workflows.
    """
    
    @staticmethod
    def create_supervisor_agent_node(supervisor: Agent, workers: List[Agent], 
                                     allow_parallel: bool = False) -> Callable:
        """
        Create a supervisor agent node with structured routing.
        
        Supervisor nodes coordinate worker agents using LangGraph's conditional routing.
        They analyze worker outputs and make delegation decisions using function calling
        (for OpenAI) or text-based routing (for other providers).
        
        Args:
            supervisor: Supervisor agent that coordinates workers
            workers: List of worker agents available for delegation
            allow_parallel: If True, allows delegating to all workers simultaneously
            
        Returns:
            LangGraph node function that executes supervisor and returns routing decision
        """
        def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
            worker_outputs = AgentNodeFactory._build_worker_outputs_summary(state, workers)
            
            user_query = ""
            for msg in reversed(state["messages"]):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break
            
            supervisor_instructions = AgentNodeFactory._build_supervisor_instructions(
                supervisor, user_query, workers, worker_outputs
            )
            
            response, routing_decision = AgentNodeFactory._execute_supervisor_with_routing(
                supervisor, state, supervisor_instructions, workers, allow_parallel
            )
            
            # Preserve existing metadata (especially pending_interrupts) when updating routing_decision
            updated_metadata = merge_metadata(
                state.get("metadata", {}),
                {"routing_decision": routing_decision}
            )
            
            return {
                "current_agent": supervisor.name,
                "messages": [{
                    "role": "assistant",
                    "content": response,
                    "agent": supervisor.name,
                    "timestamp": None
                }],
                "agent_outputs": {supervisor.name: response},
                "metadata": updated_metadata
            }
        
        return supervisor_agent_node
    
    @staticmethod
    def _build_worker_outputs_summary(state: WorkflowState, workers: List[Agent]) -> List[str]:
        worker_outputs = []
        worker_names = [w.name for w in workers]
        for worker_name in worker_names:
            if worker_name in state["agent_outputs"]:
                output = state['agent_outputs'][worker_name]
                worker_outputs.append(f"**{worker_name}**: {output}")
        return worker_outputs
    
    @staticmethod
    def _build_supervisor_instructions(supervisor: Agent, user_query: str, 
                                      workers: List[Agent],
                                      worker_outputs: List[str]) -> str:
        """Build supervisor instructions with worker context."""
        worker_list = ", ".join([f"{w.name} ({w.role})" for w in workers])
        return get_supervisor_instructions(
            role=supervisor.role,
            instructions=supervisor.instructions,
            user_query=user_query,
            worker_list=worker_list,
            worker_outputs=worker_outputs
        )
    
    @staticmethod
    def _execute_supervisor_with_routing(agent: Agent, state: WorkflowState, 
                                        input_message: str, workers: List[Agent],
                                        allow_parallel: bool = False) -> tuple[str, Dict[str, Any]]:
        """
        Execute supervisor agent with structured routing via function calling.
        
        LangGraph workflows use conditional edges based on routing decisions. This method
        uses OpenAI's function calling to get structured routing decisions. Other providers
        fallback to text-based analysis.
        
        Args:
            agent: Supervisor agent
            state: Current workflow state
            input_message: Supervisor instructions/prompt
            workers: Available worker agents
            allow_parallel: If True, allows "PARALLEL" delegation option
            
        Returns:
            Tuple of (response_content, routing_decision_dict)
        """
        if agent.provider.lower() != "openai" or agent.type != "response":
            content = AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
            return content, {"action": "finish"}
        
        client = get_llm_client(agent)
        
        tools = AgentNodeFactory._build_delegation_tool(workers, allow_parallel)
        
        enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
        
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
        routing_decision = AgentNodeFactory._extract_routing_decision(message, content)
        return routing_decision[1], routing_decision[0]
    
    @staticmethod
    def _build_delegation_tool(workers: List[Agent], allow_parallel: bool) -> List[Dict[str, Any]]:
        """Build OpenAI function tool definition for delegation."""
        if not workers:
            return []
        
        worker_name_options = [w.name for w in workers]
        if allow_parallel and len(workers) > 1:
            worker_name_options.append("PARALLEL")
        
        worker_desc_parts = [f'{w.name} ({w.role})' for w in workers]
        if allow_parallel and len(workers) > 1:
            worker_desc_parts.append("PARALLEL (delegate to ALL workers simultaneously)")
        
        return [{
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
                            "description": f"The name of the worker to delegate to. Available: {', '.join(worker_desc_parts)}"
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
        }]
    
    @staticmethod
    def _extract_routing_decision(message, content: str) -> tuple[Dict[str, Any], str]:
        """
        Extract routing decision from OpenAI function call response.
        
        Returns tuple of (routing_decision_dict, updated_content)
        """
        routing_decision = {"action": "finish"}
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "delegate_task":
                args = json.loads(tool_call.function.arguments)
                routing_decision = {
                    "action": "delegate",
                    "target_worker": args.get("worker_name"),
                    "task_description": args.get("task_description"),
                    "priority": args.get("priority", "medium")
                }
                
                delegation_text = f"\n\n**🔄 Delegating to {args['worker_name']}**: {args['task_description']}"
                content = content + delegation_text if content else delegation_text[2:]
        
        return routing_decision, content

    @staticmethod
    def _get_previous_worker_outputs(state: WorkflowState, supervisor_name: str, current_worker_name: str) -> Optional[List[str]]:
        """Get formatted list of previous worker outputs."""
        agent_outputs = state.get("agent_outputs", {})
        worker_outputs = []
        for name, output in agent_outputs.items():
            if name not in (supervisor_name, current_worker_name):
                worker_outputs.append(f"**{name}**: {output}")
        return worker_outputs if worker_outputs else None
    
    @staticmethod
    def _build_worker_context(state: WorkflowState, worker: Agent, supervisor: Agent) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Build context data for worker based on context mode.
        
        Returns:
            tuple of (supervisor_output/context_data, previous_worker_outputs)
        """
        context_mode = getattr(worker, 'context', 'least') or 'least'
        supervisor_output = state["agent_outputs"].get(supervisor.name, "")
        
        if context_mode == "full":
            return supervisor_output, AgentNodeFactory._get_previous_worker_outputs(state, supervisor.name, worker.name)
        elif context_mode == "summary":
            routing_decision = state.get("metadata", {}).get("routing_decision", {})
            return routing_decision.get("task_description", supervisor_output), None
        else:  # "least"
            return None, None
    
    @staticmethod
    def create_worker_agent_node(worker: Agent, supervisor: Agent) -> Callable:
        """
        Create a worker agent node for supervisor workflows.
        
        Worker nodes execute tasks delegated by the supervisor. They receive
        supervisor instructions and the original user query as context.
        
        Args:
            worker: Worker agent that executes delegated tasks
            supervisor: Supervisor agent that coordinates this worker
            
        Returns:
            LangGraph node function that executes worker and updates state
        """
        def worker_agent_node(state: WorkflowState) -> Dict[str, Any]:
            # Get original user query
            user_query = ""
            for msg in reversed(state["messages"]):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break
            
            context_data, previous_worker_outputs = AgentNodeFactory._build_worker_context(state, worker, supervisor)
            
            worker_instructions = get_worker_agent_instructions(
                role=worker.role,
                instructions=worker.instructions,
                user_query=user_query,
                supervisor_output=context_data,
                previous_worker_outputs=previous_worker_outputs
            )
        
            response = AgentNodeFactory._execute_agent(worker, state, worker_instructions, [], 0)
            
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
        
        enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
        
        # Use appropriate executor based on agent type and provider
        llm_client = get_llm_client(agent)
        if agent.provider.lower() == "openai" and agent.type == "response":
            executor = ResponseAPIExecutor(agent)
            result = executor.execute(llm_client, enhanced_instructions, stream=False)
        else:
            # For HITL, we need to persist executors across workflow steps
            # Store executor in workflow state metadata to maintain state
            executor_key = f"workflow_executor_{agent.name}"
            
            # Initialize agent_executors in session_state if needed
            if "agent_executors" not in st.session_state:
                st.session_state.agent_executors = {}
            
            # Get or create executor from session_state (this preserves the checkpointer instance)
            # Tools are loaded automatically by CreateAgentExecutor from CustomTool registry
            if executor_key not in st.session_state.agent_executors:
                # Create new executor (tools are loaded automatically from agent.tools)
                executor = CreateAgentExecutor(agent)
                st.session_state.agent_executors[executor_key] = executor
            else:
                # Reuse existing executor (has the same checkpointer with checkpoint data)
                executor = st.session_state.agent_executors[executor_key]
                # Update tools if agent's tools have changed
                from ..utils import CustomTool
                executor.tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
            
            # Also store metadata for reference (thread_id, etc.)
            if "executors" not in state.get("metadata", {}):
                state["metadata"]["executors"] = {}
            state["metadata"]["executors"][executor_key] = {"thread_id": executor.thread_id,}
            
            # Use thread_id from state for consistent execution
            thread_id = executor.thread_id
            config = {"configurable": {"thread_id": thread_id}}
            
            result = executor.execute(llm_client, input_message, stream=False, config=config)
            
            # If there's an interrupt, store it in workflow state using state functions
            if result.get("__interrupt__"):
                interrupt_update = set_pending_interrupt(state, agent.name, result, executor_key)
                # Update state metadata
                state["metadata"].update(interrupt_update["metadata"])
                # Return empty content - workflow will pause and wait for human decision
                return ""
            
            return result.get("content", "")
        
        return result.get("content", "")
    
    @staticmethod
    def create_tool_calling_agent_node(calling_agent: Agent, tool_agents: List[Agent]) -> Callable:
        """
        Create a tool calling agent node implementing the "agent-as-tools" pattern.
        
        LangGraph workflow pattern where agents are exposed as tools to a calling agent.
        The calling agent uses OpenAI function calling to invoke tool agents synchronously.
        This differs from supervisor pattern in that the calling agent maintains control.
        
        Args:
            calling_agent: Agent that can call other agents as tools
            tool_agents: List of agents exposed as callable tools
            
        Returns:
            LangGraph node function for tool calling pattern
        """
        tool_agents_map = {agent.name: agent for agent in tool_agents}
        
        def tool_calling_agent_node(state: WorkflowState) -> Dict[str, Any]:
            user_query = ""
            for msg in reversed(state["messages"]):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break
            
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

        client = get_llm_client(agent)
        
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

            if not message.tool_calls:
                return message.content or ""
            
            for tool_call in message.tool_calls:
                tool_result = AgentNodeFactory._execute_tool_call(
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
    def _execute_tool_call(tool_call, tool_agents_map: Dict[str, Agent], 
                           state: WorkflowState) -> str:
        """Execute a tool call by invoking the corresponding agent."""
        tool_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            task = args.get("task", "")
            tool_agent = tool_agents_map.get(tool_name)
            
            if not tool_agent:
                return f"Error: Agent {tool_name} not found"
            
            return AgentNodeFactory._execute_tool_agent(tool_agent, task, state)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
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

