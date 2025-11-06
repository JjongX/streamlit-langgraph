import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
from langchain_core.tools import StructuredTool

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
    Supports both handoff and tool calling delegation modes.
    """
    
    @staticmethod
    def _extract_user_query(state: WorkflowState) -> str:
        """
        Extract user query from state messages.
        """
        user_query = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                user_query = msg["content"]
                break
        return user_query
    
    @staticmethod
    def _execute_agent(agent: Agent, state: WorkflowState, input_message: str, 
                      context_messages: List[str], agent_responses_count: int) -> str:
        """
        Execute an agent with the given input and return the response.
                
        Args:
            agent: Agent to execute
            state: Current workflow state
            input_message: Input message/prompt for the agent
            context_messages: Context messages (unused for now)
            agent_responses_count: Number of agent responses (unused for now)
            
        Returns:
            Agent response content as string
        """
        enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
        
        llm_client = get_llm_client(agent)
        executor_key = f"workflow_executor_{agent.name}"
        
        if "agent_executors" not in st.session_state:
            st.session_state.agent_executors = {}
        
        if agent.provider.lower() == "openai" and agent.type == "response":
            if agent.human_in_loop and agent.interrupt_on:
                if executor_key not in st.session_state.agent_executors:
                    executor = ResponseAPIExecutor(agent)
                    st.session_state.agent_executors[executor_key] = executor
                else:
                    executor = st.session_state.agent_executors[executor_key]
                
                if "executors" not in state.get("metadata", {}):
                    state["metadata"]["executors"] = {}
                state["metadata"]["executors"][executor_key] = {"thread_id": executor.thread_id}
                
                thread_id = executor.thread_id
                config = {"configurable": {"thread_id": thread_id}}
                
                result = executor.execute(llm_client, enhanced_instructions, stream=False, config=config)
                
                if result.get("__interrupt__"):
                    interrupt_update = set_pending_interrupt(state, agent.name, result, executor_key)
                    state["metadata"].update(interrupt_update["metadata"])
                    return ""
                
                content = result.get("content", "")
                return content
            else:
                executor = ResponseAPIExecutor(agent)
                result = executor.execute(llm_client, enhanced_instructions, stream=False)
                return result.get("content", "")
        else:
            if executor_key not in st.session_state.agent_executors:
                executor = CreateAgentExecutor(agent)
                st.session_state.agent_executors[executor_key] = executor
            else:
                executor = st.session_state.agent_executors[executor_key]
                from ..utils import CustomTool
                executor.tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
            
            if "executors" not in state.get("metadata", {}):
                state["metadata"]["executors"] = {}
            state["metadata"]["executors"][executor_key] = {"thread_id": executor.thread_id,}
            
            thread_id = executor.thread_id
            config = {"configurable": {"thread_id": thread_id}}
            
            result = executor.execute(llm_client, input_message, stream=False, config=config)
            
            if result.get("__interrupt__"):
                interrupt_update = set_pending_interrupt(state, agent.name, result, executor_key)
                state["metadata"].update(interrupt_update["metadata"])
                return ""
            
            return result.get("content", "")
    
    class HandoffDelegation:
        """
        Inner class handling handoff delegation pattern where agents transfer control between nodes.
        
        Supervisor nodes coordinate worker agents using LangGraph's conditional routing.
        They analyze worker outputs and make delegation decisions using function calling
        (for OpenAI) or text-based routing (for other providers).
        """
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
            # Return the node function directly (closure captures worker and supervisor)
            def worker_agent_node(state: WorkflowState) -> Dict[str, Any]:
                user_query = AgentNodeFactory._extract_user_query(state)
                
                context_data, previous_worker_outputs = AgentNodeFactory.HandoffDelegation._build_worker_context(
                    state, worker, supervisor
                )
                
                worker_instructions = get_worker_agent_instructions(
                    role=worker.role,
                    instructions=worker.instructions,
                    user_query=user_query,
                    supervisor_output=context_data,
                    previous_worker_outputs=previous_worker_outputs
                )
            
                response = AgentNodeFactory._execute_agent(worker, state, worker_instructions, [], 0)
                
                executor_key = f"workflow_executor_{worker.name}"
                pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
                
                if executor_key in pending_interrupts:
                    return {
                        "current_agent": worker.name,
                        "metadata": state.get("metadata", {}),
                    }
                
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
        def _build_worker_outputs_summary(state: WorkflowState, workers: List[Agent]) -> List[str]:
            """Build summary of worker outputs from state."""
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
            uses OpenAI's function calling to get structured routing decisions for ResponseAPIExecutor.
            For CreateAgentExecutor, it uses LangChain's tool calling mechanism to enable delegation.
            Other providers fallback to text-based analysis.
            
            Args:
                agent: Supervisor agent
                state: Current workflow state
                input_message: Supervisor instructions/prompt
                workers: Available worker agents
                allow_parallel: If True, allows "PARALLEL" delegation option
                
            Returns:
                Tuple of (response_content, routing_decision_dict)
            """
            # Handle OpenAI ResponseAPIExecutor with direct function calling
            if agent.provider.lower() == "openai" and agent.type == "response":
                client = get_llm_client(agent)
                
                tools = AgentNodeFactory.HandoffDelegation._build_delegation_tool(workers, allow_parallel)
                
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
                        response = client.chat.completions.create(
                            model=agent.model,
                            messages=messages,
                            temperature=agent.temperature
                        )
                
                message = response.choices[0].message
                content = message.content or ""
                routing_decision = AgentNodeFactory.HandoffDelegation._extract_routing_decision(message, content)
                return routing_decision[1], routing_decision[0]
            
            # Handle CreateAgentExecutor with LangChain tool calling for delegation
            # Check if we have workers to delegate to
            if workers:
                return AgentNodeFactory.HandoffDelegation._execute_supervisor_with_langchain_routing(
                    agent, state, input_message, workers, allow_parallel
                )
            
            # No workers available, fallback to text-based
            content = AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
            return content, {"action": "finish"}
        
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
        def _execute_supervisor_with_langchain_routing(agent: Agent, state: WorkflowState,
                                                       input_message: str, workers: List[Agent],
                                                       allow_parallel: bool = False) -> tuple[str, Dict[str, Any]]:
            """
            Execute supervisor agent with LangChain tool calling for delegation.
            
            This method enables CreateAgentExecutor to delegate tasks using LangChain's
            tool calling mechanism, similar to how ResponseAPIExecutor uses OpenAI function calling.
            
            Args:
                agent: Supervisor agent (using CreateAgentExecutor)
                state: Current workflow state
                input_message: Supervisor instructions/prompt
                workers: Available worker agents
                allow_parallel: If True, allows "PARALLEL" delegation option
                
            Returns:
                Tuple of (response_content, routing_decision_dict)
            """
            # Build delegation tool as LangChain StructuredTool
            delegation_tool = AgentNodeFactory.HandoffDelegation._build_delegation_langchain_tool(workers, allow_parallel)
            
            if not delegation_tool:
                # No workers available, fallback to text-based
                content = AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
                return content, {"action": "finish"}
            
            # Get or create executor
            llm_client = get_llm_client(agent)
            executor_key = f"workflow_executor_{agent.name}"
            
            if "agent_executors" not in st.session_state:
                st.session_state.agent_executors = {}
            
            # Create executor with delegation tool temporarily added
            if executor_key not in st.session_state.agent_executors:
                # Get existing tools from agent
                from ..utils import CustomTool
                existing_tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
                # Add delegation tool
                all_tools = existing_tools + [delegation_tool]
                executor = CreateAgentExecutor(agent, tools=all_tools)
                st.session_state.agent_executors[executor_key] = executor
            else:
                # Reuse existing executor but temporarily add delegation tool
                executor = st.session_state.agent_executors[executor_key]
                # Get existing tools
                existing_tools = executor.tools.copy() if executor.tools else []
                # Temporarily add delegation tool (if not already present)
                tool_names = [tool.name for tool in existing_tools]
                if "delegate_task" not in tool_names:
                    executor.tools = existing_tools + [delegation_tool]
                    # Rebuild agent with updated tools
                    executor.agent_obj = None  # Force rebuild
            
            # Store metadata
            if "executors" not in state.get("metadata", {}):
                state["metadata"]["executors"] = {}
            state["metadata"]["executors"][executor_key] = {"thread_id": executor.thread_id}
            
            # Execute with enhanced instructions
            enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
            config = {"configurable": {"thread_id": executor.thread_id}}
            
            with st.spinner(f"🤖 {agent.name} is working..."):
                # Invoke agent directly to get access to raw output with messages
                if executor.agent_obj is None:
                    executor._build_agent(llm_client)
                
                # Check for interrupts first via streaming
                interrupt_data = executor._detect_interrupt_in_stream(config, enhanced_instructions)
                if interrupt_data:
                    result = executor._create_interrupt_response(interrupt_data, executor.thread_id, config)
                    interrupt_update = set_pending_interrupt(state, agent.name, result, executor_key)
                    state["metadata"].update(interrupt_update["metadata"])
                    return "", {"action": "finish"}
                
                # Invoke agent and get raw output
                out = executor.agent_obj.invoke(
                    {"messages": [{"role": "user", "content": enhanced_instructions}]},
                    config=config
                )
                
                # Check for interrupt in output
                if isinstance(out, dict) and "__interrupt__" in out:
                    result = executor._create_interrupt_response(out["__interrupt__"], executor.thread_id, config)
                    interrupt_update = set_pending_interrupt(state, agent.name, result, executor_key)
                    state["metadata"].update(interrupt_update["metadata"])
                    return "", {"action": "finish"}
            
            # Extract routing decision from agent output messages
            routing_decision = AgentNodeFactory.HandoffDelegation._extract_routing_decision_from_langchain(
                out, enhanced_instructions
            )
            
            return routing_decision[1], routing_decision[0]
        
        @staticmethod
        def _build_delegation_langchain_tool(workers: List[Agent], allow_parallel: bool) -> Optional[StructuredTool]:
            """Build LangChain StructuredTool for delegation."""
            if not workers:
                return None
            
            worker_name_options = [w.name for w in workers]
            if allow_parallel and len(workers) > 1:
                worker_name_options.append("PARALLEL")
            
            worker_desc_parts = [f'{w.name} ({w.role})' for w in workers]
            if allow_parallel and len(workers) > 1:
                worker_desc_parts.append("PARALLEL (delegate to ALL workers simultaneously)")
            
            # Create a dummy function that will be called when tool is invoked
            # The actual routing decision is extracted from the tool call itself
            def delegate_task(worker_name: str, task_description: str, priority: str = "medium") -> str:
                """
                Delegate a task to a specialist worker agent.
                
                This function is called when the supervisor decides to delegate.
                The actual routing happens in the workflow, not here.
                """
                return f"Task delegated to {worker_name}: {task_description}"
            
            # Create tool description
            tool_description = (
                f"Delegate a task to a specialist worker agent. Use this when you need a specialist to handle specific work. "
                f"Available workers: {', '.join(worker_desc_parts)}"
            )
            
            # Create StructuredTool with proper schema
            from pydantic import BaseModel, Field
            
            class DelegationParams(BaseModel):
                worker_name: str = Field(
                    description=f"The name of the worker to delegate to. Available: {', '.join(worker_desc_parts)}",
                    enum=worker_name_options
                )
                task_description: str = Field(description="Clear description of what the worker should do")
                priority: str = Field(
                    default="medium",
                    description="Priority level of this task",
                    enum=["high", "medium", "low"]
                )
            
            tool = StructuredTool.from_function(
                func=delegate_task,
                name="delegate_task",
                description=tool_description,
                args_schema=DelegationParams
            )
            
            return tool
        
        @staticmethod
        def _extract_routing_decision_from_langchain(out: Any, prompt: str) -> tuple[Dict[str, Any], str]:
            """
            Extract routing decision from LangChain agent output.
            
            Checks the agent's output messages for tool calls to the delegation tool.
            
            Args:
                out: The output from agent_obj.invoke() which may contain messages
                prompt: The original prompt (for context)
                
            Returns tuple of (routing_decision_dict, updated_content)
            """
            routing_decision = {"action": "finish"}
            content = ""
            
            # Extract content and check for tool calls in messages
            try:
                # The output can be a dict with 'messages' key or just messages
                messages = None
                if isinstance(out, dict):
                    if 'messages' in out:
                        messages = out['messages']
                    elif 'output' in out:
                        content = str(out['output'])
                elif hasattr(out, 'messages'):
                    messages = out.messages
                elif hasattr(out, 'content'):
                    content = out.content
                
                # Check messages for tool calls
                if messages:
                    # Find the last AIMessage with tool_calls
                    from langchain_core.messages import AIMessage
                    
                    for msg in reversed(messages):
                        # Check if it's an AIMessage with tool_calls
                        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                            # Look for delegate_task tool call
                            for tool_call in msg.tool_calls:
                                if tool_call.get("name") == "delegate_task" or (
                                    isinstance(tool_call, dict) and tool_call.get("name") == "delegate_task"
                                ):
                                    # Extract arguments - tool_call can be dict or object
                                    if isinstance(tool_call, dict):
                                        args = tool_call.get("args", {})
                                    else:
                                        # Try to get args attribute
                                        args = getattr(tool_call, "args", {})
                                    
                                    # If args is a string (JSON), parse it
                                    if isinstance(args, str):
                                        args = json.loads(args)
                                    
                                    routing_decision = {
                                        "action": "delegate",
                                        "target_worker": args.get("worker_name"),
                                        "task_description": args.get("task_description"),
                                        "priority": args.get("priority", "medium")
                                    }
                                    
                                    delegation_text = f"\n\n**🔄 Delegating to {args.get('worker_name')}**: {args.get('task_description')}"
                                    # Get content from the message or use extracted content
                                    if hasattr(msg, 'content') and msg.content:
                                        content = msg.content
                                    if content:
                                        content = content + delegation_text
                                    else:
                                        content = delegation_text[2:]  # Remove leading \n\n
                                    return routing_decision, content
                        
                        # Also check if message has content (for final response)
                        if hasattr(msg, 'content') and msg.content and not content:
                            content = msg.content
                
                # If no tool calls found, extract content normally
                if not content:
                    if isinstance(out, dict):
                        if 'output' in out:
                            content = str(out['output'])
                        elif 'messages' in out and out['messages']:
                            last_msg = out['messages'][-1]
                            if hasattr(last_msg, 'content'):
                                content = last_msg.content
                            else:
                                content = str(last_msg)
                    elif hasattr(out, 'content'):
                        content = out.content
                    else:
                        content = str(out)
            except Exception:
                # Fallback: just extract text content
                if not content:
                    if isinstance(out, dict) and 'output' in out:
                        content = str(out['output'])
                    elif isinstance(out, str):
                        content = out
                    else:
                        content = str(out)
            
            return routing_decision, content or ""
        
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
                return supervisor_output, AgentNodeFactory.HandoffDelegation._get_previous_worker_outputs(
                    state, supervisor.name, worker.name
                )
            elif context_mode == "summary":
                routing_decision = state.get("metadata", {}).get("routing_decision", {})
                return routing_decision.get("task_description", supervisor_output), None
            else:  # "least"
                return None, None
    
    class ToolCallingDelegation:
        """
        Inner class handling tool calling delegation pattern where agents are exposed as tools.
        
        LangGraph workflow pattern where agents are exposed as tools to a calling agent.
        The calling agent uses OpenAI function calling to invoke tool agents synchronously.
        This differs from supervisor pattern in that the calling agent maintains control.
        """
        
        
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
            """
            if agent.provider.lower() != "openai":
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
                    tool_result = AgentNodeFactory.ToolCallingDelegation._execute_tool_call(
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
                
                return AgentNodeFactory.ToolCallingDelegation._execute_tool_agent(tool_agent, task, state)
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
    
    @staticmethod
    def create_supervisor_agent_node(supervisor: Agent, workers: List[Agent], 
                                     allow_parallel: bool = False,
                                     delegation_mode: str = "handoff") -> Callable:
        """
        Create a supervisor agent node with structured routing.
        
        Supervisor nodes coordinate worker agents using LangGraph's conditional routing.
        They analyze worker outputs and make delegation decisions using function calling
        (for OpenAI) or text-based routing (for other providers).
        
        Args:
            supervisor: Supervisor agent that coordinates workers
            workers: List of worker agents available for delegation
            allow_parallel: If True, allows delegating to all workers simultaneously
            delegation_mode: "handoff" (default) or "tool_calling" delegation mode
            
        Returns:
            LangGraph node function that executes supervisor and returns routing decision
        """
        # Handoff delegation mode
        if delegation_mode == "handoff":
            # Closure captures supervisor, workers, and allow_parallel
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                # Don't execute supervisor if there's a pending interrupt
                pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
                if pending_interrupts:
                    return {"current_agent": supervisor.name, "metadata": state.get("metadata", {})}
                
                worker_outputs = AgentNodeFactory.HandoffDelegation._build_worker_outputs_summary(state, workers)
                
                user_query = AgentNodeFactory._extract_user_query(state)

                supervisor_instructions = AgentNodeFactory.HandoffDelegation._build_supervisor_instructions(
                    supervisor, user_query, workers, worker_outputs
                )
                response, routing_decision = AgentNodeFactory.HandoffDelegation._execute_supervisor_with_routing(
                    supervisor, state, supervisor_instructions, workers, allow_parallel
                )
                
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

        else:  # delegation_mode == "tool_calling"
            tool_agents_map = {agent.name: agent for agent in workers}
            # Closure captures supervisor, workers, and tool_agents_map
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                user_query = AgentNodeFactory._extract_user_query(state)
                agent_tools = AgentNodeFactory.ToolCallingDelegation._create_agent_tools(workers, state)
                response = AgentNodeFactory.ToolCallingDelegation._execute_agent_with_tools(
                    supervisor, state, user_query, agent_tools, tool_agents_map
                )
                
                return {
                    "current_agent": supervisor.name,
                    "messages": [{
                        "role": "assistant",
                        "content": response,
                        "agent": supervisor.name,
                        "timestamp": None
                    }],
                    "agent_outputs": {supervisor.name: response}
                }
            
            return supervisor_agent_node
    
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
        return AgentNodeFactory.HandoffDelegation.create_worker_agent_node(worker, supervisor)
