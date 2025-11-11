import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..agent import Agent, ResponseAPIExecutor, CreateAgentExecutor, AgentManager
from .prompts import SupervisorPromptBuilder, ToolCallingPromptBuilder
from .state import WorkflowState, WorkflowStateManager

class AgentNodeFactory:
    """Factory for creating LangGraph agent nodes with handoff and tool calling delegation modes."""

    @staticmethod
    def create_supervisor_agent_node(supervisor: Agent, workers: List[Agent], 
                                     allow_parallel: bool = False,
                                     delegation_mode: str = "handoff") -> Callable:
        """Create a supervisor agent node with structured routing."""
        if delegation_mode == "handoff":
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
                if pending_interrupts:
                    return {"current_agent": supervisor.name, "metadata": state.get("metadata", {})}
                
                worker_outputs = AgentNodeFactory.HandoffDelegation._build_worker_outputs_summary(state, workers)
                user_query = AgentNodeFactory._extract_user_query(state)
                supervisor_instructions = SupervisorPromptBuilder.get_supervisor_instructions(
                    role=supervisor.role,
                    instructions=supervisor.instructions,
                    user_query=user_query,
                    worker_list=", ".join([f"{w.name} ({w.role})" for w in workers]),
                    worker_outputs=worker_outputs
                )
                response, routing_decision = AgentNodeFactory.HandoffDelegation._execute_supervisor_with_routing(
                    supervisor, state, supervisor_instructions, workers, allow_parallel
                )
                return {
                    "current_agent": supervisor.name,
                    "messages": [{"role": "assistant", "content": response, "agent": supervisor.name}],
                    "agent_outputs": {supervisor.name: response},
                    "metadata": WorkflowStateManager.merge_metadata(state.get("metadata", {}), {"routing_decision": routing_decision})
                }
            return supervisor_agent_node
        else: # tool calling delegation mode
            tool_agents_map = {agent.name: agent for agent in workers}
            def supervisor_agent_node(state: WorkflowState) -> Dict[str, Any]:
                user_query = AgentNodeFactory._extract_user_query(state)
                agent_tools = AgentNodeFactory.ToolCallingDelegation._create_agent_tools(workers, state)
                response = AgentNodeFactory.ToolCallingDelegation._execute_agent_with_tools(
                    supervisor, state, user_query, agent_tools, tool_agents_map
                )
                return {
                    "current_agent": supervisor.name,
                    "messages": [{"role": "assistant", "content": response, "agent": supervisor.name}],
                    "agent_outputs": {supervisor.name: response}
                }
            return supervisor_agent_node
    
    @staticmethod
    def create_worker_agent_node(worker: Agent, supervisor: Agent) -> Callable:
        """Create a worker agent node for workflow patterns (handoff delegation mode only)."""
        return AgentNodeFactory.HandoffDelegation.create_worker_agent_node(worker, supervisor)

    @staticmethod
    def _extract_user_query(state: WorkflowState) -> str:
        """Extract user query from state messages."""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                return msg["content"]
        return ""
    
    @staticmethod
    def _execute_agent(agent: Agent, state: WorkflowState, input_message: str, 
                      context_messages: List[str], agent_responses_count: int) -> str:
        """
        Execute an agent and return the response.
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
        llm_client = AgentManager.get_llm_client(agent)
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
                
                config = {"configurable": {"thread_id": executor.thread_id}}
                conversation_messages = state.get("messages", [])
                result = executor.execute(llm_client, enhanced_instructions, stream=False, config=config, messages=conversation_messages)
                if result.get("__interrupt__"):
                    interrupt_update = WorkflowStateManager.set_pending_interrupt(state, agent.name, result, executor_key)
                    state["metadata"].update(interrupt_update["metadata"])
                    return ""
                return result.get("content", "")
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
                from ..utils import CustomTool # lazy import to avoid circular import
                executor.tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
            
            if "executors" not in state.get("metadata", {}):
                state["metadata"]["executors"] = {}
            state["metadata"]["executors"][executor_key] = {"thread_id": executor.thread_id}
            
            config = {"configurable": {"thread_id": executor.thread_id}}
            conversation_messages = state.get("messages", [])
            result = executor.execute(llm_client, input_message, stream=False, config=config, messages=conversation_messages)
            if result.get("__interrupt__"):
                interrupt_update = WorkflowStateManager.set_pending_interrupt(state, agent.name, result, executor_key)
                state["metadata"].update(interrupt_update["metadata"])
                return ""
            return result.get("content", "")
    
    class HandoffDelegation:
        """Inner class for handoff delegation pattern where agents transfer control between nodes."""
        
        @staticmethod
        def create_worker_agent_node(worker: Agent, supervisor: Agent) -> Callable:
            """Create a worker agent node for supervisor workflows."""
            def worker_agent_node(state: WorkflowState) -> Dict[str, Any]:
                user_query = AgentNodeFactory._extract_user_query(state)
                context_data, previous_worker_outputs = AgentNodeFactory.HandoffDelegation._build_worker_context(
                    state, worker, supervisor
                )
                worker_instructions = SupervisorPromptBuilder.get_worker_agent_instructions(
                    role=worker.role, instructions=worker.instructions, user_query=user_query,
                    supervisor_output=context_data, previous_worker_outputs=previous_worker_outputs
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
                    "messages": [{"role": "assistant", "content": response, "agent": worker.name}],
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
        def _execute_supervisor_with_routing(agent: Agent, state: WorkflowState, 
                                            input_message: str, workers: List[Agent],
                                            allow_parallel: bool = False) -> tuple[str, Dict[str, Any]]:
            """
            Execute supervisor agent with structured routing via function calling.
            
            Routes to appropriate executor based on agent type:
            - ResponseAPIExecutor (OpenAI response type) -> uses OpenAI function calling directly
            - CreateAgentExecutor (LangChain) -> uses LangChain tool calling
            
            Args:
                agent: Supervisor agent
                state: Current workflow state
                input_message: Supervisor instructions/prompt
                workers: Available worker agents
                allow_parallel: If True, allows "PARALLEL" delegation option
                
            Returns:
                Tuple of (response_content, routing_decision_dict)
            """
            if agent.provider.lower() == "openai" and agent.type == "response":
                # Use ResponseAPIExecutor approach - OpenAI function calling
                return AgentNodeFactory.HandoffDelegation._execute_with_response_api_executor(
                    agent, state, input_message, workers, allow_parallel
                )
            else:
                # Use CreateAgentExecutor approach - LangChain tool calling
                return AgentNodeFactory.HandoffDelegation._execute_with_create_agent_executor(
                    agent, state, input_message, workers, allow_parallel
                )
        
        @staticmethod
        def _execute_with_response_api_executor(agent: Agent, state: WorkflowState,
                                               input_message: str, workers: List[Agent],
                                               allow_parallel: bool) -> tuple[str, Dict[str, Any]]:
            """Execute supervisor using ResponseAPIExecutor approach with OpenAI function calling."""
            client = AgentManager.get_llm_client(agent)
            tools = AgentNodeFactory.HandoffDelegation._build_openai_delegation_tool(workers, allow_parallel)
            enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
            messages = [{"role": "user", "content": enhanced_instructions}]
            
            with st.spinner(f"🤖 {agent.name} is working..."):
                response = client.chat.completions.create(
                    model=agent.model, messages=messages, temperature=agent.temperature,
                    tools=tools, tool_choice="auto" if tools else None
                )
            message = response.choices[0].message
            content = message.content or ""
            routing_decision = AgentNodeFactory.HandoffDelegation._extract_openai_routing_decision(message, content)
            return routing_decision[1], routing_decision[0]

        @staticmethod
        def _build_openai_delegation_tool(workers: List[Agent], allow_parallel: bool) -> List[Dict[str, Any]]:
            """Build OpenAI function tool definition for delegation."""
            if not workers:
                return []
            
            worker_name_options = [w.name for w in workers]
            worker_desc_parts = [f'{w.name} ({w.role})' for w in workers]
            if allow_parallel and len(workers) > 1:
                worker_name_options.append("PARALLEL")
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
        def _extract_openai_routing_decision(message, content: str) -> tuple[Dict[str, Any], str]:
            """Extract routing decision from OpenAI function call response."""
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
        def _execute_with_create_agent_executor(agent: Agent, state: WorkflowState,
                                               input_message: str, workers: List[Agent],
                                               allow_parallel: bool) -> tuple[str, Dict[str, Any]]:
            """Execute supervisor using CreateAgentExecutor approach with LangChain tool calling."""
            # Check if we have workers to delegate to
            if not workers:
                content = AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
                return content, {"action": "finish"}
            
            # Build delegation tool as LangChain StructuredTool
            delegation_tool = AgentNodeFactory.HandoffDelegation._build_langchain_delegation_tool(workers, allow_parallel)
            if not delegation_tool:
                content = AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
                return content, {"action": "finish"}
            
            llm_client = AgentManager.get_llm_client(agent)
            executor_key = f"workflow_executor_{agent.name}"
            if "agent_executors" not in st.session_state:
                st.session_state.agent_executors = {}
            
            # Create executor with delegation tool temporarily added
            if executor_key not in st.session_state.agent_executors:
                from ..utils import CustomTool # lazy import to avoid circular import
                existing_tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
                executor = CreateAgentExecutor(agent, tools=existing_tools + [delegation_tool])
                st.session_state.agent_executors[executor_key] = executor
            else:
                # Reuse existing executor but temporarily add delegation tool
                executor = st.session_state.agent_executors[executor_key]
                existing_tools = executor.tools.copy() if executor.tools else []
                if "delegate_task" not in [tool.name for tool in existing_tools]:
                    executor.tools = existing_tools + [delegation_tool]
                    executor.agent_obj = None
            
            if "executors" not in state.get("metadata", {}):
                state["metadata"]["executors"] = {}
            state["metadata"]["executors"][executor_key] = {"thread_id": executor.thread_id}
            
            # Execute with enhanced instructions
            enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
            config = {"configurable": {"thread_id": executor.thread_id}}
            
            with st.spinner(f"🤖 {agent.name} is working..."):
                if executor.agent_obj is None:
                    executor._build_agent(llm_client)
                
                # Check for interrupts first via streaming
                interrupt_data = executor._detect_interrupt_in_stream(config, enhanced_instructions)
                if interrupt_data:
                    result = executor._create_interrupt_response(interrupt_data, executor.thread_id, config)
                    interrupt_update = WorkflowStateManager.set_pending_interrupt(state, agent.name, result, executor_key)
                    state["metadata"].update(interrupt_update["metadata"])
                    return "", {"action": "finish"}
                # Invoke agent and get raw output
                out = executor.agent_obj.invoke(
                    {"messages": [{"role": "user", "content": enhanced_instructions}]}, config=config
                )
                # Check for interrupts in output
                if isinstance(out, dict) and "__interrupt__" in out:
                    result = executor._create_interrupt_response(out["__interrupt__"], executor.thread_id, config)
                    interrupt_update = WorkflowStateManager.set_pending_interrupt(state, agent.name, result, executor_key)
                    state["metadata"].update(interrupt_update["metadata"])
                    return "", {"action": "finish"}
            # Extract routing decision from LangChain output
            routing_decision = AgentNodeFactory.HandoffDelegation._extract_langchain_routing_decision(out, enhanced_instructions)
            return routing_decision[1], routing_decision[0]
        
        @staticmethod
        def _build_langchain_delegation_tool(workers: List[Agent], allow_parallel: bool) -> Optional[StructuredTool]:
            """Build LangChain StructuredTool for delegation."""
            if not workers:
                return None
            
            worker_name_options = [w.name for w in workers]
            worker_desc_parts = [f'{w.name} ({w.role})' for w in workers]
            if allow_parallel and len(workers) > 1:
                worker_name_options.append("PARALLEL")
                worker_desc_parts.append("PARALLEL (delegate to ALL workers simultaneously)")
            
            # Create a dummy function that will be called when tool is invoked
            # Function required by StructuredTool - return value is not used, routing decision is extracted from tool call
            def delegate_task(worker_name: str, task_description: str, priority: str = "medium") -> str:
                return f"Task delegated to {worker_name}: {task_description}"
            tool_description = (
                f"Delegate a task to a specialist worker agent. Use this when you need a specialist to handle specific work. "
                f"Available workers: {', '.join(worker_desc_parts)}"
            )
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

            return StructuredTool.from_function(
                func=delegate_task,
                name="delegate_task",
                description=tool_description,
                args_schema=DelegationParams
            )
        
        @staticmethod
        def _extract_langchain_routing_decision(out: Any, prompt: str) -> tuple[Dict[str, Any], str]:
            """Extract routing decision from LangChain agent output."""
            routing_decision = {"action": "finish"}
            content = ""
            
            try:
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
                
                if messages:
                    from langchain_core.messages import AIMessage
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
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
                                    if hasattr(msg, 'content') and msg.content:
                                        content = msg.content
                                    content = content + delegation_text if content else delegation_text[2:] # # Remove leading \n\n
                                    return routing_decision, content
                        # Also check if message has content (for final response)
                        if hasattr(msg, 'content') and msg.content and not content:
                            content = msg.content
                
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
                    content = str(out.get('output', '')) if isinstance(out, dict) and 'output' in out else (out if isinstance(out, str) else str(out))
            
            return routing_decision, content or ""

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
            """Build context data for worker based on context mode."""
            context_mode = getattr(worker, 'context', 'least') or 'least'
            supervisor_output = state["agent_outputs"].get(supervisor.name, "")
            
            if context_mode == "full":
                return supervisor_output, AgentNodeFactory.HandoffDelegation._get_previous_worker_outputs(
                    state, supervisor.name, worker.name
                )
            elif context_mode == "summary":
                routing_decision = state.get("metadata", {}).get("routing_decision", {})
                return routing_decision.get("task_description", supervisor_output), None
            else: # least
                return None, None
    
    class ToolCallingDelegation:
        """Inner class for tool calling delegation pattern where agents are exposed as tools."""
        
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
                return AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)

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
                return AgentNodeFactory._execute_agent(tool_agent, state, tool_instructions, [], 0)
            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"
