# Handoff delegation pattern implementation for agent nodes.

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...agent import Agent, AgentManager
from ...core.executor_registry import ExecutorRegistry
from ...state import WorkflowState, WorkflowStateManager
from .base import AgentNodeBase, create_message_with_id
from ..prompts import SupervisorPromptBuilder


class HandoffDelegation:
    """Handoff delegation pattern where agents transfer control between nodes."""
    
    @staticmethod
    def create_worker_agent_node(worker: Agent, supervisor: Agent) -> Callable:
        """Create a worker agent node for supervisor workflows."""
        def worker_agent_node(state: WorkflowState) -> Dict[str, Any]:
            user_query = AgentNodeBase.extract_user_query(state)
            context_data, previous_worker_outputs = HandoffDelegation._build_worker_context(
                state, worker, supervisor
            )
            worker_instructions = SupervisorPromptBuilder.get_worker_agent_instructions(
                role=worker.role, instructions=worker.instructions, user_query=user_query,
                supervisor_output=context_data, previous_worker_outputs=previous_worker_outputs
            )
            response = AgentNodeBase.execute_agent(worker, state, worker_instructions, [], 0)
            
            executor_key = f"workflow_executor_{worker.name}"
            pending_interrupts = state.get("metadata", {}).get("pending_interrupts", {})
            if executor_key in pending_interrupts:
                return {
                    "current_agent": worker.name,
                    "metadata": state.get("metadata", {}),
                }
            return {
                "current_agent": worker.name,
                "messages": [create_message_with_id("assistant", response, worker.name)],
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
                                        allow_parallel: bool = False) -> Tuple[str, Dict[str, Any]]:
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
            return HandoffDelegation._execute_with_response_api_executor(
                agent, state, input_message, workers, allow_parallel
            )
        else:
            # Use CreateAgentExecutor approach - LangChain tool calling
            return HandoffDelegation._execute_with_create_agent_executor(
                agent, state, input_message, workers, allow_parallel
            )
    
    @staticmethod
    def _execute_with_response_api_executor(agent: Agent, state: WorkflowState,
                                           input_message: str, workers: List[Agent],
                                           allow_parallel: bool) -> Tuple[str, Dict[str, Any]]:
        """Execute supervisor using ResponseAPIExecutor approach with OpenAI function calling."""
        client = AgentManager.get_llm_client(agent)
        tools = HandoffDelegation._build_openai_delegation_tool(workers, allow_parallel)
        enhanced_instructions = f"You are {agent.role}. {agent.instructions}\n\nCurrent task: {input_message}"
        messages = [{"role": "user", "content": enhanced_instructions}]
        
        with st.spinner(f"🤖 {agent.name} is working..."):
            response = client.chat.completions.create(
                model=agent.model, messages=messages, temperature=agent.temperature,
                tools=tools, tool_choice="auto" if tools else None
            )
        message = response.choices[0].message
        content = message.content or ""
        routing_decision = HandoffDelegation._extract_openai_routing_decision(message, content)
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
    def _extract_openai_routing_decision(message, content: str) -> Tuple[Dict[str, Any], str]:
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
                                           allow_parallel: bool) -> Tuple[str, Dict[str, Any]]:
        """Execute supervisor using CreateAgentExecutor approach with LangChain tool calling."""
        # Check if we have workers to delegate to
        if not workers:
            content = AgentNodeBase.execute_agent(agent, state, input_message, [], 0)
            return content, {"action": "finish"}
        
        # Build delegation tool as LangChain StructuredTool
        delegation_tool = HandoffDelegation._build_langchain_delegation_tool(workers, allow_parallel)
        if not delegation_tool:
            content = AgentNodeBase.execute_agent(agent, state, input_message, [], 0)
            return content, {"action": "finish"}
        
        llm_client = AgentManager.get_llm_client(agent)
        from ...utils import CustomTool  # lazy import to avoid circular import
        
        # Get or create executor with delegation tool temporarily added
        existing_tools = CustomTool.get_langchain_tools(agent.tools) if agent.tools else []
        executor = ExecutorRegistry.get_or_create(agent, executor_type="workflow", tools=existing_tools + [delegation_tool])
        
        # Reuse existing executor but temporarily add delegation tool if needed
        if executor.tools and "delegate_task" not in [tool.name for tool in executor.tools]:
            existing_tools = executor.tools.copy() if executor.tools else []
            executor.tools = existing_tools + [delegation_tool]
            if hasattr(executor, 'agent_obj'):
                executor.agent_obj = None
        
        if "executors" not in state.get("metadata", {}):
            state["metadata"]["executors"] = {}
        executor_key = ExecutorRegistry._get_executor_key(agent.name, "workflow")
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
        routing_decision = HandoffDelegation._extract_langchain_routing_decision(out, enhanced_instructions)
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
    def _extract_langchain_routing_decision(out: Any, prompt: str) -> Tuple[Dict[str, Any], str]:
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
                                content = content + delegation_text if content else delegation_text[2:]  # Remove leading \n\n
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
            return supervisor_output, HandoffDelegation._get_previous_worker_outputs(
                state, supervisor.name, worker.name
            )
        elif context_mode == "summary":
            routing_decision = state.get("metadata", {}).get("routing_decision", {})
            return routing_decision.get("task_description", supervisor_output), None
        else:  # least
            return None, None

