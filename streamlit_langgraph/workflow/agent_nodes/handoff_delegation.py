# Handoff delegation pattern implementation for agent nodes.

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...agent import Agent, get_llm_client
from ...utils.text_extraction import extract_text_from_content
from ...core.runtime import RuntimeHooks
from ...core.state import WorkflowState, WorkflowStateManager
from ...core.executor.response_api import ResponseAPIExecutor
from .factory import AgentNodeBase


class HandoffDelegation:
    """Handoff delegation pattern for supervisor-worker workflows."""

    def __init__(self, runtime: RuntimeHooks, agent_executor: AgentNodeBase):
        self.runtime = runtime
        self.agent_executor = agent_executor

    # Public API
    def execute_supervisor_with_routing(
        self,
        agent: Agent,
        state: WorkflowState,
        input_message: str,
        workers: List[Agent],
        allow_parallel: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute supervisor agent with structured routing via function calling.
        
        Routes to appropriate executor based on executor type:
        - ResponseAPIExecutor -> uses OpenAI Responses API function calling
        - CreateAgentExecutor -> uses LangChain tool calling
        
        Args:
            agent: Supervisor agent
            state: Current workflow state
            input_message: Supervisor instructions/prompt
            workers: Available worker agents
            allow_parallel: If True, allows "PARALLEL" delegation option
            
        Returns:
            Tuple of (response_content, routing_decision_dict)
        """
        executor = self.runtime.executor_registry.get_or_create(
            agent,
            executor_type="workflow",
        )
        
        if isinstance(executor, ResponseAPIExecutor):
            return self._execute_with_response_api_executor(
                agent, state, input_message, workers, allow_parallel
            )
        else:
            return self._execute_with_create_agent_executor(
                agent, state, input_message, workers, allow_parallel
            )
    
    @staticmethod
    def build_worker_context(
        state: WorkflowState,
        worker: Agent,
        supervisor: Agent,
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Build context data for worker based on context mode."""
        context_mode = worker.context
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
    
    @staticmethod
    def build_worker_outputs_summary(
        state: WorkflowState,
        workers: List[Agent],
    ) -> List[str]:
        """Build summary of worker outputs from state."""
        worker_outputs = []
        worker_names = [w.name for w in workers]
        for worker_name in worker_names:
            if worker_name in state["agent_outputs"]:
                output = state['agent_outputs'][worker_name]
                worker_outputs.append(f"**{worker_name}**: {output}")
        return worker_outputs
    
    # Private execution
    def _execute_with_response_api_executor(
        self,
        agent: Agent,
        state: WorkflowState,
        input_message: str,
        workers: List[Agent],
        allow_parallel: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute supervisor using ResponseAPIExecutor approach with Response API function calling."""
        if not workers:
            return self._finish_without_delegation(agent, state, input_message)
        
        delegation_tool = HandoffDelegation._build_openai_delegation_tool(workers, allow_parallel)
        if not delegation_tool:
            return self._finish_without_delegation(agent, state, input_message)
        
        executor = self.runtime.executor_registry.get_or_create(
            agent,
            executor_type="workflow",
        )
        conversation_messages, file_messages, vector_store_ids = HandoffDelegation._extract_state_context(state)
        executor.set_vector_store_ids(vector_store_ids)

        with self.runtime.spinner_context(f"🤖 {agent.name} is working..."):
            out = executor.invoke_response_api(
                prompt=input_message,
                messages=conversation_messages,
                file_messages=file_messages,
                delegation_tool=delegation_tool if delegation_tool else None,
            )
        
        # Extract routing decision from Response API output
        content, routing_decision = HandoffDelegation._extract_response_api_routing_decision(
            out,
            input_message,
        )
        return content, routing_decision
    
    def _execute_with_create_agent_executor(
        self,
        agent: Agent,
        state: WorkflowState,
        input_message: str,
        workers: List[Agent],
        allow_parallel: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute supervisor using CreateAgentExecutor approach with LangChain tool calling."""
        if not workers:
            return self._finish_without_delegation(agent, state, input_message)
        
        delegation_tool = HandoffDelegation._build_langchain_delegation_tool(workers, allow_parallel)
        if not delegation_tool:
            return self._finish_without_delegation(agent, state, input_message)
        
        llm_client = get_llm_client(agent)
        
        executor = self.runtime.executor_registry.get_or_create(
            agent,
            executor_type="workflow",
            tools=agent.get_tools(),
        )
        HandoffDelegation._ensure_delegation_tool(executor, delegation_tool)
        
        executor_key = f"workflow_executor_{agent.name}"
        config, workflow_thread_id = WorkflowStateManager.get_or_create_workflow_config(state, executor_key)
        
        conversation_messages, file_messages, _ = HandoffDelegation._extract_state_context(state)
        
        with self.runtime.spinner_context(f"🤖 {agent.name} is working..."):
            if executor.agent_obj is None:
                executor.build_agent(llm_client)
            
            interrupt_data = None
            if executor.agent.human_in_loop and executor.agent.interrupt_on:
                # Use executor's message conversion for interrupt detection
                langchain_messages = executor.convert_to_langchain_messages(
                    conversation_messages, input_message, file_messages
                )
                interrupt_data = executor.detect_interrupt_in_stream(config, langchain_messages)
            
            if interrupt_data:
                result = executor.create_interrupt_response(interrupt_data, workflow_thread_id, config)
                interrupt_update = WorkflowStateManager.set_pending_interrupt(state, agent.name, result, executor_key)
                state["metadata"].update(interrupt_update["metadata"])
                return "", {"action": "finish"}
            
            # Use executor's invoke_agent method which properly handles conversation history
            out = executor.invoke_agent(
                llm_client=llm_client,
                prompt=input_message,
                messages=conversation_messages,
                file_messages=file_messages,
                config=config
            )
            
            if isinstance(out, dict) and "__interrupt__" in out:
                result = executor.create_interrupt_response(out["__interrupt__"], workflow_thread_id, config)
                interrupt_update = WorkflowStateManager.set_pending_interrupt(state, agent.name, result, executor_key)
                state["metadata"].update(interrupt_update["metadata"])
                return "", {"action": "finish"}
        
        content, routing_decision = HandoffDelegation._extract_langchain_routing_decision(
            out,
            input_message,
        )
        return content, routing_decision
    
    def _finish_without_delegation(
        self,
        agent: Agent,
        state: WorkflowState,
        input_message: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute agent directly when delegation is not possible."""
        response = self.agent_executor.execute_agent(agent, state, input_message)
        return response.get("content", ""), {"action": "finish"}
    
    # Private Tool Building Methods
    @staticmethod
    def _build_worker_options(workers: List[Agent], allow_parallel: bool) -> Tuple[List[str], List[str]]:
        """Build worker name options and description parts for delegation tools."""
        worker_name_options = [w.name for w in workers]
        worker_desc_parts = [f'{w.name} ({w.role})' for w in workers]
        if allow_parallel and len(workers) > 1:
            worker_name_options.append("PARALLEL")
            worker_desc_parts.append("PARALLEL (delegate to ALL workers simultaneously)")
        return worker_name_options, worker_desc_parts
    
    @staticmethod
    def _build_delegation_parameters(workers: List[Agent], allow_parallel: bool) -> Dict[str, Any]:
        """Build common delegation parameters for both tool formats."""
        worker_name_options, worker_desc_parts = HandoffDelegation._build_worker_options(workers, allow_parallel)
        
        parameters_dict = {
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
        
        return {
            "worker_name_options": worker_name_options,
            "worker_desc_parts": worker_desc_parts,
            "parameters_dict": parameters_dict
        }
    
    @staticmethod
    def _build_openai_delegation_tool(workers: List[Agent], allow_parallel: bool) -> List[Dict[str, Any]]:
        """Build OpenAI function tool definition for delegation in Response API format."""
        if not workers:
            return []
        
        params = HandoffDelegation._build_delegation_parameters(workers, allow_parallel)
        # Return Response API format
        return [{
            "type": "function",
            "name": "delegate_task",
            "description": "Delegate a task to a specialist worker agent. Use this when you need a specialist to handle specific work.",
            "parameters": params["parameters_dict"]
        }]
    
    @staticmethod
    def _build_langchain_delegation_tool(workers: List[Agent], allow_parallel: bool) -> Optional[StructuredTool]:
        """Build LangChain StructuredTool for delegation."""
        if not workers:
            return None
        
        params = HandoffDelegation._build_delegation_parameters(workers, allow_parallel)
        worker_name_options = params["worker_name_options"]
        worker_desc_parts = params["worker_desc_parts"]
        
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
    
    # Private Extraction/Parsing Methods
    @staticmethod
    def _extract_state_context(state: WorkflowState) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[str]]]:
        """Extract shared context (messages, files, vector stores) from workflow state."""
        metadata = state.get("metadata", {})
        return (state.get("messages", []), metadata.get("file_messages"), metadata.get("vector_store_ids"))
    
    @staticmethod
    def _extract_response_api_routing_decision(out: Any, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Extract routing decision from Response API output."""
        routing_decision = {"action": "finish"}
        content_parts: List[str] = []

        output_items = HandoffDelegation._get_response_api_output_items(out)
        for item in output_items:
            delegate_args = HandoffDelegation._parse_delegate_task_call(item)
            if delegate_args is not None:
                routing_decision = {
                    "action": "delegate",
                    "target_worker": delegate_args.get("worker_name"),
                    "task_description": delegate_args.get("task_description"),
                    "priority": delegate_args.get("priority", "medium"),
                }
                prefix = "".join(content_parts)
                delegation_text = (
                    f"\n\n**🔄 Delegating to {delegate_args.get('worker_name')}**: "
                    f"{delegate_args.get('task_description')}"
                )
                if prefix:
                    return prefix + delegation_text, routing_decision
                return delegation_text[2:], routing_decision

            item_text = HandoffDelegation._extract_response_api_item_text(item)
            if item_text:
                content_parts.append(item_text)

        return "".join(content_parts), routing_decision

    @staticmethod
    def _get_response_api_output_items(out: Any) -> List[Any]:
        """Get output items from a Response API dict payload."""
        if not isinstance(out, dict):
            return []
        output_items = out.get("output", [])
        return output_items if isinstance(output_items, list) else []

    @staticmethod
    def _parse_delegate_task_call(item: Any) -> Optional[Dict[str, Any]]:
        """Parse delegate_task arguments from a Response API function_call item."""
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "function_call":
            return None
        function_name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        if function_name != "delegate_task":
            return None
        arguments = item.get("arguments", "{}") if isinstance(item, dict) else getattr(item, "arguments", "{}")
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(parsed, dict):
            raise RuntimeError("delegate_task arguments must be a JSON object")
        return parsed

    @staticmethod
    def _extract_response_api_item_text(item: Any) -> str:
        """Extract readable text from a single Response API output item."""
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "output_text":
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
            if not text:
                text = item.get("content", "") if isinstance(item, dict) else getattr(item, "content", "")
            return str(text) if text else ""
        if item_type == "code_interpreter_call":
            return HandoffDelegation._extract_code_interpreter_text(item)
        if item_type == "message":
            return HandoffDelegation._extract_message_item_text(item)
        return ""

    @staticmethod
    def _extract_code_interpreter_text(item: Any) -> str:
        """Extract code and outputs from a code_interpreter_call item."""
        text_parts: List[str] = []
        code = item.get("code", "") if isinstance(item, dict) else getattr(item, "code", "")
        if not code:
            code = item.get("input", "") if isinstance(item, dict) else getattr(item, "input", "")
        if code:
            text_parts.append(f"\n\n```python\n{code}\n```\n\n")

        output = item.get("output") if isinstance(item, dict) else getattr(item, "output", None)
        if isinstance(output, list):
            for output_item in output:
                if not isinstance(output_item, dict):
                    continue
                output_type = output_item.get("type", "")
                if output_type == "text" and output_item.get("text"):
                    text_parts.append(str(output_item.get("text")))
                elif output_type == "image":
                    text_parts.append("\n[Code generated an image]\n")
        elif isinstance(output, str):
            text_parts.append(output)

        return "".join(text_parts)

    @staticmethod
    def _extract_message_item_text(item: Any) -> str:
        """Extract concatenated text from a Response API message item."""
        content_blocks = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        if not isinstance(content_blocks, list):
            return ""
        text_parts: List[str] = []
        for block in content_blocks:
            if hasattr(block, "text"):
                text_parts.append(str(block.text))
                continue
            if isinstance(block, dict):
                if block.get("text"):
                    text_parts.append(str(block.get("text")))
                continue
            if isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    
    @staticmethod
    def _extract_langchain_routing_decision(out: Any, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Extract routing decision from LangChain agent output."""
        routing_decision = {"action": "finish"}
        content = ""
        
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
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        # Handle different tool_call formats
                        tool_name = None
                        tool_args = None
                        
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("args", {})
                        else:
                            tool_name = getattr(tool_call, "name", None)
                            tool_args = getattr(tool_call, "args", {})
                        
                        if tool_name == "delegate_task":
                            if isinstance(tool_args, str):
                                tool_args = json.loads(tool_args)
                            
                            routing_decision = {
                                "action": "delegate",
                                "target_worker": tool_args.get("worker_name"),
                                "task_description": tool_args.get("task_description"),
                                "priority": tool_args.get("priority", "medium")
                            }
                            
                            delegation_text = f"\n\n**🔄 Delegating to {tool_args.get('worker_name')}**: {tool_args.get('task_description')}"
                            if hasattr(msg, 'content') and msg.content:
                                content = extract_text_from_content(msg.content)
                            content = content + delegation_text if content else delegation_text[2:]
                            return content, routing_decision
                
                if hasattr(msg, 'content') and msg.content and not content:
                    content = extract_text_from_content(msg.content)
        
        if not content:
            if isinstance(out, dict):
                if 'output' in out:
                    content = str(out['output'])
                elif 'messages' in out and out['messages']:
                    last_msg = out['messages'][-1]
                    if hasattr(last_msg, 'content'):
                        content = extract_text_from_content(last_msg.content)
                    else:
                        content = str(last_msg)
            elif hasattr(out, 'content'):
                content = extract_text_from_content(out.content)
            else:
                content = str(out)
        
        return content or "", routing_decision
    
    # Private Utility Methods
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
    def _ensure_delegation_tool(executor: Any, delegation_tool: StructuredTool) -> None:
        """Attach delegation tool to executor once and invalidate cached agent if needed."""
        if not delegation_tool:
            return
        existing_tools = getattr(executor, "tools", None) or []
        tool_names = [getattr(tool, "name", None) for tool in existing_tools if hasattr(tool, "name")]
        if delegation_tool.name not in tool_names:
            executor.tools = list(existing_tools) + [delegation_tool]
            if hasattr(executor, "agent_obj"):
                executor.agent_obj = None
