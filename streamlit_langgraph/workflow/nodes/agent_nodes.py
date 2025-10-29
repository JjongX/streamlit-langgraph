from typing import Callable, List, Dict, Any
import openai
import os
import streamlit as st

from ...agent import Agent, ResponseAPIExecutor, CreateAgentExecutor
from ..state import WorkflowState
from langchain.chat_models import init_chat_model

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
                new_messages = state["messages"] + [{
                    "role": "assistant",
                    "content": response,
                    "agent": agent.name,
                    "timestamp": None
                }]
                state_updates["messages"] = new_messages
                
            return state_updates
        
        return agent_node
    
    @staticmethod
    def create_supervisor_agent_node(supervisor: Agent, workers: List[Agent]) -> Callable:
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
            
            action_guidance = (
                "\n\n📋 WORKFLOW PROGRESS:"
                f"\n✅ Workers used: {workers_used_list}"
                f"\n⏳ Workers not yet used: {', '.join(unused_workers) if unused_workers else 'None'}"
                "\n\nYOUR DECISION:"
                "\n- Analyze what work still needs to be done"
                "\n- Determine which specialist can best handle it"
                "\n- Use the 'delegate_task' function to assign work to a specialist"
                "\n\nYOUR OPTIONS:"
                "\n1. **Delegate to Worker**: Use the delegate_task function to assign tasks to a specialist"
                f"\n   - Available workers: {', '.join(unused_workers) if unused_workers else 'All workers used'}"
                "\n2. **Complete Workflow**: When all required work is complete, provide the final output without calling delegate_task."
                "\n\n💡 Think carefully about which worker to delegate to based on their specializations."
            )

            supervisor_instructions = f"""You are supervising the following workers: {worker_list}

User's Request: {user_query}

Workers Used So Far: {workers_used_list}

Worker Outputs So Far:
{chr(10).join(worker_outputs) if worker_outputs else "No worker outputs yet"}

{action_guidance}

DELEGATION:
• To delegate: Call the 'delegate_task' function with the worker name and task details
• To complete: Provide your final response without calling any function
"""
            response, routing_decision = AgentNodeFactory._execute_supervisor_with_routing(
                supervisor, state, supervisor_instructions, workers, workers_used
            )
            
            updated_agent_outputs = state["agent_outputs"].copy()
            updated_agent_outputs[supervisor.name] = response

            return {
                "current_agent": supervisor.name,
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": response,
                    "agent": supervisor.name,
                    "timestamp": None
                }],
                "agent_outputs": updated_agent_outputs,
                "metadata": {**state["metadata"], "routing_decision": routing_decision}
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
            
            worker_instructions = f"""Original Request: {user_query}

Supervisor Instructions: {supervisor_output}

Your Role: {worker.role} - {worker.instructions}

Please complete the task assigned by your supervisor."""
            
            response = AgentNodeFactory._execute_agent(worker, state, worker_instructions, [], 0)
            
            updated_agent_outputs = state["agent_outputs"].copy()
            updated_agent_outputs[worker.name] = response
            
            return {
                "current_agent": worker.name,
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": response,
                    "agent": worker.name,
                    "timestamp": None
                }],
                "agent_outputs": updated_agent_outputs
            }
        
        return worker_agent_node
    
    @staticmethod
    def _execute_agent(agent: Agent, state: WorkflowState, input_message: str, 
                      context_messages: List[str], agent_responses_count: int) -> str:
        """Execute an agent with the given input and return the response."""
        
        enhanced_instructions = f"""You are {agent.role}. {agent.instructions}

{f"Recent conversation context: {chr(10).join(context_messages[-3:])}" if context_messages else ""}

Current task: {input_message}"""
        
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
                                        workers_used: set) -> tuple[str, Dict[str, Any]]:
        """Execute supervisor agent with structured routing via function calling.
        
        Note: Function calling routing only supported for OpenAI. Other providers
        will fallback to text-based routing.
        """
        
        try:
            # Only OpenAI supports function calling for routing currently
            if agent.provider.lower() != "openai":
                content = AgentNodeFactory._execute_agent(agent, state, input_message, [], 0)
                return content, {"action": "finish"}
            
            client = openai.OpenAI()
            
            # Define delegation function for structured routing
            available_workers = [w for w in workers if w.name not in workers_used]
            
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
                                "enum": [w.name for w in available_workers] if available_workers else ["none"],
                                "description": f"The name of the worker to delegate to. Available: {', '.join([f'{w.name} ({w.role})' for w in available_workers])}"
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
            
            enhanced_instructions = f"""You are {agent.role}. {agent.instructions}

Current task: {input_message}"""
            
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
