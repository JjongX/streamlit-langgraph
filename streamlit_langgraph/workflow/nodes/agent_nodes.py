from typing import Callable, List, Dict, Any
import openai
import os
import streamlit as st

from ...agent import Agent
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
        """Create a supervisor agent node that delegates tasks to workers."""
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
                "\n- Choose the appropriate action"
                "\n\nYOUR OPTIONS:"
                "\n1. **Delegate to Worker**: End your response with \"HANDOFF to [worker_name]\" to assign tasks"
                f"\n   - Available workers: {', '.join(unused_workers) if unused_workers else 'All workers used'}"
                "\n2. **Complete Workflow**: When all required work is complete, provide the final output. The workflow will automatically finish when the supervisor stops delegating."
                "\n\n💡 Think carefully about which worker to delegate to based on their specializations."
            )

            supervisor_instructions = f"""You are supervising the following workers: {worker_list}

User's Request: {user_query}

Workers Used So Far: {workers_used_list}

Worker Outputs So Far:
{chr(10).join(worker_outputs) if worker_outputs else "No worker outputs yet"}

{action_guidance}

DELEGATION FORMAT (when needed):
• To delegate: End your response with "HANDOFF to [worker_name]"
• To complete: When you have provided the final output and no further delegation is needed, the workflow will finish automatically.

"""
            response = AgentNodeFactory._execute_agent(supervisor, state, supervisor_instructions, [], 0)

            handoff_command = ""
            if "HANDOFF to " in response:
                handoff_start = response.rfind("HANDOFF to ")
                handoff_command = response[handoff_start:].strip()
                handoff_command = handoff_command.split('\n')[0].strip().rstrip('.')
                target_worker = handoff_command.replace("HANDOFF to ", "").strip()
                if target_worker in workers_used:
                    handoff_command = ""
            
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
                "metadata": {**state["metadata"], "handoff_command": handoff_command} # Handoff command is used to determine which worker to delegate to
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
        
        try:
            # Get API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                error_msg = "OpenAI API key not found in environment variables"
                return f"❌ {error_msg}"
            
            client = openai.OpenAI(api_key=api_key)
            
            enhanced_instructions = f"""You are {agent.role}. {agent.instructions}

{f"Recent conversation context: {chr(10).join(context_messages[-3:])}" if context_messages else ""}

Current task: {input_message}"""
            
            messages = [{"role": "user", "content": enhanced_instructions}]
            
            with st.spinner(f"🤖 {agent.name} is working..."):
                response = client.responses.create(
                    model=agent.model,
                    input=messages,
                    temperature=agent.temperature,
                    stream=False
                )
            
            content = ""
            if hasattr(response, 'output') and isinstance(response.output, list):
                for message in response.output:
                    if hasattr(message, 'content') and isinstance(message.content, list):
                        for content_item in message.content:
                            if hasattr(content_item, 'text'):
                                content += content_item.text
            else:
                if hasattr(response, 'content'):
                    content = str(response.content)
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = str(response.message.content)
            
            return content
                
        except Exception as e:
            error_message = f"Error executing agent {agent.name}: {str(e)}"
            st.error(error_message)
            return error_message
