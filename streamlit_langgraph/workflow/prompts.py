# Prompt templates and builders for different workflow patterns.

from typing import List, Optional

SUPERVISOR_PROMPT_TEMPLATE = """You are {role}.

You are supervising the following workers: {worker_list}

User's Request: {user_query}

Worker Outputs So Far:
{worker_outputs}

YOUR DECISION:
- Analyze what work still needs to be done
- Determine which specialist can best handle it
- Use the 'delegate_task' function to assign work to a specialist

YOUR OPTIONS:
1. **Delegate to Worker**: Use the delegate_task function to assign tasks to a specialist
2. **Complete Workflow**: When all required work is complete, provide the final response without calling delegate_task.

💡 Think carefully about which worker to delegate to based on their specializations.
"""

NETWORK_PROMPT_TEMPLATE = """You are {role}.

You are part of a collaborative network with the following peers: {peer_list}

User's Request: {user_query}

Peer Outputs So Far:
{peer_outputs}

YOUR DECISION:
- Analyze what work still needs to be done
- Determine if another peer agent should handle the next step
- Use the 'delegate_task' function to hand off to a peer if needed

YOUR OPTIONS:
1. **Hand off to Peer**: Use delegate_task to hand off work to a peer agent who can better handle the next step
2. **Complete Workflow**: When the task is fully complete, provide the final response without calling delegate_task

💡 Collaborate with your peers based on their specializations to deliver the best result.
"""

WORKER_TOOL_PROMPT_TEMPLATE = """Task: {task}

Your role: {role}
Your instructions: {instructions}

Complete this task and return the result. Be concise and focused on the specific task.
"""


class SupervisorPromptBuilder:    
    @staticmethod
    def get_supervisor_instructions(
        role: str, 
        instructions: str, 
        user_query: str,
        worker_list: str, 
        worker_outputs: List[str]
    ) -> str:
        """Get full supervisor instructions template."""
        outputs_text = "\n".join(worker_outputs) if worker_outputs else "No worker outputs yet"
        return SUPERVISOR_PROMPT_TEMPLATE.format(
            role=role,
            user_query=user_query,
            worker_list=worker_list,
            worker_outputs=outputs_text
        )
    
    @staticmethod
    def get_worker_agent_instructions(
        role: str, 
        instructions: str, 
        user_query: str, 
        supervisor_output: Optional[str] = None, 
        previous_worker_outputs: Optional[List[str]] = None
    ) -> str:
        """Get instructions for worker agents in supervisor workflows."""
        instruction_parts = [
            f"Original Request: {user_query}",
            f"Your Role: {role}"
        ]
        
        if supervisor_output:
            instruction_parts.append(f"\nSupervisor Instructions: {supervisor_output}")
        
        if previous_worker_outputs:
            instruction_parts.append(
                f"\nPrevious Worker Results:\n{chr(10).join(previous_worker_outputs)}"
            )
        
        instruction_parts.append("\nPlease complete the task assigned to you.")
        
        return chr(10).join(instruction_parts)


class NetworkPromptBuilder:    
    @staticmethod
    def get_network_agent_instructions(
        role: str, 
        instructions: str, 
        user_query: str,
        peer_list: str, 
        peer_outputs: List[str]
    ) -> str:
        """Get full network agent instructions template."""
        outputs_text = "\n".join(peer_outputs) if peer_outputs else "No peer outputs yet"
        return NETWORK_PROMPT_TEMPLATE.format(
            role=role,
            user_query=user_query,
            peer_list=peer_list,
            peer_outputs=outputs_text
        )



class ToolCallingPromptBuilder:
    """Builder class for creating tool calling agent prompts."""
    
    @staticmethod
    def get_worker_tool_instructions(role: str, instructions: str, task: str) -> str:
        """Get instructions for worker agent invoked as a tool."""
        return WORKER_TOOL_PROMPT_TEMPLATE.format(
            role=role,
            instructions=instructions,
            task=task
        )
