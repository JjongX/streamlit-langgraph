# Prompt templates.

from typing import List, Optional

# Supervisor Prompt Templates
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

# Tool Calling Prompt Templates
WORKER_TOOL_PROMPT_TEMPLATE = """Task: {task}

Your role: {role}
Your instructions: {instructions}

Complete this task and return the result. Be concise and focused on the specific task.
"""


class SupervisorPromptBuilder:
    """Builder class for creating supervisor and worker agent prompts."""
    
    @staticmethod
    def get_supervisor_instructions(
        role: str, instructions: str, user_query: str,
        worker_list: str, worker_outputs: List[str]) -> str:
        """
        Get full supervisor instructions template.
        """
        outputs_text = "\n".join(worker_outputs) if worker_outputs else "No worker outputs yet"
        return SUPERVISOR_PROMPT_TEMPLATE.format(
            role=role,
            user_query=user_query,
            worker_list=worker_list,
            worker_outputs=outputs_text
        )
    
    @staticmethod
    def get_worker_agent_instructions(
        role: str, instructions: str, user_query: str, 
        supervisor_output: Optional[str] = None, previous_worker_outputs: Optional[List[str]] = None) -> str:
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
    
class ToolCallingPromptBuilder:
    """Builder class for creating tool calling agent prompts."""
    
    @staticmethod
    def get_worker_tool_instructions(role: str, instructions: str, task: str) -> str:
        """
        Get instructions for worker agent invoked as a tool.
        """
        return WORKER_TOOL_PROMPT_TEMPLATE.format(
            role=role,
            instructions=instructions,
            task=task
        )
