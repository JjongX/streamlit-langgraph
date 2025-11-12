from typing import List, Optional

# Common prompt fragments
_ROLE_INTRO = "You are {role}. {instructions}"
_USER_QUERY_HEADER = "User's Request: {user_query}"
_WORKER_LIST_HEADER = "You are supervising the following workers: {worker_list}"
_WORKER_OUTPUTS_HEADER = "Worker Outputs So Far:\n{worker_outputs}"
_TASK_HEADER = "Task: {task}"
_ROLE_HEADER = "Your role: {role}"
_INSTRUCTIONS_HEADER = "Your instructions: {instructions}"

# Supervisor Prompt Templates
SUPERVISOR_PROMPT_TEMPLATE = f"""{_ROLE_INTRO}

{_WORKER_LIST_HEADER}

{_USER_QUERY_HEADER}

{_WORKER_OUTPUTS_HEADER}

YOUR DECISION:
- Analyze what work still needs to be done
- Determine which specialist can best handle it
- Use the 'delegate_task' function to assign work to a specialist

YOUR OPTIONS:
1. **Delegate to Worker**: Use the delegate_task function to assign tasks to a specialist
2. **Complete Workflow**: When all required work is complete, provide the final response without calling delegate_task.

💡 Think carefully about which worker to delegate to based on their specializations.
"""

SEQUENTIAL_ROUTE_GUIDANCE = """When delegating sequentially:
- Delegate to one worker at a time
- Wait for worker response before deciding next action
- Use worker outputs to inform next delegation
"""

# Tool Calling Prompt Templates
ORCHESTRATOR_TOOL_PROMPT_TEMPLATE = f"""{_ROLE_INTRO}

You have access to specialized agents that can help you. When you need their expertise, call them as tools.
After they complete their task, they will return results to you, and you should synthesize the final response.
"""

WORKER_TOOL_PROMPT_TEMPLATE = f"""{_TASK_HEADER}

{_ROLE_HEADER}
{_INSTRUCTIONS_HEADER}

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
            instructions=instructions,
            user_query=user_query,
            worker_list=worker_list,
            worker_outputs=outputs_text
        )
    
    @staticmethod
    def get_worker_agent_instructions(
        role: str, instructions: str, user_query: str, 
        supervisor_output: Optional[str] = None, previous_worker_outputs: Optional[List[str]] = None) -> str:
        """
        Get instructions for worker agents in supervisor workflows.
        """
        instruction_parts = [
            f"Original Request: {user_query}",
            f"Your Role: {role} - {instructions}"
        ]
        
        if supervisor_output:
            instruction_parts.append(f"\nSupervisor Instructions: {supervisor_output}")
        
        if previous_worker_outputs:
            instruction_parts.append(
                f"\nPrevious Worker Results:\n{chr(10).join(previous_worker_outputs)}"
            )
        
        instruction_parts.append("\nPlease complete the task assigned to you.")
        
        return chr(10).join(instruction_parts)
    
    @staticmethod
    def get_sequential_route_guidance() -> str:
        """
        Get guidance for sequential supervisor routing decisions.
        """
        return SEQUENTIAL_ROUTE_GUIDANCE


class ToolCallingPromptBuilder:
    """Builder class for creating tool calling agent prompts."""
    
    @staticmethod
    def get_orchestrator_tool_instructions(role: str, instructions: str) -> str:
        """
        Get instructions for orchestrator agent that calls workers as tools.
        """
        return ORCHESTRATOR_TOOL_PROMPT_TEMPLATE.format(
            role=role,
            instructions=instructions
        )
    
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

