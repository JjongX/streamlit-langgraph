from typing import List, Optional

class SupervisorPromptBuilder:
    """Builder class for creating supervisor and worker agent prompts."""
    
    @staticmethod
    def get_supervisor_instructions(
        role: str, instructions: str, user_query: str,
        worker_list: str, worker_outputs: List[str]) -> str:
        """
        Get full supervisor instructions template.
        
        Args:
            role: Supervisor's role
            instructions: Supervisor's specific instructions
            user_query: Original user query
            worker_list: Formatted list of workers with roles
            worker_outputs: List of formatted worker outputs
            
        Returns:
            Complete supervisor instruction template
        """
        return f"""You are {role}. {instructions}

You are supervising the following workers: {worker_list}

User's Request: {user_query}

Worker Outputs So Far:
{chr(10).join(worker_outputs) if worker_outputs else "No worker outputs yet"}

YOUR DECISION:
- Analyze what work still needs to be done
- Determine which specialist can best handle it
- Use the 'delegate_task' function to assign work to a specialist

YOUR OPTIONS:
1. **Delegate to Worker**: Use the delegate_task function to assign tasks to a specialist
2. **Complete Workflow**: When all required work is complete, provide the final response without calling delegate_task.

💡 Think carefully about which worker to delegate to based on their specializations.
"""
    
    @staticmethod
    def get_worker_agent_instructions(
        role: str, instructions: str, user_query: str, 
        supervisor_output: Optional[str] = None, previous_worker_outputs: Optional[List[str]] = None) -> str:
        """
        Get instructions for worker agents in supervisor workflows.
        
        Args:
            role: Worker's role
            instructions: Worker's specific instructions
            user_query: Original user request
            supervisor_output: Supervisor's instructions/output (optional based on context mode)
            previous_worker_outputs: Previous worker outputs (optional, only for "full" context mode)
            
        Returns:
            Worker instruction template
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
        
        Returns:
            Guidance string for sequential routing
        """
        return (
            "When delegating sequentially:\n"
            "- Delegate to one worker at a time\n"
            "- Wait for worker response before deciding next action\n"
            "- Use worker outputs to inform next delegation"
        )


class ToolCallingPromptBuilder:
    """Builder class for creating tool calling agent prompts."""
    
    @staticmethod
    def get_tool_calling_agent_instructions(role: str, instructions: str) -> str:
        """
        Get instructions for agents that can call other agents as tools.
        
        Args:
            role: Agent's role
            instructions: Agent's specific instructions
            
        Returns:
            Instruction template for tool calling agents
        """
        return f"""You are {role}. {instructions}

You have access to specialized agents that can help you. When you need their expertise, call them as tools.
After they complete their task, they will return results to you, and you should synthesize the final response."""
    
    @staticmethod
    def get_tool_agent_instructions(role: str, instructions: str, task: str) -> str:
        """
        Get instructions for agents that are invoked as tools.
        
        Tool agents receive only the task description, not full context.
        They execute and return results synchronously.
        
        Args:
            role: Agent's role
            instructions: Agent's specific instructions
            task: Specific task description
            
        Returns:
            Instruction template for tool agents
        """
        return f"""Task: {task}

Your role: {role}
Your instructions: {instructions}

Complete this task and return the result. Be concise and focused on the specific task."""

