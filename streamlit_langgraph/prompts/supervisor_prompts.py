from typing import List

def get_supervisor_instructions(
    role: str, instructions: str, user_query: str,
    worker_list: str, worker_outputs: List[str]
) -> str:
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


def get_supervisor_sequential_route_guidance() -> str:
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

