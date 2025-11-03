from typing import List, Set

def get_supervisor_action_guidance(workers_used: Set[str], unused_workers: List[str]) -> str:
    """
    Get action guidance for supervisor agents.
    
    Args:
        workers_used: Set of worker names that have been used
        unused_workers: List of worker names not yet used
        
    Returns:
        Formatted action guidance string
    """
    workers_used_list = ", ".join(workers_used) if workers_used else "None"
    
    return (
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


def get_supervisor_instructions(
    role: str, instructions: str, user_query: str,
    worker_list: str, workers_used_list: str, worker_outputs: List[str],action_guidance: str
) -> str:
    """
    Get full supervisor instructions template.
    
    Args:
        role: Supervisor's role
        instructions: Supervisor's specific instructions
        user_query: Original user query
        worker_list: Formatted list of workers with roles
        workers_used_list: Comma-separated list of used workers
        worker_outputs: List of formatted worker outputs
        action_guidance: Action guidance string
        
    Returns:
        Complete supervisor instruction template
    """
    return f"""You are {role}. {instructions}

You are supervising the following workers: {worker_list}

User's Request: {user_query}

Workers Used So Far: {workers_used_list}

Worker Outputs So Far:
{chr(10).join(worker_outputs) if worker_outputs else "No worker outputs yet"}

{action_guidance}

DELEGATION:
• To delegate: Call the 'delegate_task' function with the worker name and task details
• To complete: Provide your final response without calling any function
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

