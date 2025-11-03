from typing import List

def get_basic_agent_instructions(
    role: str, instructions: str, user_query: str,
    context_messages: List[str] = None
) -> str:
    """
    Get basic agent instructions for standard agent execution.
    
    Args:
        role: Agent's role
        instructions: Agent's specific instructions
        user_query: Current task/user query/input message
        context_messages: Optional recent conversation context (last 3 messages)
        
    Returns:
        Enhanced instruction string for basic agents
    """
    context_part = ""
    if context_messages:
        context_part = f"Recent conversation context: {chr(10).join(context_messages[-3:])}"
    
    return f"""You are {role}. {instructions}

{context_part}

Current task: {user_query}"""

def get_worker_agent_instructions(role: str, instructions: str, user_query: str, supervisor_output: str) -> str:
    """
    Get instructions for worker agents in supervisor workflows.
    
    Args:
        role: Worker's role
        instructions: Worker's specific instructions
        user_query: Original user request
        supervisor_output: Supervisor's instructions/output
        
    Returns:
        Worker instruction template
    """
    return f"""Original Request: {user_query}

Supervisor Instructions: {supervisor_output}

Your Role: {role} - {instructions}

Please complete the task assigned by your supervisor."""

