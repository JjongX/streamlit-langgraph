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


def get_enhanced_agent_instructions(
    role: str, instructions: str, user_query: str,
    context: str = "", tool_descriptions: List[str] = None, file_context_note: str = ""
) -> str:
    """
    Get enhanced agent instructions for chat interface execution.
    
    This includes tool descriptions and file context information.
    
    Args:
        role: Agent's role
        instructions: Agent's specific instructions
        user_query: User's query/prompt
        context: Optional current conversation context
        tool_descriptions: Optional list of formatted tool descriptions
        file_context_note: Optional note about uploaded files
        
    Returns:
        Enhanced instruction string with tool and context information
    """
    tools_section = chr(10).join(tool_descriptions) if tool_descriptions else "No special tools available"
    
    context_section = f"\n\nCurrent conversation context:\n{context}" if context else ""
    
    return f"""You are {role}. {instructions}

Available tools:
{tools_section}{file_context_note}{context_section}

User: {user_query}
Assistant:"""


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

