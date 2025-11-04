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

