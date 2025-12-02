# Message utility functions.

import uuid
from typing import Any, Dict


def create_message_with_id(role: str, content: str, agent: str) -> Dict[str, Any]:
    """Helper to create a message with a unique ID."""
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "agent": agent
    }

