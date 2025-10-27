"""
Workflow pattern implementations for streamlit-langgraph.

This module provides core multi-agent orchestration patterns:
- Supervisor: A supervisor agent coordinates and delegates to worker agents
"""

from .supervisor import SupervisorPattern

__all__ = [
    'SupervisorPattern'
]
