"""
Workflow pattern implementations for streamlit-langgraph.

This module provides core multi-agent orchestration patterns:
- Supervisor: A supervisor agent coordinates and delegates to worker agents
- Hierarchical: A top supervisor coordinates sub-supervisors, each managing their own teams
"""

from .supervisor import SupervisorPattern
from .hierarchical import HierarchicalPattern, SupervisorTeam

__all__ = [
    'SupervisorPattern',
    'HierarchicalPattern',
    'SupervisorTeam'
]
