"""Middleware module for streamlit-langgraph.

This module provides middleware functionality following LangChain's middleware pattern.
See: https://docs.langchain.com/oss/python/langchain/middleware

Currently includes:
- General interrupt management (InterruptManager)
- Human-in-the-Loop (HITL) middleware for interrupt handling
"""

from .interrupts import InterruptManager
from .hitl import HITLHandler, HITLUtils

__all__ = [
    "InterruptManager",  # General interrupt management (shared)
    "HITLHandler",  # HITL-specific approval UI/UX handler
    "HITLUtils",    # HITL-specific approval/decision utilities
]

