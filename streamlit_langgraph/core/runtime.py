"""Runtime hooks for core workflow execution.

This module defines a small, UI-agnostic interface that the core workflow uses
to access optional UI behaviors (stream rendering, spinners) without importing
any UI framework. The UI layer supplies these hooks when needed.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, ContextManager, Optional, Protocol

from .executor.registry import ExecutorRegistry


class StreamRendererProtocol(Protocol):
    """UI-agnostic interface for rendering streaming agent output."""

    def render(self, agent, stream, message_id: str) -> str:
        """Render stream events and return the full response text."""


@dataclass
class RuntimeHooks:
    """Runtime hooks for workflow execution.

    The core workflow relies on these hooks without knowing the UI.
    A UI layer (e.g., Streamlit) can provide concrete implementations.
    """

    executor_registry: ExecutorRegistry
    stream_renderer: Optional[StreamRendererProtocol] = None
    spinner: Optional[Callable[[str], ContextManager]] = None

    def spinner_context(self, message: str) -> ContextManager:
        """Return a spinner context manager; fall back to a no-op."""
        if self.spinner:
            return self.spinner(message)

        @contextmanager
        def _noop_spinner() -> ContextManager[None]:
            yield None

        return _noop_spinner()
