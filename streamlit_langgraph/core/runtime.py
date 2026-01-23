# Runtime hooks for core workflow execution.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, ContextManager, Optional, Protocol

from .executor.registry import ExecutorRegistry


class StreamRenderer(Protocol):
    """Protocol for rendering streaming agent output."""

    def render(self, agent, stream, message_id: str) -> str:
        """Render stream events and return the full response text."""


@dataclass
class RuntimeHooks:
    """Runtime hooks for workflow execution."""

    executor_registry: ExecutorRegistry
    stream_renderer: Optional[StreamRenderer] = None
    spinner: Optional[Callable[[str], ContextManager]] = None

    def spinner_context(self, message: str) -> ContextManager:
        """Return a spinner context manager if provided."""
        if self.spinner:
            return self.spinner(message)
        @contextmanager
        def _noop_spinner() -> ContextManager[None]:
            yield None

        return _noop_spinner()
