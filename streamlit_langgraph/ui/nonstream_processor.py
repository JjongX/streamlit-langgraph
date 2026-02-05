# Non-stream response processing for UI rendering and text extraction.

from typing import Any, Dict, List


class NonStreamProcessor:
    """Process non-streamed responses for display and extraction."""

    def process_nonstream(self, section: Any, response: Dict[str, Any]) -> str:
        """Render a non-stream response and return the final text content."""
        content = response.get("content", "")
        self._render_reasoning_blocks(section, response.get("blocks", []))
        if content:
            self._render_text(section, content)
        else:
            section.stream()
        return content

    def _render_reasoning_blocks(self, section: Any, blocks: List[Dict[str, Any]]) -> None:
        """Render reasoning blocks before final text."""
        if not blocks:
            return
        for block_data in blocks:
            if block_data.get("category") == "reasoning":
                section.update("reasoning", block_data.get("content", ""))
        section.stream()

    def _render_text(self, section: Any, content: str) -> None:
        """Render the final response text."""
        section.update("text", content)
        section.stream()
