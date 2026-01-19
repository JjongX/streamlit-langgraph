"""Helpers for extracting text from mixed content payloads."""

from typing import Any


def extract_text_from_content(content: Any) -> str:
    """Extract text from content."""
    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
            elif hasattr(block, "text") and block.text:
                text_parts.append(str(block.text))
        return "".join(text_parts)

    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        if "content" in content:
            return extract_text_from_content(content.get("content"))
        return str(content)

    if hasattr(content, "content"):
        return extract_text_from_content(content.content)
    if hasattr(content, "text") and content.text:
        return str(content.text)

    return str(content) if content else ""
