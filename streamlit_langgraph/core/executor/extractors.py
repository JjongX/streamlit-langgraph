# Pure extraction helpers for LLM responses.

from typing import Any, List

from langchain_core.messages import AIMessage

from ...utils.text_extraction import extract_text_from_content


def extract_langchain_text(out: Any) -> str:
    """Extract text content from LangChain agent output."""
    if isinstance(out, dict):
        if out.get("output"):
            return extract_text_from_content(out["output"])

        messages = out.get("messages") or []
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return extract_text_from_content(msg.content) if msg.content else ""

            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return extract_text_from_content(last_message.content)
            return str(last_message) if last_message else ""

    if isinstance(out, str):
        return out

    if hasattr(out, "content"):
        return extract_text_from_content(out.content)

    return str(out) if out else ""


def extract_langchain_reasoning(out: Any) -> List[str]:
    """Extract reasoning texts from LangChain agent output."""
    messages = []
    if isinstance(out, dict):
        messages = out.get("messages") or []
    elif isinstance(out, AIMessage):
        messages = [out]

    texts: List[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "reasoning":
                if block.get("reasoning"):
                    texts.append(str(block.get("reasoning")).strip())
                summary = block.get("summary")
                if isinstance(summary, list):
                    texts.extend(
                        str(item.get("text")).strip()
                        for item in summary
                        if isinstance(item, dict) and item.get("text")
                    )
                elif isinstance(summary, dict) and summary.get("text"):
                    texts.append(str(summary.get("text")).strip())
            elif block_type == "reasoning_summary_text":
                if block.get("text"):
                    texts.append(str(block.get("text")).strip())

    return [text for text in texts if text]


def extract_response_api_text(response: Any) -> str:
    """Extract text content from OpenAI Response API response."""
    if not response:
        return ""

    text_parts = []

    def get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    output_items = None
    if hasattr(response, "output") and response.output:
        output_items = response.output
    elif hasattr(response, "items") and response.items:
        output_items = response.items

    if output_items:
        for item in output_items:
            item_type = get_value(item, "type")
            if item_type == "output_text":
                text = get_value(item, "text") or extract_text_from_content(get_value(item, "content"))
                if text:
                    text_parts.append(str(text))
            elif item_type == "code_interpreter_call":
                code = get_value(item, "code") or get_value(item, "input")
                if code:
                    text_parts.append(f"\n\n```python\n{code}\n```\n\n")
                output = get_value(item, "output")
                if output:
                    if isinstance(output, list):
                        for output_item in output:
                            output_type = get_value(output_item, "type")
                            if output_type == "text":
                                out_text = get_value(output_item, "text")
                                if out_text:
                                    text_parts.append(str(out_text))
                            elif output_type == "image":
                                text_parts.append("\n[Code generated an image]\n")
                    else:
                        text_parts.append(str(output))
            elif item_type == "message":
                content_blocks = get_value(item, "content", [])
                for block in content_blocks:
                    block_text = get_value(block, "text")
                    if block_text:
                        text_parts.append(str(block_text))

    if not text_parts:
        output_text = getattr(response, "output_text", None)
        if output_text:
            extracted = extract_text_from_content(output_text)
            if extracted:
                text_parts.append(extracted)

    if not text_parts:
        text = get_value(response, "text") or extract_text_from_content(get_value(response, "content"))
        if text:
            text_parts.append(str(text))

    return "".join(text_parts) if text_parts else str(response) if response else ""


def extract_reasoning_blocks(response: Any) -> List[str]:
    """Extract reasoning texts from a Responses API object."""
    if not response:
        return []

    def val(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    summary_texts: List[str] = []
    reasoning = val(response, "reasoning")
    summary = val(reasoning, "summary") if reasoning else None
    if summary and not isinstance(summary, str):
        text = val(summary, "text")
        if text:
            summary_texts.append(str(text).strip())

    output_items = val(response, "output") or val(response, "items") or []
    for item in output_items:
        item_type = val(item, "type")
        if item_type == "reasoning":
            summary = val(item, "summary")
            if isinstance(summary, list):
                texts = [str(val(s, "text")).strip() for s in summary if val(s, "text")]
                if texts:
                    summary_texts.append("\n\n".join(t for t in texts if t))
            elif summary and not isinstance(summary, str):
                text = val(summary, "text")
                if text:
                    summary_texts.append(str(text).strip())
        elif item_type == "reasoning_summary_text":
            text = val(item, "text")
            if text:
                summary_texts.append(str(text).strip())

    return summary_texts
