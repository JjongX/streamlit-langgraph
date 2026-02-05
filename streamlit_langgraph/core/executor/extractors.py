# Pure extraction helpers for LLM responses.

from typing import Any, List, Tuple

from langchain_core.messages import AIMessage

from ...utils.text_extraction import extract_text_from_content


def _extract_langchain_text_base(out: Any) -> Tuple[str, Any]:
    """Return (text, response_msg) from LangChain output."""
    text = ""
    response_msg = None
    
    if isinstance(out, dict):
        if out.get("output"):
            text = extract_text_from_content(out["output"])
        else:
            messages = out.get("messages") or []
            if messages:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        text = extract_text_from_content(msg.content) if msg.content else ""
                        response_msg = msg
                        break
                
                if not text:
                    last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        text = extract_text_from_content(last_message.content)
                    else:
                        text = str(last_message) if last_message else ""
                    response_msg = last_message

    elif isinstance(out, str):
        text = out

    elif hasattr(out, "content"):
        text = extract_text_from_content(out.content)
        response_msg = out

    else:
        text = str(out) if out else ""
    
    return (text, response_msg)


def extract_langchain_text(out: Any) -> str:
    """Extract plain text from LangChain output."""
    text, _ = _extract_langchain_text_base(out)
    return text


def extract_langchain_text_with_gemini_extras(out: Any) -> str:
    """Gemini-only: append code execution + grounding citations when present."""
    text, response_msg = _extract_langchain_text_base(out)
    if response_msg:
        text = _append_gemini_extras(text, response_msg)
    return text


def _append_gemini_extras(text: str, response_msg: Any) -> str:
    """Append Gemini code execution and grounding info to text."""
    extras = []
    
    # Check for code execution results
    code_results = extract_gemini_code_execution(response_msg)
    if code_results:
        code_text = format_gemini_code_execution(code_results)
        if code_text:
            extras.append(code_text)
    
    # Check for grounding metadata (Google Search results)
    grounding = extract_gemini_grounding_metadata(response_msg)
    if grounding and grounding.get("grounding_chunks"):
        # Don't duplicate text, just add citations
        chunks = grounding.get("grounding_chunks", [])
        if chunks:
            citations = []
            for i, chunk in enumerate(chunks):
                uri = chunk.get("uri", "")
                title = chunk.get("title", f"Source {i+1}")
                if uri:
                    citations.append(f"[{i+1}] [{title}]({uri})")
            if citations:
                extras.append("**Sources:**\n" + "\n".join(citations))
    
    if extras:
        return text + "\n\n" + "\n\n".join(extras)
    
    return text


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


# Gemini-specific extractors

def extract_gemini_grounding_metadata(response: Any) -> dict:
    """Extract Google Search grounding metadata from a Gemini AIMessage."""
    if not response:
        return {}
    
    # Get response_metadata from AIMessage
    metadata = None
    if hasattr(response, "response_metadata"):
        metadata = response.response_metadata
    elif isinstance(response, dict):
        metadata = response.get("response_metadata", {})
    
    if not metadata:
        return {}
    
    grounding_metadata = metadata.get("grounding_metadata", {})
    if not grounding_metadata:
        return {}
    
    result = {}
    
    # Extract web search queries
    web_queries = grounding_metadata.get("web_search_queries", [])
    if web_queries:
        result["web_search_queries"] = web_queries
    
    # Extract grounding chunks (citations)
    chunks = grounding_metadata.get("grounding_chunks", [])
    if chunks:
        result["grounding_chunks"] = [
            {"uri": chunk.get("web", {}).get("uri", ""), "title": chunk.get("web", {}).get("title", "")}
            for chunk in chunks if chunk.get("web")
        ]
    
    # Extract grounding supports (text segments mapped to sources)
    supports = grounding_metadata.get("grounding_supports", [])
    if supports:
        result["grounding_supports"] = supports
    
    return result


def extract_gemini_code_execution(response: Any) -> List[dict]:
    """Extract code execution blocks from a Gemini AIMessage."""
    if not response:
        return []
    
    # Get content from AIMessage
    content = None
    if hasattr(response, "content"):
        content = response.content
    elif isinstance(response, dict):
        content = response.get("content")
    
    if not isinstance(content, list):
        return []
    
    results = []
    for block in content:
        if not isinstance(block, dict):
            continue
        
        # Look for Gemini 2.5/3 content blocks
        if block.get("type") == "executable_code" or "executable_code" in block:
            exec_code = block.get("executable_code", block)
            code = exec_code.get("code", "")
            if code:
                results.append({
                    "type": "code",
                    "code": code,
                    "language": exec_code.get("language", "python")
                })
        
        if block.get("type") == "code_execution_result" or "code_execution_result" in block:
            exec_result = block.get("code_execution_result", block)
            output = exec_result.get("output", "")
            if output:
                results.append({
                    "type": "output",
                    "output": output
                })

        # LangChain server tool blocks (Gemini code execution).
        if block.get("type") == "server_tool_call" and block.get("name") == "code_interpreter":
            args = block.get("args", {}) if isinstance(block.get("args"), dict) else {}
            code = args.get("code", "") or args.get("input", "")
            if code:
                results.append({
                    "type": "code",
                    "code": code,
                    "language": "python",
                })

        if block.get("type") == "server_tool_result" and block.get("name") == "code_interpreter":
            # Gemini often returns a plain string under `output`
            output = block.get("output", "")
            if isinstance(output, str) and output:
                results.append({
                    "type": "output",
                    "output": output,
                })
    
    return results


def format_gemini_grounding_citations(text: str, grounding_metadata: dict) -> str:
    """Append a Sources section from grounding metadata."""
    if not grounding_metadata or not grounding_metadata.get("grounding_chunks"):
        return text
    
    chunks = grounding_metadata.get("grounding_chunks", [])
    if not chunks:
        return text
    
    # Build citation list
    citations = []
    for i, chunk in enumerate(chunks):
        uri = chunk.get("uri", "")
        title = chunk.get("title", f"Source {i+1}")
        if uri:
            citations.append(f"[{i+1}] [{title}]({uri})")
    
    if citations:
        return f"{text}\n\n**Sources:**\n" + "\n".join(citations)
    
    return text


def format_gemini_code_execution(code_results: List[dict]) -> str:
    """Format code execution blocks as markdown."""
    if not code_results:
        return ""
    
    parts = []
    for result in code_results:
        if result.get("type") == "code":
            lang = result.get("language", "python")
            code = result.get("code", "")
            parts.append(f"```{lang}\n{code}\n```")
        elif result.get("type") == "output":
            output = result.get("output", "")
            parts.append(f"**Output:**\n```\n{output}\n```")
    
    return "\n\n".join(parts)
