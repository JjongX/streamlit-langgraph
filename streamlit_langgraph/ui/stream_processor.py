# Stream processor for handling different streaming formats.

import base64
from typing import Any, Dict, Iterable, List

from langchain_core.messages import AIMessage

from ..core.executor.extractors import (
    extract_gemini_grounding_metadata,
    format_gemini_grounding_citations,
)


class StreamProcessor:
    """
    Processes streaming responses from different sources.

    Handles:
    - OpenAI Responses API stream events
    - LangChain stream_mode="messages" format (tuple events)
    - LangChain stream_mode="updates" format (dict events)
    """

    def __init__(self, client=None, container_id=None):
        self._client = client
        self._container_id = container_id

    def process_stream(self, section, stream_iter) -> str:
        """Process a stream iterator and return the accumulated text response."""
        stream_type = None
        full_response = ""

        for event in stream_iter:
            if stream_type is None:
                stream_type = self._detect_stream_type(event)

            if stream_type == "responses_api":
                delta = self._process_responses_api_stream(event, section)
            elif stream_type == "langchain_messages":
                token, _ = event
                delta = self._process_langchain_message_token(token, section)
            else:  # langchain_updates
                delta = self._process_langchain_update_event(event, section)

            if delta:
                full_response += delta

        return full_response

    def _detect_stream_type(self, event: Any) -> str:
        """Detect stream event type from the first event."""
        if isinstance(event, tuple) and len(event) == 2:
            return "langchain_messages"
        if isinstance(event, dict):
            return "langchain_updates"
        if hasattr(event, "type"):
            return "responses_api"
        raise RuntimeError(f"Unsupported stream event format: {type(event).__name__}")

    def _process_langchain_update_event(self, event: Dict[str, Any], section) -> str:
        """Process a LangChain updates-format event dict."""
        if not isinstance(event, dict):
            return ""

        deltas: List[str] = []
        for update_value in event.values():
            for token in self._iter_ai_messages(update_value):
                delta = self._process_langchain_message_token(token, section)
                if delta:
                    deltas.append(delta)
        return "".join(deltas)

    def _iter_ai_messages(self, value: Any) -> Iterable[AIMessage]:
        """Yield AIMessage objects found in LangChain update payloads."""
        if isinstance(value, AIMessage):
            yield value
            return
        if isinstance(value, list):
            for item in value:
                yield from self._iter_ai_messages(item)
            return
        if isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        yield msg
            message = value.get("message")
            if isinstance(message, AIMessage):
                yield message

    def _process_responses_api_stream(self, event, section) -> str:
        """Process a single streaming event from OpenAI Responses API."""
        if event.type == "response.output_text.delta":
            section.update("text", event.delta)
            section.stream()
            return event.delta
        if event.type == "response.reasoning_summary_text.delta":
            summary_delta = getattr(event, "delta", "")
            if summary_delta:
                section.update("reasoning", summary_delta)
                section.stream()
            return ""
        if event.type == "response.code_interpreter_call_code.delta":
            section.update("code", event.delta)
            section.stream()
            return ""
        if event.type == "response.image_generation_call.partial_image":
            image_bytes = base64.b64decode(event.partial_image_b64)
            item_id = getattr(event, "item_id", None)
            filename = f"{item_id}.{getattr(event, 'output_format', 'png')}" if item_id else "image.png"
            section.update("generated_image", image_bytes, filename=filename, file_id=item_id)
            section.stream()
            return ""
        if event.type == "response.output_text.annotation.added":
            annotation = event.annotation
            if annotation["type"] == "container_file_citation":
                self._render_container_file_annotation(
                    section,
                    file_id=annotation["file_id"],
                    filename=annotation["filename"],
                    container_id=self._container_id,
                )
        return ""

    def _process_langchain_message_token(self, token, section) -> str:
        """Process a single LangChain stream_mode='messages' token."""
        if not isinstance(token, AIMessage):
            return ""

        content_blocks = self._resolve_content_blocks(token)
        if content_blocks is None:
            return ""
        if isinstance(content_blocks, str):
            section.update("text", content_blocks)
            section.stream()
            return content_blocks

        text_parts: List[str] = []
        for block in content_blocks:
            self._process_langchain_content_block(block, section, text_parts)

        self._gemini_grounding(token, section)
        return self._flush_text_parts(section, text_parts)

    def _resolve_content_blocks(self, token: AIMessage) -> Any:
        """Resolve AIMessage content into a list of blocks or plain text."""
        content_blocks = getattr(token, "content_blocks", None)
        if content_blocks is not None:
            return content_blocks

        content = token.content
        if isinstance(content, list):
            return content
        if isinstance(content, str) and content:
            return content
        return None

    def _process_langchain_content_block(self, block: Any, section, text_parts: List[str]) -> None:
        """Process one LangChain content block."""
        if isinstance(block, str):
            text_parts.append(block)
            return
        if not isinstance(block, dict):
            return

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
            self._message_annotations(block, section)
            return
        if block_type in ("reasoning", "reasoning_summary_text"):
            reasoning_text = self._extract_reasoning_block_text(block)
            if reasoning_text:
                section.update("reasoning", reasoning_text)
                section.stream()
            return
        if block_type == "server_tool_call":
            self._message_tool_call(block, section)
            return
        if block_type == "server_tool_result":
            self._message_tool_result(block, section, text_parts)
            return
        if block_type == "executable_code":
            self._gemini_executable_code(block, section)
            return
        if block_type == "code_execution_result":
            self._gemini_code_execution_result(block, section)

    @staticmethod
    def _flush_text_parts(section, text_parts: List[str]) -> str:
        """Append and render text delta from collected parts."""
        if not text_parts:
            return ""
        delta = "".join(text_parts)
        if not delta:
            return ""
        section.update("text", delta)
        section.stream()
        return delta

    def _render_container_file_annotation(
        self,
        section,
        file_id: str,
        filename: str,
        container_id: str,
    ) -> None:
        """Fetch and render file citation output from a container."""
        if not (file_id and container_id and self._client):
            return
        file_content = self._client.containers.files.content.retrieve(
            file_id=file_id,
            container_id=container_id,
        )
        file_bytes = file_content.read()
        if not file_bytes:
            return
        is_image = filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
        if is_image:
            section.update("image", file_bytes, filename=filename, file_id=file_id)
        section.update("download", file_bytes, filename=filename, file_id=file_id)
        section.stream()

    def _gemini_executable_code(self, block, section) -> None:
        """Render executable_code."""
        exec_code = block.get("executable_code", block)
        if isinstance(exec_code, dict):
            code = exec_code.get("code", "")
        else:
            code = getattr(exec_code, "code", "")
        if code:
            section.update("code", code)
            section.stream()

    def _gemini_code_execution_result(self, block, section) -> None:
        """Render code_execution_result."""
        exec_result = block.get("code_execution_result", block)
        if isinstance(exec_result, dict):
            output = exec_result.get("output", "")
        else:
            output = getattr(exec_result, "output", "")
        if output:
            section.update("text", f"\n**Output:**\n```\n{output}\n```\n")
            section.stream()

    def _gemini_grounding(self, token, section) -> None:
        """Render grounding citations, if present."""
        grounding_metadata = extract_gemini_grounding_metadata(token)
        citations_text = format_gemini_grounding_citations("", grounding_metadata).strip()
        if citations_text:
            section.update("text", "\n\n" + citations_text)
            section.stream()

    def _message_annotations(self, block, section):
        """Process container_file_citation annotations in text blocks."""
        annotations = block.get("annotations", [])
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            annotation_value = (
                annotation.get("value", {})
                if annotation.get("type") == "non_standard_annotation"
                else annotation
            )
            if annotation_value.get("type") == "container_file_citation":
                self._render_container_file_annotation(
                    section,
                    file_id=annotation_value.get("file_id", ""),
                    filename=annotation_value.get("filename", ""),
                    container_id=annotation_value.get("container_id", ""),
                )

    def _message_tool_call(self, block, section):
        """Process server_tool_call blocks (code_interpreter, etc.)."""
        if block.get("name") != "code_interpreter":
            return
        args = block.get("args", {})
        code = args.get("code", "") or args.get("input", "")
        if not code:
            return
        if (not section.empty and section.last_block.category == "code" and section.last_block.content):
            previous_code = section.last_block.content
            if code.startswith(previous_code):
                delta = code[len(previous_code):]
                if delta:
                    section.update("code", delta)
                    section.stream()
                    return
            section.blocks[-1] = section.display_manager.create_block("code", code)
            section.stream()
            return
        section.update("code", code)
        section.stream()

    def _message_tool_result(self, block, section, text_parts):
        """Process server_tool_result blocks (outputs from tool execution)."""
        outputs = block.get("output", block.get("outputs", []))
        if isinstance(outputs, str) and outputs:
            if block.get("name") == "code_interpreter":
                text_parts.append(f"\n**Output:**\n```\n{outputs}\n```\n")
            else:
                text_parts.append(str(outputs))
            return
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs else []

        for output in outputs:
            if not isinstance(output, dict):
                continue
            output_type = output.get("type", "")
            if output_type == "text":
                output_text = output.get("text", "")
                if output_text:
                    if block.get("name") == "code_interpreter":
                        text_parts.append(f"\n**Output:**\n```\n{output_text}\n```\n")
                    else:
                        text_parts.append(output_text)
            elif output_type == "image":
                image_data = output.get("image", {})
                if isinstance(image_data, dict):
                    image_data_str = image_data.get("data", "") or image_data.get("base64", "")
                    if image_data_str:
                        image_bytes = base64.b64decode(image_data_str)
                        filename = f"code_output_{block.get('tool_call_id', '')}.png"
                        section.update("image", image_bytes, filename=filename)
                        section.update("download", image_bytes, filename=filename)
                        section.stream()

    def _extract_reasoning_block_text(self, block: dict) -> str:
        """Extract reasoning text from a content block."""
        if not isinstance(block, dict):
            return ""
        if block.get("type") == "reasoning_summary_text":
            text = block.get("text", "")
            return str(text) if text is not None else ""
        if block.get("type") == "reasoning":
            if block.get("reasoning") is not None:
                return str(block.get("reasoning"))
            if block.get("text") is not None:
                return str(block.get("text"))
            summary = block.get("summary")
            if isinstance(summary, list):
                texts = [
                    str(item.get("text"))
                    for item in summary
                    if isinstance(item, dict) and item.get("text")
                ]
                return "\n\n".join(t for t in texts if t is not None)
            if isinstance(summary, dict) and summary.get("text"):
                return str(summary.get("text"))
        return ""
