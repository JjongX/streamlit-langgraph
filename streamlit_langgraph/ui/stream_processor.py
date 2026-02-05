# Stream processor for handling different streaming formats.

import base64

from langchain_core.messages import AIMessage


class StreamProcessor:
    """
    Processes streaming responses from different sources.
    
    Handles:
    - OpenAI Responses API stream events
    - LangChain stream_mode="messages" format (tokens)
    """
    
    def __init__(self, client=None, container_id=None):
        self._client = client
        self._container_id = container_id
    
    def process_stream(self, section, stream_iter) -> str:
        """
        Main entry point for processing streaming responses.
        
        Detects the stream format and routes to appropriate handler:
        - Responses API: Direct OpenAI Responses API stream events
        - LangChain: LangChain stream_mode="messages" format (tuples) or "updates" format (dicts)
        
        Args:
            section: Display section to update
            stream_iter: Stream iterator (format varies by source)
            
        Returns:
            Full accumulated response text
        """
        stream_type = None
        full_response = ""
        
        for event in stream_iter:
            if stream_type is None:
                if isinstance(event, tuple) and len(event) == 2:
                    stream_type = 'langchain_messages'
                elif hasattr(event, 'type'):
                    stream_type = 'responses_api'
                else:
                    raise ValueError(f"Unknown stream type: {event}")
            
            if stream_type == 'responses_api':
                delta = self._process_responses_api_stream(event, section)
                if delta:
                    full_response += delta
            elif stream_type == 'langchain_messages':
                token, _ = event
                delta = self._process_langchain_message_token(token, section)
                if delta:
                    full_response += delta
        
        return full_response
    
    def _process_responses_api_stream(self, event, section) -> str:
        """
        Process a single streaming event from OpenAI Responses API.

        Args:
            event: Responses API stream event object
            section: Display section to update
            
        Returns:
            Delta text to append to response
        """
        if event.type == "response.output_text.delta":
            section.update("text", event.delta)
            section.stream()
            return event.delta
        elif event.type == "response.reasoning_summary_text.delta":
            # Handle streaming reasoning summary text
            summary_delta = getattr(event, 'delta', '')
            if summary_delta:
                section.update("reasoning", summary_delta)
                section.stream()
        elif event.type == "response.code_interpreter_call_code.delta":
            section.update("code", event.delta)
            section.stream()
        elif event.type == "response.image_generation_call.partial_image":
            image_bytes = base64.b64decode(event.partial_image_b64)
            item_id = getattr(event, 'item_id', None)
            filename = f"{item_id}.{getattr(event, 'output_format', 'png')}" if item_id else "image.png"
            section.update("generated_image", image_bytes, filename=filename, file_id=item_id)
            section.stream()
        elif event.type == "response.output_text.annotation.added":
            annotation = event.annotation
            if annotation["type"] == "container_file_citation":
                file_id = annotation["file_id"]
                filename = annotation["filename"]
                file_bytes = None
                if self._client and self._container_id:
                    file_content = self._client.containers.files.content.retrieve(
                        file_id=file_id, container_id=self._container_id
                    )
                    file_bytes = file_content.read()
                if file_bytes:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                        section.update("image", file_bytes, filename=filename, file_id=file_id)
                        section.update("download", file_bytes, filename=filename, file_id=file_id)
                        section.stream()
                    else:
                        section.update("download", file_bytes, filename=filename, file_id=file_id)
                        section.stream()
                        
        return ""
    
    def _process_langchain_message_token(self, token, section) -> str:
        """Process a single LangChain stream_mode='messages' token."""
        if not isinstance(token, AIMessage):
            return ""
        
        # Handle content_blocks if available
        content_blocks = getattr(token, 'content_blocks', None)
        
        # Gemini may emit list content blocks directly.
        if content_blocks is None:
            content = token.content
            if isinstance(content, list):
                content_blocks = content
            elif isinstance(content, str) and content:
                # Simple text content
                section.update("text", content)
                section.stream()
                return content
        
        if not content_blocks:
            return ""
        
        text_parts = []
        for block in content_blocks:
            if not isinstance(block, dict):
                if isinstance(block, str):
                    text_parts.append(block)
                continue
                
            block_type = block.get('type')
            if block_type == 'text':
                text = block.get('text', '')
                if text:
                    text_parts.append(text)
                self._message_annotations(block, section)
            elif block_type in ('reasoning', 'reasoning_summary_text'):
                reasoning_text = self._extract_reasoning_block_text(block)
                if reasoning_text:
                    section.update("reasoning", reasoning_text)
                    section.stream()
            elif block_type == 'server_tool_call':
                self._message_tool_call(block, section)
            elif block_type == 'server_tool_result':
                self._message_tool_result(block, section, text_parts)
            # Gemini-only (LangChain): code execution blocks
            elif block_type == 'executable_code':
                self._gemini_executable_code(block, section)
            elif block_type == 'code_execution_result':
                self._gemini_code_execution_result(block, section)
        
        self._gemini_grounding(token, section)
        
        if text_parts:
            delta = ''.join(text_parts)
            if delta:
                section.update("text", delta)
                section.stream()
                return delta
        
        return ""

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
        response_metadata = getattr(token, "response_metadata", None)
        if not response_metadata:
            return
        grounding_metadata = response_metadata.get("grounding_metadata", {})
        if not grounding_metadata:
            return
        chunks = grounding_metadata.get("grounding_chunks", [])
        if not chunks:
            return
        citations = []
        for i, chunk in enumerate(chunks):
            web = chunk.get("web", {})
            uri = web.get("uri", "")
            title = web.get("title", f"Source {i+1}")
            if uri:
                citations.append(f"[{i+1}] [{title}]({uri})")
        if citations:
            section.update("text", "\n\n**Sources:**\n" + "\n".join(citations))
            section.stream()

    def _message_annotations(self, block, section):
        """Process container_file_citation annotations in text blocks."""
        annotations = block.get('annotations', [])
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            
            annotation_value = annotation.get('value', {}) if annotation.get('type') == 'non_standard_annotation' else annotation
            
            if annotation_value.get('type') == 'container_file_citation':
                file_id = annotation_value.get('file_id', '')
                filename = annotation_value.get('filename', '')
                container_id = annotation_value.get('container_id', '')
                
                if file_id and container_id and self._client:
                    file_content = self._client.containers.files.content.retrieve(
                        file_id=file_id, container_id=container_id
                    )
                    file_bytes = file_content.read()
                    if file_bytes:
                        is_image = filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
                        if is_image:
                            section.update("image", file_bytes, filename=filename)
                        section.update("download", file_bytes, filename=filename, file_id=file_id)
                        section.stream()
    
    def _message_tool_call(self, block, section):
        """Process server_tool_call blocks (code_interpreter, etc.)."""
        if block.get('name') == 'code_interpreter':
            args = block.get('args', {})
            code = args.get('code', '') or args.get('input', '')
            if code:
                # It is for stream delta updates but not supported currently
                # Read more docs and langcahin's scripts to validate that it is not supported
                if (not section.empty and 
                    section.last_block.category == "code" and 
                    section.last_block.content):
                    previous_code = section.last_block.content
                    if code.startswith(previous_code):
                        delta = code[len(previous_code):]
                        if delta:
                            section.update("code", delta)
                            section.stream()
                            return
                    section.blocks[-1] = section.display_manager.create_block("code", code)
                    section.stream()
                else:
                    section.update("code", code)
                    section.stream()
    
    def _message_tool_result(self, block, section, text_parts):
        """Process server_tool_result blocks (outputs from tool execution)."""
        outputs = block.get('output', block.get('outputs', []))
        # Gemini code execution often returns a plain string under `output`
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
            
            output_type = output.get('type', '')
            if output_type == 'text':
                output_text = output.get('text', '')
                if output_text:
                    # For code execution, render tool output clearly as a fenced block
                    if block.get("name") == "code_interpreter":
                        text_parts.append(f"\n**Output:**\n```\n{output_text}\n```\n")
                    else:
                        text_parts.append(output_text)
            elif output_type == 'image':
                image_data = output.get('image', {})
                if isinstance(image_data, dict):
                    image_data_str = image_data.get('data', '') or image_data.get('base64', '')
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
            # Streaming emits reasoning deltas under the "reasoning" key.
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

    
