# Stream processor for handling different streaming formats.

import base64

from langchain_core.messages import AIMessage


class StreamProcessor:
    """
    Processes streaming responses from different sources.
    
    Handles:
    - OpenAI Responses API stream events
    - LangChain stream_mode="messages" format (tokens)
    - LangChain stream_mode="updates" format (events)
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
        events_list = []
        stream_type = None
        full_response = ""
        
        for event in stream_iter:
            events_list.append(event)
            
            if stream_type is None:
                if isinstance(event, tuple) and len(event) == 2:
                    stream_type = 'langchain_messages'
                elif hasattr(event, 'type'):
                    stream_type = 'responses_api'
                elif isinstance(event, dict):
                    stream_type = 'langchain_updates'
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
            elif stream_type == 'langchain_updates':
                partial_response = self._process_langchain_updates_event(event, section, full_response)
                if partial_response and len(partial_response) > len(full_response):
                    full_response = partial_response
        
        return full_response
    
    def _process_langchain_message_token(self, token, section) -> str:
        """
        Process a single token from LangChain stream_mode="messages".
        
        This method handles LangChain's stream_mode="messages" format, which returns
        (token, metadata) tuples where token is an AIMessage with content_blocks attribute.
        
        Handles various content block types:
        - text: Regular text content
        - server_tool_call: Tool calls (code_interpreter, file_search, etc.)
        - server_tool_result: Tool execution results (text, images, etc.)
        
        Args:
            token: AIMessage with content_blocks attribute from stream_mode="messages"
            section: Display section to update
            
        Returns:
            Delta text to append to response
        """
        if not isinstance(token, AIMessage) or not hasattr(token, 'content_blocks') or not token.content_blocks:
            return ""
        
        text_parts = []
        for block in token.content_blocks:
            if not isinstance(block, dict):
                continue
                
            block_type = block.get('type')
            if block_type == 'text':
                text = block.get('text', '')
                if text:
                    text_parts.append(text)
                self._message_annotations(block, section)
            elif block_type == 'server_tool_call':
                self._message_tool_call(block, section)
            elif block_type == 'server_tool_result':
                self._message_tool_result(block, section, text_parts)
        
        if text_parts:
            delta = ''.join(text_parts)
            if delta:
                section.update("text", delta)
                section.stream()
                return delta
        
        return ""
    
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
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs else []
        
        for output in outputs:
            if not isinstance(output, dict):
                continue
            
            output_type = output.get('type', '')
            if output_type == 'text':
                output_text = output.get('text', '')
                if output_text:
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
    
    def _process_langchain_updates_event(self, event: dict, section, accumulated_content: str = "") -> str:
        """
        Process a single LangChain/LangGraph stream event incrementally.
        
        LangChain streams return dictionaries with node names as keys,
        containing state updates with messages.
        
        Args:
            event: Single stream event (dict)
            section: Display section to update
            accumulated_content: Previously accumulated content
            
        Returns:
            Updated full response content
        """
        if not isinstance(event, dict):
            return accumulated_content
        
        for _, state_update in event.items():
            if isinstance(state_update, dict):
                if 'messages' in state_update:
                    messages = state_update['messages']
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, 'content') and msg.content:
                                if isinstance(msg.content, str):
                                    new_content = msg.content
                                    if new_content.startswith(accumulated_content):
                                        delta = new_content[len(accumulated_content):]
                                        if delta:
                                            section.update("text", delta)
                                            section.stream()
                                            return new_content
                                    else:
                                        section.update("text", new_content)
                                        section.stream()
                                        return new_content
                                elif isinstance(msg.content, list):
                                    text_parts = []
                                    for block in msg.content:
                                        if isinstance(block, dict) and block.get('type') == 'text':
                                            text_parts.append(block.get('text', ''))
                                        elif isinstance(block, str):
                                            text_parts.append(block)
                                    if text_parts:
                                        new_content = ''.join(text_parts)
                                        if new_content != accumulated_content:
                                            if new_content.startswith(accumulated_content):
                                                delta = new_content[len(accumulated_content):]
                                                if delta:
                                                    section.update("text", delta)
                                                    section.stream()
                                            else:
                                                section.update("text", new_content)
                                                section.stream()
                                            return new_content
                elif 'output' in state_update:
                    output = state_update['output']
                    if output and str(output) != accumulated_content:
                        section.update("text", str(output))
                        section.stream()
                        return str(output)
        
        return accumulated_content
    
    def _process_responses_api_stream(self, event, section) -> str:
        """
        Process a single streaming event from OpenAI Responses API.

        This method handles the Responses API stream events directly.
        It is not currently used in the package, but is maintained for future use.

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
        elif event.type == "response.code_interpreter_call_code.delta":
            section.update("code", event.delta)
            section.stream()
        elif event.type == "response.image_generation_call.partial_image":
            image_bytes = base64.b64decode(event.partial_image_b64)
            filename = f"{getattr(event, 'item_id', 'image')}.{getattr(event, 'output_format', 'png')}"
            section.update("image", image_bytes, filename=filename, file_id=getattr(event, 'item_id', None))
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
    
    def _extract_final_content(self, events_list, stream_type, section) -> str:
        """
        Extract final content from stream events when streaming didn't yield content.
        
        Args:
            events_list: List of all stream events
            stream_type: Detected stream type ('responses_api', 'langchain_messages', 'langchain_updates')
            section: Display section to update
            
        Returns:
            Extracted final content text
        """
        if not events_list:
            return ""
        
        last_event = events_list[-1]
        full_response = ""
        
        if stream_type == 'langchain_messages' and isinstance(last_event, tuple):
            # Extract from last token in messages mode
            token, metadata = last_event
            if isinstance(token, AIMessage):
                if hasattr(token, 'content_blocks') and token.content_blocks:
                    text_parts = []
                    for block in token.content_blocks:
                        if isinstance(block, dict):
                            block_type = block.get('type')
                            if block_type == 'text':
                                text_parts.append(block.get('text', ''))
                            elif block_type == 'server_tool_call':
                                tool_name = block.get('name', '')
                                if tool_name == 'code_interpreter':
                                    args = block.get('args', {})
                                    code = args.get('code', '') or args.get('input', '')
                                    if code:
                                        if (not section.empty and 
                                            section.last_block.category == "code" and 
                                            section.last_block.content):
                                            previous_code = section.last_block.content
                                            if code.startswith(previous_code):
                                                delta = code[len(previous_code):]
                                                if delta:
                                                    section.update("code", delta)
                                                    section.stream()
                                            elif previous_code.startswith(code):
                                                section.blocks[-1] = section.display_manager.create_block("code", code)
                                                section.stream()
                                            else:
                                                section.blocks[-1] = section.display_manager.create_block("code", code)
                                                section.stream()
                                        else:
                                            section.update("code", code)
                                            section.stream()
                            elif block_type == 'server_tool_result':
                                tool_call_id = block.get('tool_call_id', '')
                                print(f"[DEBUG] _extract_final_content - server_tool_result block detected. tool_call_id: {tool_call_id}")
                                print(f"[DEBUG] _extract_final_content - Full block keys: {list(block.keys())}")
                                outputs = block.get('output', block.get('outputs', []))
                                print(f"[DEBUG] _extract_final_content - Outputs type: {type(outputs)}, Outputs: {outputs}")
                                if not isinstance(outputs, list):
                                    outputs = [outputs] if outputs else []
                                for idx, output in enumerate(outputs):
                                    print(f"[DEBUG] _extract_final_content - Output {idx}: type={type(output)}, content={output}")
                                    if isinstance(output, dict):
                                        output_type = output.get('type', '')
                                        print(f"[DEBUG] _extract_final_content - Output type: {output_type}")
                                        if output_type == 'text':
                                            output_text = output.get('text', '')
                                            if output_text:
                                                text_parts.append(output_text)
                                        elif output_type == 'image':
                                            print(f"[DEBUG] _extract_final_content - Image output detected!")
                                            image_data = output.get('image', {})
                                            print(f"[DEBUG] _extract_final_content - image_data type: {type(image_data)}")
                                            image_data_str = None
                                            if isinstance(image_data, dict):
                                                image_data_str = image_data.get('data', '') or image_data.get('base64', '')
                                            elif isinstance(image_data, str):
                                                image_data_str = image_data
                                            if image_data_str:
                                                image_bytes = base64.b64decode(image_data_str)
                                                filename = f"code_output_{tool_call_id}.png"
                                                print(f"[DEBUG] _extract_final_content - Creating image block")
                                                section.update("image", image_bytes, filename=filename)
                                                section.update("download", image_bytes, filename=filename)
                                                section.stream()
                    if text_parts:
                        full_response = ''.join(text_parts)
                        if full_response:
                            section.update("text", full_response)
                            section.stream()
        
        return full_response

