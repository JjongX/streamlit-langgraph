# Shared conversation history management for executors.

from typing import Any, Dict, List, Optional

from ...ui.display_manager import Block, Section, DisplayManager


def extract_text_from_content(content: Any) -> str:
    """Extract text from content."""
    if not content:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get('type')
                if block_type == 'text':
                    text_parts.append(block.get('text', ''))
                elif 'text' in block:
                    text_parts.append(block.get('text', ''))
            elif isinstance(block, str):
                text_parts.append(block)
            # Handle objects with .text attribute like ResponseOutputText
            elif hasattr(block, 'text'):
                text_parts.append(str(block.text) if block.text else "")
        return ''.join(text_parts) if text_parts else ""
    
    if isinstance(content, dict):
        # Try common dict patterns
        if 'text' in content:
            return str(content.get('text', ''))
        if 'content' in content:
            return extract_text_from_content(content.get('content'))
        # Try to extract from nested structures
        return str(content) if content else ""
    
    # Handle objects with attributes
    if hasattr(content, 'content'):
        return extract_text_from_content(content.content)
    if hasattr(content, 'text'):
        return str(content.text) if content.text else ""
    
    return str(content) if content else ""


class ConversationHistoryMixin:
    """Mixin class providing conversation history management for executors."""
    
    def _init_conversation_history(self, agent):
        """Initialize conversation history tracking."""
        self._original_system_message = agent.system_message or ""
        self._history_display_manager = DisplayManager(config=None, state_manager=None)
        self._conversation_history: List[Section] = []
        self._processed_message_ids: set = set()
        self._conversation_history_mode = getattr(agent, 'conversation_history_mode', 'filtered')
    
    def _convert_message_to_blocks(self, content: Any) -> List[Block]:
        """Convert message content to Block objects."""
        blocks = []
        
        # Skip empty/None content
        if not content:
            return blocks
        
        if isinstance(content, str):
            blocks.append(self._history_display_manager.create_block("text", content=content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "input_text":
                        blocks.append(self._history_display_manager.create_block("text", content=block.get("text", "")))
                    elif block_type == "input_file":
                        blocks.append(self._history_display_manager.create_block("text", content=f"[File: {block.get('file_id', 'unknown')}]"))
                    elif block_type == "output_text":
                        blocks.append(self._history_display_manager.create_block("text", content=block.get("text", "")))
                    elif block_type == "text":
                        blocks.append(self._history_display_manager.create_block("text", content=block.get("text", "")))
        else:
            blocks.append(self._history_display_manager.create_block("text", content=str(content)))
        
        return blocks
    
    def _add_to_conversation_history(self, role: str, blocks: List[Block]) -> None:
        """Add a message to conversation history using Section."""
        # Skip if history is disabled
        if self._conversation_history_mode == "disable":
            return
        if blocks:
            section = Section(self._history_display_manager, role, blocks=blocks)
            self._conversation_history.append(section)
    
    def _get_conversation_history_sections_dict(self) -> List[Dict[str, Any]]:
        """
        Get conversation history as sections_dict format.
        
        Returns empty list if history is disabled.
        Filters out code/reasoning blocks if mode is "filtered".
        Includes all blocks if mode is "full".
        """
        # Return empty if history is disabled
        if self._conversation_history_mode == "disable":
            return []
        
        sections_dict = [section.to_dict() for section in self._conversation_history if not section.empty]
        
        # return all sections without filtering
        if self._conversation_history_mode == "full":
            return [{"role": section_dict.get("role"), "blocks": section_dict.get("blocks", [])} 
                   for section_dict in sections_dict]
        
        # filter out code and reasoning blocks to save tokens
        filtered_sections = []
        for section_dict in sections_dict:
            filtered_blocks = [
                block for block in section_dict.get("blocks", [])
                if block.get("category") not in ["code", "reasoning"]
            ]
            if filtered_blocks:
                filtered_sections.append({"role": section_dict.get("role"), "blocks": filtered_blocks})
        
        return filtered_sections
    
    def _process_file_messages(self, file_messages: Optional[List]) -> None:
        """Process file messages and add to conversation history."""
        if not file_messages:
            return
        
        for file_msg in file_messages:
            if not isinstance(file_msg, dict):
                continue
            
            role = file_msg.get("role", "user")
            content = file_msg.get("content", [])
            if role != "user" or not content:
                continue
            
            file_blocks = []
            for block in content if isinstance(content, list) else [content]:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "input_file":
                        file_blocks.append(self._history_display_manager.create_block(
                            "text", content=f"[File: {block.get('file_id', 'unknown')}]"
                        ))
                    elif block_type == "input_text":
                        file_blocks.append(self._history_display_manager.create_block(
                            "text", content=block.get("text", "")
                        ))
            
            if file_blocks:
                temp_id = f"file_msg_{len(self._conversation_history)}"
                if temp_id not in self._processed_message_ids:
                    self._add_to_conversation_history(role, file_blocks)
                    self._processed_message_ids.add(temp_id)
    
    def _update_conversation_history_from_messages(
        self, 
        messages: Optional[List[Dict[str, Any]]], 
        file_messages: Optional[List] = None
    ) -> None:
        """Update conversation history from workflow_state messages."""
        # Process regular messages
        if messages:
            for msg in messages:
                msg_id = msg.get("id")
                if not msg_id or msg_id in self._processed_message_ids:
                    continue
                
                role = msg.get("role", "")
                content = msg.get("content", "")
                if not content:
                    continue
                
                blocks = self._convert_message_to_blocks(content)
                if blocks:
                    self._add_to_conversation_history(role, blocks)
                    self._processed_message_ids.add(msg_id)
        
        # Process file messages
        self._process_file_messages(file_messages)

