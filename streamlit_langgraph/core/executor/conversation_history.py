# Shared conversation history management for executors.

from typing import Any, Dict, List

from ...ui.display_manager import Block, Section, DisplayManager
from ...utils.text_extraction import extract_text_from_content


class ConversationHistoryMixin:
    """Mixin class providing conversation history management for executors."""
    
    def _init_conversation_history(self, agent):
        """Initialize conversation history tracking."""
        self._original_system_message = f"You are a {agent.role}. {agent.instructions}"
        self._history_display_manager = DisplayManager(config=None, state_manager=None)
        self._conversation_history: List[Section] = []
        self._processed_message_ids: set = set()
        self._conversation_history_mode = getattr(agent, 'conversation_history_mode', 'filtered')

    def _text_block(self, text: str) -> Block:
        """Create a text block using the display manager."""
        return self._history_display_manager.create_block("text", content=text)
    
    def _convert_message_to_blocks(self, content: Any) -> List[Block]:
        """Convert message content to Block objects."""
        blocks = []
        if not content:
            return blocks

        if isinstance(content, str):
            blocks.append(self._text_block(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    blocks.append(self._text_block(block))
                    continue
                if isinstance(block, dict):
                    block_type = block.get("type")
                    text_content = None
                    if block_type in ("input_text", "output_text", "text"):
                        text_content = block.get("text", "")
                    elif block_type == "input_file":
                        text_content = f"[File: {block.get('file_id', 'unknown')}]"
                    if text_content is not None:
                        blocks.append(self._text_block(text_content))
        else:
            blocks.append(self._text_block(str(content)))

        return blocks
    
    def _add_to_conversation_history(self, role, blocks):
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
    
    def _process_file_messages(self, file_messages):
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
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "input_file":
                    file_blocks.append(self._text_block(f"[File: {block.get('file_id', 'unknown')}]"))
                elif block_type == "input_text":
                    file_blocks.append(self._text_block(block.get("text", "")))

            if file_blocks:
                temp_id = f"file_msg_{len(self._conversation_history)}"
                if temp_id not in self._processed_message_ids:
                    self._add_to_conversation_history(role, file_blocks)
                    self._processed_message_ids.add(temp_id)
    
    def _update_conversation_history_from_messages(self, messages, file_messages=None):
        """Update conversation history from workflow_state messages."""
        if self._conversation_history_mode == "disable":
            return
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
