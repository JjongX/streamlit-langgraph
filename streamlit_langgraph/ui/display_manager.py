# Display management for Streamlit UI components.

import base64
import os
from typing import Any, Dict, List, Optional

import streamlit as st

from ..utils import MIME_TYPES


class Block:
    """
    Individual content unit within a Section.
    
    A Block represents a single piece of content (text, code, reasoning, or download)
    that will be rendered as part of a chat message.
    """
    def __init__(
        self,
        display_manager: "DisplayManager",
        category: str,
        content: Optional[str] = None,
        filename: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> None:
        self.display_manager = display_manager
        self.category = category
        self.content = content or ""
        self.filename = filename
        self.file_id = file_id

    def write(self) -> None:
        """Render this block's content to the Streamlit interface."""
        if self.category == "text":
            st.markdown(self.content)
        elif self.category == "code":
            with st.expander("", expanded=False, icon=":material/code:"):
                st.code(self.content)
        elif self.category == "reasoning":
            with st.expander("", expanded=False, icon=":material/lightbulb:"):
                st.markdown(self.content)
        elif self.category == "image":
            if self.content:
                st.image(self.content, caption=self.filename)
        elif self.category == "download":
            self._render_download()
    
    def _render_download(self) -> None:
        """Render download button for file content."""
        _, file_extension = os.path.splitext(self.filename)
        st.download_button(
            label=self.filename,
            data=self.content,
            file_name=self.filename,
            mime=MIME_TYPES[file_extension.lstrip(".")],
            key=self.display_manager._download_button_key,
        )
        self.display_manager._download_button_key += 1


class Section:
    """
    Container for Blocks representing a single chat message.
    
    A Section groups multiple Blocks together to form a complete chat message
    from either a user or assistant. It handles streaming updates and rendering.
    """
    def __init__(
        self,
        display_manager: "DisplayManager",
        role: str,
        blocks: Optional[List[Block]] = None,
    ) -> None:
        self.display_manager = display_manager
        self.role = role
        self.blocks = blocks or []
        self.delta_generator = st.empty()
        self._section_index = None  # Track which index this section is saved at
    
    @property
    def empty(self) -> bool:
        return len(self.blocks) == 0

    @property
    def last_block(self) -> Optional[Block]:
        return None if self.empty else self.blocks[-1]
    
    def update(self, category: str, content: str, filename: Optional[str] = None, 
               file_id: Optional[str] = None) -> None:
        """
        Add or append content to this section.
        
        If the last block has the same category and is streamable, content is appended.
        Otherwise, a new block is created.
        """
        if self.empty:
             # Create first block
            self.blocks = [self.display_manager.create_block(
                category, content, filename=filename, file_id=file_id
            )]
        elif (category in ["text", "code", "reasoning"] and 
              self.last_block.category == category):
            # Append to existing block for same category
            self.last_block.content += content
        else:
            # Create new block for different category
            self.blocks.append(self.display_manager.create_block(
                category, content, filename=filename, file_id=file_id
            ))
    
    def stream(self) -> None:
        """Render this section and all its blocks to the Streamlit interface."""
        avatar = (self.display_manager.config.user_avatar if self.role == "user" 
                 else self.display_manager.config.assistant_avatar)
        with self.delta_generator:
            with st.chat_message(self.role, avatar=avatar):
                for block in self.blocks:
                    block.write()
                # Show agent name if available
                if hasattr(self, '_agent_info') and "agent" in self._agent_info:
                    st.caption(f"Agent: {self._agent_info['agent']}")
        
        # Always save section to session state for persistence across reruns
        self._save_to_session_state()
    
    def _save_to_session_state(self) -> None:
        """Save section data to session state for persistence."""
        section_data = {
            "role": self.role,
            "blocks": [],
            "agent_info": getattr(self, '_agent_info', {}),
            "message_id": getattr(self, '_message_id', None)
        }
        
        for block in self.blocks:
            block_data = {
                "category": block.category,
                "filename": block.filename,
                "file_id": block.file_id
            }
            # For binary content (images, downloads), store as base64 or reference
            if block.category in ["image", "download"] and block.content:
                import base64
                if isinstance(block.content, bytes):
                    block_data["content_b64"] = base64.b64encode(block.content).decode('utf-8')
                else:
                    block_data["content"] = block.content
            else:
                block_data["content"] = block.content
            
            section_data["blocks"].append(block_data)
        
        # Update existing section if it exists, otherwise append new one
        # This prevents duplicate sections during streaming
        if self._section_index is not None and self._section_index < len(st.session_state.display_sections):
            # Update existing section in place
            st.session_state.display_sections[self._section_index] = section_data
        else:
            # Append new section and remember its index
            st.session_state.display_sections.append(section_data)
            self._section_index = len(st.session_state.display_sections) - 1


class DisplayManager:
    """Manages UI rendering for chat messages."""
    
    def __init__(self, config):
        """Initialize DisplayManager with UI configuration."""
        self.config = config
        self._sections = []
        self._download_button_key = 0
    
    def create_block(self, category, content=None, filename=None, file_id=None) -> Block:
        """Create a new Block instance."""
        return Block(self, category, content=content, filename=filename, file_id=file_id)

    def add_section(self, role, blocks=None) -> Section:
        """Create and add a new Section for a chat message."""
        section = Section(self, role, blocks=blocks)
        self._sections.append(section)
        return section
    
    def render_message_history(self) -> None:
        """Render historical messages from session state."""
        display_sections = st.session_state.get("display_sections", [])
        
        for section_data in display_sections:
            avatar = (self.config.user_avatar if section_data["role"] == "user" 
                     else self.config.assistant_avatar)
            
            with st.chat_message(section_data["role"], avatar=avatar):
                for block_data in section_data.get("blocks", []):
                    category = block_data.get("category")
                    if category == "text":
                        st.markdown(block_data.get("content", ""))
                    elif category == "image":
                        if "content_b64" in block_data:
                            content = base64.b64decode(block_data["content_b64"])
                            st.image(content, caption=block_data.get("filename"))
                        elif "content" in block_data and block_data["content"]:
                            st.image(block_data["content"], caption=block_data.get("filename"))
                    elif category == "download":
                        if "content_b64" in block_data:
                            content = base64.b64decode(block_data["content_b64"])
                        else:
                            content = block_data.get("content", b"")
                        if content:
                            _, file_extension = os.path.splitext(block_data.get("filename", ""))
                            st.download_button(
                                label=block_data.get("filename", "Download"),
                                data=content,
                                file_name=block_data.get("filename", "file"),
                                mime=MIME_TYPES.get(file_extension.lstrip("."), "application/octet-stream"),
                                key=f"download_{block_data.get('file_id', self._download_button_key)}",
                            )
                            self._download_button_key += 1
                    elif category == "code":
                        with st.expander("", expanded=False, icon=":material/code:"):
                            st.code(block_data.get("content", ""))
                    elif category == "reasoning":
                        with st.expander("", expanded=False, icon=":material/lightbulb:"):
                            st.markdown(block_data.get("content", ""))
                
                if "agent_info" in section_data and "agent" in section_data["agent_info"]:
                    st.caption(f"Agent: {section_data['agent_info']['agent']}")
    
    def render_welcome_message(self) -> None:
        """Render welcome message if configured."""
        if self.config.welcome_message:
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.markdown(self.config.welcome_message)
    
    def render_workflow_message(self, message: Dict[str, Any]) -> bool:
        """Render a single workflow message."""
        msg_id = message.get("id")
        if not msg_id:
            return False
        
        # Check if already displayed using display_sections
        display_sections = st.session_state.get("display_sections", [])
        displayed_ids = {s.get("message_id") for s in display_sections if s.get("message_id")}
        if msg_id in displayed_ids:
            return False
        
        # Only render assistant messages with valid agents
        if (message.get("role") == "assistant" and 
            message.get("agent") and 
            message.get("agent") != "system"):
            
            section = self.add_section("assistant")
            section._agent_info = {"agent": message.get("agent", "Assistant")}
            section._message_id = msg_id
            section.update("text", message.get("content", ""))
            section.stream()
            
            return True
        
        return False

