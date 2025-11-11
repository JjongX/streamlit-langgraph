# Display management for Streamlit UI components.

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
                # Render all blocks
                for block in self.blocks:
                    block.write()
                # Show agent name if available
                if hasattr(self, '_agent_info') and "agent" in self._agent_info:
                    st.caption(f"Agent: {self._agent_info['agent']}")


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
    
    def render_message_history(self, messages: List[Dict[str, Any]]) -> None:
        """Render historical messages from session state."""
        for message in messages:
            # Skip system messages and workflow completion messages
            if message.get("role") == "system":
                continue
            if message.get("role") == "assistant" and message.get("agent") in ["END", "__end__", None]:
                continue
            
            avatar = self.config.user_avatar if message["role"] == "user" else self.config.assistant_avatar
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "agent" in message:
                    st.caption(f"Agent: {message['agent']}")
    
    def render_welcome_message(self) -> None:
        """Render welcome message if configured."""
        if self.config.welcome_message:
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.markdown(self.config.welcome_message)
    
    def render_workflow_message(self, message: Dict[str, Any]) -> bool:
        """
        Render a single workflow message.
        
        Checks if message is already displayed and renders it if not.
        
        Args:
            message: Message dictionary with 'role', 'content', 'agent', and 'id'
            
        Returns:
            True if message was rendered, False if already displayed
        """
        msg_id = message.get("id")
        if not msg_id:
            return False
        
        # Check if already displayed
        session_message_ids = {msg.get("id") for msg in st.session_state.messages if msg.get("id")}
        if msg_id in session_message_ids:
            return False
        
        # Only render assistant messages with valid agents
        if (message.get("role") == "assistant" and 
            message.get("agent") and 
            message.get("agent") != "system"):
            
            section = self.add_section("assistant")
            section._agent_info = {"agent": message.get("agent", "Assistant")}
            section.update("text", message.get("content", ""))
            section.stream()
            
            # Add to session_state immediately to prevent duplicates on reruns
            session_msg = {
                "id": msg_id,
                "role": message.get("role"),
                "content": message.get("content", "")
            }
            if "agent" in message:
                session_msg["agent"] = message["agent"]
            st.session_state.messages.append(session_msg)
            
            return True
        
        return False

