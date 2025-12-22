# Display management for Streamlit UI components.

import base64
import os
from typing import Any, Dict, List, Optional, Union

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
        content: Optional[Union[str, bytes]] = None,
        filename: Optional[str] = None,
        file_id: Optional[str] = None,
    ):
        self.display_manager = display_manager
        self.category = category
        self.content = content if content is not None else ("" if category not in ["image", "generated_image", "download"] else b"")
        self.filename = filename
        self.file_id = file_id

    def write(self):
        """Render this block's content to the Streamlit interface."""
        if self.category == "text":
            st.markdown(self.content)
        elif self.category == "code":
            with st.expander("", expanded=False, icon=":material/code:"):
                st.code(self.content)
        elif self.category == "reasoning":
            with st.expander("", expanded=False, icon=":material/lightbulb:"):
                st.markdown(self.content)
        elif self.category in ["image", "generated_image"]:
            if self.content:
                st.image(self.content, caption=self.filename)
        elif self.category == "download":
            self._render_download()
    
    def _render_download(self):
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
    ):
        self.display_manager = display_manager
        self.role = role
        self.blocks = blocks or []
        self.delta_generator = st.empty()
        self._section_index = None
    
    @property
    def empty(self) -> bool:
        return len(self.blocks) == 0

    @property
    def last_block(self) -> Optional[Block]:
        return None if self.empty else self.blocks[-1]
    
    def update(self, category, content, filename=None, file_id=None):
        """
        Add or append content to this section.
        
        If the last block has the same category and is streamable, content is appended.
        For generated_image, if the last block is also generated_image with the same file_id,
        the content is replaced (for partial image updates).
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
        elif (category == "generated_image" and 
              self.last_block.category == "generated_image" and
              self.last_block.file_id == file_id and file_id is not None):
            # Replace content for partial image updates (same file_id)
            self.last_block.content = content
        else:
            # Create new block for different category
            self.blocks.append(self.display_manager.create_block(
                category, content, filename=filename, file_id=file_id
            ))
    
    def stream(self):
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary format for serialization."""
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
            if block.category in ["image", "generated_image", "download"] and block.content:
                import base64
                if isinstance(block.content, bytes):
                    block_data["content_b64"] = base64.b64encode(block.content).decode('utf-8')
                else:
                    block_data["content"] = block.content
            else:
                block_data["content"] = block.content
            
            section_data["blocks"].append(block_data)
        
        return section_data
    
    def _save_to_session_state(self):
        """Save section data to workflow_state."""
        section_data = self.to_dict()
        
        if not self.display_manager.state_manager:
            raise ValueError("state_manager is required. workflow_state must be the single source of truth.")
        self._section_index = self.display_manager.state_manager.update_display_section(
            self._section_index, section_data
        )


class DisplayManager:
    """Manages UI rendering for chat messages."""
    
    def __init__(self, config=None, state_manager=None):
        """
        Initialize DisplayManager with UI configuration.
        
        Args:
            config: UI configuration (optional, for non-UI use cases like executors)
            state_manager: StateSynchronizer instance for accessing workflow_state
        """
        self.config = config
        self.state_manager = state_manager
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
    
    def render_message_history(self):
        """Render historical messages from workflow_state."""
        if not self.state_manager:
            raise ValueError("state_manager is required. workflow_state must be the single source of truth.")
        display_sections = self.state_manager.get_display_sections()
        
        for section_data in display_sections:
            avatar = (self.config.user_avatar if section_data["role"] == "user" 
                     else self.config.assistant_avatar)
            
            with st.chat_message(section_data["role"], avatar=avatar):
                for block_data in section_data.get("blocks", []):
                    category = block_data.get("category")
                    if category == "text":
                        st.markdown(block_data.get("content", ""))
                    elif category in ["image", "generated_image"]:
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
    
    def render_welcome_message(self):
        """Render welcome message if configured."""
        if self.config.welcome_message:
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.markdown(self.config.welcome_message)
    
    def render_workflow_message(self, message):
        """Render a single workflow message."""
        msg_id = message.get("id")
        if not msg_id:
            return False
        
        if not self.state_manager:
            raise ValueError("state_manager is required. workflow_state must be the single source of truth.")
        displayed_ids = self.state_manager.get_displayed_message_ids()
        
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

