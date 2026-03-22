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
        elif self.category == "parallel_workers":
            self._render_parallel_workers()
    
    def _render_download(self):
        """Render download button for file content."""
        _, file_extension = os.path.splitext(self.filename)
        mime_type = MIME_TYPES.get(file_extension.lstrip("."), "application/octet-stream")
        st.download_button(
            label=self.filename,
            data=self.content,
            file_name=self.filename,
            mime=mime_type,
            key=self.display_manager._download_button_key,
        )
        self.display_manager._download_button_key += 1

    def _render_parallel_workers(self):
        """Render grouped parallel worker status/output panels."""
        if not isinstance(self.content, dict):
            return

        title = self.content.get("title", "Parallel Execution")
        st.markdown(f"**{title}**")

        worker_items = self.content.get("workers", [])
        if not isinstance(worker_items, list):
            return

        for worker_item in worker_items:
            if not isinstance(worker_item, dict):
                continue
            agent = worker_item.get("agent", "Worker")
            status = worker_item.get("status", "Pending")
            expander_title = f"{agent} - {status}"
            expanded = status != "Completed"
            with st.expander(expander_title, expanded=expanded):
                events = worker_item.get("events", [])
                if isinstance(events, list):
                    for event in events:
                        if event:
                            st.caption(str(event))
                result_text = worker_item.get("result")
                if result_text:
                    st.markdown(result_text)


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

    def update_parallel_workers(self, content: Dict[str, Any]) -> None:
        """Create or replace grouped parallel worker block."""
        if self.empty:
            self.blocks = [self.display_manager.create_block("parallel_workers", content)]
            return

        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            if block.category == "parallel_workers":
                self.blocks[i] = self.display_manager.create_block("parallel_workers", content)
                return

        self.blocks.append(self.display_manager.create_block("parallel_workers", content))
    
    def stream(self):
        """Render this section and persist it."""
        avatar = (self.display_manager.config.user_avatar if self.role == "user" 
                 else self.display_manager.config.assistant_avatar)
        with self.delta_generator:
            with st.chat_message(self.role, avatar=avatar):
                for block in self.blocks:
                    block.write()
                # Show agent name if available
                if hasattr(self, '_agent_info') and "agent" in self._agent_info:
                    st.caption(f"Agent: {self._agent_info['agent']}")
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
            state_manager: Streamlit-backed state manager for accessing workflow_state
        """
        self.config = config
        self.state_manager = state_manager
        self._sections = []
        self._download_button_key = 0
        self._parallel_sections: Dict[str, Dict[str, Any]] = {}
    
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
                    block = self._block_from_serialized_data(block_data)
                    block.write()
                
                if "agent_info" in section_data and "agent" in section_data["agent_info"]:
                    st.caption(f"Agent: {section_data['agent_info']['agent']}")

    def _block_from_serialized_data(self, block_data: Dict[str, Any]) -> Block:
        """Rehydrate a serialized block dict to a Block instance."""
        content: Any
        if "content_b64" in block_data:
            content = base64.b64decode(block_data["content_b64"])
        else:
            content = block_data.get("content")
        return self.create_block(
            block_data.get("category", "text"),
            content=content,
            filename=block_data.get("filename"),
            file_id=block_data.get("file_id"),
        )
    
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
        
        if self._render_parallel_grouped_message(message):
            return True

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

    def _render_parallel_grouped_message(self, message: Dict[str, Any]) -> bool:
        """Render parallel worker status/output inside a single grouped section."""
        if message.get("role") != "assistant":
            return False
        agent = message.get("agent")
        if not agent or agent == "system":
            return False

        is_status_event = bool(message.get("is_status_event"))
        run_key = None
        if self.state_manager and hasattr(self.state_manager, "get_latest_user_message_id"):
            run_key = self.state_manager.get_latest_user_message_id()
        if not run_key:
            return False

        parallel_state = self._parallel_sections.get(run_key)
        if not is_status_event and parallel_state is None:
            return False
        if not is_status_event and parallel_state is not None and agent not in parallel_state["workers"]:
            return False

        if parallel_state is None:
            section = self.add_section("assistant")
            section._agent_info = {"agent": "Parallel Execution"}
            section._message_id = message.get("id")
            parallel_state = {"section": section, "workers": {}}
            self._parallel_sections[run_key] = parallel_state

        workers = parallel_state["workers"]
        worker_entry = workers.setdefault(
            agent,
            {"agent": agent, "status": "Pending", "events": [], "result": ""},
        )

        if is_status_event:
            status_text = str(message.get("content", "")).strip()
            if status_text:
                worker_entry["events"].append(status_text)
            normalized = status_text.lower()
            if "started" in normalized:
                worker_entry["status"] = "Running"
            elif "completed" in normalized:
                worker_entry["status"] = "Completed"
        else:
            result_text = message.get("content", "")
            worker_entry["result"] = result_text
            worker_entry["status"] = "Completed"
            if "[status] Completed" not in worker_entry["events"]:
                worker_entry["events"].append("[status] Completed")

        panel_content = {
            "title": "Parallel Execution",
            "workers": [workers[name] for name in workers],
        }
        parallel_state["section"].update_parallel_workers(panel_content)
        parallel_state["section"].stream()
        return True
