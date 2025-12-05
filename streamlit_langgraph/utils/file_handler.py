# File handling utilities for OpenAI API integration.

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

MIME_TYPES = {
    "txt" : "text/plain",
    "csv" : "text/csv",
    "tsv" : "text/tab-separated-values",
    "html": "text/html",
    "yaml": "text/yaml",
    "md"  : "text/markdown",
    "png" : "image/png",
    "jpg" : "image/jpeg",
    "jpeg": "image/jpeg",
    "gif" : "image/gif",
    "xml" : "application/xml",
    "json": "application/json",
    "pdf" : "application/pdf",
    "zip" : "application/zip",
    "tar" : "application/x-tar",
    "gz"  : "application/gzip",
    "xls" : "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "doc" : "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "ppt" : "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class FileHandler:
    """Handler for managing file uploads with OpenAI API integration."""
    
    FILE_SEARCH_EXTENSIONS = [
        ".c", ".cpp", ".cs", ".css", ".doc", ".docx", ".go", 
        ".html", ".java", ".js", ".json", ".md", ".pdf", ".php", 
        ".pptx", ".py", ".rb", ".sh", ".tex", ".ts", ".txt"
    ]

    CODE_INTERPRETER_EXTENSIONS = [
        ".c", ".cs", ".cpp", ".csv", ".doc", ".docx", ".html", 
        ".java", ".json", ".md", ".pdf", ".php", ".pptx", ".py", 
        ".rb", ".tex", ".txt", ".css", ".js", ".sh", ".ts", ".tsv", 
        ".jpeg", ".jpg", ".gif", ".pkl", ".png", ".tar", ".xlsx", 
        ".xml", ".zip"
    ]

    VISION_EXTENSIONS = [".png", ".jpeg", ".jpg", ".webp", ".gif"]
    
    @dataclass
    class FileInfo:
        """Comprehensive information about uploaded or processed files."""
        name: str
        path: str
        size: int
        type: str
        content: Optional[bytes] = None
        metadata: Dict[str, Any] = None
        openai_file_id: Optional[str] = None
        vision_file_id: Optional[str] = None
        input_messages: List[Dict[str, Any]] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
            if self.input_messages is None:
                self.input_messages = []
        
        @property
        def extension(self) -> str:
            """Get file extension."""
            return Path(self.name).suffix.lower()
    
    def __init__(
        self, 
        temp_dir: Optional[str] = None, 
        openai_client=None,
        model=None,
        allow_file_search: Optional[bool] = False,
        allow_code_interpreter: Optional[bool] = False,
        container_id: Optional[str] = None,
    ):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.files: Dict[str, FileHandler.FileInfo] = {}
        self.openai_client = openai_client
        self.model = model
        self.allow_file_search = allow_file_search
        self.allow_code_interpreter = allow_code_interpreter
        self._container_id = container_id
        self._tracked_files: List[FileHandler.FileInfo] = []
        self._dynamic_vector_store = None
        
        # Auto-create container if code_interpreter is enabled but no container_id provided
        if self.allow_code_interpreter and not self._container_id and self.openai_client:
            try:
                container = self.openai_client.containers.create(name="streamlit-langgraph")
                self._container_id = container.id
            except Exception:
                pass
        
        if "file_handler_vector_stores" not in st.session_state:
            st.session_state.file_handler_vector_stores = []
    
    def update_settings(
        self,
        allow_file_search: Optional[bool] = None,
        allow_code_interpreter: Optional[bool] = None,
        container_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:
        """Update FileHandler settings dynamically."""
        if model is not None:
            self.model = model
        if allow_file_search is not None:
            self.allow_file_search = allow_file_search
        if allow_code_interpreter is not None:
            self.allow_code_interpreter = allow_code_interpreter
        if container_id is not None:
            self._container_id = container_id
    
    def track(self, uploaded_file: UploadedFile) -> "FileHandler.FileInfo":
        """Tracks a file uploaded by the user.
        
        Args:
            uploaded_file: An UploadedFile object from Streamlit.
            
        Returns:
            FileInfo: The tracked file information.
        """
        file_path = Path(os.path.join(self.temp_dir, uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        file_ext = file_path.suffix.lower()
        file_type = MIME_TYPES.get(file_ext.lstrip("."), "application/octet-stream")
        
        file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name
        
        file_info = FileHandler.FileInfo(
            name=file_path.name,
            path=str(file_path),
            size=file_path.stat().st_size if file_path.exists() else 0,
            type=file_type,
            content=None,
            metadata={'file_id': file_id, 'extension': file_ext, 'uploaded_at': None}
        )
        
        if not self.openai_client:
            self._tracked_files.append(file_info)
            if file_id:
                self.files[file_id] = file_info
            return file_info
        
        if not hasattr(self.openai_client, 'files'):
            raise ValueError(
                "OpenAI client is not properly configured. "
                "The client must have a 'files' attribute for file operations. "
                "Please ensure the OpenAI client is correctly initialized."
            )

        file_ext_lower = file_path.suffix.lower()
        openai_file = None
        vision_file = None
        skip_file_search = False

        if file_ext_lower == ".pdf":
            with open(file_path, "rb") as f:
                openai_file = self.openai_client.files.create(file=f, purpose="user_data")
            file_info.input_messages.append({
                "role": "user",
                "content": [{"type": "input_file", "file_id": openai_file.id}]
            })

        if file_ext_lower in FileHandler.VISION_EXTENSIONS:
            vision_file = self.openai_client.files.create(file=file_path, purpose="vision")
            file_info.input_messages.append({
                "role": "user",
                "content": [{"type": "input_image", "file_id": vision_file.id}]
            })

        if (self.allow_code_interpreter and 
            self._container_id and 
            file_ext_lower in FileHandler.CODE_INTERPRETER_EXTENSIONS):
            if file_ext_lower in FileHandler.VISION_EXTENSIONS:
                openai_file = vision_file
            if openai_file is None:
                with open(file_path, "rb") as f:
                    openai_file = self.openai_client.files.create(file=f, purpose="user_data")
            self.openai_client.containers.files.create(
                container_id=self._container_id,
                file_id=openai_file.id,
            )
            
            # Store file info for conversation context
            file_info.metadata['container_file_id'] = openai_file.id
            file_info.metadata['container_id'] = self._container_id

        if (self.allow_file_search and 
            not skip_file_search and 
            file_ext_lower in FileHandler.FILE_SEARCH_EXTENSIONS):
            if openai_file is None:
                with open(file_path, "rb") as f:
                    openai_file = self.openai_client.files.create(file=f, purpose="user_data")
            
            if self._dynamic_vector_store is None:
                if "file_handler_vector_stores" in st.session_state and st.session_state.file_handler_vector_stores:
                    existing_vs_id = st.session_state.file_handler_vector_stores[0]
                    try:
                        self._dynamic_vector_store = self.openai_client.vector_stores.retrieve(existing_vs_id)
                    except Exception:
                        self._dynamic_vector_store = self.openai_client.vector_stores.create(
                            name="streamlit-langgraph"
                        )
                        if self._dynamic_vector_store.id not in st.session_state.file_handler_vector_stores:
                            st.session_state.file_handler_vector_stores.append(self._dynamic_vector_store.id)
                else:
                    self._dynamic_vector_store = self.openai_client.vector_stores.create(
                        name="streamlit-langgraph"
                    )
                    if "file_handler_vector_stores" not in st.session_state:
                        st.session_state.file_handler_vector_stores = []
                    if self._dynamic_vector_store.id not in st.session_state.file_handler_vector_stores:
                        st.session_state.file_handler_vector_stores.append(self._dynamic_vector_store.id)
            
            self.openai_client.vector_stores.files.create(
                vector_store_id=self._dynamic_vector_store.id,
                file_id=openai_file.id
            )
            
            result = self.openai_client.vector_stores.retrieve(
                vector_store_id=self._dynamic_vector_store.id,
            )
            while result.status != "completed":
                time.sleep(1)
                result = self.openai_client.vector_stores.retrieve(
                    vector_store_id=self._dynamic_vector_store.id,
                )

        if openai_file:
            file_info.openai_file_id = openai_file.id
        if vision_file:
            file_info.vision_file_id = vision_file.id
        
        # Add comprehensive file message for conversation context
        # This helps the model understand what files are available across turns
        file_context_parts = [f"Uploaded file: {file_path.name}"]
        if file_info.metadata.get('container_file_id'):
            file_context_parts.append("Available in code interpreter container")
            file_context_parts.append(f"File ID: {file_info.metadata['container_file_id']}")
            file_context_parts.append("You can access this file using Python code in the code interpreter")
        elif file_info.openai_file_id:
            file_context_parts.append(f"File ID: {file_info.openai_file_id}")
        
        file_info.input_messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": " | ".join(file_context_parts)}]
        })

        self._tracked_files.append(file_info)
        if file_id:
            self.files[file_id] = file_info
        
        return file_info
    
    def get_openai_input_messages(self) -> List[Dict[str, Any]]:
        """Get OpenAI input messages for all tracked files."""
        messages = []
        for file_info in self._tracked_files:
            messages.extend(file_info.input_messages)
        return messages

    def get_vector_store_ids(self) -> List[str]:
        """Get vector store IDs for file search."""
        vector_store_ids = []
        
        if self._dynamic_vector_store:
            vector_store_ids.append(self._dynamic_vector_store.id)
        
        if "file_handler_vector_stores" in st.session_state:
            for vs_id in st.session_state.file_handler_vector_stores:
                if vs_id not in vector_store_ids:
                    vector_store_ids.append(vs_id)
        
        return vector_store_ids
