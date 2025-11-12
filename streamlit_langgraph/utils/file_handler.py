# File handling utilities for OpenAI API integration.

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """
    Handler for managing file uploads with OpenAI API integration.
    """
    
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
        # OpenAI integration fields
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
    
    def __init__(self, temp_dir: Optional[str] = None, openai_client=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.files: Dict[str, FileHandler.FileInfo] = {}
        self.openai_client = openai_client
    
    def save_uploaded_file(self, uploaded_file, file_id: Optional[str] = None) -> "FileHandler.FileInfo":
        """
        Save an uploaded file and process it for OpenAI integration.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            file_id: Optional custom file ID
            
        Returns:
            FileInfo: Information about the saved file
        """
        if file_id is None:
            file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name
        
        file_path = os.path.join(self.temp_dir, uploaded_file.name)
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        file_ext = Path(uploaded_file.name).suffix.lower()
        file_type = MIME_TYPES.get(file_ext)
        
        file_info = FileHandler.FileInfo(
            name=uploaded_file.name,
            path=file_path,
            size=len(uploaded_file.getvalue()),
            type=file_type,
            content=uploaded_file.getvalue(),
            metadata={
                'file_id': file_id,
                'extension': file_ext,
                'uploaded_at': None
            }
        )
        
        if self.openai_client:
            self._process_file_for_openai(file_info)
        
        self.files[file_id] = file_info
        return file_info
    
    def get_openai_input_messages(self) -> List[Dict[str, Any]]:
        """Get OpenAI input messages for all files.
        
        Returns:
            List[Dict]: List of OpenAI input messages for files
        """
        messages = []
        for file_info in self.files.values():
            messages.extend(file_info.input_messages)
        return messages

    def _process_file_for_openai(self, file_info: "FileHandler.FileInfo") -> None:
        """
        Process a file for OpenAI integration and update its input_messages.
        
        Handles different file types: PDFs, images (vision), and text files.
        Creates appropriate OpenAI file objects and adds them to input_messages.
        """
        if not self.openai_client:
            return
        
        file_ext = file_info.extension
        file_path = Path(file_info.path)
        
        file_info.input_messages.append({
            "role": "user", 
            "content": f"File locally available at: {file_path}"
        })
        
        if file_ext == ".pdf":
            with open(file_path, "rb") as f:
                openai_file = self.openai_client.files.create(file=f, purpose="user_data")
                file_info.openai_file_id = openai_file.id
            file_info.input_messages.append({
                "role": "user",
                "content": [{"type": "input_file", "file_id": openai_file.id}]
            })
        elif file_ext in FileHandler.VISION_EXTENSIONS:
            with open(file_path, "rb") as f:
                vision_file = self.openai_client.files.create(file=f, purpose="vision")
                file_info.vision_file_id = vision_file.id
            file_info.input_messages.append({
                "role": "user",
                "content": [{"type": "input_image", "file_id": vision_file.id}]
            })
        elif file_ext in [".txt", ".md", ".json", ".csv", ".py", ".js", ".html", ".xml"]:
            with open(file_path, "rb") as f:
                openai_file = self.openai_client.files.create(file=f, purpose="user_data")
                file_info.openai_file_id = openai_file.id
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_info.input_messages.append({
                    "role": "user",
                    "content": f"Content of {file_path.name}:\n```\n{content[:2000]}{'...' if len(content) > 2000 else ''}\n```"
                })



