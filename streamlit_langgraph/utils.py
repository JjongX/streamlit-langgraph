import inspect
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

FILE_SEARCH_EXTENSIONS = [
    ".c", ".cpp", ".cs", ".css", ".doc", ".docx", ".go", 
    ".html", ".java", ".js", ".json", ".md", ".pdf", ".php", 
    ".pptx", ".py", ".rb", ".sh", ".tex", ".ts", ".txt"
]

CODE_INTERPRETER_EXTENSIONS = [
    ".c", ".cs", ".cpp", ".csv", ".doc", ".docx", ".html", 
    ".java", ".json", ".md", ".pdf", ".php", ".pptx", ".py", 
    ".rb", ".tex", ".txt", ".css", ".js", ".sh", ".ts", ".csv", 
    ".jpeg", ".jpg", ".gif", ".pkl", ".png", ".tar", ".xlsx", 
    ".xml", ".zip"
]

VISION_EXTENSIONS = [".png", ".jpeg", ".jpg", ".webp", ".gif"]

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
    
    @property
    def capabilities(self) -> List[str]:
        """Get list of capabilities for this file type."""
        caps = []
        ext = self.extension
        
        if ext in VISION_EXTENSIONS:
            caps.append("image analysis")
        if ext in CODE_INTERPRETER_EXTENSIONS:
            caps.append("code execution")
        if ext in FILE_SEARCH_EXTENSIONS:
            caps.append("text search")
        if ext == ".pdf":
            caps.append("PDF processing")
        
        return caps or ["basic file handling"]
    
    def get_summary(self) -> str:
        """Get a summary string for this file."""
        cap_str = ", ".join(self.capabilities)
        return f"📄 **{self.name}** ({self.size} bytes) - Available for: {cap_str}"

class FileHandler:
    """
    Handler for managing file uploads with OpenAI API integration.
    """
    
    def __init__(self, temp_dir: Optional[str] = None, openai_client=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.files: Dict[str, FileInfo] = {}
        self.openai_client = openai_client
    
    def save_uploaded_file(self, uploaded_file, file_id: Optional[str] = None) -> FileInfo:
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
        
        file_info = FileInfo(
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
        
        # Process for OpenAI integration if client is available
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
    
    def get_file_context_summary(self) -> str:
        """Get a summary of all uploaded files for context.
        
        Returns:
            str: Summary of uploaded files
        """
        if not self.files:
            return "No files uploaded."
        
        return "\n".join(file_info.get_summary() for file_info in self.files.values())

    def _process_file_for_openai(self, file_info: FileInfo) -> None:
        """Process a file for OpenAI integration and update its input_messages."""
        if not self.openai_client:
            return
        
        file_ext = file_info.extension
        file_path = Path(file_info.path)
        
        # Add local file path message
        file_info.input_messages.append({
            "role": "user", 
            "content": f"File locally available at: {file_path}"
        })
        
        try:
            # Handle PDF files with special processing
            if file_ext == ".pdf":
                with open(file_path, "rb") as f:
                    openai_file = self.openai_client.files.create(file=f, purpose="user_data")
                    file_info.openai_file_id = openai_file.id
                
                # Try to add as input file for direct PDF processing
                try:
                    file_info.input_messages.append({
                        "role": "user",
                        "content": [{"type": "input_file", "file_id": openai_file.id}]
                    })
                except Exception:
                    pass
            
            # Handle vision files (images)
            elif file_ext in VISION_EXTENSIONS:
                with open(file_path, "rb") as f:
                    vision_file = self.openai_client.files.create(file=f, purpose="vision")
                    file_info.vision_file_id = vision_file.id
                
                file_info.input_messages.append({
                    "role": "user",
                    "content": [{"type": "input_image", "file_id": vision_file.id}]
                })
            
            # Handle other file types that can be processed as text
            elif file_ext in [".txt", ".md", ".json", ".csv", ".py", ".js", ".html", ".xml"]:
                with open(file_path, "rb") as f:
                    openai_file = self.openai_client.files.create(file=f, purpose="user_data")
                    file_info.openai_file_id = openai_file.id
                
                # Add file content as text in the message
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        file_info.input_messages.append({
                            "role": "user",
                            "content": f"Content of {file_path.name}:\n```\n{content[:2000]}{'...' if len(content) > 2000 else ''}\n```"
                        })
                except Exception:
                    # Fallback to file reference
                    file_info.input_messages.append({
                        "role": "user",
                        "content": [{"type": "input_file", "file_id": openai_file.id}]
                    })
            
        except Exception as e:
            # Fallback: at least provide file information
            file_info.input_messages.append({
                "role": "user",
                "content": f"File {file_path.name} is available but could not be processed: {str(e)}"
            })

@dataclass
class CustomTool:
    """
    A custom tool that can be used by agents in the multiagent system.
    """
    name: str
    description: str
    function: Callable
    parameters: Optional[Dict[str, Any]] = None
    return_direct: bool = False

    _registry = {}

    @classmethod
    def register_tool(cls, name: str, description: str, function: Callable, **kwargs) -> "CustomTool":
        """
        Register a custom tool and add it to the class-level registry.
        """
        tool = cls(
            name=name,
            description=description,
            function=function,
            **kwargs
        )
        cls._registry[name] = tool
        return tool
    
    @classmethod
    def tool(cls, name: str, description: str, **kwargs):
        """
        Decorator for registering functions as tools.
        
        Example:
            @CustomTool.tool("calculator", "Performs basic arithmetic")
            def calculate(expression: str) -> float:
                return eval(expression)
        """
        def decorator(func: Callable) -> Callable:
            cls.register_tool(name, description, func, **kwargs)
            return func
        return decorator
    
    def __post_init__(self):
        """Extract function parameters if not provided."""
        if self.parameters is None:
            self.parameters = self._extract_parameters()
    
    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameters from function signature."""
        sig = inspect.signature(self.function)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            # Try to extract type from annotation
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is str:
                    param_info["type"] = "string"
                elif param.annotation is int:
                    param_info["type"] = "integer"
                elif param.annotation is float:
                    param_info["type"] = "number"
                elif param.annotation is bool:
                    param_info["type"] = "boolean"
                elif param.annotation is list:
                    param_info["type"] = "array"
                elif param.annotation is dict:
                    param_info["type"] = "object"
            
            parameters["properties"][param_name] = param_info
            
            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return parameters
