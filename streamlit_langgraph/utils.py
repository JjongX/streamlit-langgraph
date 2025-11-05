import inspect
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from langchain_core.tools import StructuredTool
# from langchain.tools import StructuredTool

from .agent import CreateAgentExecutor, get_llm_client
from .workflow.state import (
    clear_pending_interrupt,
    get_hitl_decision,
    get_pending_interrupts,
    set_hitl_decision,
    set_pending_interrupt,
)

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
        elif file_ext in VISION_EXTENSIONS:
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
    
    @classmethod
    def get_langchain_tools(cls, tool_names: Optional[List[str]] = None) -> List[Any]:
        """
        Convert CustomTool registry items to LangChain tools.
        
        Args:
            tool_names: Optional list of tool names to include. If None, includes all registered tools.
        
        Returns:
            List of LangChain tool objects
        """
        tools = []
        registry = cls._registry
        
        if tool_names:
            registry = {name: registry[name] for name in tool_names if name in registry}
        
        for tool_name, custom_tool in registry.items():
            try:
                tool = StructuredTool.from_function(
                    func=custom_tool.function,
                    name=tool_name,
                    description=custom_tool.description,
                )
                tools.append(tool)
            except Exception:
                # Skip tools that fail to convert
                continue
        
        return tools
    
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

class HITLUtils:
    """
    Utility class for Human-in-the-Loop (HITL) functionality.
    
    Provides static methods for handling interrupts, formatting decisions,
    and checking permissions in HITL workflows.
    """
    
    @staticmethod
    def extract_action_requests_from_interrupt(interrupt_raw: Any) -> List[Dict[str, Any]]:
        """
        Extract action_requests from Interrupt objects.
        
        Args:
            interrupt_raw: Can be a list of Interrupt objects, a dict, or other formats
            
        Returns:
            List of action request dictionaries with keys like 'name', 'args', 'description', 'id'
        """
        if not interrupt_raw:
            return []
        
        if isinstance(interrupt_raw, list):
            return HITLUtils._extract_from_list(interrupt_raw)
        elif isinstance(interrupt_raw, dict):
            return HITLUtils._extract_from_dict(interrupt_raw)
        
        return []
    
    @staticmethod
    def _extract_from_list(interrupt_list: List[Any]) -> List[Dict[str, Any]]:
        """Extract action requests from a list of interrupt items."""
        result = []
        for item in interrupt_list:
            if hasattr(item, 'value'):
                result.extend(HITLUtils._extract_from_dict(item.value))
            elif isinstance(item, dict):
                result.extend(HITLUtils._extract_from_dict(item))
        return result
    
    @staticmethod
    def _extract_from_dict(interrupt_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract action requests from a dict."""
        if 'action_requests' in interrupt_dict:
            return interrupt_dict['action_requests']
        return [interrupt_dict]
    
    @staticmethod
    def format_decisions(decisions: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Format decisions for HumanInTheLoopMiddleware.
        
        The middleware expects decisions in format:
        [{"type": "approve|reject|edit", "edit": {...} if edit}]
        
        Args:
            decisions: List of decision dicts or None values
            
        Returns:
            List of formatted decision dictionaries
        """
        formatted_decisions = []
        for decision in decisions:
            if decision:
                formatted_decisions.append(decision)
            else:
                # If no decision, default to approve
                formatted_decisions.append({"type": "approve"})
        return formatted_decisions
    
    @staticmethod
    def check_edit_allowed(agent_interrupt_on: Optional[Dict[str, Any]], tool_name: str) -> bool:
        """
        Check if editing is allowed for a tool based on agent's interrupt_on configuration.
        
        Args:
            agent_interrupt_on: The agent's interrupt_on configuration dict
            tool_name: Name of the tool to check
            
        Returns:
            True if editing is allowed, False otherwise
        """
        if not agent_interrupt_on:
            return True  # Default to allowing edit if not configured
        
        tool_config = agent_interrupt_on.get(tool_name, {})
        if isinstance(tool_config, dict):
            allowed_decisions = tool_config.get("allowed_decisions", ["approve", "reject", "edit"])
            return "edit" in allowed_decisions
        
        return True  # Default to allowing edit
    
    @staticmethod
    def parse_edit_input(edit_text: str, default_input: Any) -> tuple:
        """
        Parse user edit input, attempting to parse as JSON if it looks like JSON.
        
        Args:
            edit_text: The text input from the user
            default_input: The default input value to use if parsing fails or text is empty
            
        Returns:
            Tuple of (parsed_input, error_message). error_message is None if successful.
        """
        import json
        
        if not edit_text.strip():
            return default_input, None
        
        # If it looks like JSON (starts with { or [), try to parse it
        if edit_text.strip().startswith('{') or edit_text.strip().startswith('['):
            try:
                parsed = json.loads(edit_text)
                return parsed, None
            except (json.JSONDecodeError, ValueError):
                return None, "Invalid JSON. Please fix the input format."
        
        # Try to parse as JSON anyway, but fallback to string if it fails
        try:
            parsed = json.loads(edit_text)
            return parsed, None
        except (json.JSONDecodeError, ValueError):
            return edit_text, None
    
    @staticmethod
    def extract_action_info(action: Any, action_index: int) -> tuple:
        """Extract tool name, input, and ID from action."""
        if isinstance(action, dict):
            tool_name = action.get("name", action.get("tool", "Unknown"))
            tool_input = action.get("args", action.get("input", {}))
            action_id = action.get("id", f"action_{action_index}")
        else:
            tool_name = str(action)
            tool_input = {}
            action_id = f"action_{action_index}"
        return tool_name, tool_input, action_id
    
    @staticmethod
    def find_pending_action(decisions: List[Optional[Dict[str, Any]]]) -> Optional[int]:
        """Find the first action that needs a decision."""
        for i, decision in enumerate(decisions):
            if decision is None:
                return i
        return None
    
    @staticmethod
    def has_pending_interrupts(workflow_state: Optional[Dict[str, Any]]) -> bool:
        """
        Check if there are any pending interrupts in the workflow state.
        
        Args:
            workflow_state: The workflow state dictionary (can be None)
            
        Returns:
            True if there are valid pending interrupts, False otherwise
        """
        if not workflow_state:
            return False
        
        pending_interrupts = get_pending_interrupts(workflow_state)
        
        # Check if any interrupt has valid __interrupt__ data
        for value in pending_interrupts.values():
            if isinstance(value, dict) and value.get("__interrupt__"):
                return True
        
        return False
    
    @staticmethod
    def get_valid_interrupts(workflow_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract and filter valid interrupts from workflow state."""
        pending_interrupts = get_pending_interrupts(workflow_state)
        return {
            key: value for key, value in pending_interrupts.items()
            if isinstance(value, dict) and value.get("__interrupt__")
        }
    
    @staticmethod
    def initialize_decisions(workflow_state: Dict[str, Any], executor_key: str, 
                            num_actions: int) -> List[Optional[Dict[str, Any]]]:
        """Initialize decisions list from workflow state or create new one."""
        decisions = get_hitl_decision(workflow_state, executor_key)
        if decisions is None or len(decisions) != num_actions:
            decisions = [None] * num_actions
        return decisions
    
    @staticmethod
    def clear_interrupt_and_decisions(workflow_state: Dict[str, Any], executor_key: str):
        """Clear interrupt and decisions from workflow state."""
        if "pending_interrupts" in workflow_state.get("metadata", {}):
            workflow_state["metadata"]["pending_interrupts"].pop(executor_key, None)
        HITLUtils.clear_decisions(workflow_state, executor_key)
    
    @staticmethod
    def clear_decisions(workflow_state: Dict[str, Any], executor_key: str):
        """Clear decisions for an executor."""
        if "hitl_decisions" in workflow_state.get("metadata", {}):
            decisions_key = f"{executor_key}_decisions"
            workflow_state["metadata"]["hitl_decisions"].pop(decisions_key, None)

class HITLHandler:
    """Handler for Human-in-the-Loop (HITL) interrupt processing and UI."""
    
    def __init__(self, agent_manager, config):
        """
        Initialize HITL handler with dependencies.
        
        Args:
            agent_manager: AgentManager instance for accessing agents
            config: UIConfig instance for UI settings
        """
        self.agent_manager = agent_manager
        self.config = config
    
    def handle_pending_interrupts(self, workflow_state: Dict[str, Any]) -> bool:
        """
        Display UI for pending human-in-the-loop interrupts and handle user decisions.
        
        Args:
            workflow_state: Current workflow state
            
        Returns:
            True if interrupts were found and handled (should block further processing)
            False if no interrupts (should continue normal processing)
        """
        if not workflow_state:
            return False
        
        valid_interrupts = HITLUtils.get_valid_interrupts(workflow_state)
        if not valid_interrupts:
            return False
        
        st.markdown("---")
        st.markdown("### ⚠️ **Human Approval Required**")
        st.info("The workflow has paused and is waiting for your approval.")
        
        # Process the first valid interrupt
        for executor_key, interrupt_data in valid_interrupts.items():
            if self.process_interrupt(workflow_state, executor_key, interrupt_data):
                return True
        
        return False
    
    def process_interrupt(self, workflow_state: Dict[str, Any], executor_key: str, 
                          interrupt_data: Dict[str, Any]) -> bool:
        """
        Process a single interrupt - returns True if handled.
        
        Args:
            workflow_state: Current workflow state
            executor_key: Key identifying the executor
            interrupt_data: Interrupt data dictionary
            
        Returns:
            True if interrupt was handled, False otherwise
        """
        agent_name = interrupt_data.get("agent", "Unknown")
        interrupt_raw = interrupt_data.get("__interrupt__", [])
        original_config = interrupt_data.get("config", {})
        thread_id = interrupt_data.get("thread_id")
        
        interrupt_info = HITLUtils.extract_action_requests_from_interrupt(interrupt_raw)
        if not interrupt_info:
            st.error("⚠️ Error: Could not extract action details from interrupt.")
            return False
        
        executor = self.get_or_create_executor(executor_key, agent_name, thread_id, workflow_state)
        if executor is None:
            return False
        
        decisions = HITLUtils.initialize_decisions(workflow_state, executor_key, len(interrupt_info))
        pending_action_index = HITLUtils.find_pending_action(decisions)
        
        if pending_action_index is None:
            return self.resume_with_decisions(workflow_state, executor_key, executor, agent_name, 
                                             decisions, original_config, thread_id)
        
        self.display_action_approval_ui(executor_key, executor, agent_name, interrupt_info, 
                                        pending_action_index, decisions, workflow_state)
        return True
    
    def get_or_create_executor(self, executor_key: str, agent_name: str, 
                               thread_id: str, workflow_state: Dict[str, Any]):
        """
        Get existing executor or create a new one.
        
        Args:
            executor_key: Key identifying the executor
            agent_name: Name of the agent
            thread_id: Thread ID for the executor
            workflow_state: Current workflow state
            
        Returns:
            CreateAgentExecutor instance or None
        """
        executor = st.session_state.agent_executors.get(executor_key)
        if executor is None:
            agent = self.agent_manager.agents.get(agent_name)
            if agent and thread_id:
                executor = CreateAgentExecutor(agent, thread_id=thread_id)
                st.session_state.agent_executors[executor_key] = executor
        
        if executor is None:
            # Clear invalid interrupt
            clear_update = clear_pending_interrupt(workflow_state, executor_key)
            workflow_state["metadata"].update(clear_update["metadata"])
        
        return executor
    
    def resume_with_decisions(self, workflow_state: Dict[str, Any], executor_key: str,
                               executor, agent_name: str,
                               decisions: List[Dict[str, Any]], original_config: Dict[str, Any],
                               thread_id: str) -> bool:
        """
        Resume execution with all decisions made.
        
        Args:
            workflow_state: Current workflow state
            executor_key: Key identifying the executor
            executor: CreateAgentExecutor instance
            agent_name: Name of the agent
            decisions: List of user decisions
            original_config: Original execution config
            thread_id: Thread ID for the executor
            
        Returns:
            True if execution was resumed
        """
        clear_update = clear_pending_interrupt(workflow_state, executor_key)
        workflow_state["metadata"].update(clear_update["metadata"])
        
        formatted_decisions = HITLUtils.format_decisions(decisions)
        
        if executor.agent_obj is None:
            llm_client = get_llm_client(executor.agent)
            executor._build_agent(llm_client)
        
        resume_config = original_config or {"configurable": {"thread_id": thread_id}}
        
        with st.spinner("Processing your decision..."):
            resume_response = executor.resume(formatted_decisions, config=resume_config)
        
        if resume_response and resume_response.get("__interrupt__"):
            interrupt_update = set_pending_interrupt(workflow_state, agent_name, resume_response, executor_key)
            workflow_state["metadata"].update(interrupt_update["metadata"])
            HITLUtils.clear_decisions(workflow_state, executor_key)
            st.rerun()
        
        if resume_response and resume_response.get("content"):
            # Add response message if not duplicate
            content = resume_response.get("content")
            message_exists = False
            for msg in st.session_state.messages:
                if (msg.get("role") == "assistant" and msg.get("agent") == agent_name and msg.get("content") == content):
                    message_exists = True
                    break
            
            if not message_exists:
                st.session_state.messages.append({"role": "assistant", "content": content, "agent": agent_name})
        
        HITLUtils.clear_interrupt_and_decisions(workflow_state, executor_key)
        st.rerun()
    
    def display_action_approval_ui(self, executor_key: str, executor,
                                    agent_name: str, interrupt_info: List[Dict[str, Any]],
                                    action_index: int, decisions: List[Optional[Dict[str, Any]]],
                                    workflow_state: Dict[str, Any]):
        """
        Display UI for approving/rejecting/editing an action.
        
        Args:
            executor_key: Key identifying the executor
            executor: CreateAgentExecutor instance
            agent_name: Name of the agent
            interrupt_info: List of action information dictionaries
            action_index: Index of the current action to process
            decisions: List of user decisions
            workflow_state: Current workflow state
        """
        import json
        
        action = interrupt_info[action_index]
        tool_name, tool_input, action_id = HITLUtils.extract_action_info(action, action_index)
        
        agent_interrupt_on = getattr(executor.agent, 'interrupt_on', None)
        allow_edit = HITLUtils.check_edit_allowed(agent_interrupt_on, tool_name)
        
        def handle_approve():
            self.handle_decision(workflow_state, executor_key, decisions, action_index, {"type": "approve"})
        
        def handle_reject():
            self.handle_decision(workflow_state, executor_key, decisions, action_index, {"type": "reject"})
        
        def handle_edit(edit_text):
            parsed_input, error_msg = HITLUtils.parse_edit_input(edit_text, tool_input)
            if error_msg:
                st.error(error_msg)
            else:
                self.handle_decision(workflow_state, executor_key, decisions, action_index,
                                    {"type": "edit", "input": parsed_input})
                                    
        with st.container():
            st.markdown("---")
            st.markdown(f"**Agent:** {agent_name} is requesting approval to execute the following action:")
            st.write(f"**Tool:** `{tool_name}`")
            if tool_input:
                st.json(tool_input)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Approve", key=f"approve_{executor_key}_{action_id}"):
                    handle_approve()
            with col2:
                if st.button("❌ Reject", key=f"reject_{executor_key}_{action_id}"):
                    handle_reject()
            with col3:
                if allow_edit:
                    edit_key = f"edit_{executor_key}_{action_id}"
                    edit_btn_key = f"edit_btn_{executor_key}_{action_id}"
                    default_value = json.dumps(tool_input, indent=2) if tool_input else ""
                    
                    edit_text = st.text_area(
                        f"Edit {tool_name} input (optional)",
                        value=default_value, key=edit_key, height=100
                    )
                    
                    if st.button("✏️ Approve with Edit", key=edit_btn_key):
                        handle_edit(edit_text)
    
    def handle_decision(self, workflow_state: Dict[str, Any], executor_key: str,
                       decisions: List[Optional[Dict[str, Any]]], action_index: int,
                       decision: Dict[str, Any]):
        """
        Handle user decision and update workflow state.
        
        Args:
            workflow_state: Current workflow state
            executor_key: Key identifying the executor
            decisions: List of user decisions
            action_index: Index of the action being decided
            decision: Decision dictionary (type: approve/reject/edit)
        """        
        decisions[action_index] = decision
        decision_update = set_hitl_decision(workflow_state, executor_key, decisions)
        workflow_state["metadata"].update(decision_update["metadata"])
        st.rerun()
