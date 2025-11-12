# Main chat interface for Streamlit and LangGraph.

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from .agent import Agent, AgentManager
from .core.execution_coordinator import ExecutionCoordinator
from .core.executor import WorkflowExecutor
from .state import StateCoordinator, WorkflowStateManager
from .ui import DisplayManager
from .utils import FileHandler, CustomTool, HITLHandler, HITLUtils


@dataclass
class UIConfig:
    """Configuration for the Streamlit UI interface."""
    title: str
    page_icon: Optional[str] = "🤖"
    page_layout: str = "wide"
    stream: bool = True  # Enable/disable streaming responses
    enable_file_upload: bool = True
    show_sidebar: bool = True  # Show default sidebar
    user_avatar: Optional[str] = "👤"
    assistant_avatar: Optional[str] = "🤖"
    placeholder: str = "Type your message here..."
    welcome_message: Optional[str] = None


class LangGraphChat:
    """
    Main class for creating agent chat interfaces with Streamlit and LangGraph.
    """
    
    def __init__(
        self,
        workflow=None,
        agents: Optional[List[Agent]] = None,
        config: Optional[UIConfig] = None,
        custom_tools: Optional[List[CustomTool]] = None):
        """
        Initialize the LangGraph Chat interface.

        Args:
            workflow: LangGraph workflow (StateGraph)
            agents: List of agents to use
            config: Chat configuration
            custom_tools: List of custom tools
        """
        self.config = config or UIConfig()
        self.agent_manager = AgentManager()
        self.workflow = workflow
        self.workflow_executor = WorkflowExecutor() if workflow else None
        
        # Initialize agents
        if agents:
            if not workflow and len(agents) > 1:
                raise ValueError(
                    "Multiple agents require a workflow. "
                    "Either provide a workflow parameter or use a single agent."
                )
            for agent in agents:
                if agent.human_in_loop and not workflow:
                    raise ValueError("Human-in-the-loop is only available for multiagent workflows.")
                self.agent_manager.add_agent(agent)
        # Register custom tools
        if custom_tools:
            for tool in custom_tools:
                CustomTool.register_tool(
                    tool.name, tool.description, tool.function, 
                    parameters=tool.parameters, return_direct=tool.return_direct
                )
        # Initialize LLM client from first agent
        first_agent = next(iter(self.agent_manager.agents.values()))
        self.llm = AgentManager.get_llm_client(first_agent)
        openai_client = self.llm if hasattr(self.llm, 'files') else None
        self.file_handler = FileHandler(openai_client=openai_client)
        
        # Initialize Streamlit session state
        self._init_session_state()
        # Initialize state coordinator
        self.state_coordinator = StateCoordinator()
        
        # Initialize display manager
        self.display_manager = DisplayManager(self.config)
        
        # Initialize execution coordinator
        self.execution_coordinator = ExecutionCoordinator(
            workflow_executor=self.workflow_executor,
            agent_manager=self.agent_manager,
            llm_client=self.llm,
            config=self.config
        )
        
        self.hitl_handler = HITLHandler(self.agent_manager, self.config, self.state_coordinator)
    
    def create_block(self, category, content=None, filename=None, file_id=None):
        """Create a new Block instance."""
        return self.display_manager.create_block(category, content=content, filename=filename, file_id=file_id)

    def add_section(self, role, blocks=None):
        """Create and add a new Section for a chat message."""
        return self.display_manager.add_section(role, blocks=blocks)
    
    def _init_session_state(self):
        """Initialize Streamlit session state with required keys."""
        
        if "workflow_state" not in st.session_state:
            st.session_state.workflow_state = WorkflowStateManager.create_initial_state()
        if "agent_executors" not in st.session_state:
            st.session_state.agent_executors = {}  # Executor objects (not data)
        if "messages" not in st.session_state:
            st.session_state.messages = []  # Performance cache for rendering
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []  # File objects (not in workflow_state)
    
    def run(self):
        """Run the main chat interface."""
        st.set_page_config(
            page_title=self.config.title,
            page_icon=self.config.page_icon,
            layout=self.config.page_layout
        )
        st.title(self.config.title)
        
        # Render default sidebar if enabled in config
        if self.config.show_sidebar:
            self._render_sidebar()
        self._render_chat_interface()
    
    def _render_sidebar(self):
        """Render the sidebar with controls and information."""
        with st.sidebar:
            # Agent information
            st.header("Agent Configuration")
            agents = list(self.agent_manager.agents.values())
            if agents:
                for agent in agents:
                    with st.expander(f"{agent.name}", expanded=False):
                        st.write(f"**Role:** {agent.role}")
                        st.write(f"**Instructions:** {agent.instructions[:100]}...")
                        # Display capabilities
                        capabilities = []
                        if hasattr(agent, 'allow_file_search') and agent.allow_file_search:
                            capabilities.append("📁 File Search")
                        if hasattr(agent, 'allow_code_interpreter') and agent.allow_code_interpreter:
                            capabilities.append("💻 Code Interpreter")
                        if hasattr(agent, 'allow_web_search') and agent.allow_web_search:
                            capabilities.append("🌐 Web Search")
                        if hasattr(agent, 'tools') and agent.tools:
                            capabilities.append(f"🛠️ {len(agent.tools)} Custom Tools")
                        if capabilities:
                            st.write("**Capabilities:**")
                            for cap in capabilities:
                                st.write(f"- {cap}")
            # Add chat-specific controls
            st.header("Controls")
            if st.button("Reset All", type="secondary"):
                st.session_state.clear()
                st.rerun()
    
    def _render_chat_interface(self):
        """Render the main chat interface."""
        # Sync at the START of rendering
        self.state_coordinator.sync_to_session_state()
        
        # Display welcome message
        if not st.session_state.messages:
            self.display_manager.render_welcome_message()

        # Check for pending interrupts FIRST - workflow_state is the single source of truth
        workflow_state = st.session_state.workflow_state
        if HITLUtils.has_pending_interrupts(workflow_state):
            interrupt_handled = self.hitl_handler.handle_pending_interrupts(workflow_state)
            if interrupt_handled:
                return  # Don't process messages or show input while handling interrupts

        # Render message history
        self.display_manager.render_message_history(st.session_state.messages)

        # Render user input
        if prompt := st.chat_input(
            self.config.placeholder, accept_file=self.config.enable_file_upload
        ):
            self._handle_user_input(prompt)
    
    def _handle_user_input(self, chat_input):
        """Handle user input and generate responses."""
        if self.config.enable_file_upload: # File upload enabled; it has text and files attribute
            prompt = chat_input.text
            files = getattr(chat_input, 'files', [])
        else: # File upload disabled (plain string)  
            prompt = str(chat_input)
            files = []

        if files:
            self._process_file_uploads(files)
        
        workflow_state = st.session_state.workflow_state
        if "metadata" not in workflow_state:
            workflow_state["metadata"] = {}
        
        self.state_coordinator.add_user_message(prompt)
        
        section = self.add_section("user")
        section.update("text", prompt)
        for uploaded_file in files:
            section.update("text", f"\n:material/attach_file: `{uploaded_file.name}`")
        section.stream()

        # Clear HITL state before new request
        self.state_coordinator.clear_hitl_state()
        
        # Generate response
        with st.spinner("Thinking..."):
            response = self._generate_response(prompt)

        # Handle workflow completion
        if response.get("agent") == "workflow-completed":
            return
        # Handle interrupts from human-in-the-loop
        if response.get("__interrupt__"):
            st.rerun()
        
        if response and "stream" in response:
            # Handle streaming response from OpenAI Responses API
            section = self.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}
            
            full_response = ""
            for event in response["stream"]:
                if hasattr(event, 'type'):
                    full_response += self._process_stream_event(event, section)
            
            response["content"] = full_response
        else:
            # Handle non-streaming response from agent
            section = self.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}
            section.update("text", response["content"])
            section.stream()

        # Add assistant response to state if not a workflow control message
        if (response.get("content") and 
            response.get("agent") not in ["workflow", "workflow-completed"]):
            self.state_coordinator.add_assistant_message(
                response["content"], 
                response["agent"]
            )
    
    def _process_file_uploads(self, files):
        """Process uploaded files and update workflow state."""
        for uploaded_file in files:
            if uploaded_file not in st.session_state.uploaded_files:
                file_info = self.file_handler.save_uploaded_file(uploaded_file)
                st.session_state.uploaded_files.append(uploaded_file)
                # Add file metadata to workflow state (not content)
                self.state_coordinator.update_workflow_state({
                    "files": [{k: v for k, v in file_info.__dict__.items() if k != "content"}]
                }, auto_sync=False)
    
    def _process_stream_event(self, event, section) -> str:
        """
        Process a single streaming event from OpenAI Responses API.
        
        Handles various event types: text deltas, code interpreter output,
        image generation, and file citations.
        """
        if event.type == "response.output_text.delta":
            section.update("text", event.delta)
            section.stream()
            return event.delta
        elif event.type == "response.code_interpreter_call_code.delta":
            section.update("code", event.delta)
            section.stream()
        elif event.type == "response.image_generation_call.partial_image":
            image_bytes = base64.b64decode(event.partial_image_b64)
            filename = f"{getattr(event, 'item_id', 'image')}.{getattr(event, 'output_format', 'png')}"
            section.update("image", image_bytes, filename=filename, file_id=getattr(event, 'item_id', None))
            section.stream()
        elif event.type == "response.output_text.annotation.added":
            annotation = event.annotation
            if annotation["type"] == "container_file_citation":
                file_id = annotation["file_id"]
                filename = annotation["filename"]
                file_bytes = None
                
                if hasattr(self, '_client') and hasattr(self, '_container_id') and self._client and self._container_id:
                    file_content = self._client.containers.files.content.retrieve(
                        file_id=file_id, container_id=self._container_id
                    )
                    file_bytes = file_content.read()
                    
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                    section.update("image", file_bytes, filename=filename, file_id=file_id)
                    section.update("download", file_bytes, filename=filename, file_id=file_id)
                    section.stream()
                else:
                    section.update("download", file_bytes, filename=filename, file_id=file_id)
                    section.stream()
        return ""
    
    def _generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using the configured workflow or dynamically selected agents."""
        try:
            if self.workflow_executor:
                # Use workflow execution
                return self._run_workflow(prompt)
            elif self.agent_manager.agents:
                # Single agent mode (validated to be exactly 1 agent in __init__)
                agent = list(self.agent_manager.agents.values())[0]
                return self._run_agent(prompt, agent)
        except Exception as e:
            # Show error to user 
            st.error(f"**Error**: {str(e)}")
            # Return empty response to prevent further processing
            return {"role": "assistant", "content": "", "agent": "system"}
    
    def _run_workflow(self, prompt: str) -> Dict[str, Any]:
        """
        Run the multiagent workflow and orchestrate UI updates.
        
        Coordinates workflow execution with display callbacks, HITL handling,
        and state synchronization.
        """
        def display_callback(msg, msg_id):
            """Callback to display agent responses as they complete during workflow execution."""
            self.display_manager.render_workflow_message(msg)
        
        # Use execution coordinator for workflow execution
        result_state = self.execution_coordinator.execute_workflow(
            self.workflow, prompt, display_callback=display_callback
        )

        if HITLUtils.has_pending_interrupts(result_state):
            st.session_state.workflow_state = result_state
            self.state_coordinator.sync_to_session_state()
            st.rerun()
        else:
            self.state_coordinator.clear_hitl_state()

        st.session_state.workflow_state = result_state
        self.state_coordinator.sync_to_session_state()
        
        return {
            "role": "assistant",
            "content": "",
            "agent": "workflow-completed"
        }
    
    def _run_agent(self, prompt: str, agent: Agent) -> Dict[str, Any]:
        """
        Run a single agent (non-workflow mode) and orchestrate UI updates.
        
        Coordinates agent execution with HITL handling and state synchronization.
        """
        file_messages = self.file_handler.get_openai_input_messages()
        
        # Use execution coordinator for single agent execution
        response = self.execution_coordinator.execute_single_agent(
            agent, prompt, file_messages=file_messages
        )
        
        # Handle interrupts from ResponseAPIExecutor
        if response.get("__interrupt__"):
            self.state_coordinator.set_pending_interrupt(
                agent.name,
                response,
                "single_agent_executor"
            )
            return response
        
        # Update workflow_state with agent response using coordinator
        if response.get("content"):
            self.state_coordinator.add_assistant_message(
                response.get("content", ""),
                response.get("agent", agent.name)
            )
        
        return response
