# Main chat interface.

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple

import streamlit as st
from langgraph.graph import StateGraph

from .agent import Agent, get_llm_client
from .core.executor import WorkflowExecutor
from .core.executor.registry import ExecutorRegistry
from .core.executor.response_api import ResponseAPIExecutor
from .core.middleware import HITLUtils
from .core.runtime import RuntimeHooks
from .core.state import WorkflowState, WorkflowStateManager
from .ui import (
    DisplayManager,
    HITLHandler,
    NonStreamProcessor,
    StreamProcessor,
    StreamlitStateSynchronizer,
    StreamlitStreamRenderer,
)
from .utils import FileHandler, CustomTool


@dataclass
class UIConfig:
    """
    Streamlit UI configuration.
    
    Attributes:
        title: Application title shown in browser tab and header
        page_icon: Favicon emoji or path to image file
        page_layout: Page layout mode ("wide" or "centered")
        stream: Enable streaming responses
        enable_file_upload: File upload configuration (False, True, "multiple", or "directory")
        show_sidebar: Show default sidebar (set False for custom)
        user_avatar: Avatar for user messages (emoji or image path)
        assistant_avatar: Avatar for assistant messages (emoji or image path)
        placeholder: Placeholder text for chat input
        welcome_message: Welcome message shown at start (supports Markdown)
        file_callback: Optional callback to preprocess files before upload.
            Can return a single file path (str) or a tuple (main_file_path, additional_files)
            where additional_files can be a directory path or list of file paths.
            Additional files will be automatically uploaded to code_interpreter container if enabled.
    """
    title: str = "LangGraph Chat"
    page_icon: Optional[str] = "🤖"
    page_layout: str = "wide"
    stream: bool = True
    # Might change to a boolean and default to getting multiple if set to True.
    enable_file_upload: Union[bool, Literal["multiple", "directory"]] = "multiple"
    show_sidebar: bool = True
    user_avatar: Optional[str] = "👤"
    assistant_avatar: Optional[str] = "🤖"
    placeholder: str = "Type your message here..."
    welcome_message: Optional[str] = None
    file_callback: Optional[Callable[[str], Union[str, Tuple[str, Union[str, List[str], Path]]]]] = None


class LangGraphChat:
    """
    Main chat interface for Streamlit and LangGraph workflows.
    
    This class manages the entire chat interface, including UI rendering,
    message handling, file processing, and workflow execution.
    """
    
    def __init__(
        self,
        workflow: Optional[StateGraph] = None,
        agents: Optional[List[Agent]] = None,
        config: Optional[UIConfig] = None,
        custom_tools: Optional[List[CustomTool]] = None,
    ):
        """
        Initialize the LangGraph Chat interface.

        Args:
            workflow: LangGraph workflow (StateGraph) for multi-agent scenarios
            agents: List of agents to use
            config: Chat configuration
            custom_tools: List of custom tools to register
        """
        self.workflow = workflow
        self.agents = self._validate_and_build_agents(workflow, agents)
        self.config = config or UIConfig()

        self._init_session_state()
        self.state_manager = StreamlitStateSynchronizer()
        self.display_manager = DisplayManager(self.config, state_manager=self.state_manager)
        self._init_registry_and_components(workflow, custom_tools)

    def _validate_and_build_agents(
        self,
        workflow: Optional[StateGraph],
        agents: Optional[List[Agent]],
    ) -> Dict[str, Agent]:
        """Validate `workflow`/`agents` combination and build agent map."""
        agent_map: Dict[str, Agent] = {}
        if agents:
            if not workflow and len(agents) > 1:
                raise ValueError(
                    "Multiple agents require a workflow. "
                    "Either provide a workflow parameter or use a single agent."
                )
            for agent in agents:
                if agent.human_in_loop and not workflow:
                    raise ValueError("Human-in-the-loop is only available for multiagent workflows.")
                agent_map[agent.name] = agent

        if not agent_map:
            raise ValueError("At least one agent is required to initialize LangGraphChat.")
        return agent_map

    def _init_registry_and_components(
        self,
        workflow: Optional[StateGraph],
        custom_tools: Optional[List[CustomTool]],
    ) -> None:
        """Initialize all registry and components."""
        self._initialize_runtime(workflow)
        self._register_custom_tools(custom_tools)
        _, openai_client = self._initialize_file_and_clients()
        self._initialize_ui_components(openai_client)

    def _initialize_runtime(self, workflow: Optional[StateGraph]) -> None:
        """Initialize runtime hooks and workflow executor."""
        workflow_runtime = getattr(workflow, "runtime_hooks", None) if workflow else None
        self.runtime = workflow_runtime or RuntimeHooks(executor_registry=ExecutorRegistry())
        if self.runtime.spinner is None:
            self.runtime.spinner = st.spinner
        self.workflow_executor = WorkflowExecutor(self.runtime.executor_registry)

    def _register_custom_tools(self, custom_tools: Optional[List[CustomTool]]) -> None:
        """Register user-provided custom tools."""
        if not custom_tools:
            return
        for tool in custom_tools:
            CustomTool.register_tool(
                tool.name,
                tool.description,
                tool.function,
                parameters=tool.parameters,
                return_direct=tool.return_direct,
            )

    def _initialize_file_and_clients(self) -> tuple[Agent, Any]:
        """Initialize file handler and model clients."""
        first_agent = next(iter(self.agents.values()))
        openai_client = None
        if (
            first_agent.provider.lower() == "openai"
            and ExecutorRegistry.has_openai_native_tools(first_agent)
        ):
            executor = self.runtime.executor_registry.get_or_create(
                first_agent, executor_type="single_agent"
            )
            if isinstance(executor, ResponseAPIExecutor):
                openai_client = getattr(executor, "openai_client", None)

        self.file_handler = FileHandler(
            openai_client=openai_client,
            model=first_agent.model,
            allow_file_search=first_agent.allow_file_search,
            allow_code_interpreter=first_agent.allow_code_interpreter,
            container_id=first_agent.container_id,
            preprocessing_callback=self.config.file_callback,
        )
        self._sync_container_ids(first_agent)

        vector_store_ids = self.file_handler.get_vector_store_ids()
        self.llm = get_llm_client(first_agent, vector_store_ids=vector_store_ids)
        self._client = openai_client
        self._container_id = first_agent.container_id
        return first_agent, openai_client

    def _sync_container_ids(self, first_agent: Agent) -> None:
        """Sync container_id from FileHandler to all code-interpreter agents."""
        if self.file_handler._container_id and not first_agent.container_id:
            first_agent.container_id = self.file_handler._container_id
        if self.file_handler._container_id:
            Agent.sync_container_ids(list(self.agents.values()))

    def _initialize_ui_components(self, openai_client: Any) -> None:
        """Initialize UI processors and interrupt handling."""
        if self.runtime.stream_renderer is None:
            self.runtime.stream_renderer = StreamlitStreamRenderer(
                config=self.config,
                state_manager=self.state_manager,
                client=openai_client,
            )
        self.interrupt_handler = HITLHandler(
            self.agents, self.config,
            self.state_manager, self.display_manager, self.runtime.executor_registry,
        )
        self.stream_processor = StreamProcessor(client=self._client, container_id=self._container_id)
        self.nonstream_processor = NonStreamProcessor()
    
    def _init_session_state(self):
        """Initialize all Streamlit session state variables in one place."""
        if "workflow_state" not in st.session_state:
            st.session_state.workflow_state = WorkflowState(
                messages=[],
                current_agent=None,
                agent_outputs={},
                files=[],
                metadata={"stream": self.config.stream},
            )
        else:
            WorkflowStateManager.ensure_stream_flag(
                st.session_state.workflow_state,
                self.config.stream,
            )
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "uploaded_files_set" not in st.session_state:
            st.session_state.uploaded_files_set = set()
    
    def _get_workflow_state(self) -> Dict[str, Any]:
        """Get workflow state, initializing metadata if needed."""
        workflow_state = st.session_state.workflow_state
        WorkflowStateManager.ensure_stream_flag(workflow_state, self.config.stream)
        return workflow_state

    def run(self):
        """Run the main chat interface."""
        st.set_page_config(
            page_title=self.config.title,
            page_icon=self.config.page_icon,
            layout=self.config.page_layout
        )
        st.title(self.config.title)
        
        if self.config.show_sidebar:
            self._render_sidebar()
        self._render_chat_interface()

    
    def _update_file_messages_in_state(self, force=False):
        """Update file messages and vector store IDs in workflow state."""
        workflow_state = self._get_workflow_state()
        self.file_handler.update_vector_store_metadata(self.state_manager, workflow_state, force=force)
    
    def _process_file_uploads(self, files):
        """Process uploaded files and update workflow state."""
        for uploaded_file in files:
            file_id = getattr(uploaded_file, 'file_id', None) or uploaded_file.name
            if file_id not in st.session_state.uploaded_files_set:
                file_info = self.file_handler.track(uploaded_file)
                st.session_state.uploaded_files.append(uploaded_file)
                st.session_state.uploaded_files_set.add(file_id)
                file_dict = {k: v for k, v in file_info.__dict__.items() if k != "content"}
                self.state_manager.update_workflow_state({"files": [file_dict]})
        
        if self.file_handler._container_id:
            all_agents = list(self.agents.values())
            Agent.sync_container_ids(all_agents)
        
        self._update_file_messages_in_state(force=True)

    def _run_agent(self, prompt, agent):
        """Run single agent (HITL not supported - use workflows for HITL)."""
        # Get file messages and vector store IDs from state
        workflow_state = self._get_workflow_state()
        metadata = workflow_state.get("metadata", {})
        file_messages, vector_store_ids = metadata.get("file_messages"), metadata.get("vector_store_ids")
        self.llm = FileHandler.ensure_llm_vector_ids(agent, self.llm, vector_store_ids)
        
        response = self.workflow_executor.execute_agent(
            agent,
            prompt,
            llm_client=self.llm,
            config=self.config,
            file_messages=file_messages,
            vector_store_ids=vector_store_ids,
            messages=workflow_state.get("messages", []),
        )
        
        if agent.container_id:
            self._container_id = agent.container_id
            self.stream_processor._container_id = agent.container_id
            self.file_handler.update_settings(
                allow_file_search=agent.allow_file_search,
                allow_code_interpreter=agent.allow_code_interpreter,
                container_id=agent.container_id,
            )

        return response
    
    def _run_workflow(self, prompt):
        """Execute multiagent workflow and handle UI updates."""
        self.state_manager.update_workflow_state({"metadata": {"stream": self.config.stream}})
        WorkflowStateManager.ensure_stream_flag(st.session_state.workflow_state, self.config.stream)
        self._update_file_messages_in_state()
        
        result_state = self.workflow_executor.execute_workflow(
            self.workflow,
            display_callback=self.display_manager.render_workflow_message,
            initial_state=st.session_state.workflow_state,
            displayed_message_ids=self.state_manager.get_displayed_message_ids(),
        )

        if HITLUtils.has_pending_interrupts(result_state):
            WorkflowStateManager.preserve_display_sections(
                st.session_state.workflow_state, result_state
            )
            st.session_state.workflow_state = result_state
            st.rerun()
        else:
            self.state_manager.clear_hitl_state()

        WorkflowStateManager.preserve_display_sections(
            st.session_state.workflow_state, result_state
        )
        st.session_state.workflow_state = result_state
        
        return {"id": str(uuid.uuid4()), "role": "assistant", "content": "", "agent": "workflow-completed"}
    
    def _generate_response(self, prompt):
        """Generate response using the configured workflow or dynamically selected agents."""
        if self.workflow:
            return self._run_workflow(prompt)
        elif self.agents:
            agent = next(iter(self.agents.values()))
            return self._run_agent(prompt, agent)
        return {"id": str(uuid.uuid4()), "role": "assistant", "content": "", "agent": "system"}

    def _handle_user_input(self, chat_input):
        """Handle user input and generate responses."""
        if self.config.enable_file_upload:
            prompt = chat_input.text
            files = getattr(chat_input, 'files', [])
        else:
            prompt = str(chat_input)
            files = []

        # Add user message to state
        self.state_manager.add_user_message(prompt)
        section = self.display_manager.add_section("user")
        section.update("text", prompt)
        for uploaded_file in files:
            section.update("text", f"\n:material/attach_file: `{uploaded_file.name}`")
        section.stream()
        
        if files:
            with st.spinner("Processing files..."):
                self._process_file_uploads(files)
        
        self.state_manager.clear_hitl_state()
        
        with st.spinner("Thinking..."):
            response = self._generate_response(prompt)

        if response.get("agent") == "workflow-completed":
            return
        if response.get("__interrupt__"):
            st.rerun()

        section = self.display_manager.add_section("assistant")
        section._agent_info = {"agent": response["agent"]}
        if "stream" in response:
            stream_iter = response["stream"]
            full_response = self.stream_processor.process_stream(section, stream_iter)
            response["content"] = full_response
        else:
            full_response = self.nonstream_processor.process_nonstream(section, response)
            response["content"] = full_response

        self._persist_assistant_message(response)

    def _persist_assistant_message(self, response: dict) -> None:
        """Persist assistant responses to workflow state when applicable."""
        content = response.get("content")
        agent = response.get("agent")
        if content and agent not in ["workflow", "workflow-completed"]:
            self.state_manager.add_assistant_message(content, agent)

    def _render_sidebar(self):
        """Render the sidebar with controls and information."""
        with st.sidebar:
            st.header("Agent Configuration")
            agents = list(self.agents.values())
            if agents:
                for agent in agents:
                    with st.expander(f"{agent.name}", expanded=False):
                        st.write(f"**Role:** {agent.role}")
                        st.write(f"**Instructions:** {agent.instructions[:100]}...")
                        capabilities = []
                        if agent.allow_file_search:
                            capabilities.append("📁 File Search")
                        if agent.allow_code_interpreter:
                            capabilities.append("💻 Code Interpreter")
                        if agent.allow_web_search:
                            capabilities.append("🌐 Web Search")
                        if agent.tools:
                            capabilities.append(f"🛠️ {len(agent.tools)} Custom Tools")
                        if capabilities:
                            st.write("**Capabilities:**")
                            for cap in capabilities:
                                st.write(f"- {cap}")
            st.header("Controls")
            if st.button("Reset All", type="secondary"):
                self.file_handler.reset()
                self._container_id = None
                self.stream_processor._container_id = None
                st.session_state.clear()
                self._init_session_state()
                
                st.rerun()

    def _render_chat_interface(self):
        """Render the main chat interface."""
        display_sections = self.state_manager.get_display_sections()
        if not display_sections:
            self.display_manager.render_welcome_message()

        workflow_state = self._get_workflow_state()
        if HITLUtils.has_pending_interrupts(workflow_state):
            interrupt_handled = self.interrupt_handler.handle_pending_interrupts(workflow_state)
            if interrupt_handled:
                return  # Don't process messages or show input while handling interrupts

        self.display_manager.render_message_history()
        if prompt := st.chat_input(
            self.config.placeholder, accept_file=self.config.enable_file_upload
        ):
            self._handle_user_input(prompt)
