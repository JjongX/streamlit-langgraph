# Main chat interface.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from .agent import Agent, AgentManager
from .core.executor import WorkflowExecutor
from .core.executor.registry import ExecutorRegistry
from .core.state import StateSynchronizer, WorkflowStateManager
from .core.middleware import HITLHandler, HITLUtils
from .ui import DisplayManager, StreamProcessor
from .utils import FileHandler, CustomTool


@dataclass
class UIConfig:
    """Configuration for the Streamlit UI interface."""
    title: str
    page_icon: Optional[str] = "🤖"
    page_layout: str = "wide"
    stream: bool = True
    enable_file_upload: bool = True
    show_sidebar: bool = True
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
        self._init_session_state()
        self.agent_manager = AgentManager()
        self.state_manager = StateSynchronizer()
        self.display_manager = DisplayManager(self.config, state_manager=self.state_manager)
        self.workflow = workflow
        self.workflow_executor = WorkflowExecutor()
        
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
        if custom_tools:
            for tool in custom_tools:
                CustomTool.register_tool(
                    tool.name, tool.description, tool.function, 
                    parameters=tool.parameters, return_direct=tool.return_direct
                )
        
        first_agent = next(iter(self.agent_manager.agents.values()))
        
        openai_client = None
        if (first_agent.provider.lower() == "openai" and
            ExecutorRegistry.has_native_tools(first_agent)):
            executor = ExecutorRegistry().get_or_create(first_agent, executor_type="single_agent")
            from .core.executor.response_api import ResponseAPIExecutor
            if isinstance(executor, ResponseAPIExecutor):
                openai_client = executor.openai_client
        
        self.file_handler = FileHandler(
            openai_client=openai_client,
            model=first_agent.model,
            allow_file_search=first_agent.allow_file_search,
            allow_code_interpreter=first_agent.allow_code_interpreter,
            container_id=first_agent.container_id,
        )
        
        vector_store_ids = self.file_handler.get_vector_store_ids()
        self.llm = AgentManager.get_llm_client(first_agent, vector_store_ids=vector_store_ids)
        self._client = openai_client
        self._container_id = first_agent.container_id
        self.interrupt_handler = HITLHandler(self.agent_manager, self.config, self.state_manager, self.display_manager)
        self.stream_processor = StreamProcessor(client=self._client, container_id=self._container_id)
    
    def _init_session_state(self):
        """Initialize all Streamlit session state variables in one place."""
        
        if "workflow_state" not in st.session_state:
            st.session_state.workflow_state = WorkflowStateManager.create_initial_state()
        if "agent_executors" not in st.session_state:
            st.session_state.agent_executors = {}
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

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
    
    def _render_sidebar(self):
        """Render the sidebar with controls and information."""
        with st.sidebar:
            st.header("Agent Configuration")
            agents = list(self.agent_manager.agents.values())
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
                st.session_state.clear()
                st.rerun()
    
    def _render_chat_interface(self):
        """Render the main chat interface."""
        display_sections = self.state_manager.get_display_sections()
        if not display_sections:
            self.display_manager.render_welcome_message()

        workflow_state = st.session_state.workflow_state
        if HITLUtils.has_pending_interrupts(workflow_state):
            interrupt_handled = self.interrupt_handler.handle_pending_interrupts(workflow_state)
            if interrupt_handled:
                return  # Don't process messages or show input while handling interrupts

        self.display_manager.render_message_history()
        if prompt := st.chat_input(
            self.config.placeholder, accept_file=self.config.enable_file_upload
        ):
            self._handle_user_input(prompt)
    
    def _handle_user_input(self, chat_input):
        """Handle user input and generate responses."""
        if self.config.enable_file_upload:
            prompt = chat_input.text
            files = getattr(chat_input, 'files', [])
        else:
            prompt = str(chat_input)
            files = []

        if files:
            self._process_file_uploads(files)
        
        workflow_state = st.session_state.workflow_state
        if "metadata" not in workflow_state:
            workflow_state["metadata"] = {}
        
        self.state_manager.add_user_message(prompt)
        
        section = self.display_manager.add_section("user")
        section.update("text", prompt)
        for uploaded_file in files:
            section.update("text", f"\n:material/attach_file: `{uploaded_file.name}`")
        section.stream()

        self.state_manager.clear_hitl_state()
        
        with st.spinner("Thinking..."):
            response = self._generate_response(prompt)

        if response.get("agent") == "workflow-completed":
            return
        if response.get("__interrupt__"):
            st.rerun()
        
        if response and "stream" in response:
            section = self.display_manager.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}
            stream_iter = response["stream"]
            full_response = self.stream_processor.process_stream(section, stream_iter)
            response["content"] = full_response
        else:
            section = self.display_manager.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}
            section.update("text", response["content"])
            section.stream()

        if (response.get("content") and 
            response.get("agent") not in ["workflow", "workflow-completed"]):
            self.state_manager.add_assistant_message(
                response["content"], 
                response["agent"]
            )
    
    def _process_file_uploads(self, files):
        """Process uploaded files and update workflow state."""
        for uploaded_file in files:
            if uploaded_file not in st.session_state.uploaded_files:
                file_info = self.file_handler.save_uploaded_file(uploaded_file)
                st.session_state.uploaded_files.append(uploaded_file)
                self.state_manager.update_workflow_state({
                    "files": [{k: v for k, v in file_info.__dict__.items() if k != "content"}]
                })
    
    def _generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using the configured workflow or dynamically selected agents."""
        if self.workflow:
            return self._run_workflow(prompt)
        elif self.agent_manager.agents:
            agent = list(self.agent_manager.agents.values())[0]
            return self._run_agent(prompt, agent)
        return {"role": "assistant", "content": "", "agent": "system"}
    
    def _run_workflow(self, prompt: str) -> Dict[str, Any]:
        """Run the multiagent workflow and orchestrate UI updates."""
        workflow_state = st.session_state.workflow_state
        if "metadata" not in workflow_state:
            workflow_state["metadata"] = {}
        workflow_state["metadata"]["stream"] = self.config.stream
        
        def display_callback(msg, msg_id):
            self.display_manager.render_workflow_message(msg)
        
        result_state = self.workflow_executor.execute_workflow(
            self.workflow, display_callback=display_callback
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
        
        return {
            "role": "assistant",
            "content": "",
            "agent": "workflow-completed"
        }
    
    def _run_agent(self, prompt: str, agent: Agent) -> Dict[str, Any]:
        """
        Run a single agent and orchestrate UI updates.
        
        Note: HITL is not supported for single agents. Use workflows for HITL functionality.
        """
        file_messages = self.file_handler.get_openai_input_messages()
        vector_store_ids = self.file_handler.get_vector_store_ids()
        
        if agent.allow_file_search and vector_store_ids:
            current_vector_ids = getattr(self.llm, '_vector_store_ids', None)
            if current_vector_ids != vector_store_ids:
                self.llm = AgentManager.get_llm_client(agent, vector_store_ids=vector_store_ids)
        
        response = self.workflow_executor.execute_agent(
            agent, prompt,
            llm_client=self.llm,
            config=self.config,
            file_messages=file_messages
        )
        
        if agent.container_id:
            self._container_id = agent.container_id
            self.stream_processor._container_id = agent.container_id
            self.file_handler.update_settings(
                allow_file_search=agent.allow_file_search,
                allow_code_interpreter=agent.allow_code_interpreter,
                container_id=agent.container_id,
            )
        
        if response.get("content"):
            self.state_manager.add_assistant_message(
                response.get("content", ""),
                response.get("agent", agent.name)
            )
        
        return response
