import base64
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from .agent import Agent, AgentManager, ResponseAPIExecutor, CreateAgentExecutor, get_llm_client
from .utils import FileHandler, CustomTool, MIME_TYPES
from .workflow import WorkflowExecutor, create_initial_state

@dataclass
class UIConfig:
    """Configuration for the Streamlit UI interface."""
    title: str
    page_icon: Optional[str] = "🤖"
    page_layout: str = "wide"
    stream: bool = True  # Enable/disable streaming responses
    enable_file_upload: bool = True
    user_avatar: Optional[str] = "👤"
    assistant_avatar: Optional[str] = "🤖"
    placeholder: str = "Type your message here..."
    welcome_message: Optional[str] = None
    info_message: Optional[str] = None

class LangGraphChat:
    """
    Main class for creating agent chat interfaces with Streamlit and LangGraph.
    """
    
    def __init__(
        self,
        workflow=None,
        agents: Optional[List[Agent]] = None,
        config: Optional[UIConfig] = None,
        custom_tools: Optional[List[CustomTool]] = None,
    ):
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
            for agent in agents:
                self.agent_manager.add_agent(agent)
        # Register custom tools
        if custom_tools:
            for tool in custom_tools:
                CustomTool.register_tool(
                    tool.name, tool.description, tool.function, 
                    parameters=tool.parameters, return_direct=tool.return_direct
                )
        self.llm = self._initialize_llm()
        openai_client = self.llm if hasattr(self.llm, 'files') else None
        self.file_handler = FileHandler(openai_client=openai_client)
        self._init_session_state()
        self._sections = []
        self._download_button_key = 0
    
    class Block:
        """A block of content in the chat."""
        def __init__(
            self,
            chat: "LangGraphChat",
            category: str,
            content: Optional[str] = None,
            filename: Optional[str] = None,
            file_id: Optional[str] = None,
        ) -> None:
            self.chat = chat
            self.category = category
            self.content = content or ""
            self.filename = filename
            self.file_id = file_id

        def iscategory(self, category) -> bool:
            """Checks if the block belongs to the specified category."""
            return self.category == category

        def write(self) -> None:
            """Renders the block's content to the chat."""
            if self.category == "text":
                st.markdown(self.content)
            elif self.category == "code":
                with st.expander("", expanded=False, icon=":material/code:"):
                    st.code(self.content)
            elif self.category == "reasoning":
                with st.expander("", expanded=False, icon=":material/lightbulb:"):
                    st.markdown(self.content)
            elif self.category == "download":
                _, file_extension = os.path.splitext(self.filename)
                st.download_button(
                    label=self.filename,
                    data=self.content,
                    file_name=self.filename,
                    mime=MIME_TYPES[file_extension.lstrip(".")],
                    key=self.chat._download_button_key,
                )
                self.chat._download_button_key += 1

    class Section:
        """A section of the chat."""
        def __init__(
            self,
            chat: "LangGraphChat",
            role: str,
            blocks: Optional[List["LangGraphChat.Block"]] = None,
        ) -> None:
            self.chat = chat
            self.role = role
            self.blocks = blocks or []
            self.delta_generator = st.empty()
            
        @property
        def empty(self) -> bool:
            """Returns True if the section has no blocks."""
            return len(self.blocks) == 0

        @property
        def last_block(self) -> Optional["LangGraphChat.Block"]:
            """Returns the last block in the section or None if empty."""
            return None if self.empty else self.blocks[-1]

        def update(self, category, content, filename=None, file_id=None) -> None:
            """Updates the section with new content, appending or extending existing blocks."""
            if self.empty:
                self.blocks = [self.chat.create_block(
                    category, content, filename=filename, file_id=file_id
                )]
            elif category in ["text", "code", "reasoning"] and self.last_block.iscategory(category):
                self.last_block.content += content
            else:
                self.blocks.append(self.chat.create_block(
                    category, content, filename=filename, file_id=file_id
                ))

        def update_and_stream(self, category, content, filename=None, file_id=None) -> None:
            """Updates the section and streams the update live to the UI."""
            self.update(category, content, filename=filename, file_id=file_id)
            self.stream()

        def stream(self) -> None:
            """Renders the section content using Streamlit's delta generator."""
            with self.delta_generator:
                with st.chat_message(self.role, avatar=self.chat.config.user_avatar if self.role == "user" else self.chat.config.assistant_avatar):
                    for block in self.blocks:
                        block.write()
                    if hasattr(self, '_agent_info') and "agent" in self._agent_info:
                        st.caption(f"Agent: {self._agent_info['agent']}")

    def create_block(self, category, content=None, filename=None, file_id=None) -> "Block":
        return self.Block(self, category, content=content, filename=filename, file_id=file_id)

    def add_section(self, role, blocks=None) -> "Section":
        section = self.Section(self, role, blocks=blocks)
        self._sections.append(section)
        return section
    
    def _initialize_llm(self):
        """
        Initialize an LLM client based on the first agent's configuration.
        
        Note: Currently uses the first agent's provider/model/type for the entire chat session.
        """
        first_agent = next(iter(self.agent_manager.agents.values()))
        return get_llm_client(first_agent)

    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "current_agent" not in st.session_state:
            st.session_state.current_agent = None
        
        if "workflow_state" not in st.session_state:
            st.session_state.workflow_state = create_initial_state()
        
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        
        if "agent_outputs" not in st.session_state:
            st.session_state.agent_outputs = {}
    
    def run(self):
        """Run the main chat interface."""
        st.set_page_config(
            page_title=self.config.title,
            page_icon=self.config.page_icon,
            layout=self.config.page_layout
        )
        st.title(self.config.title)
        
        if self.config.info_message:
            st.info(self.config.info_message)
        
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
                        if hasattr(agent, 'allow_web_search') and agent.allow_web_search:
                            capabilities.append("🌐 Web Search")
                        if hasattr(agent, 'allow_code_interpreter') and agent.allow_code_interpreter:
                            capabilities.append("💻 Code Interpreter")
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
        # Display welcome message
        if self.config.welcome_message and not st.session_state.messages:
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.markdown(self.config.welcome_message)

        for message in st.session_state.messages:
            if message.get("role") == "assistant" and (message.get("agent") == "END" or message.get("agent") is None):
                continue
            avatar = self.config.user_avatar if message["role"] == "user" else self.config.assistant_avatar
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "agent" in message:
                    st.caption(f"Agent: {message['agent']}")

        if prompt := st.chat_input(
            self.config.placeholder, 
            accept_file=self.config.enable_file_upload
        ):
            self._handle_user_input(prompt)
    
    def _handle_user_input(self, chat_input):
        """Handle user input and generate responses."""
        if hasattr(chat_input, 'text'):
            prompt = chat_input.text
            files = getattr(chat_input, 'files', [])
        else:
            prompt = str(chat_input)
            files = []
        
        if files:
            self._process_file_uploads(files)
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message and attached files in the chat
        with st.chat_message("user", avatar=self.config.user_avatar):
            st.markdown(prompt)
            if files:
                for uploaded_file in files:
                    st.markdown(f":material/attach_file: `{uploaded_file.name}`")
        
        user_message = {"role": "user", "content": prompt, "agent": None, "timestamp": None}
        st.session_state.workflow_state["messages"].append(user_message)
        
        with st.spinner("Thinking..."):
            response = self._generate_response(prompt)

        if response.get("agent") == "workflow-completed":
            return
        
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
            section.update_and_stream("text", response["content"])

        if (response.get("content") and 
            response.get("agent") not in ["workflow", "workflow-completed"]):
            st.session_state.messages.append(response)
    
    def _process_file_uploads(self, files):
        """Process uploaded files and update workflow state."""
        for uploaded_file in files:
            if uploaded_file not in st.session_state.uploaded_files:
                file_info = self.file_handler.save_uploaded_file(uploaded_file)
                st.session_state.uploaded_files.append(uploaded_file)
                if "files" in st.session_state.workflow_state:
                    st.session_state.workflow_state["files"].append({
                        k: v for k, v in file_info.__dict__.items() if k != "content"
                    })
    
    def _process_stream_event(self, event, section) -> str:
        """
        Process a single streaming event from OpenAI Responses API.
        
        Handles various event types: text deltas, code interpreter output,
        image generation, and file citations.
        """
        if event.type == "response.output_text.delta":
            section.update_and_stream("text", event.delta)
            return event.delta
        elif event.type == "response.code_interpreter_call_code.delta":
            section.update_and_stream("code", event.delta)
        elif event.type == "response.image_generation_call.partial_image":
            image_bytes = base64.b64decode(event.partial_image_b64)
            filename = f"{getattr(event, 'item_id', 'image')}.{getattr(event, 'output_format', 'png')}"
            section.update_and_stream("image", image_bytes, filename=filename, file_id=getattr(event, 'item_id', None))
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
                    section.update_and_stream("image", file_bytes, filename=filename, file_id=file_id)
                    section.update_and_stream("download", file_bytes, filename=filename, file_id=file_id)
                else:
                    section.update_and_stream("download", file_bytes, filename=filename, file_id=file_id)
        return ""
    
    def _generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using the configured workflow or dynamically selected agents."""
        try:
            if self.workflow_executor:
                # Use workflow execution
                return self._execute_workflow(prompt)
            elif self.agent_manager.agents:
                # Use simple agent selection - first available agent
                agent = next(iter(self.agent_manager.agents.values()))
                return self._execute_agent(prompt, agent)
        except Exception as e:
            traceback.print_exc()
            return {
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}",
                "agent": "system"
            }
    
    def _execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow with sequential agent display.
        
        Uses a display callback to show agent responses as they complete during
        workflow execution. This provides real-time feedback in the Streamlit UI.
        """
        st.session_state.workflow_displayed_count = 0
        
        def display_agent_response(state):
            """Callback to display agent responses as they complete during workflow execution."""
            if state and "messages" in state:
                assistant_messages = [
                    msg for msg in state["messages"] 
                    if msg["role"] == "assistant" and msg.get("agent") and msg.get("agent") != "system"
                ]
                
                if len(assistant_messages) > st.session_state.workflow_displayed_count:
                    new_messages = assistant_messages[st.session_state.workflow_displayed_count:]
                    for new_msg in new_messages:
                        agent_name = new_msg.get("agent", "Assistant")
                        section = self.add_section("assistant")
                        section._agent_info = {"agent": agent_name}
                        section.update_and_stream("text", new_msg['content'])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": new_msg['content'],
                            "agent": agent_name
                        })
                        st.session_state.workflow_displayed_count += 1
        
        try:
            result_state = self.workflow_executor.execute_workflow(
                self.workflow, prompt, display_callback=display_agent_response
            )
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"❌ **Error executing workflow: {str(e)}**",
                "agent": "workflow"
            }
        
        st.session_state.workflow_state = result_state
        return {
            "role": "assistant",
            "content": "",
            "agent": "workflow-completed"
        }

    def _build_context(self) -> str:
        """Build context string from recent conversation history and uploaded files."""
        context_parts = []
        recent_messages = st.session_state.messages[-5:] if st.session_state.messages else []
        
        for msg in recent_messages:
            role = msg["role"].title()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            context_parts.append(f"{role}: {content}")
        
        if st.session_state.uploaded_files:
            file_context = self.file_handler.get_file_context_summary()
            context_parts.append(f"\n--- Uploaded Files ---\n{file_context}")
        
        return "\n".join(context_parts) if context_parts else "No previous context"
    
    def _execute_agent(self, prompt: str, agent: Agent) -> Dict[str, Any]:
        """Execute prompt with a specific agent for single-agent responses."""
        file_messages = self.file_handler.get_openai_input_messages()
        
        if agent.type == "response":
            executor = ResponseAPIExecutor(agent)
            return executor.execute(self.llm, prompt, stream=self.config.stream, file_messages=file_messages)
        else:
            executor = CreateAgentExecutor(agent, tools=[])
            return executor.execute(self.llm, prompt, stream=False)

    