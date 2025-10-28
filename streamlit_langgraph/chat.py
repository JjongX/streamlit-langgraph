import streamlit as st
import os
import openai
from langchain.chat_models import init_chat_model
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import base64

from .agent import Agent, AgentManager, ResponseAPIExecutor, CreateAgentExecutor
from .workflow import WorkflowExecutor, create_initial_state
from .utils import CustomTool, FileHandler, MIME_TYPES

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
        # Initialize workflow
        self.workflow = workflow
        self.workflow_executor = WorkflowExecutor() if workflow else None
        # Initialize agents
        if agents:
            for agent in agents:
                self.agent_manager.add_agent(agent)
        # Register custom tools
        if custom_tools:
            for tool in custom_tools:
                CustomTool.register_tool(tool.name, tool.description, tool.function, parameters=tool.parameters, return_direct=tool.return_direct)

        # Initialize LLM (provider-flexible)
        self.llm = self._initialize_llm()
        # Initialize FileHandler with OpenAI client if available
        openai_client = self.llm if hasattr(self.llm, 'files') else None
        self.file_handler = FileHandler(openai_client=openai_client)
        # Initialize Streamlit session state
        self._init_session_state()
        # Initialize sections for streaming
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

    @property
    def last_section(self) -> Optional["Section"]:
        """Returns the last section of the chat."""
        return self._sections[-1] if self._sections else None
    
    def _initialize_llm(self):
        """Initialize an LLM client based on the first agent's provider/model/type."""
        if not self.agent_manager.agents:
            st.error("No agents defined. Please define at least one agent with provider, model, and type.")
            st.stop()
        first_agent = next(iter(self.agent_manager.agents.values()))
        provider = first_agent.provider.lower()
        model_name = first_agent.model
        agent_type = first_agent.type
        if not provider or not model_name or not agent_type:
            st.error("Each agent must specify provider, model, and type ('response' or 'agent').")
            st.stop()
        if provider == "openai" and agent_type == "response":
            api_key = None
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
            else:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Please add OPENAI_API_KEY to .streamlit/secrets.toml or set the environment variable.")
                st.info("Create .streamlit/secrets.toml and add: OPENAI_API_KEY = \"your-api-key\"")
                st.stop()
            try:
                client = openai.OpenAI(api_key=api_key)
                client._provider = "openai"
                return client
            except Exception:
                raise
        # For agent_type == 'agent' or non-OpenAI providers, use LangChain
        try:
            if model_name:
                chat_model = init_chat_model(model=model_name)
            else:
                chat_model = init_chat_model()
            setattr(chat_model, "_provider", provider)
            return chat_model
        except Exception as e:
            st.error(f"Failed to initialize LLM for provider '{provider}': {str(e)}")
            st.stop()
    
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
            st.title("🤖 Agent Configuration")
            
            agents = list(self.agent_manager.agents.values())
            if agents:
                st.subheader("Active Agents")
                for agent in agents:
                    with st.expander(f"🤖 {agent.name}", expanded=False):
                        st.write(f"**Role:** {agent.role}")
                        st.write(f"**Instructions:** {agent.instructions[:100]}...")
                        
                        # Display capabilities
                        capabilities = []
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

        for i, message in enumerate(st.session_state.messages):
            if message.get("role") == "assistant" and (message.get("agent") == "END" or message.get("agent") is None):
                continue
            avatar = self.config.user_avatar if message["role"] == "user" else self.config.assistant_avatar
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                # Display agent info for assistant messages
                if message["role"] == "assistant" and "agent" in message:
                    st.caption(f"Agent: {message['agent']}")

        # Chat input with file upload support
        if prompt := st.chat_input(
            self.config.placeholder, 
            accept_file=self.config.enable_file_upload
        ):
            self._handle_user_input(prompt)
    
    def _handle_user_input(self, chat_input):
        """Handle user input and generate responses."""
        # Extract text and files from chat input
        if hasattr(chat_input, 'text'):
            # New format with text and files
            prompt = chat_input.text
            files = getattr(chat_input, 'files', [])
        else:
            # Backward compatibility - just text
            prompt = str(chat_input)
            files = []
        
        # Handle file uploads if any
        if files:
            for uploaded_file in files:
                if uploaded_file not in st.session_state.uploaded_files:
                    file_info = self.file_handler.save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_files.append(uploaded_file)
                    # Optionally, add file info to workflow state if needed
                    if "files" in st.session_state.workflow_state:
                        st.session_state.workflow_state["files"].append({
                            k: v for k, v in file_info.__dict__.items() if k != "content"
                        })
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user", avatar=self.config.user_avatar):
            st.markdown(prompt)
            
            # Display uploaded files if any
            if files:
                for uploaded_file in files:
                    st.markdown(f":material/attach_file: `{uploaded_file.name}`")
        
        # Update workflow state
        user_message = {"role": "user", "content": prompt, "agent": None, "timestamp": None}
        st.session_state.workflow_state["messages"].append(user_message)
        
        # Generate response
        with st.spinner("Thinking..."):
            response = self._generate_response(prompt)

        # Display response based on type
        if response.get("agent") == "workflow-completed":
            # Workflow execution completed - responses were already displayed sequentially
            pass  # No additional display needed
        elif response and "stream" in response:
            # Streaming single agent response using Section
            section = self.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}

            full_response = ""
            try:
                for event in response["stream"]:
                    # Handle OpenAI Responses API events properly
                    if hasattr(event, 'type'):
                        # Handle text output delta events
                        if event.type == "response.output_text.delta":
                            content_delta = event.delta
                            full_response += content_delta
                            section.update_and_stream("text", content_delta)
                        # Handle code interpreter code output events
                        elif event.type == "response.code_interpreter_call_code.delta":
                            code_delta = event.delta
                            section.update_and_stream("code", code_delta)
                        # Handle image generation or plot output events
                        elif event.type == "response.image_generation_call.partial_image":
                            image_bytes = base64.b64decode(event.partial_image_b64)
                            section.update_and_stream("image", image_bytes, filename=f"{getattr(event, 'item_id', 'image')}.{getattr(event, 'output_format', 'png')}", file_id=getattr(event, 'item_id', None))
                        # Handle file download or container file citation events
                        elif event.type == "response.output_text.annotation.added":
                            annotation = event.annotation
                            if annotation["type"] == "container_file_citation":
                                file_id = annotation["file_id"]
                                filename = annotation["filename"]
                                file_bytes = None
                                if hasattr(self, '_client') and hasattr(self, '_container_id') and self._client and self._container_id:
                                    file_content = self._client.containers.files.content.retrieve(
                                        file_id=file_id,
                                        container_id=self._container_id
                                    )
                                    file_bytes = file_content.read()
                                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                                    # Show both image and download button for image files
                                    section.update_and_stream("image", file_bytes, filename=filename, file_id=file_id)
                                    section.update_and_stream("download", file_bytes, filename=filename, file_id=file_id)
                                else:
                                    section.update_and_stream("download", file_bytes, filename=filename, file_id=file_id)
                        # Handle other event types as needed
                        elif event.type == "response.completed":
                            pass  # Response completed
                    # Fallback: Handle ResponseTextDeltaEvent objects directly
                    elif hasattr(event, 'delta') and event.delta:
                        content_delta = str(event.delta)
                        full_response += content_delta
                        section.update_and_stream("text", content_delta)
                    # Handle ResponseTextDoneEvent - text contains the complete text
                    elif hasattr(event, 'text') and event.text:
                        content_delta = str(event.text)
                        full_response += content_delta
                        section.update_and_stream("text", content_delta)
                    # Legacy format for chat completions - delta.content structure
                    elif hasattr(event, 'delta') and hasattr(event.delta, 'content'):
                        if event.delta.content:
                            content_delta = event.delta.content
                            full_response += content_delta
                            section.update_and_stream("text", content_delta)
                    # Fallback for backwards compatibility with chat completions
                    elif hasattr(event, 'choices') and len(event.choices) > 0:
                        if hasattr(event.choices[0], 'delta') and hasattr(event.choices[0].delta, 'content'):
                            if event.choices[0].delta.content:
                                content_delta = event.choices[0].delta.content
                                full_response += content_delta
                                section.update_and_stream("text", content_delta)

                response["content"] = full_response  # Update response for session state

            except Exception as e:
                import traceback
                traceback.print_exc()
                section.update_and_stream("text", f"Streaming error: {str(e)}")
                response["content"] = f"Streaming error: {str(e)}"
        elif response and "content" in response and response["content"]:
            # Single agent response or non-workflow response using Section
            section = self.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}
            section.update_and_stream("text", response["content"])
        else:
            # Error case
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.error("No response content was generated.")

        # Add assistant response to chat only if it has content and isn't a workflow completion indicator
        if (response.get("content") and 
            response.get("agent") not in ["workflow", "workflow-completed"]):
            # Clean response before adding to session state (remove stream object)
            clean_response = {
                "role": response["role"],
                "content": response["content"],
                "agent": response["agent"]
            }
            st.session_state.messages.append(clean_response)
    
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
            else:
                return {
                    "role": "assistant",
                    "content": "No agents are available. Please define agents in your application.",
                    "agent": "system"
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}",
                "agent": "system"
            }
    
    def _execute_workflow(self, prompt: str) -> Dict[str, Any]:
        """Execute the LangGraph workflow with sequential agent display."""
        
        # Reset displayed count for this new workflow execution
        st.session_state.workflow_displayed_count = 0
        
        # Execute workflow with streaming display callback
        def display_agent_response(state):
            """Callback to display agent responses as they complete."""
            if state and "messages" in state:
                # Get assistant messages from the workflow state
                assistant_messages = [
                    msg for msg in state["messages"] 
                    if msg["role"] == "assistant" and msg.get("agent") and msg.get("agent") != "system"
                ]
                
                # Check if there are new assistant messages to display
                if len(assistant_messages) > st.session_state.workflow_displayed_count:
                    # Get the new messages that haven't been displayed yet
                    new_messages = assistant_messages[st.session_state.workflow_displayed_count:]
                    
                    for new_msg in new_messages:
                        agent_name = new_msg.get("agent", "Assistant")
                        
                        # Display the agent response sequentially using Section
                        section = self.add_section("assistant")
                        section._agent_info = {"agent": agent_name}
                        section.update_and_stream("text", new_msg['content'])
                        
                        # Add to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": new_msg['content'],
                            "agent": agent_name
                        })
                        
                        # Update the displayed count
                        st.session_state.workflow_displayed_count += 1
        
        try:
            # Execute workflow with display callback for sequential updates
            result_state = self.workflow_executor.execute_workflow(
                self.workflow, 
                prompt, 
                display_callback=display_agent_response
            )
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"❌ **Error executing workflow: {str(e)}**",
                "agent": "workflow"
            }
        
        st.session_state.workflow_state = result_state
        
        agent_messages = [
            msg for msg in result_state["messages"]
            if (
                msg["role"] == "assistant"
                and msg.get("agent")
                and msg.get("agent") not in ["workflow", "END", "__end__"]
            )
        ]
        
        if not agent_messages:
            return {
                "role": "assistant",
                "content": "❌ **No agent responses generated from workflow**",
                "agent": "workflow"
            }
        
        last_agent = result_state["messages"][-1].get("agent") if result_state["messages"] else None
        if last_agent in ["END", "__end__"]:
            completion_message = "✅ **Workflow completed successfully**"
        else:
            completion_message = ""
        
        return {
            "role": "assistant",
            "content": completion_message,
            "agent": "workflow-completed"
        }

    def _build_context(self) -> str:
        """Build context string from conversation history."""
        context_parts = []
        
        # Recent messages
        recent_messages = st.session_state.messages[-5:] if st.session_state.messages else []
        for msg in recent_messages:
            role = msg["role"].title()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            context_parts.append(f"{role}: {content}")
        
        # Uploaded files summary
        if st.session_state.uploaded_files:
            file_context = self.file_handler.get_file_context_summary()
            context_parts.append(f"\n--- Uploaded Files ---\n{file_context}")
        
        return "\n".join(context_parts) if context_parts else "No previous context"
    
    def _execute_agent(self, prompt: str, agent: Agent) -> Dict[str, Any]:
        """Execute prompt with a specific agent, using explicit type for routing."""
        if not self.llm:
            return {
                "role": "assistant",
                "content": f"As a {agent.role}, I would help you, but I need an OpenAI API key to function properly. Please add OPENAI_API_KEY to .streamlit/secrets.toml",
                "agent": agent.name
            }
        try:
            context = self._build_context()
            tool_descriptions = []
            if agent.allow_web_search:
                tool_descriptions.append("🌐 **Web Search**: Search the internet for current information")
            if agent.allow_code_interpreter:
                tool_descriptions.append("🐍 **Code Interpreter**: Execute Python code and create visualizations")
            if agent.allow_file_search:
                tool_descriptions.append("📁 **File Search**: Search through uploaded documents")
            for tool in agent.tools:
                if tool in CustomTool._registry:
                    tool_obj = CustomTool._registry[tool]
                    tool_descriptions.append(f"🔧 **{tool_obj.name}**: {tool_obj.description}")
            file_context_note = ""
            if st.session_state.uploaded_files:
                file_context_note = f"""

IMPORTANT: The user has uploaded {len(st.session_state.uploaded_files)} file(s). The file contents are included in the conversation context below. You can read, analyze, and discuss these files directly. When referring to file contents, be specific and helpful."""
            enhanced_instructions = f"""You are {agent.role}. {agent.instructions}

Available tools:
{chr(10).join(tool_descriptions) if tool_descriptions else "No special tools available"}{file_context_note}

Current conversation context:
{context}

User: {prompt}
Assistant:"""
            # agent_provider = agent.provider.lower()
            file_messages = self.file_handler.get_openai_input_messages()
            if agent.type == "response":
                executor = ResponseAPIExecutor(agent)
                return executor.execute(self.llm, enhanced_instructions, stream=self.config.stream, file_messages=file_messages)
            else:
                executor = CreateAgentExecutor(agent, tools=[])
                return executor.execute(self.llm, prompt, stream=False)
        except Exception as e:
            return {
                "role": "assistant",
                "content": f"I encountered an error while processing your request: {str(e)}",
                "agent": agent.name
            }
    
    def add_agent(self, agent: Agent):
        """Add an agent to the chat."""
        self.agent_manager.add_agent(agent)
    
    def remove_agent(self, agent_name: str):
        """Remove an agent from the chat."""
        self.agent_manager.remove_agent(agent_name)
    
    def add_tool(self, tool: CustomTool):
        """Add a custom tool to the chat."""
        CustomTool.register_tool(tool.name, tool.description, tool.function, parameters=tool.parameters, return_direct=tool.return_direct)
    
    def set_workflow(self, workflow):
        """Set or update the workflow."""
        self.workflow = workflow
        self.workflow_executor = WorkflowExecutor() if workflow else None
