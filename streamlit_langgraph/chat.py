# Main chat interface.

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from .agent import Agent, AgentManager
from .core.executor import WorkflowExecutor
from .core.state import StateSynchronizer, WorkflowStateManager
from .core.middleware import HITLHandler, HITLUtils
from .ui import DisplayManager
from .utils import FileHandler, CustomTool


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
        # Configuration and session state
        self.config = config or UIConfig()
        self._init_session_state()
        # Core managers
        self.agent_manager = AgentManager()
        self.state_manager = StateSynchronizer()
        self.display_manager = DisplayManager(self.config)
        # Workflow setup
        self.workflow = workflow
        self.workflow_executor = WorkflowExecutor()
        
        # Agent and tool setup
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
        
        # LLM and file handling
        first_agent = next(iter(self.agent_manager.agents.values()))
        self.llm = AgentManager.get_llm_client(first_agent)
        openai_client = self.llm if hasattr(self.llm, 'files') else None
        self.file_handler = FileHandler(openai_client=openai_client)
        self._client = self.llm if hasattr(self.llm, 'containers') else None
        self._container_id = None
        
        # Handlers
        self.interrupt_handler = HITLHandler(self.agent_manager, self.config, self.state_manager, self.display_manager)
    
    def _init_session_state(self):
        """Initialize all Streamlit session state variables in one place."""
        
        if "workflow_state" not in st.session_state:
            st.session_state.workflow_state = WorkflowStateManager.create_initial_state()
        if "agent_executors" not in st.session_state:
            st.session_state.agent_executors = {}
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []  # File objects (not in workflow_state)
        if "display_sections" not in st.session_state:
            st.session_state.display_sections = []  # UI sections for persistence across reruns

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
            st.header("Agent Configuration")
            agents = list(self.agent_manager.agents.values())
            if agents:
                for agent in agents:
                    with st.expander(f"{agent.name}", expanded=False):
                        st.write(f"**Role:** {agent.role}")
                        st.write(f"**Instructions:** {agent.instructions[:100]}...")
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
            st.header("Controls")
            if st.button("Reset All", type="secondary"):
                st.session_state.clear()
                st.rerun()
    
    def _render_chat_interface(self):
        """Render the main chat interface."""
        if not st.session_state.display_sections:
            self.display_manager.render_welcome_message()

        # Check for pending interrupts FIRST - workflow_state is the single source of truth
        workflow_state = st.session_state.workflow_state
        if HITLUtils.has_pending_interrupts(workflow_state):
            interrupt_handled = self.interrupt_handler.handle_pending_interrupts(workflow_state)
            if interrupt_handled:
                return  # Don't process messages or show input while handling interrupts

        # Render message history
        self.display_manager.render_message_history()
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
        
        self.state_manager.add_user_message(prompt)
        
        section = self.display_manager.add_section("user")
        section.update("text", prompt)
        for uploaded_file in files:
            section.update("text", f"\n:material/attach_file: `{uploaded_file.name}`")
        section.stream()

        # Clear HITL state before new request
        self.state_manager.clear_hitl_state()
        
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
            # Handle streaming response - detect format and route to appropriate handler
            section = self.display_manager.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}
            
            stream_iter = response["stream"]
            full_response = self._process_stream(section, stream_iter)
            response["content"] = full_response
        else:
            # Handle non-streaming response from agent
            section = self.display_manager.add_section("assistant")
            section._agent_info = {"agent": response["agent"]}
            section.update("text", response["content"])
            section.stream()

        # Add assistant response to state if not a workflow control message
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
                # Add file metadata to workflow state (not content)
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
        """
        Run the multiagent workflow and orchestrate UI updates.
        
        Coordinates workflow execution with display callbacks, HITL handling,
        and state synchronization.
        """
        def display_callback(msg, msg_id):
            """Callback to display agent responses as they complete during workflow execution."""
            self.display_manager.render_workflow_message(msg)
        
        # Use WorkflowExecutor directly (orchestrator logic merged in)
        result_state = self.workflow_executor.execute_workflow(
            self.workflow, display_callback=display_callback
        )

        if HITLUtils.has_pending_interrupts(result_state):
            st.session_state.workflow_state = result_state
            st.rerun()
        else:
            self.state_manager.clear_hitl_state()

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
        
        # Use WorkflowExecutor directly (orchestrator logic merged in)
        response = self.workflow_executor.execute_agent(
            agent, prompt,
            llm_client=self.llm,
            config=self.config,
            file_messages=file_messages
        )
        
        # Update container_id if agent has code interpreter (for file retrieval)
        if agent.container_id:
            self._container_id = agent.container_id
        
        # Update workflow_state with agent response using state manager
        if response.get("content"):
            self.state_manager.add_assistant_message(
                response.get("content", ""),
                response.get("agent", agent.name)
            )
        
        return response

    def _process_stream(self, section, stream_iter) -> str:
        """
        Main entry point for processing streaming responses.
        
        Detects the stream format and routes to appropriate handler:
        - Responses API: Direct OpenAI Responses API stream events
        - LangChain: LangChain stream_mode="messages" format (tuples) or "updates" format (dicts)
        
        Args:
            section: Display section to update
            stream_iter: Stream iterator (format varies by source)
            
        Returns:
            Full accumulated response text
        """
        events_list = []
        stream_type = None
        full_response = ""
        
        try:
            for event in stream_iter:
                events_list.append(event)
                
                # Detect stream type from first event
                if stream_type is None:
                    if isinstance(event, tuple) and len(event) == 2:
                        stream_type = 'langchain_messages'
                    elif hasattr(event, 'type'):
                        stream_type = 'responses_api'
                    elif isinstance(event, dict):
                        stream_type = 'langchain_updates'
                    else:
                        stream_type = 'unknown'
                
                # Route to appropriate handler
                if stream_type == 'responses_api':
                    delta = self._process_responses_api_stream(event, section)
                    if delta:
                        full_response += delta
                elif stream_type == 'langchain_messages':
                    token, metadata = event
                    delta = self._process_langchain_message_token(token, section)
                    if delta:
                        full_response += delta
                elif stream_type == 'langchain_updates':
                    partial_response = self._process_langchain_stream_event(event, section, full_response)
                    if partial_response and len(partial_response) > len(full_response):
                        full_response = partial_response
            
            # Fallback: extract from final state if no content was streamed
            if not full_response and events_list:
                full_response = self._extract_final_content(events_list, stream_type, section)
                
        except (StopIteration, TypeError, AttributeError):
            # Empty stream or invalid format - try to extract from final state
            if events_list:
                full_response = self._extract_final_content(events_list, stream_type, section)
        
        return full_response
    
    def _process_responses_api_stream(self, event, section) -> str:
        """
        Process a single event from OpenAI Responses API stream.
        
        This handles direct Responses API streaming (bypassing LangChain wrapper).
        
        Args:
            event: Responses API stream event object
            section: Display section to update
            
        Returns:
            Delta text to append to response
        """
        return self._process_stream_event(event, section)
    
    def _process_langchain_message_token(self, token, section) -> str:
        """
        Process a single token from LangChain stream_mode="messages".
        
        According to LangChain docs, stream_mode="messages" returns (token, metadata) tuples
        where token is an AIMessage with content_blocks attribute.
        
        Args:
            token: AIMessage with content_blocks attribute
            section: Display section to update
            
        Returns:
            Delta text to append to response
        """
        from langchain_core.messages import AIMessage
        
        if not isinstance(token, AIMessage):
            return ""
        
        # Extract text from content_blocks
        # content_blocks is a list of dicts: [{'type': 'text', 'text': '...'}, ...]
        if hasattr(token, 'content_blocks') and token.content_blocks:
            text_parts = []
            for block in token.content_blocks:
                if isinstance(block, dict):
                    if block.get('type') == 'text' and block.get('text'):
                        text_parts.append(block.get('text', ''))
            if text_parts:
                delta = ''.join(text_parts)
                if delta:
                    section.update("text", delta)
                    section.stream()
                    return delta
        
        # Fallback: try content attribute
        if hasattr(token, 'content') and token.content:
            if isinstance(token.content, str) and token.content:
                section.update("text", token.content)
                section.stream()
                return token.content
            elif isinstance(token.content, list):
                text_parts = []
                for block in token.content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif isinstance(block, str):
                        text_parts.append(block)
                if text_parts:
                    delta = ''.join(text_parts)
                    if delta:
                        section.update("text", delta)
                        section.stream()
                        return delta
        
        return ""
    
    def _process_langchain_stream_event(self, event: dict, section, accumulated_content: str = "") -> str:
        """
        Process a single LangChain/LangGraph stream event incrementally.
        
        LangChain streams return dictionaries with node names as keys,
        containing state updates with messages.
        
        Args:
            event: Single stream event (dict)
            section: Display section to update
            accumulated_content: Previously accumulated content
            
        Returns:
            Updated full response content
        """
        from langchain_core.messages import AIMessage
        
        if not isinstance(event, dict):
            return accumulated_content
        
        # LangGraph stream format: {node_name: state_update}
        for node_name, state_update in event.items():
            if isinstance(state_update, dict):
                # Check for messages in state update
                if 'messages' in state_update:
                    messages = state_update['messages']
                    # Find new AIMessages
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if hasattr(msg, 'content') and msg.content:
                                if isinstance(msg.content, str):
                                    new_content = msg.content
                                    # Only append new content (delta)
                                    if new_content.startswith(accumulated_content):
                                        delta = new_content[len(accumulated_content):]
                                        if delta:
                                            section.update("text", delta)
                                            section.stream()
                                            return new_content
                                    else:
                                        # Full replacement (new content doesn't start with accumulated)
                                        section.update("text", new_content)
                                        section.stream()
                                        return new_content
                                elif isinstance(msg.content, list):
                                    # Handle list content (Responses API format)
                                    text_parts = []
                                    for block in msg.content:
                                        if isinstance(block, dict) and block.get('type') == 'text':
                                            text_parts.append(block.get('text', ''))
                                        elif isinstance(block, str):
                                            text_parts.append(block)
                                    if text_parts:
                                        new_content = ''.join(text_parts)
                                        # Check if this is new content
                                        if new_content != accumulated_content:
                                            if new_content.startswith(accumulated_content):
                                                delta = new_content[len(accumulated_content):]
                                                if delta:
                                                    section.update("text", delta)
                                                    section.stream()
                                            else:
                                                section.update("text", new_content)
                                                section.stream()
                                            return new_content
                # Check for output in state update
                elif 'output' in state_update:
                    output = state_update['output']
                    if output and str(output) != accumulated_content:
                        section.update("text", str(output))
                        section.stream()
                        return str(output)
        
        return accumulated_content
    
    def _process_stream_event(self, event, section) -> str:
        """
        Process a single streaming event from OpenAI Responses API.
        
        Handles various event types: text deltas, code interpreter output,
        image generation, and file citations.
        
        Args:
            event: Responses API stream event object
            section: Display section to update
            
        Returns:
            Delta text to append to response
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
                if file_bytes:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                        section.update("image", file_bytes, filename=filename, file_id=file_id)
                        section.update("download", file_bytes, filename=filename, file_id=file_id)
                        section.stream()
                    else:
                        section.update("download", file_bytes, filename=filename, file_id=file_id)
                        section.stream()
                        
        return ""
    
    def _extract_final_content(self, events_list, stream_type, section) -> str:
        """
        Extract final content from stream events when streaming didn't yield content.
        
        Args:
            events_list: List of all stream events
            stream_type: Detected stream type ('responses_api', 'langchain_messages', 'langchain_updates')
            section: Display section to update
            
        Returns:
            Extracted final content text
        """
        if not events_list:
            return ""
        
        last_event = events_list[-1]
        full_response = ""
        
        if stream_type == 'langchain_messages' and isinstance(last_event, tuple):
            # Extract from last token in messages mode
            token, metadata = last_event
            from langchain_core.messages import AIMessage
            if isinstance(token, AIMessage):
                if hasattr(token, 'content_blocks') and token.content_blocks:
                    text_parts = []
                    for block in token.content_blocks:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                    if text_parts:
                        full_response = ''.join(text_parts)
                elif hasattr(token, 'content') and token.content:
                    if isinstance(token.content, str):
                        full_response = token.content
                    elif isinstance(token.content, list):
                        text_parts = []
                        for block in token.content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_parts.append(block.get('text', ''))
                            elif isinstance(block, str):
                                text_parts.append(block)
                        if text_parts:
                            full_response = ''.join(text_parts)
                if full_response:
                    section.update("text", full_response)
                    section.stream()
        elif stream_type == 'langchain_updates' and isinstance(last_event, dict):
            # Handle updates mode format
            for node_name, state_update in last_event.items():
                if isinstance(state_update, dict) and 'messages' in state_update:
                    from langchain_core.messages import AIMessage
                    messages = state_update['messages']
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                            if isinstance(msg.content, str) and msg.content:
                                full_response = msg.content
                                section.update("text", full_response)
                                section.stream()
                                break
                        if full_response:
                            break
        
        return full_response
