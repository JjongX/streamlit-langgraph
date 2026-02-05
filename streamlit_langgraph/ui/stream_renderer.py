# Streamlit renderer for streaming agent output.

from .display_manager import DisplayManager
from .stream_processor import StreamProcessor
from .streamlit_state import StreamlitStateSynchronizer


class StreamlitStreamRenderer:
    """Render streaming responses into Streamlit chat sections."""

    def __init__(self, config, state_manager: StreamlitStateSynchronizer, client=None):
        self.config = config
        self.state_manager = state_manager
        self.client = client

    def render(self, agent, stream, message_id: str) -> str:
        """Render stream events and return the full response text."""
        display_manager = DisplayManager(config=self.config, state_manager=self.state_manager)
        stream_processor = StreamProcessor(
            client=self.client,
            container_id=getattr(agent, "container_id", None),
        )
        section = display_manager.add_section("assistant")
        section._agent_info = {"agent": agent.name}
        section._message_id = message_id
        return stream_processor.process_stream(section, stream)
