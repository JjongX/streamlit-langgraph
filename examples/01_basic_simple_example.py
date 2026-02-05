import os
import streamlit as st
import streamlit_langgraph as slg


def main():    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/01_basic_simple.yaml")
    assistant = slg.Agent(config_path)
    config = slg.UIConfig(
        title="Simple Chat Assistant",
        page_icon="💬",
        welcome_message="""Welcome to the **Simple Chat Assistant**!

Single-agent example. Try the sample prompts below to test different features.

**Sample prompts (FAQ):**
- **General chat**: "What is 2+2?" or "Tell me a short joke."
- **Web search**: "What is the latest news about AI today?"
- **File search**: "Summarize the file I uploaded." (enable `allow_file_search` in config)
- **Code interpreter**: "Plot a sine wave from 0 to 2π" or "Run: print('Hello')"
- **Image generation**: "Generate an image of a sunset over the ocean." (enable `allow_image_generation` in config)""",
    )
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            agents=[assistant],
            config=config
        )
    st.session_state.chat.run()

if __name__ == "__main__":
    main()
