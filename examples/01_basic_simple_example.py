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

💬 **Single Agent Mode**: This example demonstrates direct single-agent interaction without workflow complexity.

I'm a helpful AI assistant ready to chat with you about anything. I can help with:

- 💡 **Questions & Answers**: Ask me about any topic
- 🤔 **Problem Solving**: Work through challenges together  
- 📚 **Learning**: Explain concepts and ideas
- 🔍 **Research**: Search the web for current information
- 💬 **Conversation**: Just chat about your day!

What would you like to talk about?""",
    )
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            agents=[assistant],
            config=config
        )
    st.session_state.chat.run()

if __name__ == "__main__":
    main()
