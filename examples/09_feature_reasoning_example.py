import os
import streamlit as st
import streamlit_langgraph as slg


def main():    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/09_feature_reasoning.yaml")
    assistant = slg.Agent(config_path)
    config = slg.UIConfig(
        title="Reasoning Chat Assistant",
        page_icon="🧠",
        welcome_message="""Welcome to the **Reasoning Chat Assistant**!

💬 **Single Agent Mode with Reasoning**: This example demonstrates how to use OpenAI's reasoning effort feature to see the AI's thinking process.

🧠 **Reasoning Effort**: This assistant uses `reasoning_effort: medium` to show its thinking process. You'll see collapsible reasoning blocks (💡) that reveal how the AI thinks through problems step by step!

**Note**: Reasoning models (like gpt-5.2) don't support custom temperature settings. The reasoning_effort parameter controls how deeply the model thinks before responding.

I'm a helpful AI assistant ready to chat with you about anything. I can help with:

- 💡 **Questions & Answers**: Ask me about any topic (see my reasoning!)
- 🤔 **Problem Solving**: Work through challenges together (watch me think!)
- 📚 **Learning**: Explain concepts and ideas with detailed reasoning
- 🔍 **Research**: Search the web for current information
- 💬 **Complex Conversations**: Discuss nuanced topics with structured thinking

Try asking me a complex question or problem to see my reasoning process in action!

What would you like to explore?""",
    )
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            agents=[assistant],
            config=config,
        )
    st.session_state.chat.run()

if __name__ == "__main__":
    main()
