import streamlit_langgraph as slg


def main():    
    assistant = slg.Agent(
        name="assistant",
        role="Helpful Assistant", 
        instructions=(
            "You are a helpful assistant that can answer questions and have conversations. "
            "If you do not know the answer, just state that you do not know."
        ),
        provider="openai",
        model="gpt-4.1",
        temperature=0.7,
        # Native OpenAI tools are automatically handled via Responses API when enabled
        allow_file_search=True,
        allow_code_interpreter=True,
        allow_web_search=True,
        allow_image_generation=True,
        enable_logging=True,
    )
    config = slg.UIConfig(
        title="Simple Chat Assistant",
        page_icon="💬",
        welcome_message="""Welcome to the **Simple Chat Assistant**!

💬 **Single Agent Mode**: This example demonstrates direct single-agent interaction without workflow complexity.

I'm a helpful AI assistant ready to chat with you about anything. I can help with:

💡 **Questions & Answers**: Ask me about any topic
🤔 **Problem Solving**: Work through challenges together  
📚 **Learning**: Explain concepts and ideas
🔍 **Research**: Search the web for current information
💬 **Conversation**: Just chat about your day!

What would you like to talk about?""",
        enable_file_upload=True,
    )
    chat = slg.LangGraphChat(
        agents=[assistant],
        config=config
    )
    chat.run()

if __name__ == "__main__":
    main()
