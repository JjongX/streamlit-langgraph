from streamlit_langgraph import Agent, UIConfig, LangGraphChat

def main():    
    # Create a simple assistant
    assistant = Agent(
        name="assistant",
        role="Helpful Assistant", 
        instructions="You are a helpful assistant that can answer questions and have conversations.",
        provider="openai",
        model="gpt-4.1-mini",
        temperature=0.7,
        ## Response mode
        type="response",
        allow_file_search=True,
        allow_code_interpreter=True,
        allow_web_search=True,
        ## Agent mode
        # type="agent",
    )
    # Create UI configuration
    config = UIConfig(
        title="Simple Chat Assistant",
        page_icon="💬",
        stream=True,
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
    # Create the chat interface (single agent, no workflow needed)
    chat = LangGraphChat(
        agents=[assistant],
        config=config
    )
    # Run the chat
    chat.run()

if __name__ == "__main__":
    main()
