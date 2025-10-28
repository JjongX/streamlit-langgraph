from streamlit_langgraph import Agent, UIConfig, LangGraphChat

def main():
    # Define a simple agent using type='agent' (LangChain agent)
    assistant = Agent(
        name="assistant",
        role="Helpful Assistant",
        instructions="You are a helpful assistant that can answer questions and have conversations.",
        type="agent",  # Use LangChain agent, not OpenAI response API
        provider="openai",
        model="gpt-4.1"  
    )
    config = UIConfig(
        title="Simple LangChain Agent Example",
        welcome_message="Hello! Ask me anything. I am powered by a LangChain agent."
    )
    chat = LangGraphChat(agents=[assistant], config=config)
    chat.run()

if __name__ == "__main__":
    main()
