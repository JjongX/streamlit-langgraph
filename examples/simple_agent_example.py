from streamlit_langgraph import Agent, UIConfig, LangGraphChat

def main():
    # Define a simple agent using type='agent' (LangChain agent)
    simple_agent = Agent(
        name="SimpleLangChainAgent",
        role="Helpful Assistant",
        instructions="You are a helpful assistant that answers user questions using LangChain agent capabilities.",
        type="agent",  # Use LangChain agent, not OpenAI response API
        provider="openai",
        model="gpt-4.1"  
    )
    config = UIConfig(
        title="Simple LangChain Agent Example",
        welcome_message="Hello! Ask me anything. I am powered by a LangChain agent."
    )
    chat = LangGraphChat(agents=[simple_agent], config=config)
    chat.run()

if __name__ == "__main__":
    main()
