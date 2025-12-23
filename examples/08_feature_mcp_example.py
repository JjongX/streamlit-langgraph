# Example demonstrating MCP (Model Context Protocol) tool integration.

import streamlit as st
import streamlit_langgraph as slg


def main():    
    mcp_servers = {
        # stdio transport
        # "math": {
        #     "transport": "stdio",
        #     "command": "python",
        #     "args": [os.path.join(os.path.dirname(__file__), "mcp_servers", "math_server.py")]
        # },
        # http transport - need mcp at the end of the url for agent
        # "math": {
        #     "transport": "streamable_http",
        #     "url": "http://127.0.0.1:8000/mcp"
        # }
    }
    
    agent = slg.Agent(
        name="calculator_agent",
        role="Math Assistant",
        instructions="You are a helpful math assistant that can perform calculations using MCP tools.",
        provider="openai",
        model="gpt-4o-mini",
        tools=[],
        mcp_servers=mcp_servers
    )
    # Create chat interface
    config = slg.UIConfig(
        title="MCP Tools Chat",
        welcome_message="""Welcome to the **MCP Tools Example**!

This example demonstrates using MCP (Model Context Protocol) tools with agents.

## 🧮 Available Tools:
- **Add**: Add two numbers together
- **Multiply**: Multiply two numbers
- **Subtract**: Subtract one number from another
- **Divide**: Divide one number by another

## ❓ Example Queries:
- *"What is 15 + 27?"*
- *"Multiply 8 by 9"*
- *"Calculate 100 divided by 4"*
- *"Subtract 45 from 100"*
- *"What's 12 * 7 + 5?"*
"""
    )
    
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            agents=[agent],
            config=config
        )
    st.session_state.chat.run()


if __name__ == "__main__":
    main()

