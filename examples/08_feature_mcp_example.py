# Example demonstrating MCP (Model Context Protocol) tool integration.

import os
import streamlit as st
import streamlit_langgraph as slg


def main():    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/08_mcp_example.yaml")
    agent = slg.Agent(config_path)
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

