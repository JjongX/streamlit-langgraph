
# streamlit-langgraph

A Python package that integrates Streamlit’s intuitive web interface with LangGraph’s advanced multi-agent orchestration. Build interactive AI applications featuring multiple specialized agents collaborating in customizable workflows.

If you’re using Streamlit with a single agent, consider [streamlit-openai](https://github.com/sbslee/streamlit-openai/tree/main) instead. This project is inspired by that work, especially its integration with the OpenAI API.

## Main Goal

The main goals of this package are:

1. **Seamless Integration of Streamlit and LangGraph:** Enables users to leverage Streamlit’s rapid UI development capabilities alongside LangGraph’s flexible agent orchestration. This integration allows for real-time interaction, monitoring, and control of agent workflows directly from the browser, making multi-agent systems more accessible and transparent.

2. **Lowering the Barrier to Multi-Agent Orchestration:** Multi-agent frameworks like AutoGen, CrewAI, and OpenAI Agents SDK offer various approaches, each with its own pros and cons. LangGraph provides granular control and advanced features, but comes with a steep learning curve and fragmented documentation. This package abstracts away much of the complexity, offering simple interfaces and templates so users can focus on designing agent logic.

3. **Ready-to-Use Multi-Agent Architectures:** In real-world applications, common agent architectures such as supervisor, hierarchical, and networked systems are frequently needed. This package includes these patterns natively, allowing users to select and customize them without reinventing the wheel.

4. **Enhanced Compatibility with OpenAI Response API:** While LangChain supports OpenAI’s chat completion API, the newer Response API introduces advanced features like code interpretation and file search. By targeting compatibility with the Response API, this package ensures users can take advantage of the latest capabilities from OpenAI, making their agent workflows more powerful and versatile.

5. **Extensibility to Other LLMs:** The landscape of large language models is rapidly evolving, with alternatives like Gemini, Claude, and various local models offering unique strengths. This package is designed to be extensible, so users can experiment with different LLMs and select the best model for their specific use case, whether it’s coding, reasoning, or domain-specific tasks.

Ultimately, streamlit-langgraph aims to empower users to build, experiment with, and deploy multi-agent orchestration systems quickly and intuitively, while remaining flexible enough to adapt to new models and workflows as the field evolves.

## Status

This project is in **pre-alpha**. Features and APIs are subject to change, and not all goals are fully implemented.

**Note:** Uses `langchain`/`langgraph` version 1.0.1.
