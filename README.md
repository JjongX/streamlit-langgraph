
# streamlit-langgraph

[![PyPI version](https://badge.fury.io/py/streamlit-langgraph.svg)](https://badge.fury.io/py/streamlit-langgraph)

A Python package that integrates Streamlit’s intuitive web interface with LangGraph’s advanced multi-agent orchestration. Build interactive AI applications featuring multiple specialized agents collaborating in customizable workflows.

If you’re using Streamlit with a single agent, consider [streamlit-openai](https://github.com/sbslee/streamlit-openai/tree/main) instead. This project is inspired by that work, especially its integration with the OpenAI API.

## Main Goal

To build successful multi-agent systems, defining agent instructions, tasks, and context is more important than the actual orchestration logic. As illustrated by statements from two different frameworks:

###### [LangChain - Customizing agent context](https://docs.langchain.com/oss/python/langchain/multi-agent#customizing-agent-context)
```
At the heart of multi-agent design is **context engineering** - deciding what information each agent sees. LangChain gives you fine-grained control over:

- Which parts of the conversation or state are passed to each agent.
- Specialized prompts tailored to subagents.
- Inclusion/exclusion of intermediate reasoning.
- Customizing input/output formats per agent.

The quality of your system **heavily depends** on **context engineering**. The goal is to ensure that each agent has access to the correct data it needs to perform its task, whether it’s acting as a tool or as an active agent.
```

###### [CrewAI - The 80/20 Rule: Focus on Tasks Over Agents](https://docs.crewai.com/en/guides/agents/crafting-effective-agents#the-80%2F20-rule%3A-focus-on-tasks-over-agents)
```
The 80/20 Rule: Focus on Tasks Over Agents

When building effective AI systems, remember this crucial principle:

80% of your effort should go into designing tasks, and only 20% into defining agents

.Why? Because even the most perfectly defined agent will fail with poorly designed tasks, but well-designed tasks can elevate even a simple agent. This means:

- Spend most of your time writing clear task instructions
- Define detailed inputs and expected outputs
- Add examples and context to guide execution
- Dedicate the remaining time to agent role, goal, and backstory

This doesn’t mean agent design isn’t important - it absolutely is. But task design is where most execution failures occur, so prioritize accordingly.
```

With that in mind, this package is designed so users can focus on defining agents and tasks, rather than worrying about agent orchestration or UI implementation details.

---

The main goals of this package are:

1. **Seamless Integration of Streamlit and LangGraph:** Combine Streamlit’s rapid UI development with LangGraph’s flexible agent orchestration. This integration enables real-time interaction, monitoring, and control of agent workflows directly from the browser, making multi-agent systems more accessible and transparent.

2. **Lowering the Barrier to Multi-Agent Orchestration:** Frameworks like AutoGen, CrewAI, and the OpenAI Agents SDK each offer different approaches to multi-agent systems, with their own strengths and trade-offs. LangGraph provides powerful control and flexibility but comes with a steep learning curve and fragmented documentation. This package abstracts away much of that complexity, offering simple interfaces and templates so users can focus on agent logic rather than infrastructure.

3. **Ready-to-Use Multi-Agent Architectures:** Many real-world applications rely on standard agent architectures such as supervisor, hierarchical, and networked systems. This package includes these patterns out of the box, allowing users to select and customize them without reinventing the wheel.

4. **Enhanced Compatibility with OpenAI Response API:** While LangChain supports OpenAI’s Chat Completions API, it does not support the newer Response API, which introduces advanced capabilities such as code execution and file search. By targeting compatibility with the Response API, this package ensures that users can take advantage of the latest OpenAI features, making their multi-agent workflows more powerful and flexible.

5. **Extensibility to Other LLMs:** The landscape of large language models is evolving rapidly, with alternatives like Gemini, Claude, and various local models offering unique advantages. This package is designed to be extensible, enabling users to experiment with different LLMs and select the best model for their specific use case — whether it’s reasoning, coding, or domain-specific tasks.

Ultimately, **streamlit-langgraph** aims to empower users to build, experiment with, and deploy multi-agent orchestration systems quickly and intuitively — while remaining flexible enough to adapt as new models and workflows emerge.

## Status

This project is in **pre-alpha**. Features and APIs are subject to change, and not all goals have been fully implemented yet.

**Note:** Uses `langchain`/`langgraph` version `1.0.1`.
