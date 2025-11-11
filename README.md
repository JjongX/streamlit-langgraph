# streamlit-langgraph

[![PyPI version](https://badge.fury.io/py/streamlit-langgraph.svg)](https://badge.fury.io/py/streamlit-langgraph)

A Python package that integrates Streamlit's intuitive web interface with LangGraph's advanced multi-agent orchestration. Build interactive AI applications featuring multiple specialized agents collaborating in customizable workflows.

If you're using Streamlit with a single agent, consider [streamlit-openai](https://github.com/sbslee/streamlit-openai/tree/main) instead. This project is inspired by that work, especially its integration with the OpenAI Response API.

**streamlit-langgraph** is designed for multi-agent systems where multiple specialized agents collaborate to solve complex tasks.

## Table of Contents

- [Main Goals](#main-goals)
- [Status](#status)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Agent Configuration](#agent-configuration)
  - [Workflow Patterns](#workflow-patterns)
  - [Executor Types](#executor-types)
  - [Context Modes](#context-modes)
  - [Human-in-the-Loop](#human-in-the-loop-hitl)
  - [Custom Tools](#custom-tools)
- [Configuration Files](#configuration-files)
- [UI Customization](#ui-customization)
- [Architecture Patterns](#architecture-patterns)
- [Examples](#examples)
  - [Simple Single Agent](#simple-single-agent)
  - [Supervisor Sequential](#supervisor-sequential)
  - [Supervisor Parallel](#supervisor-parallel)
  - [Hierarchical Workflow](#hierarchical-workflow)
  - [Human-in-the-Loop](#human-in-the-loop)
- [API Reference](#api-reference)
- [License](#license)

## Main Goals

To build successful multi-agent systems, defining agent instructions, tasks, and context is more important than the actual orchestration logic. As illustrated by:

**[LangChain - Customizing agent context](https://docs.langchain.com/oss/python/langchain/multi-agent#customizing-agent-context)**:
> At the heart of multi-agent design is **context engineering** - deciding what information each agent sees... The quality of your system **heavily depends** on **context engineering**.

**[CrewAI - The 80/20 Rule](https://docs.crewai.com/en/guides/agents/crafting-effective-agents#the-80%2F20-rule%3A-focus-on-tasks-over-agents)**:
> 80% of your effort should go into designing tasks, and only 20% into defining agents... well-designed tasks can elevate even a simple agent.

With that in mind, this package is designed so users can focus on defining agents and tasks, rather than worrying about agent orchestration or UI implementation details.

**Key Features:**

1. **Seamless Integration of Streamlit and LangGraph:** Combine Streamlit's rapid UI development with LangGraph's flexible agent orchestration for real-time interaction and monitoring.

2. **Lowering the Barrier to Multi-Agent Orchestration:** Abstract away LangGraph's complexity with simple interfaces and templates.

3. **Ready-to-Use Multi-Agent Architectures:** Include standard patterns (supervisor, hierarchical, networked) out of the box.

4. **Enhanced Compatibility with OpenAI Response API:** Support OpenAI's newer Response API for advanced capabilities like code execution and file search.

5. **Extensibility to Other LLMs:** Design for easy integration with Gemini, Claude, and local models.

## Status

This project is in **pre-alpha**. Features and APIs are subject to change.

**Note:** Uses `langchain`/`langgraph` version `1.0.1`.

## Installation

```bash
pip install streamlit-langgraph
```

## Quick Start

### Single Agent (Simple)

```python
from streamlit_langgraph import Agent, UIConfig, LangGraphChat

# Define your agent
assistant = Agent(
    name="assistant",
    role="AI Assistant",
    instructions="You are a helpful AI assistant.",
    type="response",  # or "agent" for LangChain agents
    provider="openai",
    model="gpt-4.1-mini"
)

# Configure UI
config = UIConfig(
    title="My AI Assistant",
    welcome_message="Hello! How can I help you today?"
)

# Create and run chat interface
chat = LangGraphChat(agents=[assistant], config=config)
chat.run()
```

Run with: `streamlit run your_app.py`

### Multi-Agent Workflow

```python
from streamlit_langgraph import AgentManager, UIConfig, LangGraphChat
from streamlit_langgraph.workflow import WorkflowBuilder

# Load agents from YAML
agents = AgentManager.load_from_yaml("configs/my_agents.yaml")

# Create workflow
supervisor = agents[0]
workers = agents[1:]

builder = WorkflowBuilder()
workflow = builder.create_supervisor_workflow(
    supervisor=supervisor,
    workers=workers,
    execution_mode="sequential",
    delegation_mode="handoff"
)

# Create chat with workflow
chat = LangGraphChat(workflow=workflow, agents=agents)
chat.run()
```

## Core Concepts

### Agent Configuration

Agents are configured with:

```python
Agent(
    name="analyst",              # Unique identifier
    role="Data Analyst",         # Agent's role description
    instructions="...",          # Detailed task instructions
    type="response",             # "response" or "agent"
    provider="openai",           # LLM provider
    model="gpt-4.1-mini",       # Model name
    temperature=0.0,             # Response randomness
    tools=["tool1", "tool2"],   # Available tools
    context="full",              # Context mode
    human_in_loop=True,          # Enable HITL
    interrupt_on={...}           # HITL configuration
)
```

### Workflow Patterns

#### **Supervisor Pattern**
A supervisor agent coordinates worker agents:
- **Sequential**: Workers execute one at a time
- **Parallel**: Workers can execute simultaneously
- **Handoff**: Full context transfer between agents
- **Tool Calling**: Workers called as tools

#### **Hierarchical Pattern**
Multiple supervisor teams coordinated by a top supervisor:
- Top supervisor delegates to sub-supervisors
- Each sub-supervisor manages their own team
- Multi-level organizational structure

### Executor Types

#### **ResponseAPIExecutor** (`type="response"`)
- Uses OpenAI's Response API
- Supports advanced features (code interpreter, file search)
- Streaming responses
- Native tool calling

#### **CreateAgentExecutor** (`type="agent"`)
- Uses LangChain agents
- ReAct-style reasoning
- Broader LLM support
- LangChain tools integration

### Context Modes

Control how much context each agent receives:

#### **`full`** (Default)
- Agent sees **all messages** and previous worker outputs
- Best for: Tasks requiring complete conversation history
- Use case: Analysis, synthesis, decision-making

#### **`summary`**
- Agent sees **summarized context** from previous steps
- Best for: Tasks that need overview but not details
- Use case: High-level coordination, routing decisions

#### **`least`**
- Agent sees **only supervisor instructions** for their task
- Best for: Focused, independent tasks
- Use case: Specialized computations, API calls

```python
Agent(
    name="analyst",
    role="Data Analyst",
    instructions="Analyze the provided data",
    type="response",
    context="least"  # Sees only task instructions
)
```

### Human-in-the-Loop (HITL)

Enable human approval for critical agent actions:

#### **Key Features**
- **Tool Execution Approval**: Human reviews tool calls before execution
- **Decision Types**: Approve, Reject, or Edit tool inputs
- **Interrupt-Based**: Workflow pauses until human decision

#### **Use Cases**
- Sensitive operations (data deletion, API calls)
- Financial transactions
- Content moderation
- Compliance requirements

```python
Agent(
    name="executor",
    role="Action Executor",
    instructions="Execute approved actions",
    type="response",
    tools=["delete_data", "send_email"],
    human_in_loop=True,  # Enable HITL
    interrupt_on={
        "delete_data": {
            "allowed_decisions": ["approve", "reject"]
        },
        "send_email": {
            "allowed_decisions": ["approve", "reject", "edit"]
        }
    },
    hitl_description_prefix="Action requires approval"
)
```

#### **HITL Decision Types**
- **Approve**: Execute tool with provided inputs
- **Reject**: Skip tool execution, continue workflow
- **Edit**: Modify tool inputs before execution

### Custom Tools

Extend agent capabilities by registering custom functions as tools:

#### **Creating a Custom Tool**

```python
from streamlit_langgraph import CustomTool

def analyze_data(data: str, method: str = "standard") -> str:
    """
    Analyze data using specified method.
    
    This docstring is shown to the LLM, so be descriptive about:
    - What the tool does
    - When to use it
    - What each parameter means
    
    Args:
        data: The data to analyze (JSON string, CSV, etc.)
        method: Analysis method - "standard", "advanced", or "quick"
    
    Returns:
        Analysis results with insights and recommendations
    """
    # Your tool logic here
    result = f"Analyzed {len(data)} characters using {method} method"
    return result

# Register the tool
CustomTool.register_tool(
    name="analyze_data",
    description=(
        "Analyze structured data using various methods. "
        "Use this when you need to process and extract insights from data. "
        "Supports JSON, CSV, and plain text formats."
    ),
    function=analyze_data
)
```

#### **Using Tools in Agents**

```python
# Reference registered tools by name
agent = Agent(
    name="analyst",
    role="Data Analyst",
    instructions="Use analyze_data tool to process user data",
    type="response",
    tools=["analyze_data"]  # Tool name from registration
)
```

#### **Tool Best Practices**

1. **Descriptive Docstrings**: LLM uses these to understand when/how to use the tool
2. **Type Hints**: Help with parameter validation and documentation
3. **Clear Names**: Use descriptive names that indicate purpose
4. **Error Handling**: Return error messages as strings, don't raise exceptions
5. **Return Strings**: Always return string results for LLM consumption

#### **Tool with HITL**

```python
def delete_records(record_ids: str, reason: str) -> str:
    """
    Delete records from database. REQUIRES APPROVAL.
    
    Args:
        record_ids: Comma-separated list of record IDs
        reason: Justification for deletion
    
    Returns:
        Confirmation message with deleted record count
    """
    ids = record_ids.split(",")
    return f"Deleted {len(ids)} records. Reason: {reason}"

CustomTool.register_tool(
    name="delete_records",
    description="Delete database records (requires human approval)",
    function=delete_records
)

# Agent with HITL for this tool
agent = Agent(
    name="admin",
    role="Database Administrator",
    instructions="Manage database operations",
    type="response",
    tools=["delete_records"],
    human_in_loop=True,
    interrupt_on={
        "delete_records": {
            "allowed_decisions": ["approve", "reject", "edit"]
        }
    }
)
```

## Configuration Files

Agents can be configured using YAML files:

```yaml
- name: supervisor
  role: Project Manager
  type: response
  instructions: |
    You coordinate tasks and delegate to specialists.
    Analyze user requests and assign work appropriately.
  provider: openai
  model: gpt-4.1-mini
  temperature: 0.0
  tools:
    - tool_name
  context: full

- name: worker
  role: Specialist
  type: response
  instructions: |
    You handle specific tasks delegated by the supervisor.
  provider: openai
  model: gpt-4.1-mini
  temperature: 0.0
```

### HITL Configuration

```yaml
- name: analyst
  role: Data Analyst
  type: response
  instructions: "..."
  tools:
    - analyze_data
  human_in_loop: true
  interrupt_on:
    analyze_data:
      allowed_decisions:
        - approve
        - reject
        - edit
  hitl_description_prefix: "Action requires approval"
```

## UI Customization

```python
from streamlit_langgraph import UIConfig

config = UIConfig(
    title="My Multiagent App",
    welcome_message="Welcome! Ask me anything.",
    user_avatar="👤",
    assistant_avatar="🤖",
    page_icon="🤖",
    enable_file_upload=True,
    show_sidebar=True,  # Set to False to define custom sidebar
    stream=True
)

chat = LangGraphChat(workflow=workflow, agents=agents, config=config)
chat.run()
```

### Custom Sidebar

```python
import streamlit as st

config = UIConfig(show_sidebar=False)  # Disable default sidebar
chat = LangGraphChat(workflow=workflow, agents=agents, config=config)

# Define your own sidebar
with st.sidebar:
    st.header("Custom Sidebar")
    option = st.selectbox("Choose option", ["A", "B", "C"])
    # Your custom controls

chat.run()
```

## Architecture Patterns

### Pattern Selection Guide

| Pattern | Use Case | Execution | Best For |
|---------|----------|-----------|----------|
| **Supervisor Sequential** | Tasks need full context from previous steps | Sequential | Research, analysis pipelines |
| **Supervisor Parallel** | Independent tasks can run simultaneously | Parallel | Data processing, multi-source queries |
| **Hierarchical** | Complex multi-level organization | Sequential | Large teams, department structure |

## Examples

All examples are in the `examples/` directory.

### Simple Single Agent

**File**: `examples/simple_example.py`

Basic chat interface with a single agent. No workflow orchestration.

```bash
streamlit run examples/simple_example.py
```

### Supervisor Sequential

**File**: `examples/supervisor_sequential_example.py`

Supervisor coordinates workers sequentially. Workers execute one at a time with full context.

**Config**: `examples/configs/supervisor_sequential.yaml`

```bash
streamlit run examples/supervisor_sequential_example.py
```

### Supervisor Parallel

**File**: `examples/supervisor_parallel_example.py`

Supervisor delegates tasks to multiple workers who can work in parallel.

**Config**: `examples/configs/supervisor_parallel.yaml`

```bash
streamlit run examples/supervisor_parallel_example.py
```

### Hierarchical Workflow

**File**: `examples/hierarchical_example.py`

Multi-level organization with top supervisor managing sub-supervisor teams.

**Config**: `examples/configs/hierarchical.yaml`

```bash
streamlit run examples/hierarchical_example.py
```

### Human-in-the-Loop

**File**: `examples/human_in_the_loop_example.py`

Demonstrates HITL with tool execution approval. Users can approve, reject, or edit tool calls before execution.

**Config**: `examples/configs/human_in_the_loop.yaml`

```bash
streamlit run examples/human_in_the_loop_example.py
```

**Features**:
- Custom tools with approval workflow
- Sentiment analysis example
- Review escalation with edit capability

## API Reference

### Core Classes

#### `Agent`: Agent configuration dataclass.

**Key Parameters**:
- `name`: Unique identifier
- `role`: Role description
- `instructions`: Detailed task instructions
- `type`: "response" or "agent"
- `provider`: LLM provider (openai, anthropic, etc.)
- `model`: Model name
- `tools`: List of tool names
- `human_in_loop`: Enable HITL
- `context`: Context mode (full, summary, least)

#### `AgentManager`: Manages multiple agents and their interactions.

**Methods**:
- `load_from_yaml(path)`: Load agents from YAML
- `get_llm_client(agent)`: Get LLM client for agent
- `add_agent(agent)`: Add agent to manager
- `remove_agent(name)`: Remove agent by name

#### `UIConfig`: UI configuration.

**Parameters**:
- `title`: App title
- `welcome_message`: Initial message
- `user_avatar`: User avatar emoji/image
- `assistant_avatar`: Assistant avatar
- `enable_file_upload`: Enable file uploads
- `show_sidebar`: Show default sidebar
- `stream`: Enable streaming responses

#### `LangGraphChat`: Main chat interface.

**Parameters**:
- `workflow`: Optional LangGraph workflow
- `agents`: List of agents
- `config`: UIConfig instance

**Methods**:
- `run()`: Start the chat interface

#### `WorkflowBuilder`: Build multi-agent workflows.

**Methods**:
- `create_supervisor_workflow(supervisor, workers, execution_mode, delegation_mode)`
- `create_hierarchical_workflow(top_supervisor, supervisor_teams, execution_mode)`

#### `CustomTool`: Tool registry and management.

**Methods**:
- `register_tool(name, description, function)`: Register a tool
- `get_openai_tools(tool_names)`: Get OpenAI tool definitions
- `get_langchain_tools(tool_names)`: Get LangChain tools



## License

MIT License - see LICENSE file for details.

---

**Status**: Pre-alpha | **Python**: 3.9+ | **LangGraph**: 1.0.1

For issues and feature requests, please open an issue on GitHub.
