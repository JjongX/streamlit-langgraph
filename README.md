# streamlit-langgraph

[![PyPI version](https://badge.fury.io/py/streamlit-langgraph.svg)](https://badge.fury.io/py/streamlit-langgraph)

A Python package that integrates Streamlit's intuitive web interface with LangGraph's advanced multi-agent orchestration. Build interactive AI applications featuring multiple specialized agents collaborating in customizable workflows.

If you're using Streamlit with a single agent, consider [streamlit-openai](https://github.com/sbslee/streamlit-openai/tree/main) instead. This project is inspired by that work, especially its integration with the OpenAI Response API.

**streamlit-langgraph** is designed for multi-agent systems where multiple specialized agents collaborate to solve complex tasks.

## Table of Contents

- [Main Goal](#main-goal)
- [Status](#status)
- [Installation](#installation)
- [API Key Configuration](#api-key-configuration)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [Simple Single Agent](#simple-single-agent)
  - [Supervisor Sequential](#supervisor-sequential)
  - [Supervisor Parallel](#supervisor-parallel)
  - [Hierarchical Workflow](#hierarchical-workflow)
  - [Human-in-the-Loop](#human-in-the-loop)
- [Core Concepts](#core-concepts)
  - [Agent Configuration](#agent-configuration)
  - [Workflow Patterns](#workflow-patterns)
  - [Executor Types](#executor-types)
  - [Context Modes](#context-modes)
  - [Human-in-the-Loop](#human-in-the-loop-hitl)
  - [Custom Tools](#custom-tools)
- [Configuration](#configuration)
  - [Agent Configuration Files](#agent-configuration-files)
  - [UI Configuration](#ui-configuration)
- [API Reference](#api-reference)
  - [Agent](#agent)
  - [AgentManager](#agentmanager)
  - [UIConfig](#uiconfig)
  - [LangGraphChat](#langgraphchat)
  - [WorkflowBuilder](#workflowbuilder)
  - [WorkflowBuilder.SupervisorTeam](#workflowbuildersupervisorteam)
  - [CustomTool](#customtool)
- [License](#license)

## Main Goal

To build successful multi-agent systems, defining agent instructions, tasks, and context is more important than the actual orchestration logic. As illustrated by:

**[LangChain - Customizing agent context](https://docs.langchain.com/oss/python/langchain/multi-agent#customizing-agent-context)**:
> At the heart of multi-agent design is **context engineering** - deciding what information each agent sees... The quality of your system **heavily depends** on **context engineering**.

**[CrewAI - The 80/20 Rule](https://docs.crewai.com/en/guides/agents/crafting-effective-agents#the-80%2F20-rule%3A-focus-on-tasks-over-agents)**:
> 80% of your effort should go into designing tasks, and only 20% into defining agents... well-designed tasks can elevate even a simple agent.

With that in mind, this package is designed so users can focus on defining agents and tasks, rather than worrying about agent orchestration or UI implementation details.

**Key Features:**

1. **Seamless Integration of Streamlit and LangGraph:** Combine Streamlit's rapid UI development to turn simple Python scripts into interactive web applications with LangGraph's flexible agent orchestration for real-time interaction.

2. **Lowering the Barrier to Multi-Agent Orchestration:** Simplify multi-agent development with easy-to-use interfaces that abstract LangGraph's complexity.

3. **Ready-to-Use Multi-Agent Architectures:** Include standard patterns (supervisor, hierarchical, networked) out of the box.

4. **Enhanced Compatibility with OpenAI Response API:** Support OpenAI's newer Response API for advanced capabilities like file search and code execution.

5. **Extensibility to Other LLMs:** Design for easy integration with Gemini, Claude, and local models.

## Status

This project is in **pre-alpha**. Features and APIs are subject to change.

**Note:** Uses `langchain`/`langgraph` version `1.0.1`.

## Installation

```bash
pip install streamlit-langgraph
```

## API Key Configuration

Before running your application, you need to configure your API keys. Create a `.streamlit/config.toml` file in your project root directory:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

**File structure:**
```
your-project/
├── .streamlit/
│   └── config.toml
├── your_app.py
└── ...
```

## Quick Start

### Single Agent (Simple)

```python
import streamlit_langgraph as slg

# Define your agent
assistant = slg.Agent(
    name="assistant",
    role="AI Assistant",
    instructions="You are a helpful AI assistant.",
    type="response",  # or "agent" for LangChain agents
    provider="openai",
    model="gpt-4.1-mini"
)

# Configure UI
config = slg.UIConfig(
    title="My AI Assistant",
    welcome_message="Hello! How can I help you today?"
)

# Create and run chat interface
chat = slg.LangGraphChat(agents=[assistant], config=config)
chat.run()
```

Run with: `streamlit run your_app.py`

### Multi-Agent Workflow

```python
import streamlit_langgraph as slg

# Load agents from YAML
agents = slg.AgentManager.load_from_yaml("configs/my_agents.yaml")

# Create workflow
supervisor = agents[0]
workers = agents[1:]

builder = slg.WorkflowBuilder()
workflow = builder.create_supervisor_workflow(
    supervisor=supervisor,
    workers=workers,
    execution_mode="sequential",
    delegation_mode="handoff"
)

# Create chat with workflow
chat = slg.LangGraphChat(workflow=workflow, agents=agents)
chat.run()
```

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

## Core Concepts

### Agent Configuration

Agents are configured with:

```python
import streamlit_langgraph as slg

agent = slg.Agent(
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

#### **Pattern Selection Guide**

| Pattern | Use Case | Execution | Best For |
|---------|----------|-----------|----------|
| **Supervisor Sequential** | Tasks need full context from previous steps | Sequential | Research, analysis pipelines |
| **Supervisor Parallel** | Independent tasks can run simultaneously | Parallel | Data processing, multi-source queries |
| **Hierarchical** | Complex multi-level organization | Sequential | Large teams, department structure |

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
import streamlit_langgraph as slg

analyst = slg.Agent(
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
import streamlit_langgraph as slg

executor = slg.Agent(
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
import streamlit_langgraph as slg

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
slg.CustomTool.register_tool(
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
import streamlit_langgraph as slg

# Reference registered tools by name
agent = slg.Agent(
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
import streamlit_langgraph as slg

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

slg.CustomTool.register_tool(
    name="delete_records",
    description="Delete database records (requires human approval)",
    function=delete_records
)

# Agent with HITL for this tool
agent = slg.Agent(
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

## Configuration

### Agent Configuration Files

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

#### HITL Configuration

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

### UI Configuration

```python
import streamlit_langgraph as slg

config = slg.UIConfig(
    title="My Multiagent App",
    welcome_message="Welcome! Ask me anything.",
    user_avatar="👤",
    assistant_avatar="🤖",
    page_icon="🤖",
    enable_file_upload=True,
    show_sidebar=True,  # Set to False to define custom sidebar
    stream=True
)

chat = slg.LangGraphChat(workflow=workflow, agents=agents, config=config)
chat.run()
```

#### Custom Sidebar

```python
import streamlit as st
import streamlit_langgraph as slg

config = slg.UIConfig(show_sidebar=False)  # Disable default sidebar
chat = slg.LangGraphChat(workflow=workflow, agents=agents, config=config)

# Define your own sidebar
with st.sidebar:
    st.header("Custom Sidebar")
    option = st.selectbox("Choose option", ["A", "B", "C"])
    # Your custom controls

chat.run()
```

## API Reference

---

### `Agent`

**Description**: Core class for defining individual agents with their configurations.

**Constructor Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique identifier for the agent |
| `role` | `str` | Required | Brief description of the agent's role |
| `instructions` | `str` | Required | Detailed instructions guiding agent behavior |
| `type` | `str` | `"response"` | Executor type: `"response"` (OpenAI Response API) or `"agent"` (LangChain) |
| `provider` | `str` | `"openai"` | LLM provider: `"openai"`, `"anthropic"`, `"google"`, etc. |
| `model` | `str` | `"gpt-4o-mini"` | Model name (e.g., `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`) |
| `temperature` | `float` | `0.0` | Sampling temperature (0.0 to 2.0) |
| `tools` | `List[str]` | `[]` | List of tool names available to the agent |
| `context` | `str` | `"full"` | Context mode: `"full"`, `"summary"`, or `"least"` |
| `human_in_loop` | `bool` | `False` | Enable human-in-the-loop approval for tool execution |
| `interrupt_on` | `Dict` | `{}` | HITL configuration per tool |
| `hitl_description_prefix` | `str` | `""` | Prefix for HITL approval messages |
| `allow_code_interpreter` | `bool` | `False` | Enable code interpreter (Response API only) |
| `allow_file_search` | `bool` | `False` | Enable file search (Response API only) |
| `allow_web_search` | `bool` | `False` | Enable web search (Response API only) |

**Example**:
```python
import streamlit_langgraph as slg

agent = slg.Agent(
    name="analyst",
    role="Data Analyst",
    instructions="Analyze data and provide insights",
    type="response",
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    tools=["analyze_data", "visualize"],
    context="full",
    human_in_loop=True,
    interrupt_on={
        "analyze_data": {
            "allowed_decisions": ["approve", "reject", "edit"]
        }
    }
)
```

---

### `AgentManager`

**Description**: Manages multiple agents and handles agent loading/retrieval.

**Class Methods**:

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `load_from_yaml(path)` | `path: str` | `List[Agent]` | Load agents from YAML configuration file |
| `get_llm_client(agent)` | `agent: Agent` | LLM client | Get configured LLM client for an agent |

**Instance Methods**:

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `add_agent(agent)` | `agent: Agent` | `None` | Add agent to the manager |
| `remove_agent(name)` | `name: str` | `None` | Remove agent by name |
| `get_agent(name)` | `name: str` | `Agent` | Retrieve agent by name |

**Example**:
```python
import streamlit_langgraph as slg

# Load from YAML
agents = slg.AgentManager.load_from_yaml("config/agents.yaml")

# Or create manager and add agents
manager = slg.AgentManager()
manager.add_agent(my_agent)
agent = manager.get_agent("analyst")
```

---

### `UIConfig`

**Description**: Configuration for Streamlit UI customization.

**Constructor Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | `"LangGraph Chat"` | Application title shown in browser tab and header |
| `page_icon` | `str` | `"🤖"` | Favicon emoji or path to image file |
| `welcome_message` | `str` | `None` | Welcome message shown at start (supports Markdown) |
| `user_avatar` | `str` | `"👤"` | Avatar for user messages (emoji or image path) |
| `assistant_avatar` | `str` | `"🤖"` | Avatar for assistant messages (emoji or image path) |
| `stream` | `bool` | `True` | Enable streaming responses |
| `enable_file_upload` | `bool` | `False` | Show file upload widget |
| `show_sidebar` | `bool` | `True` | Show default sidebar (set False for custom) |
| `placeholder` | `str` | `None` | Placeholder text for chat input |
| `show_agent_info` | `bool` | `True` | Show agent name in messages |

**Example**:
```python
import streamlit_langgraph as slg

config = slg.UIConfig(
    title="My AI Team",
    page_icon="🚀",
    welcome_message="Welcome to **My AI Team**!",
    user_avatar="👨‍💼",
    assistant_avatar="🤖",
    stream=True,
    enable_file_upload=True,
    show_sidebar=True,
    placeholder="Ask me anything..."
)
```

---

### `LangGraphChat`

**Description**: Main interface for running chat applications with single or multiple agents.

**Constructor Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workflow` | `StateGraph` | `None` | Compiled LangGraph workflow (for multi-agent) |
| `agents` | `List[Agent]` | Required | List of agents in the application |
| `config` | `UIConfig` | `UIConfig()` | UI configuration |
| `custom_tools` | `List[CustomTool]` | `None` | List of custom tools to register |

**Methods**:

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `run()` | None | `None` | Start the Streamlit chat interface |

**Example**:
```python
import streamlit_langgraph as slg

# Single agent
chat = slg.LangGraphChat(
    agents=[assistant],
    config=config
)
chat.run()

# Multi-agent with workflow
chat = slg.LangGraphChat(
    workflow=compiled_workflow,
    agents=all_agents,
    config=config
)
chat.run()
```

---

### `WorkflowBuilder`

**Description**: Builder for creating multi-agent workflows with different patterns.

**Methods**:

#### `create_supervisor_workflow()`

Creates a supervisor pattern where one agent coordinates multiple workers.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `supervisor` | `Agent` | Required | Supervisor agent that coordinates |
| `workers` | `List[Agent]` | Required | Worker agents to be coordinated |
| `execution_mode` | `str` | `"sequential"` | `"sequential"` or `"parallel"` |
| `delegation_mode` | `str` | `"handoff"` | `"handoff"` or `"tool_calling"` |

**Returns**: `StateGraph` - Compiled workflow

**Example**:
```python
import streamlit_langgraph as slg

builder = slg.WorkflowBuilder()
workflow = builder.create_supervisor_workflow(
    supervisor=supervisor_agent,
    workers=[worker1, worker2, worker3],
    execution_mode="sequential",  # or "parallel"
    delegation_mode="handoff"      # or "tool_calling"
)
```

#### `create_hierarchical_workflow()`

Creates a hierarchical pattern with a top supervisor managing sub-supervisor teams.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_supervisor` | `Agent` | Required | Top-level supervisor |
| `supervisor_teams` | `List[SupervisorTeam]` | Required | List of sub-supervisor teams |
| `execution_mode` | `str` | `"sequential"` | Currently only `"sequential"` supported |

**Returns**: `StateGraph` - Compiled workflow

**Example**:
```python
import streamlit_langgraph as slg

# Create teams
research_team = slg.WorkflowBuilder.SupervisorTeam(
    supervisor=research_lead,
    workers=[researcher1, researcher2],
    team_name="research_team"
)

content_team = slg.WorkflowBuilder.SupervisorTeam(
    supervisor=content_lead,
    workers=[writer, editor],
    team_name="content_team"
)

# Create hierarchical workflow
builder = slg.WorkflowBuilder()
workflow = builder.create_hierarchical_workflow(
    top_supervisor=project_manager,
    supervisor_teams=[research_team, content_team],
    execution_mode="sequential"
)
```

---

##### `WorkflowBuilder.SupervisorTeam`

**Description**: Dataclass representing a sub-supervisor and their team for hierarchical workflows.

**Constructor Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `supervisor` | `Agent` | Required | Sub-supervisor agent |
| `workers` | `List[Agent]` | Required | Worker agents in this team |
| `team_name` | `str` | Auto-generated | Team identifier |

**Example**:
```python
import streamlit_langgraph as slg

team = slg.WorkflowBuilder.SupervisorTeam(
    supervisor=team_lead_agent,
    workers=[worker1, worker2, worker3],
    team_name="engineering_team"
)
```

---

### `CustomTool`

**Description**: Registry for custom tools that agents can use.

**Method**:

#### `register_tool()`

Register a custom function as a tool available to agents.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique tool name |
| `description` | `str` | Required | Description shown to LLM |
| `function` | `Callable` | Required | Python function to execute |
| `parameters` | `Dict` | Auto-extracted | Tool parameters schema |
| `return_direct` | `bool` | `False` | Return tool output directly to user |

**Returns**: `CustomTool` instance

**Example**:
```python
import streamlit_langgraph as slg

def calculate_sum(a: float, b: float) -> str:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum as a string
    """
    return str(a + b)

slg.CustomTool.register_tool(
    name="calculate_sum",
    description="Add two numbers and return the sum",
    function=calculate_sum
)

# Use in agent
agent = slg.Agent(
    name="calculator",
    role="Calculator",
    instructions="Use calculate_sum to add numbers",
    tools=["calculate_sum"]
)
```

---

## License

MIT License - see LICENSE file for details.

---

**Status**: Pre-alpha | **Python**: 3.9+ | **LangGraph**: 1.0.1

For issues and feature requests, please open an issue on GitHub.
