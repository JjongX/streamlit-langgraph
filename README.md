# streamlit-langgraph

[![PyPI version](https://badge.fury.io/py/streamlit-langgraph.svg)](https://badge.fury.io/py/streamlit-langgraph)

A Python package that integrates Streamlit's intuitive web interface with LangGraph's advanced multi-agent orchestration. Build interactive AI applications featuring multiple specialized agents collaborating in customizable workflows.

If you're using Streamlit with a single agent, consider [streamlit-openai](https://github.com/sbslee/streamlit-openai/tree/main) instead. This project is inspired by that work, especially its integration with the OpenAI Response API.

**streamlit-langgraph** is designed for multi-agent systems where multiple specialized agents collaborate to solve complex tasks.

## Table of Contents

- [Main Goal](#main-goal)
- [Status](#status)
  - [Supported LLM Providers](#supported-llm-providers)
- [Installation](#installation)
- [API Key Configuration](#api-key-configuration)
- [Quick Start](#quick-start)
- [Examples](#examples)
  - [Simple Single Agent](#simple-single-agent)
  - [File Preprocessing Callback](#file-preprocessing-callback)
  - [Supervisor Sequential](#supervisor-sequential)
  - [Supervisor Parallel](#supervisor-parallel)
  - [Hierarchical Workflow](#hierarchical-workflow)
  - [Network Workflow](#network-workflow)
  - [Human-in-the-Loop](#human-in-the-loop)
  - [MCP Tools](#mcp-tools)
- [Package Structure](#package-structure)
- [Core Logic](#core-logic)
  - [Section and Block System](#section-and-block-system)
  - [Workflow State as Single Source of Truth](#workflow-state-as-single-source-of-truth)
  - [Streamlit Session State Usage](#streamlit-session-state-usage)
- [Core Concepts](#core-concepts)
  - [Agent Configuration](#agent-configuration)
  - [UI Configuration](#ui-configuration)
  - [Workflow Patterns](#workflow-patterns)
  - [Executor Architecture](#executor-architecture)
  - [Conversation History Modes](#conversation-history-modes)
  - [Context Modes](#context-modes)
  - [Human-in-the-Loop](#human-in-the-loop-hitl)
  - [Custom Tools](#custom-tools)
  - [MCP (Model Context Protocol)](#mcp-model-context-protocol)
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

4. **Dual Executor Architecture:** Automatically selects between ResponseAPIExecutor (for native OpenAI tools) and CreateAgentExecutor (for HITL and multi-provider support), ensuring optimal performance for each use case.

5. **Extensibility to Other LLMs:** Design for easy integration with Gemini, Claude, and local models.

## Status

This project is in **pre-alpha**. Features and APIs are subject to change.

**Note:** Uses `langchain`/`langgraph` version `1.0.1`.

### Supported LLM Providers

| Provider | Support | Notes |
|----------|---------|-------|
| **OpenAI** | ✅ | Uses **ResponseAPIExecutor** (Responses API) when native tools enabled and HITL disabled. Uses **CreateAgentExecutor** (ChatCompletion API) for HITL or when native tools disabled. |
| **Anthropic (Claude)** | ❓ | May work but not explicitly tested. |
| **Google (Gemini)** | ✅ | Full support via LangChain's `init_chat_model` |
| **Other LangChain Providers** | ❓ | May work but not explicitly tested.|

**Legend:**
- ✅ **O** = Fully supported and tested
- ❌ **X** = Not supported
- ❓ **?** = May work but not explicitly tested

**Notes:**
- **OpenAI**: Automatically selects ResponseAPIExecutor (Responses API) or CreateAgentExecutor (ChatCompletion API) based on native tool configuration and HITL settings
  - ResponseAPIExecutor: Used when native tools enabled and HITL disabled
  - CreateAgentExecutor: Used for HITL scenarios or when native tools are disabled
- Support depends on LangChain's provider compatibility

## Installation

**Using pip**:

```bash
pip install streamlit-langgraph
```

**Using UV**:

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:
```bash
uv pip install streamlit-langgraph
```

Or if you're using UV for project management:
```bash
uv add streamlit-langgraph
```

## API Key Configuration

Before running your application, you need to configure your API keys. Create a `.streamlit/config.toml` file in your project root directory:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

**File structure:**:

```
your-project/
├── .streamlit/
│   └── config.toml
├── your_app.py
└── ...
```

## Quick Start

Run with: `streamlit run your_app.py`

**Single Agent (Simple)**:

```python
# your_app.py

import streamlit as st
import streamlit_langgraph as slg

# Define your agent
assistant = slg.Agent(
    name="assistant",
    role="AI Assistant",
    instructions="You are a helpful AI assistant.",
    provider="openai",
    model="gpt-4.1-mini"
)

# Configure UI
config = slg.UIConfig(
    title="My AI Assistant",
    welcome_message="Hello! How can I help you today?"
)

# Create and run chat interface
if "chat" not in st.session_state:
    st.session_state.chat = slg.LangGraphChat(
        agents=[assistant],
        config=config
    )
st.session_state.chat.run()
```

**Multi-Agent Workflow**:

```python
# your_app.py

import streamlit as st
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
if "chat" not in st.session_state:
    st.session_state.chat = slg.LangGraphChat(
        workflow=workflow,
        agents=agents
    )
st.session_state.chat.run()
```

## Examples

All examples are in the `examples/` directory.

### Simple Single Agent

**File**: `examples/01_basic_simple_example.py`

Basic chat interface with a single agent. No workflow orchestration.

```bash
streamlit run examples/01_basic_simple_example.py
```

### File Preprocessing Callback

**File**: `examples/06_feature_file_callback_example.py`

Demonstrates how to use the `file_callback` parameter to preprocess uploaded files before they are sent to OpenAI. The callback receives the file path and returns a processed file path, or optionally a tuple with additional files.

```bash
streamlit run examples/06_feature_file_callback_example.py
```

**Features**:
- Preprocess files (e.g., filter CSV columns) before upload
- Works with single agent and multi-agent workflows
- Support for returning additional files generated during preprocessing
- Automatically uploads additional CSV files to code_interpreter container

**Example - Simple Preprocessing**:
```python
import pandas as pd
import streamlit_langgraph as slg

def filter_columns(file_path: str) -> str:
    """Filter CSV to keep only columns starting with 'num_'."""
    if not file_path.endswith('.csv'):
        return file_path
    
    df = pd.read_csv(file_path)
    num_cols = [col for col in df.columns if col.startswith('num_')]
    df_filtered = df[num_cols] if num_cols else df
    
    processed_path = file_path.replace('.csv', '_filtered.csv')
    df_filtered.to_csv(processed_path, index=False)
    return processed_path

config = slg.UIConfig(
    title="File Preprocessing Example",
    file_callback=filter_columns,
)
```

**Example - Preprocessing with Additional Files**:
```python
from pathlib import Path
import streamlit_langgraph as slg

def preprocess_with_additional_files(file_path: str):
    """
    Preprocess file and generate additional CSV files.
    
    Returns:
        Tuple of (main_file_path, additional_files_directory) or
        Tuple of (main_file_path, [list_of_file_paths])
    """
    # Process the main file
    processed_path = process_main_file(file_path)
    
    # Generate additional CSV files in a directory
    output_dir = Path("outputs") / "generated_csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate multiple CSV files
    generate_csv_files(output_dir)
    
    # Return tuple: (main file, directory with additional files)
    # All CSV files in the directory will be automatically uploaded
    return (processed_path, output_dir)

config = slg.UIConfig(
    title="Multi-File Preprocessing Example",
    file_callback=preprocess_with_additional_files,
)
```

### Supervisor Sequential

**File**: `examples/02_workflow_supervisor_sequential_example.py`

Supervisor coordinates workers sequentially. Workers execute one at a time with full context.

**Config**: `examples/configs/supervisor_sequential.yaml`

```bash
streamlit run examples/02_workflow_supervisor_sequential_example.py
```

### Supervisor Parallel

**File**: `examples/03_workflow_supervisor_parallel_example.py`

Supervisor delegates tasks to multiple workers who can work in parallel.

**Config**: `examples/configs/supervisor_parallel.yaml`

```bash
streamlit run examples/03_workflow_supervisor_parallel_example.py
```

### Hierarchical Workflow

**File**: `examples/04_workflow_hierarchical_example.py`

Multi-level organization with top supervisor managing sub-supervisor teams.

**Config**: `examples/configs/hierarchical.yaml`

```bash
streamlit run examples/04_workflow_hierarchical_example.py
```

### Network Workflow

**File**: `examples/05_workflow_network_example.py`

Peer-to-peer network pattern where agents can communicate directly with any other agent. No central supervisor - agents form a mesh topology and can hand off work to any peer.

**Config**: `examples/configs/network.yaml`

```bash
streamlit run examples/05_workflow_network_example.py
```

**Features**:
- True peer-to-peer collaboration
- Agents can hand work back and forth dynamically
- No central coordinator - all agents are peers
- First agent in the list serves as the entry point
- Best for: Complex scenarios with interdependent concerns

**Use Case**: Strategic consulting teams where specialists need to collaborate dynamically, with work flowing back and forth as issues are identified and re-evaluated.

### Human-in-the-Loop

**File**: `examples/07_feature_human_in_the_loop_example.py`

Demonstrates HITL with tool execution approval. Users can approve, reject, or edit tool calls before execution.

**Config**: `examples/configs/human_in_the_loop.yaml`

```bash
streamlit run examples/07_feature_human_in_the_loop_example.py
```

**Features**:
- Custom tools with approval workflow
- Sentiment analysis example
- Review escalation with edit capability

### MCP Tools

**File**: `examples/08_feature_mcp_example.py`

Demonstrates integration with MCP (Model Context Protocol) servers to access external tools and resources.

```bash
streamlit run examples/08_feature_mcp_example.py
```

**Prerequisites**:
```bash
pip install fastmcp langchain-mcp-adapters
```

**Features**:
- Connect to MCP servers via stdio or HTTP transport
- Access tools from external MCP servers
- Works with both ResponseAPIExecutor and CreateAgentExecutor
- Example MCP servers included (math, weather)

**MCP Server Examples**:
- `examples/mcp_servers/math_server.py` - Math operations (add, multiply, subtract, divide)
- `examples/mcp_servers/weather_server.py` - Weather information

## Package Structure

This section provides an overview of the package's internal organization and module structure.

### Top-Level Modules

- **`agent.py`**: `Agent` class and `AgentManager` for agent configuration and management
- **`chat.py`**: `LangGraphChat` main interface and `UIConfig` for UI settings
- **`workflow/`**: Workflow builders and patterns (supervisor, hierarchical, network)

### Core Modules (`core/`)

**Executor (`core/executor/`):**
- `response_api.py`: `ResponseAPIExecutor` for OpenAI Responses API
- `create_agent.py`: `CreateAgentExecutor` for LangChain agents with HITL support
- `registry.py`: `ExecutorRegistry` for automatic executor selection
- `workflow.py`: `WorkflowExecutor` for workflow execution
- `conversation_history.py`: Conversation history management mixin

**State (`core/state/`):**
- `state_schema.py`: `WorkflowState` TypedDict and `WorkflowStateManager`
- `state_sync.py`: `StateSynchronizer` for syncing workflow state

**Middleware (`core/middleware/`):**
- `hitl.py`: `HITLHandler` and `HITLUtils` for human-in-the-loop
- `interrupts.py`: `InterruptManager` for interrupt handling

### UI Modules (`ui/`)

- `display_manager.py`: `DisplayManager`, `Section`, and `Block` for UI rendering
- `stream_processor.py`: `StreamProcessor` for handling streaming responses

### Utility Modules (`utils/`)

- `file_handler.py`: `FileHandler` for file upload and processing
- `custom_tool.py`: `CustomTool` registry for custom tools
- `mcp_tool.py`: `MCPToolManager` for MCP server integration

### Workflow Modules (`workflow/`)

- `builder.py`: `WorkflowBuilder` for creating workflows
- `patterns/`: Workflow pattern implementations (supervisor, hierarchical, network)
- `agent_nodes/`: Agent node factories and delegation patterns

## Core Logic

This section explains the internal architecture for rendering messages and managing state.

### Section and Block System

All chat messages are rendered through a **Section/Block** architecture:

- **Section**: Represents a single chat message (user or assistant). Contains multiple blocks.
- **Block**: Individual content units within a section:
  - `text`: Plain text content
  - `code`: Code blocks (collapsible)
  - `reasoning`: Reasoning/thinking blocks (collapsible)
  - `image`: Image content
  - `download`: Downloadable files

**Flow**:
1. User input → Creates a `Section` with `text` block
2. Agent response → Creates a `Section` with blocks based on content type
3. Streaming → Updates existing blocks or creates new ones as content arrives
4. All sections/blocks are saved to `workflow_state` for persistence

### Workflow State as Single Source of Truth

`workflow_state` is the **single source of truth** for all chat history and application state:

**Structure**:
```python
workflow_state = {
    "messages": [...],           # Conversation messages (user/assistant)
    "metadata": {
        "display_sections": [...], # UI sections/blocks for rendering
        "pending_interrupts": {...}, # HITL state
        "executors": {...},        # Executor metadata
        ...
    },
    "agent_outputs": {...},      # Agent responses by agent name
    "current_agent": "...",       # Currently active agent
    "files": [...]               # File metadata
}
```

**Key Points**:
- **All messages** (user and assistant) are stored in `workflow_state["messages"]`
- **All UI sections/blocks** are stored in `workflow_state["metadata"]["display_sections"]`
- **State persistence**: Workflow state persists across Streamlit reruns
- **Workflow execution**: LangGraph workflows read from and write to `workflow_state`
- **State synchronization**: `StateSynchronizer` manages updates to `workflow_state`
- **No fallbacks**: All state operations require `state_manager` - no direct `session_state` access for display sections

### Streamlit Session State Usage

`st.session_state` is used for **display management** and **runtime state**:

**Display Management**:
- `workflow_state`: The single source of truth (stored in session state for Streamlit persistence)
- `display_sections`: **Deprecated** - now stored in `workflow_state.metadata.display_sections`
- `agent_executors`: Runtime executor instances (not persisted in workflow_state)
- `uploaded_files`: File objects for current session (metadata stored in workflow_state)

**Key Separation**:
- **`workflow_state`**: Persistent, single source of truth for all chat data
- **`st.session_state`**: Streamlit-specific runtime state and references to workflow_state

**State Flow**:
```
User Input
  ↓
StateSynchronizer.add_user_message()
  ↓
workflow_state["messages"] updated
  ↓
DisplayManager creates Section/Block
  ↓
Section._save_to_session_state()
  ↓
workflow_state["metadata"]["display_sections"] updated
  ↓
render_message_history() reads from workflow_state
  ↓
Streamlit renders UI
```

**Benefits**:
- **Consistency**: All state in one place (`workflow_state`)
- **Persistence**: State survives Streamlit reruns
- **Workflow compatibility**: LangGraph workflows can read/write state directly
- **UI synchronization**: Display always reflects workflow_state

## Core Concepts

### Agent Configuration

Agents can be configured in two ways:

**Python Configuration:**
```python
import streamlit_langgraph as slg

agent = slg.Agent(
    name="analyst",              # Unique identifier
    role="Data Analyst",         # Agent's role description
    instructions="...",          # Detailed task instructions
    provider="openai",           # LLM provider
    model="gpt-4.1-mini",       # Model name
    temperature=0.0,             # Response randomness
    tools=["tool1", "tool2"],   # Available tools
    mcp_servers={...},          # MCP server configurations
    context="full",              # Context mode
    human_in_loop=True,          # Enable HITL
    interrupt_on={...}           # HITL configuration
)
```

**YAML File Configuration:**

Agents can be configured using YAML files for easier management:

```yaml
- name: supervisor
  role: Project Manager
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
  instructions: |
    You handle specific tasks delegated by the supervisor.
  provider: openai
  model: gpt-4.1-mini
  temperature: 0.0
```

Load the above YAML to python:
```python
import streamlit_langgraph as slg

# Load agents from YAML file
agents = slg.AgentManager.load_from_yaml("configs/agents.yaml")
supervisor = agents[0]
workers = agents[1:]
```

For complete parameter reference, see [Agent API Reference](#agent).

### UI Configuration

Configure the Streamlit interface using `UIConfig`:

```python
import streamlit as st
import streamlit_langgraph as slg

config = slg.UIConfig(
    title="My Multiagent App",
    welcome_message="Welcome! Ask me anything.",
    user_avatar="👤",
    assistant_avatar="🤖",
    page_icon="🤖",
    page_layout="wide",
    enable_file_upload="multiple",
    show_sidebar=True,
    stream=True,
    file_callback=None
)

if "chat" not in st.session_state:
    st.session_state.chat = slg.LangGraphChat(workflow=workflow, agents=agents, config=config)
st.session_state.chat.run()
```

**Custom Sidebar:**
```python
import streamlit as st
import streamlit_langgraph as slg

config = slg.UIConfig(show_sidebar=False)  # Disable default sidebar

# Define your own sidebar
with st.sidebar:
    st.header("Custom Sidebar")
    option = st.selectbox("Choose option", ["A", "B", "C"])
    # Your custom controls

if "chat" not in st.session_state:
    st.session_state.chat = slg.LangGraphChat(
        workflow=workflow,
        agents=agents,
        config=config
    )
st.session_state.chat.run()
```

For complete parameter reference, see [UIConfig API Reference](#uiconfig).

### Workflow Patterns

#### **Supervisor Pattern**
A supervisor agent coordinates worker agents:
- **Sequential**: Workers execute one at a time
- **Parallel**: Workers can execute simultaneously
- **Handoff**: Full context transfer between agents (works with both ResponseAPIExecutor and CreateAgentExecutor)
- **Tool Calling**: Workers called as tools

#### **Hierarchical Pattern**
Multiple supervisor teams coordinated by a top supervisor:
- Top supervisor delegates to sub-supervisors
- Each sub-supervisor manages their own team
- Multi-level organizational structure

#### **Network Pattern**
Peer-to-peer mesh topology where agents can communicate directly:
- No central supervisor - all agents are peers
- Any agent can hand off to any other agent
- First agent in the list serves as the entry point
- Best for: Complex scenarios with interdependent concerns where work needs to flow back and forth

#### **Pattern Selection Guide**

| Pattern | Use Case | Execution | Best For |
|---------|----------|-----------|----------|
| **Supervisor Sequential** | Tasks need full context from previous steps | Sequential | Research, analysis pipelines |
| **Supervisor Parallel** | Independent tasks can run simultaneously | Parallel | Data processing, multi-source queries |
| **Hierarchical** | Complex multi-level organization | Sequential | Large teams, department structure |
| **Network** | Interdependent concerns, dynamic collaboration | Peer-to-peer | Strategic consulting, complex problem-solving with back-and-forth |

### Executor Architecture

The system uses two executors that are automatically selected based on agent configuration:

#### **ResponseAPIExecutor**
- **When Used**: Native OpenAI tools enabled (`allow_code_interpreter`, `allow_web_search`, `allow_file_search`, `allow_image_generation`) AND HITL disabled
- **API**: Uses OpenAI's native Responses API directly
- **Features**:
  - Native tool support (code_interpreter, file_search, web_search, image_generation)
  - Custom tools support (converts LangChain tools to OpenAI function format)
  - MCP tools support (via OpenAI's MCP integration)
  - Streaming support
- **Limitations**: Does not support HITL (human-in-the-loop)

#### **CreateAgentExecutor**
- **When Used**: HITL enabled OR native tools disabled
- **API**: Uses ChatCompletion API via LangChain's `create_agent`
- **Features**:
  - Full HITL support with approval workflows
  - Multi-provider support (OpenAI, Anthropic, Google, etc.)
  - Custom tools support (LangChain StructuredTool)
  - MCP tools support (via LangChain MCP adapters)
  - Streaming support
- **Note**: When HITL is enabled, native OpenAI tools are automatically disabled (HITL requires CreateAgentExecutor)

#### **Automatic Selection**

The `ExecutorRegistry` automatically selects the appropriate executor:

```python
# Selection logic:
# - If HITL enabled → CreateAgentExecutor (native tools disabled)
# - If native tools enabled AND HITL disabled → ResponseAPIExecutor
# - Otherwise → CreateAgentExecutor
```

**Example**:
```python
# Uses ResponseAPIExecutor (native tools, no HITL)
agent = slg.Agent(
    name="assistant",
    allow_code_interpreter=True,
    allow_web_search=True
)

# Uses CreateAgentExecutor (HITL enabled)
agent = slg.Agent(
    name="assistant",
    human_in_loop=True,
    interrupt_on={"tool_name": {"allowed_decisions": ["approve", "reject"]}}
)
```

#### **Handoff Delegation Support**

Both executors work seamlessly with handoff delegation patterns:
- **ResponseAPIExecutor**: Uses OpenAI ChatCompletion API with function calling for delegation
- **CreateAgentExecutor**: Uses LangChain tool calling for delegation
- The delegation system automatically routes to the appropriate execution method based on executor type

### Conversation History Modes

Control how conversation history is managed for agents:

#### **`full`**
- Agent sees **all previous messages** in the conversation
- Best for: Tasks requiring complete conversation context
- Use case: Long-running conversations, context-dependent tasks

#### **`filtered`** (Default)
- Agent sees **filtered conversation history** (system messages and relevant context)
- Best for: Most use cases, balances context with efficiency
- Use case: General purpose agents, standard workflows

#### **`disable`**
- Agent sees **no conversation history** (only current turn)
- Best for: Stateless operations, independent tasks
- Use case: One-off computations, API calls, isolated operations

```python
import streamlit_langgraph as slg

agent = slg.Agent(
    name="analyst",
    role="Data Analyst",
    instructions="Analyze data",
    conversation_history_mode="filtered"  # Default
)
```

### Context Modes

Control how much context each agent receives from workflow execution:

#### **`full`**
- Agent sees **all messages** and previous worker outputs
- Best for: Tasks requiring complete conversation history
- Use case: Analysis, synthesis, decision-making

#### **`summary`**
- Agent sees **summarized context** from previous steps
- Best for: Tasks that need overview but not details
- Use case: High-level coordination, routing decisions

#### **`least`** (Default)
- Agent sees **only supervisor instructions** for their task
- Best for: Focused, independent tasks
- Use case: Specialized computations, API calls

```python
import streamlit_langgraph as slg

analyst = slg.Agent(
    name="analyst",
    role="Data Analyst",
    instructions="Analyze the provided data",
    context="least",  # Default: sees only task instructions
    conversation_history_mode="filtered"  # Default: filtered conversation history
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
    tools=["delete_records"],
    human_in_loop=True,
    interrupt_on={
        "delete_records": {
            "allowed_decisions": ["approve", "reject", "edit"]
        }
    }
)
```

### MCP (Model Context Protocol)

MCP (Model Context Protocol) is an open protocol for standardizing how applications provide tools and context to LLMs. This package supports connecting to MCP servers to access external tools and resources.

#### **What is MCP?**

MCP enables LLMs to interact with external systems through a standardized interface. MCP servers expose tools, resources, and prompts that agents can use, making it easy to integrate with databases, APIs, file systems, and other services.

#### **Transport Types**

MCP servers can communicate via different transport protocols:

1. **STDIO Transport** (Default)
   - Communicates through standard input/output
   - Perfect for local development and command-line tools
   - Each client spawns a new server process
   - Works with all agents (unified executor)

2. **HTTP Transport (streamable_http)**
   - Network-accessible web service
   - Supports multiple concurrent clients
   - Works with all agents (unified executor)
   - When using native OpenAI tools with Responses API: Server must be publicly accessible (not localhost)

3. **SSE Transport** (Legacy)
   - Server-Sent Events transport
   - Backward compatibility only
   - Use HTTP transport for new projects

#### **Configuring MCP Servers**

Configure MCP servers in your agent:

```python
import streamlit_langgraph as slg
import os

# STDIO transport (for local development)
mcp_servers = {
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": [os.path.join("mcp_servers", "math_server.py")]
    }
}

# HTTP transport (for network-accessible servers)
# Note: When using native OpenAI tools with Responses API, server must be publicly accessible
mcp_servers = {
    "math": {
        "transport": "http",  # or "streamable_http" (both accepted)
        "url": "http://your-server.com:8000/mcp"  # Public URL required when using Responses API
    }
}

agent = slg.Agent(
    name="calculator",
    role="Calculator",
    instructions="Use MCP tools to perform calculations",
    provider="openai",
    model="gpt-4o-mini",
    mcp_servers=mcp_servers
)
```

#### **Creating MCP Servers**

Use FastMCP to create MCP servers:

```python
# math_server.py
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run()  # STDIO transport (default)
    # Or: mcp.run(transport="http", port=8000)  # HTTP transport
```

**Running MCP Servers**:

```bash
# Using FastMCP CLI
fastmcp run math_server.py

# Using FastMCP CLI with HTTP transport
fastmcp run math_server.py --transport http
```

#### **Transport Compatibility**

| Transport | Support | Notes |
|-----------|---------|-------|
| **stdio** | ✅ Supported | Local only, perfect for development |
| **http** | ✅ Supported | Network-accessible, supports multiple clients |
| **sse** | ✅ Supported | Legacy, use HTTP instead |

**Important Notes**:
- Executor selection is automatic based on agent configuration (ResponseAPIExecutor or CreateAgentExecutor)
- When using native OpenAI tools (code_interpreter, web_search, etc.) without HITL, ResponseAPIExecutor is used
- **For ResponseAPIExecutor with MCP tools**: MCP servers must be **publicly accessible** (not localhost)
- OpenAI's servers connect to your MCP server when using Responses API, so `localhost` won't work
- For local development with native tools, use stdio transport or deploy MCP servers publicly
- For local development without native tools, stdio or localhost HTTP works fine
- CreateAgentExecutor supports all MCP transport types (stdio, HTTP, localhost)

#### **Example: Local Development**

```python
# Use stdio transport for local development
mcp_servers = {
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["math_server.py"]
    }
}

agent = slg.Agent(
    name="calculator",
    mcp_servers=mcp_servers
)
```

#### **Example: Production Deployment**

```python
# Use HTTP transport with public URL
mcp_servers = {
    "math": {
        "transport": "http",
        "url": "https://your-mcp-server.com/mcp"  # Public URL
    }
}

agent = slg.Agent(
    name="calculator",
    mcp_servers=mcp_servers
)
```

#### **MCP Server Requirements**

For agents using native OpenAI tools (Responses API) with HTTP transport:
1. MCP server must be publicly accessible (not localhost)
2. Server should bind to `0.0.0.0` (not `127.0.0.1`) to accept external connections
3. Security groups/firewalls must allow inbound traffic
4. Use HTTPS for production deployments

#### **Resources**

- [FastMCP Documentation](https://gofastmcp.com/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [LangChain MCP Integration](https://docs.langchain.com/oss/python/langchain/mcp)

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
| `provider` | `str` | `"openai"` | LLM provider: `"openai"`, `"anthropic"`, `"google"`, etc. |
| `model` | `str` | `"gpt-4.1-mini"` | Model name (e.g., `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`) |
| `system_message` | `str` | `None` | Custom system message (auto-generated from role and instructions if None) |
| `temperature` | `float` | `0.0` | Sampling temperature (0.0 to 2.0) |
| `tools` | `List[str]` | `[]` | List of tool names available to the agent |
| `mcp_servers` | `Dict[str, Dict]` | `None` | MCP server configurations (see [MCP Tools](#mcp-model-context-protocol)) |
| `context` | `str` | `"least"` | Context mode: `"full"`, `"summary"`, or `"least"` |
| `human_in_loop` | `bool` | `False` | Enable human-in-the-loop approval for tool execution |
| `interrupt_on` | `Dict` | `{}` | HITL configuration per tool |
| `hitl_description_prefix` | `str` | `"Tool execution pending approval"` | Prefix for HITL approval messages |
| `allow_code_interpreter` | `bool` | `False` | Enable code interpreter (Responses API only) |
| `container_id` | `str` | `None` | OpenAI container ID for code interpreter (auto-created if not provided) |
| `allow_file_search` | `bool` | `False` | Enable file search (Responses API only) |
| `allow_web_search` | `bool` | `False` | Enable web search (Responses API only) |
| `allow_image_generation` | `bool` | `False` | Enable image generation (Responses API only) |
| `conversation_history_mode` | `str` | `"filtered"` | Conversation history mode: `"full"`, `"filtered"`, or `"disable"` |

**Example**:
```python
import streamlit_langgraph as slg

agent = slg.Agent(
    name="analyst",
    role="Data Analyst",
    instructions="Analyze data and provide insights",
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.0,
    tools=["analyze_data", "visualize"],
    context="full",  # See all messages and previous outputs
    conversation_history_mode="filtered",  # Use filtered conversation history
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

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `agents` | `Dict[str, Agent]` | Dictionary of agents keyed by name |
| `active_agent` | `str` | Name of the currently active agent |

**Example**:
```python
import streamlit_langgraph as slg

# Load from YAML
agents = slg.AgentManager.load_from_yaml("config/agents.yaml")

# Or create manager and add agents
manager = slg.AgentManager()
manager.add_agent(my_agent)
agent = manager.agents["analyst"]  # Access via agents dictionary
```

---

### `UIConfig`

**Description**: Configuration for Streamlit UI customization.

**Constructor Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | Required | Application title shown in browser tab and header |
| `page_icon` | `str` | `"🤖"` | Favicon emoji or path to image file |
| `page_layout` | `str` | `"wide"` | Page layout mode: `"wide"` or `"centered"` |
| `stream` | `bool` | `True` | Enable streaming responses |
| `enable_file_upload` | `bool` or `str` | `"multiple"` | File upload configuration: `False`, `True`, `"multiple"`, or `"directory"` |
| `show_sidebar` | `bool` | `True` | Show default sidebar (set False for custom) |
| `user_avatar` | `str` | `"👤"` | Avatar for user messages (emoji or image path) |
| `assistant_avatar` | `str` | `"🤖"` | Avatar for assistant messages (emoji or image path) |
| `placeholder` | `str` | `"Type your message here..."` | Placeholder text for chat input |
| `welcome_message` | `str` | `None` | Welcome message shown at start (supports Markdown) |
| `file_callback` | `Callable[[str], str \| tuple]` | `None` | Optional callback to preprocess files before upload. Can return a single file path or a tuple `(main_file_path, additional_files)` where additional_files can be a directory path or list of file paths |

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
    enable_file_upload="multiple",  # Allow multiple file uploads
    file_callback=None,  # Optional: preprocess files before upload
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
import streamlit as st
import streamlit_langgraph as slg

# Single agent
if "chat" not in st.session_state:
    st.session_state.chat = slg.LangGraphChat(
        agents=[assistant],
        config=config
    )
st.session_state.chat.run()

# Multi-agent with workflow
if "chat" not in st.session_state:
    st.session_state.chat = slg.LangGraphChat(
        workflow=compiled_workflow,
        agents=all_agents,
        config=config
    )
st.session_state.chat.run()
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
| `checkpointer` | `Any` | `None` | Optional checkpointer for workflow state persistence |

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
| `checkpointer` | `Any` | `None` | Optional checkpointer for workflow state persistence |

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

#### `create_network_workflow()`

Creates a network pattern where agents can communicate peer-to-peer in a mesh topology.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Agent]` | Required | List of peer agents. First agent is the entry point |
| `checkpointer` | `Any` | `None` | Optional checkpointer for workflow state persistence |

**Returns**: `StateGraph` - Compiled workflow

**Example**:
```python
import streamlit_langgraph as slg

# Create network of peer agents
agents = [
    tech_strategist,
    business_analyst,
    risk_strategist,
    delivery_lead
]

# Create network workflow
builder = slg.WorkflowBuilder()
workflow = builder.create_network_workflow(agents=agents)
```

**Note**: In network workflows, any agent can hand off to any other agent. There is no central supervisor - all agents are peers.

---

### `CustomTool`

**Description**: Registry for custom tools that agents can use.

**Class Methods**:

#### `register_tool()`

Register a custom function as a tool available to agents.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique tool name |
| `description` | `str` | Required | Description shown to LLM |
| `function` | `Callable` | Required | Python function to execute |
| `parameters` | `Dict` | Auto-extracted | Tool parameters schema (extracted from function signature if not provided) |
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

#### `tool()` (Decorator)

Decorator for registering functions as tools.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique tool name |
| `description` | `str` | Required | Description shown to LLM |
| `**kwargs` | `Any` | - | Additional parameters (e.g., `return_direct`, `parameters`) |

**Returns**: Decorator function

**Example**:
```python
import streamlit_langgraph as slg

@slg.CustomTool.tool("calculator", "Performs basic arithmetic")
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Use in agent
agent = slg.Agent(
    name="calculator",
    role="Calculator",
    instructions="Use calculator to evaluate expressions",
    tools=["calculator"]
)
```

---

## License

MIT License - see LICENSE file for details.

---

**Status**: Pre-alpha | **Python**: 3.10+ | **LangGraph**: 1.0.1

For issues and feature requests, please open an issue on GitHub.
