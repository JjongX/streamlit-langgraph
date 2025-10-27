"""
Streamlit-LangGraph Workflow Module

This module provides a modular, extensible framework for creating and executing 
multi-agent workflows with different orchestration patterns.

## Core Components:

### State Management
- WorkflowState: Manages conversation state, agent outputs, and metadata

### Workflow Builder  
- WorkflowBuilder: Simplified interface for creating different workflow patterns

### Execution Engine
- WorkflowExecutor: Handles workflow execution with Streamlit integration

### Node Factories
- AgentNodeFactory: Creates different types of agent execution nodes
- UtilityNodeFactory: Creates utility nodes (dispatchers, aggregators, etc.)
- CoordinatorNodeFactory: Creates coordinator nodes for complex patterns

### Orchestration Patterns
- SequentialPatterns: Agents execute one after another
- ParallelPatterns: Agents execute simultaneously with result aggregation  
- SupervisorPatterns: Supervisor coordinates worker agents (sequential or parallel)

## Quick Start:

```python
from streamlit_langgraph.workflow import WorkflowBuilder, WorkflowExecutor
from streamlit_langgraph.agent import Agent

# Create agents
agents = [
    Agent("researcher", "Research Expert", "Conduct thorough research"),
    Agent("analyst", "Data Analyst", "Analyze data and findings")  
]

# Build workflow
builder = WorkflowBuilder()
workflow = builder.create_sequential_workflow(agents)

# Execute workflow
executor = WorkflowExecutor()
result = executor.execute_workflow(workflow, "Analyze market trends")
```

## Pattern Examples:

### Sequential Workflow
```python
workflow = builder.create_sequential_workflow(agents)
```

### Parallel Workflow  
```python
workflow = builder.create_parallel_workflow(agents)
```

### Supervisor Workflow
```python
supervisor = Agent("manager", "Project Manager", "Coordinate team efforts")
workflow = builder.create_supervisor_workflow(supervisor, workers, "sequential")
```


"""

from .state import WorkflowState, create_initial_state
from .builder import WorkflowBuilder  
from .executor import WorkflowExecutor, SequentialExecution, ParallelExecution
from .visualization import WorkflowVisualizer, InteractiveWorkflowBuilder
from .nodes import AgentNodeFactory, UtilityNodeFactory
from .patterns import SupervisorPattern

__all__ = [
    # Core workflow components
    'WorkflowState',
    'create_initial_state',
    'WorkflowBuilder', 
    'WorkflowExecutor',
    
    # Visualization components
    'WorkflowVisualizer',
    'InteractiveWorkflowBuilder',
    
    # Node factories
    'AgentNodeFactory',
    'UtilityNodeFactory',
    
    # Execution modes
    'SequentialExecution',
    'ParallelExecution',
    
    # Orchestration patterns
    'SupervisorPattern'
]

# Version information
__version__ = "1.0.0"
