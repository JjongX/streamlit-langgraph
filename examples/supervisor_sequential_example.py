import os

from streamlit_langgraph import UIConfig, LangGraphChat, CustomTool, load_agents_from_yaml
from streamlit_langgraph.workflow import WorkflowBuilder

def format_proposal(content: str, proposal_type: str = "academic") -> str:
    """Format content into a simple research proposal structure."""
    try:
        formatted = f"""# Research Proposal ({proposal_type.title()})

## Overview
{content[:1000] if len(content) > 1000 else content}

## Key Sections
- Background and context
- Research objectives
- Methodology approach
- Expected outcomes

---
*Formatted using format_proposal tool*
"""
        return formatted
    except Exception as e:
        return f"Formatting error: {str(e)}"

def create_supervisor_workflow_example():
    """Create a supervisor-based research workflow."""
    
    CustomTool.register_tool(
        name="format_proposal",
        description="Format content into a simple research proposal structure",
        function=format_proposal
    )
    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/supervisor_sequential.yaml")
    agents = load_agents_from_yaml(config_path)

    supervisor = agents[0]
    workers = agents[1:]

    return supervisor, workers

def main():
    """Supervisor sequential example with clean workflow pattern."""
    
    # Create a supervisor and workers
    supervisor, workers = create_supervisor_workflow_example()
    
    # Create a workflow
    builder = WorkflowBuilder()
    supervisor_workflow = builder.create_supervisor_workflow(
        supervisor=supervisor,
        workers=workers,
        execution_mode="sequential",
        delegation_mode="handoff"
    )
    
    config = UIConfig(
        title="Supervised Research Team",
        page_icon="🎓",
        stream=True,
        welcome_message="""Welcome to the **Supervised Research Team**!

**Sequential Supervised Workflow**: Our research supervisor coordinates a team of specialist agents to handle complex research projects.

## 🧠 How It Works:

### **🎯 Team Structure**
**🎯 Research Supervisor**: Project coordination and task delegation
🔍 **Information Gatherer**: Comprehensive research and data collection  
📝 **Proposal Writer**: Professional proposal creation and formatting

### **🏗️ Sequential Workflow**
1. **Supervisor** analyzes your request and creates a project plan
2. **Information Gatherer** and/or **Proposal Writer** are engaged as needed
3. **Supervisor** coordinates handoffs and ensures quality
4. **Workflow can finish at any time when the supervisor determines the task is complete**

## ❓ Example Requests:

- *"Create a research proposal on renewable energy adoption"*
- *"I need a comprehensive analysis of market trends"* 
- *"Research and write a business proposal for AI implementation"*
- *"Develop an academic research proposal for climate change studies"*
""",
        enable_file_upload=True,
        placeholder="Describe your research project or proposal needs..."
    )
    
    chat = LangGraphChat(
        workflow=supervisor_workflow,
        agents=[supervisor] + workers, # Might update later to not pass agents
        config=config
    )
    chat.run()

if __name__ == "__main__":
    main()
