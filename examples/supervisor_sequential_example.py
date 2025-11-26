import os

import streamlit_langgraph as slg


def create_supervisor_workflow_example():
    """Create a supervisor-based research workflow."""
    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/supervisor_sequential.yaml")
    agents = slg.AgentManager.load_from_yaml(config_path)

    supervisor = agents[0]
    workers = agents[1:]

    return supervisor, workers


def main():
    """Supervisor sequential example with clean workflow pattern."""
    
    # Create a supervisor and workers
    supervisor, workers = create_supervisor_workflow_example()
    # Create a workflow
    builder = slg.WorkflowBuilder()
    supervisor_workflow = builder.create_supervisor_workflow(
        supervisor=supervisor,
        workers=workers,
        execution_mode="sequential",
        delegation_mode="handoff"
    )
    
    config = slg.UIConfig(
        title="Supervised Research Team",
        page_icon="🎓",
        stream=True,
        welcome_message="""Welcome to the **Supervised Research Team**!

**Sequential Supervised Workflow**: Our research supervisor coordinates a team of specialist agents to handle complex research projects.

## 🧠 How It Works:

### **🎯 Team Structure**
**🎯 Research Supervisor**: Project coordination and task delegation
**🔍 Information Gatherer**: Comprehensive research and data collection  
**📝 Proposal Writer**: Professional proposal creation and formatting

### **🏗️ Sequential Workflow**
1. **Supervisor** analyzes your request and creates a project plan
2. **Information Gatherer** and/or **Proposal Writer** are engaged as needed
3. **Supervisor** coordinates handoffs and ensures quality
4. **Workflow can finish at any time when the supervisor determines the task is complete**

### ❓ Example Requests:
- *"Create a research proposal on renewable energy adoption"*
- *"I need a comprehensive analysis of market trends"* 
- *"Research and write a business proposal for AI implementation"*
- *"Develop an academic research proposal for climate change studies"*
""",
        enable_file_upload=True,
        placeholder="Describe your research project or proposal needs..."
    )
    
    chat = slg.LangGraphChat(
        workflow=supervisor_workflow,
        agents=[supervisor] + workers,
        config=config
    )
    chat.run()

if __name__ == "__main__":
    main()
