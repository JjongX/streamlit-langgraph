import os

from streamlit_langgraph import UIConfig, LangGraphChat, load_agents_from_yaml
from streamlit_langgraph.workflow import WorkflowBuilder, SupervisorTeam

def create_hierarchical_workflow_example():
    """Create a hierarchical workflow with multiple supervisor teams."""
    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/hierarchical.yaml")
    agents = load_agents_from_yaml(config_path)

    # Unpack agents
    project_manager = agents[0]
    research_team_lead = agents[1]
    data_researcher = agents[2]
    literature_researcher = agents[3]
    content_team_lead = agents[4]
    draft_writer = agents[5]
    content_editor = agents[6]
    
    # Create supervisor teams
    research_team = SupervisorTeam(
        supervisor=research_team_lead,
        workers=[data_researcher, literature_researcher],
        team_name="research_team"
    )
    content_team = SupervisorTeam(
        supervisor=content_team_lead,
        workers=[draft_writer, content_editor],
        team_name="content_team"
    )
    
    return project_manager, [research_team, content_team], agents


def main():
    """Hierarchical workflow example with multiple supervisor teams."""
    
    # Create hierarchical workflow components
    top_supervisor, supervisor_teams, all_agents = create_hierarchical_workflow_example()
    
    # Create the hierarchical workflow
    builder = WorkflowBuilder()
    hierarchical_workflow = builder.create_hierarchical_workflow(
        top_supervisor=top_supervisor,
        supervisor_teams=supervisor_teams,
        execution_mode="sequential"
    )
    
    config = UIConfig(
        title="Hierarchical Multi-Team Organization",
        page_icon="🏢",
        stream=True,
        welcome_message="""Welcome to the **Hierarchical Multi-Team Organization**!

**Hierarchical Workflow**: A senior project manager coordinates multiple specialized teams, each with their own supervisor and specialists.

## 🏗️ Organization Structure:

### **🎯 Senior Leadership**
**🎯 Project Manager**: Overall project coordination and team assignment

### **👥 Specialized Teams**

**🔬 Research Team** (led by Research Team Lead)
- 📊 Data Researcher: Statistics and quantitative information
- 📚 Literature Researcher: Academic papers and qualitative sources

**✍️ Content Team** (led by Content Team Lead)
- 📝 Draft Writer: Initial content creation
- ✏️ Content Editor: Refinement and polishing

## 🔄 How It Works:

1. **Project Manager** analyzes your request and assigns work to appropriate team(s)
2. **Team Supervisors** break down assignments and delegate to their specialists
3. **Specialists** complete their specific tasks and report to their supervisor
4. **Team Supervisors** synthesize team outputs and report to Project Manager
5. **Project Manager** coordinates between teams and delivers final results

## ❓ Example Requests:

- *"Research and write a comprehensive report on climate change impacts"*
- *"Create a business proposal with market research and competitive analysis"*
- *"Develop a white paper on AI adoption with supporting data and literature"*
- *"Research renewable energy trends and write an executive summary"*

**Note**: This is a complex workflow that coordinates multiple teams. Tasks will flow through the appropriate teams based on requirements.
""",
        enable_file_upload=True,
        placeholder="Describe your project that requires research and content creation..."
    )
    
    chat = LangGraphChat(
        workflow=hierarchical_workflow,
        agents=all_agents,
        config=config
    )
    chat.run()


if __name__ == "__main__":
    main()

