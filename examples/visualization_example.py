import streamlit as st
import sys
import os

# Add the parent directory to the path to import streamlit_langgraph
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_langgraph import Agent
from streamlit_langgraph.workflow import WorkflowVisualizer, InteractiveWorkflowBuilder

def create_sample_agents():
    """Create a set of sample agents for demonstration."""
    agents = [
        Agent(
            name="Researcher",
            role="Research Agent",
            instructions="Research and gather information on the given topic using web search and analysis tools",
            tools=["web_search"],
            provider="openai",
            model="gpt-4.1",
            type="response"
        ),
        Agent(
            name="Analyzer", 
            role="Analysis Agent",
            instructions="Analyze and process the researched information to extract key insights",
            tools=["code_interpreter"],
            provider="openai",
            model="gpt-4.1",
            type="response"
        ),
        Agent(
            name="Writer",
            role="Writing Agent", 
            instructions="Write comprehensive reports based on analysis results",
            tools=[],
            provider="openai",
            model="gpt-4.1",
            type="response"
        ),
        Agent(
            name="Reviewer",
            role="Review Agent",
            instructions="Review and validate the written content for accuracy and quality",
            tools=["file_search"],
            provider="openai",
            model="gpt-4.1",
            type="response"
        )
    ]
    return agents

def create_supervisor_agent():
    """Create supervisor agent for supervisor patterns."""
    supervisor = Agent(
        name="Supervisor",
        role="Workflow Supervisor",
        instructions="Coordinate and manage workflow execution, delegate tasks to appropriate agents",
        tools=[],
        provider="openai",
        model="gpt-4.1",
        type="agent"
    )
    
    return supervisor

def main():
    """Main Streamlit application for workflow visualization demo."""
    
    st.set_page_config(
        page_title="LangGraph Workflow Visualizer",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 LangGraph Workflow Pattern Visualizer")
    st.write("Explore and visualize different workflow patterns using LangGraph")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Pattern Gallery", "Interactive Builder", "Pattern Comparison"]
    )
    
    # Initialize visualizer
    visualizer = WorkflowVisualizer()
    
    if mode == "Pattern Gallery":
        st.header("📋 Workflow Pattern Gallery")
        st.write("Browse through different workflow patterns and see their visualizations.")
        
        # Create sample agents
        agents = create_sample_agents()
        supervisor = create_supervisor_agent()
        
        # Pattern selection
        pattern = st.selectbox(
            "Select a Pattern to Visualize",
            ["Sequential", "Parallel", "Supervisor (Sequential)", "Supervisor (Parallel)"]
        )
        
        if pattern == "Sequential":
            st.subheader("Sequential Workflow Pattern")
            st.write("Agents execute one after another in a predefined sequence.")
            
            with st.expander("Show Agent Details"):
                for i, agent in enumerate(agents, 1):
                    st.write(f"**Step {i}: {agent.name}** - {agent.role}")
            
            if st.button("Visualize Sequential Workflow"):
                visualizer.visualize_sequential_workflow(agents)
        
        elif pattern == "Parallel":
            st.subheader("Parallel Workflow Pattern")
            st.write("Agents execute simultaneously and results are aggregated.")
            
            aggregation = st.selectbox("Aggregation Strategy", ["concatenate", "summarize", "vote"])
            
            if st.button("Visualize Parallel Workflow"):
                visualizer.visualize_parallel_workflow(agents, aggregation)
        
        elif pattern == "Supervisor (Sequential)":
            st.subheader("Supervisor Workflow Pattern - Sequential")
            st.write("A supervisor agent coordinates sequential execution of worker agents.")
            
            if st.button("Visualize Supervisor Sequential Workflow"):
                visualizer.visualize_supervisor_workflow(supervisor, agents, "sequential")
        
        elif pattern == "Supervisor (Parallel)":
            st.subheader("Supervisor Workflow Pattern - Parallel") 
            st.write("A supervisor agent coordinates parallel execution of worker agents.")
            
            if st.button("Visualize Supervisor Parallel Workflow"):
                visualizer.visualize_supervisor_workflow(supervisor, agents, "parallel")
    
    elif mode == "Interactive Builder":
        st.header("🛠️ Interactive Workflow Builder")
        st.write("Build and visualize workflows interactively.")
        
        builder = InteractiveWorkflowBuilder()
        builder.build_workflow_interactively()
    
    elif mode == "Pattern Comparison":
        st.header("🔀 Pattern Comparison")
        st.write("Compare different workflow patterns side by side.")
        
        agents = create_sample_agents()
        supervisor = create_supervisor_agent()
        
        visualizer.compare_workflow_patterns(
            agents=agents,
            supervisor_agent=supervisor
        )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This demo showcases the visualization capabilities of the "
        "streamlit-langgraph library for different workflow patterns."
    )

if __name__ == "__main__":
    main()
