from typing import List, Optional, Any
import streamlit as st
from langgraph.graph import StateGraph

from ..agent import Agent
from .builder import WorkflowBuilder

class WorkflowVisualizer:
    """
    Provides visualization capabilities for different LangGraph workflow patterns.
    
    This class can generate and display visual representations of workflow graphs
    using LangGraph's built-in visualization features with Streamlit integration.
    """
    
    def __init__(self):
        """Initialize the workflow visualizer."""
        self.builder = WorkflowBuilder()
    
    def visualize_sequential_workflow(self, agents: List[Agent], 
                                    title: str = "Sequential Workflow") -> None:
        """
        Visualize a sequential workflow pattern.
        
        Args:
            agents (List[Agent]): List of agents in sequential order
            title (str): Title for the visualization
        """
        workflow = self.builder.create_sequential_workflow(agents)
        self._display_workflow_graph(workflow, title, 
                                   description="Agents execute one after another in sequence.")
    
    def visualize_parallel_workflow(self, agents: List[Agent], 
                                  aggregation_strategy: str = "concatenate",
                                  title: str = "Parallel Workflow") -> None:
        """
        Visualize a parallel workflow pattern.
        
        Args:
            agents (List[Agent]): List of agents executing in parallel
            aggregation_strategy (str): Strategy for combining results
            title (str): Title for the visualization
        """
        workflow = self.builder.create_parallel_workflow(agents, aggregation_strategy)
        self._display_workflow_graph(workflow, title,
                                   description=f"Agents execute simultaneously with {aggregation_strategy} aggregation.")
    
    def visualize_supervisor_workflow(self, supervisor: Agent, workers: List[Agent],
                                    execution_mode: str = "sequential",
                                    title: str = "Supervisor Workflow") -> None:
        """
        Visualize a supervisor workflow pattern.
        
        Args:
            supervisor (Agent): Supervisor agent
            workers (List[Agent]): Worker agents
            execution_mode (str): "sequential" or "parallel" worker execution
            title (str): Title for the visualization
        """
        workflow = self.builder.create_supervisor_workflow(supervisor, workers, execution_mode)
        self._display_workflow_graph(workflow, title,
                                   description=f"Supervisor coordinates {execution_mode} execution of workers.")
    

    
    def _display_workflow_graph(self, workflow: StateGraph, title: str, 
                              description: str = "") -> None:
        """
        Display a workflow graph using Streamlit.
        
        Args:
            workflow (StateGraph): Compiled workflow graph
            title (str): Title for the display
            description (str): Description of the workflow
        """
        st.subheader(title)
        if description:
            st.write(description)
        
        try:
            # Generate graph visualization using LangGraph's built-in method
            graph_image = self._generate_graph_image(workflow)
            
            if graph_image:
                st.image(graph_image, caption=f"{title} - Graph Structure")
            else:
                # Fallback: Display graph structure as text
                self._display_graph_structure_text(workflow, title)
                
        except Exception:
            st.warning(f"Could not generate visual graph for {title}. Showing text representation instead.")
            self._display_graph_structure_text(workflow, title)
    
    def _generate_graph_image(self, workflow: StateGraph) -> Optional[Any]:
        """
        Generate graph image using LangGraph's visualization capabilities.
        
        Args:
            workflow (StateGraph): Compiled workflow graph
            
        Returns:
            Optional[Any]: Graph image or None if generation fails
        """
        try:
            # Try to use LangGraph's get_graph method with visualization
            graph = workflow.get_graph()
            
            # Check if the graph has a draw method (available in newer versions)
            if hasattr(graph, 'draw_mermaid') or hasattr(graph, 'draw_png'):
                # Try different visualization methods
                if hasattr(graph, 'draw_png'):
                    return graph.draw_png()
                elif hasattr(graph, 'draw_mermaid'):
                    # For mermaid diagrams, we'll display as code
                    mermaid_code = graph.draw_mermaid()
                    return self._render_mermaid_in_streamlit(mermaid_code)
                    
        except Exception as e:
            print(f"Graph visualization error: {e}")
            return None
        
        return None
    
    def _render_mermaid_in_streamlit(self, mermaid_code: str) -> None:
        """
        Render Mermaid diagram in Streamlit.
        
        Args:
            mermaid_code (str): Mermaid diagram code
        """
        st.subheader("Workflow Graph (Mermaid)")
        st.code(mermaid_code, language="mermaid")
        
        # If streamlit-mermaid is available, use it for better rendering
        try:
            import streamlit_mermaid as stmd
            stmd.st_mermaid(mermaid_code)
        except ImportError:
            st.info("Install streamlit-mermaid for enhanced diagram rendering: `pip install streamlit-mermaid`")
    
    def _display_graph_structure_text(self, workflow: StateGraph, title: str) -> None:
        """
        Display graph structure as text when visual rendering is not available.
        
        Args:
            workflow (StateGraph): Compiled workflow graph
            title (str): Title for the display
        """
        try:
            graph = workflow.get_graph()
            
            st.subheader(f"{title} - Graph Structure")
            
            # Display nodes
            if hasattr(graph, 'nodes'):
                st.write("**Nodes:**")
                for node in graph.nodes:
                    st.write(f"- {node}")
            
            # Display edges
            if hasattr(graph, 'edges'):
                st.write("**Edges:**")
                for edge in graph.edges:
                    st.write(f"- {edge}")
                    
        except Exception as e:
            st.error(f"Could not display graph structure: {e}")
    
    def compare_workflow_patterns(self, agents: List[Agent], 
                                supervisor_agent: Optional[Agent] = None) -> None:
        """
        Display a comparison of different workflow patterns using the same set of agents.
        
        Args:
            agents (List[Agent]): Base set of agents to use across patterns
            supervisor_agent (Optional[Agent]): Agent to use as supervisor (if provided)
        """
        st.header("Workflow Pattern Comparison")
        
        # Create tabs for different patterns
        tabs = st.tabs(["Sequential", "Parallel", "Supervisor"])
        
        with tabs[0]:
            self.visualize_sequential_workflow(agents)
        
        with tabs[1]:
            self.visualize_parallel_workflow(agents)
        
        with tabs[2]:
            if supervisor_agent:
                self.visualize_supervisor_workflow(supervisor_agent, agents)
            else:
                st.warning("Supervisor agent not provided for supervisor pattern.")


class InteractiveWorkflowBuilder(WorkflowBuilder):
    """
    Interactive workflow builder that extends WorkflowBuilder with Streamlit UI.
    
    Inherits all workflow creation logic from WorkflowBuilder and adds:
    - Interactive Streamlit interface
    - Real-time workflow visualization
    - Agent configuration UI
    - Pattern selection interface
    """
    
    def __init__(self):
        """Initialize the interactive builder."""
        super().__init__()  # Initialize parent WorkflowBuilder
        self.visualizer = WorkflowVisualizer()
    
    def build_workflow_interactively(self) -> None:
        """
        Provide an interactive interface for building and visualizing workflows.
        """
        st.header("Interactive Workflow Builder")
        
        # Workflow pattern selection
        pattern = st.selectbox(
            "Select Workflow Pattern",
            ["Sequential", "Parallel", "Supervisor"]
        )
        
        # Agent configuration
        st.subheader("Configure Agents")
        num_agents = st.number_input("Number of Agents", min_value=1, max_value=10, value=3)
        
        agents = []
        for i in range(num_agents):
            with st.expander(f"Agent {i+1}"):
                name = st.text_input(f"Agent {i+1} Name", value=f"Agent_{i+1}")
                role = st.text_input(f"Agent {i+1} Role", value="Worker Agent")
                instructions = st.text_area(f"Agent {i+1} Instructions", 
                                          value=f"Perform specialized tasks as a {role}")
                tools = st.multiselect(f"Agent {i+1} Tools", 
                                     ["web_search", "code_interpreter", "file_search"])
                
                # Create agent with proper parameters
                try:
                    agent = Agent(
                        name=name, 
                        role=role, 
                        instructions=instructions,
                        tools=tools
                    )
                    agents.append(agent)
                except Exception as e:
                    st.error(f"Could not create agent {i+1}: {e}")
        
        # Pattern-specific configuration
        if pattern == "Parallel":
            aggregation = st.selectbox(
                "Aggregation Strategy", 
                ["concatenate", "summarize", "vote"]
            )
        elif pattern == "Supervisor":
            execution_mode = st.selectbox(
                "Execution Mode",
                ["sequential", "parallel"]
            )
        
        # Build and visualize button
        if st.button("Build and Visualize Workflow"):
            if not agents:
                st.error("Please configure at least one agent.")
                return
            
            try:
                if pattern == "Sequential":
                    self.visualizer.visualize_sequential_workflow(agents)
                elif pattern == "Parallel":
                    self.visualizer.visualize_parallel_workflow(agents, aggregation)
                elif pattern == "Supervisor":
                    if len(agents) < 2:
                        st.error("Supervisor pattern requires at least 2 agents (1 supervisor + 1 worker).")
                        return
                    supervisor = agents[0]
                    workers = agents[1:]
                    self.visualizer.visualize_supervisor_workflow(supervisor, workers, execution_mode)

                    
            except Exception as e:
                st.error(f"Error building workflow: {e}")
