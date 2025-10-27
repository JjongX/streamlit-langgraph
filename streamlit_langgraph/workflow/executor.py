import streamlit as st
from typing import Dict, Any, Optional, List, Callable, Union
from langgraph.graph import StateGraph, START, END

from .state import WorkflowState, create_initial_state
from ..agent import Agent
from .nodes import AgentNodeFactory, UtilityNodeFactory

class WorkflowExecutor:
    """
    Handles execution of compiled workflows with Streamlit integration.
    """
    
    def __init__(self):
        """Initialize the workflow executor."""
        pass
    
    def execute_workflow(self, workflow: StateGraph, user_input: str, 
                        display_callback: Optional[Callable] = None,
                        config: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """
        Execute a compiled workflow with the given user input.
        
        Args:
            workflow (StateGraph): Compiled workflow graph
            user_input (str): User's input/request
            display_callback (Callable): Optional callback for displaying messages
            config (Dict): Optional configuration for workflow execution
            
        Returns:
            WorkflowState: Final state after workflow execution
        """
        # Initialize workflow state
        initial_state = create_initial_state(
            messages=[{"role": "user", "content": user_input}]
        )
        
        if config:
            initial_state["metadata"].update(config)
        
        try:
            # Execute the workflow
            if display_callback:
                return self._execute_with_display(workflow, initial_state, display_callback)
            else:
                return self._execute_basic(workflow, initial_state)
                
        except Exception as e:
            st.error(f"Workflow execution failed: {str(e)}")
            # Return state with error information
            error_message = {"role": "system", "content": f"Execution error: {str(e)}", "agent": None, "timestamp": None}
            initial_state["messages"].append(error_message)
            return initial_state
    
    def _execute_basic(self, workflow: StateGraph, initial_state: WorkflowState) -> WorkflowState:
        """Execute workflow without display callbacks."""
        config = {"recursion_limit": 50}
        try:
            final_state = workflow.invoke(initial_state, config=config)
        except KeyError as e:
            if str(e) in ["'__end__'", "'END'"]:
                raise
        
        last_agent = final_state["messages"][-1].get("agent") if final_state["messages"] else None
        if last_agent in ["END", "__end__"]:
            final_state["messages"].append({
                "role": "system",
                "content": "Workflow completed successfully.",
                "agent": None,
                "timestamp": None
            })
        return final_state
    
    def _execute_with_display(self, workflow: StateGraph, initial_state: WorkflowState, 
                            display_callback: Callable) -> WorkflowState:
        """Execute workflow with real-time display updates."""
        accumulated_state = initial_state.copy()
        config = {"recursion_limit": 50}
        try:
            for node_output in workflow.stream(initial_state, config=config):
                for node_name, state_update in node_output.items():
                    if isinstance(state_update, dict):
                        if "messages" in state_update:
                            accumulated_state["messages"] = state_update["messages"]
                        if "metadata" in state_update:
                            accumulated_state["metadata"].update(state_update["metadata"])
                        if "agent_outputs" in state_update:
                            accumulated_state["agent_outputs"].update(state_update["agent_outputs"])
                        if "current_agent" in state_update:
                            accumulated_state["current_agent"] = state_update["current_agent"]
                    if display_callback:
                        display_callback(accumulated_state)
        except KeyError as e:
            if str(e) in ["'__end__'", "'END'"]:
                accumulated_state["messages"].append({
                    "role": "system",
                    "content": "Workflow completed successfully.",
                    "agent": None,
                    "timestamp": None
                })
                return accumulated_state
            else:
                st.error(f"Error during workflow execution: {str(e)}")
                raise e
        last_agent = accumulated_state["messages"][-1].get("agent") if accumulated_state["messages"] else None
        if last_agent in ["END", "__end__"]:
            accumulated_state["messages"].append({
                "role": "system",
                "content": "Workflow completed successfully.",
                "agent": None,
                "timestamp": None
            })
        return accumulated_state
    
    def get_workflow_results(self, final_state: WorkflowState) -> Dict[str, Any]:
        """
        Extract and format workflow results from the final state.
        
        Args:
            final_state (WorkflowState): The final state after workflow execution
            
        Returns:
            Dict[str, Any]: Formatted results including agent outputs and metadata
        """
        results = {
            "messages": final_state["messages"],
            "agent_outputs": final_state["agent_outputs"],
            "metadata": final_state["metadata"],
            "summary": self._create_results_summary(final_state)
        }
        
        return results
    
    def _create_results_summary(self, final_state: WorkflowState) -> str:
        """Create a summary of the workflow execution results."""
        
        # Count agent contributions
        agent_count = len([msg for msg in final_state["messages"] if msg["role"] == "assistant"])
        
        # Get the final response (last assistant message)
        final_response = ""
        for msg in reversed(final_state["messages"]):
            if msg["role"] == "assistant":
                final_response = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                break
        
        # Get participating agents
        participating_agents = list(final_state["agent_outputs"].keys())
        
        summary = f"""
Workflow Execution Summary:
- Total agent interactions: {agent_count}
- Participating agents: {', '.join(participating_agents)}
- Workflow steps completed: {len(final_state["messages"])}
- Final response: {final_response}
        """.strip()
        
        return summary
    
    def display_agent_conversation(self, final_state: WorkflowState):
        """
        Display the complete agent conversation in a structured format.
        
        Args:
            final_state (WorkflowState): The final state containing all messages
        """
        st.subheader("🔄 Agent Conversation Flow")
        
        messages = final_state["messages"]
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            agent = message.get("agent", "System")
            
            if role == "user":
                with st.chat_message("user"):
                    st.write(f"**User:** {content}")
            
            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.write(f"**{agent}:** {content}")
            
            elif role == "system":
                with st.expander(f"System Message {i+1}", expanded=False):
                    st.write(content)
    
    def display_agent_outputs(self, final_state: WorkflowState):
        """
        Display individual agent outputs in an organized manner.
        
        Args:
            final_state (WorkflowState): The final state containing agent outputs
        """
        agent_outputs = final_state["agent_outputs"]
        
        if not agent_outputs:
            st.info("No individual agent outputs to display.")
            return
        
        st.subheader("🤖 Individual Agent Contributions")
        
        # Create tabs for each agent if multiple agents
        if len(agent_outputs) > 1:
            agent_names = list(agent_outputs.keys())
            tabs = st.tabs(agent_names)
            
            for i, agent_name in enumerate(agent_names):
                with tabs[i]:
                    output = agent_outputs[agent_name]
                    st.write(output)
        else:
            # Single agent - no tabs needed
            for agent_name, output in agent_outputs.items():
                st.write(f"**{agent_name}:**")
                st.write(output)
    
    def display_workflow_metadata(self, final_state: WorkflowState):
        """
        Display workflow execution metadata and statistics.
        
        Args:
            final_state (WorkflowState): The final state containing metadata
        """
        metadata = getattr(final_state, 'metadata', {})
        
        if not metadata:
            return
        
        with st.expander("⚙️ Workflow Execution Details", expanded=False):
            st.json(metadata)


class SequentialExecution:
    """Sequential execution mode for multi-agent workflows."""
    
    @staticmethod
    def create_sequential_workflow(agents: List[Agent]) -> StateGraph:
        """
        Create a sequential workflow where agents execute one after another.
        
        Args:
            agents (List[Agent]): List of agents in execution order
        Returns:
            StateGraph: Compiled workflow graph
        """
    # No need to import START, just use the string "START" for compatibility
        if not agents:
            raise ValueError("At least one agent is required")
        graph = StateGraph(WorkflowState)
        # Add nodes for each agent
        for agent in agents:
            graph.add_node(agent.name, AgentNodeFactory.create_basic_agent_node(agent))
        # Add sequential edges
        for i in range(len(agents) - 1):
            graph.add_edge(agents[i].name, agents[i + 1].name)
        # Always start with START and end with END (from langgraph.graph)
        graph.add_edge(START, agents[0].name)
        graph.add_edge(agents[-1].name, END)
        graph.set_entry_point(START)
        return graph.compile()

class ParallelExecution:
    """Parallel execution mode for multi-agent workflows."""
    
    @staticmethod
    def create_parallel_workflow(agents: List[Agent], 
                               aggregation_strategy: Union[str, Agent] = "concatenate") -> StateGraph:
        """
        Create a parallel workflow where agents execute simultaneously.
        
        Args:
            agents (List[Agent]): List of agents to execute in parallel
            aggregation_strategy (Union[str, Agent]): How to combine results or aggregator agent
        Returns:
            StateGraph: Compiled workflow graph
        """
    # No need to import START, just use the string "START" for compatibility
        if not agents:
            raise ValueError("At least one agent is required")
        graph = StateGraph(WorkflowState)
        # Add a dispatcher node to start parallel execution
        graph.add_node("dispatcher", UtilityNodeFactory.create_dispatcher_node())
        # Add nodes for each agent
        for agent in agents:
            graph.add_node(agent.name, AgentNodeFactory.create_basic_agent_node(agent))
            graph.add_edge("dispatcher", agent.name)
        # Handle aggregation
        if isinstance(aggregation_strategy, Agent):
            aggregator_agent = aggregation_strategy
            graph.add_node(aggregator_agent.name, AgentNodeFactory.create_basic_agent_node(aggregator_agent))
            for agent in agents:
                graph.add_edge(agent.name, aggregator_agent.name)
            graph.add_edge(aggregator_agent.name, END)
        elif isinstance(aggregation_strategy, str):
            aggregator_agent = ParallelExecution._create_aggregator_agent(aggregation_strategy)
            graph.add_node(aggregator_agent.name, AgentNodeFactory.create_basic_agent_node(aggregator_agent))
            for agent in agents:
                graph.add_edge(agent.name, aggregator_agent.name)
            graph.add_edge(aggregator_agent.name, END)
        else:
            for agent in agents:
                graph.add_edge(agent.name, END)
        # Always start with START (from langgraph.graph)
        graph.add_edge(START, "dispatcher")
        graph.set_entry_point(START)
        return graph.compile()
    
    @staticmethod
    def _create_aggregator_agent(strategy: str) -> Agent:
        """Create an aggregator agent based on the aggregation strategy."""
        if strategy == "concatenate":
            return Agent(
                name="Aggregator",
                role="Result Aggregator",
                instructions="Concatenate all results from parallel agents into a single comprehensive output",
                tools=[]
            )
        elif strategy == "summarize":
            return Agent(
                name="Summarizer",
                role="Result Summarizer", 
                instructions="Summarize and synthesize results from parallel agents into a concise summary",
                tools=[]
            )
        elif strategy == "vote":
            return Agent(
                name="Voter",
                role="Result Voter",
                instructions="Analyze results from parallel agents and select the best or most relevant output",
                tools=[]
            )
        else:
            return Agent(
                name="Aggregator",
                role="Result Aggregator",
                instructions=f"Combine results from parallel agents using {strategy} strategy",
                tools=[]
            )
