from typing import Any, Callable, Dict, List

from ...agent import Agent
from ..state import WorkflowState, get_metadata

class UtilityNodeFactory:
    """Factory for creating utility nodes like dispatchers, aggregators, etc."""
    
    @staticmethod
    def create_dispatcher_node() -> Callable:
        """Create a dispatcher node for parallel workflows."""
        def dispatcher_node(state: WorkflowState) -> Dict[str, Any]:
            return {"metadata": {**state["metadata"], "parallel_execution": True}}
        
        return dispatcher_node
    
    @staticmethod
    def create_parallel_dispatcher_node(worker_agents: List[Agent]) -> Callable:
        """Create a dispatcher node for parallel execution of workers."""
        def parallel_dispatcher_node(state: WorkflowState) -> Dict[str, Any]:
            supervisor_instructions = ""
            for msg in reversed(state["messages"]):
                if msg["role"] == "assistant" and "Research_Supervisor" in msg.get("agent", ""):
                    supervisor_instructions = msg["content"]
                    break
            
            # Prepare metadata updates
            updated_metadata = state["metadata"].copy()
            updated_metadata["parallel_execution_started"] = True
            updated_metadata["parallel_workers"] = [worker.name for worker in worker_agents]
            updated_metadata["parallel_task_instructions"] = supervisor_instructions
            
            return {"metadata": updated_metadata}
        
        return parallel_dispatcher_node
    
    @staticmethod
    def create_result_aggregator_node(supervisor_agent: Agent) -> Callable:
        """Create an aggregator node to collect parallel worker results."""
        def result_aggregator_node(state: WorkflowState) -> Dict[str, Any]:
            # Collect outputs from parallel workers
            parallel_workers = get_metadata(state, "parallel_workers", [])
            worker_results = []
            
            for msg in state["messages"]:
                if (msg["role"] == "assistant" and 
                    msg.get("agent") in parallel_workers and
                    get_metadata(state, "parallel_execution_started", False)):
                    worker_results.append(f"**{msg['agent']}**: {msg['content'][:300]}...")
            
            # Create aggregated result
            if worker_results:
                aggregated_content = f"""**Parallel Execution Results Aggregated**

The following team members completed their tasks simultaneously:

{chr(10).join(worker_results)}

**Coordination Note**: All parallel tasks have been completed. Ready for supervisor review and next steps."""
                
                # Update agent outputs
                updated_agent_outputs = state["agent_outputs"].copy()
                updated_agent_outputs["Result_Aggregator"] = aggregated_content
                
                # Update metadata
                updated_metadata = state["metadata"].copy()
                updated_metadata["parallel_execution_completed"] = True
                
                return {
                    "messages": state["messages"] + [{
                        "role": "assistant",
                        "content": aggregated_content,
                        "agent": "Result_Aggregator",
                        "timestamp": None
                    }],
                    "agent_outputs": updated_agent_outputs,
                    "metadata": updated_metadata
                }
            
            # Just mark parallel execution as complete if no results
            return {"metadata": {**state["metadata"], "parallel_execution_completed": True}}
        
        return result_aggregator_node

