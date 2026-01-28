import os

import streamlit as st
import streamlit_langgraph as slg
from streamlit_langgraph.core.executor.registry import ExecutorRegistry

def verify_executor_selection(agents: list) -> dict:
    """
    Verify which executor would be selected for each agent.
    
    Returns a dict mapping agent names to their expected executor type.
    """
    results = {}
    for agent in agents:
        has_native = ExecutorRegistry.has_native_tools(agent)
        if has_native and not agent.human_in_loop:
            executor_type = "ResponseAPIExecutor"
        else:
            executor_type = "CreateAgentExecutor"
        
        results[agent.name] = {
            "provider": agent.provider,
            "model": agent.model,
            "has_native_tools": has_native,
            "human_in_loop": agent.human_in_loop,
            "executor": executor_type,
        }
    return results


def create_multi_provider_workflow():
    """Create a supervisor workflow with OpenAI + Google agents."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        "./configs/10_multi_provider_executors.yaml",
    )
    agents = slg.Agent(config_path)
    supervisor = agents[0]
    workers = agents[1:]
    
    builder = slg.WorkflowBuilder()
    workflow = builder.create_supervisor_workflow(
        supervisor=supervisor,
        workers=workers,
        execution_mode="sequential",
        delegation_mode="handoff",
    )
    return workflow, agents


def display_executor_info(agents: list):
    """Display executor selection information in the sidebar."""
    st.sidebar.markdown("### Executor Selection")
    
    executor_info = verify_executor_selection(agents)
    
    for agent_name, info in executor_info.items():
        with st.sidebar.expander(f"**{agent_name}**", expanded=True):
            st.markdown(f"**Provider:** `{info['provider']}`")
            st.markdown(f"**Model:** `{info['model']}`")
            st.markdown(f"**Native Tools:** `{info['has_native_tools']}`")
            st.markdown(f"**HITL:** `{info['human_in_loop']}`")
            
            # Color-code the executor type
            if info['executor'] == "ResponseAPIExecutor":
                st.success(f"→ {info['executor']}")
            else:
                st.info(f"→ {info['executor']}")


def main():
    """Main entry point for the multi-provider executors demo."""
    workflow, agents = create_multi_provider_workflow()
    
    # Display executor selection info in sidebar
    display_executor_info(agents)
    
    config = slg.UIConfig(
        title="Multi-Provider Executors Demo",
        page_icon="🧭",
        welcome_message="""Welcome to the **Multi-Provider Executors Demo**!

This workflow demonstrates how different executors are automatically selected based on agent configuration.

## 🔧 Executor Selection Logic

| Agent | Provider | Native Tools | Executor |
|-------|----------|--------------|----------|
| **OpenAI_Supervisor** | OpenAI | ✅ file_search | `ResponseAPIExecutor` |
| **OpenAI_Writer** | OpenAI | ❌ | `CreateAgentExecutor` |
| **Gemini_Analyst** | Google | ❌ | `CreateAgentExecutor` |

## 💡 Try These Prompts

- **"Compare the benefits of solar vs wind energy"** - Tests both workers
- **"Write a short summary about quantum computing"** - Tests OpenAI Writer
- **"Analyze the pros and cons of remote work"** - Tests Gemini Analyst
- **"What are the key differences between Python and JavaScript?"** - Tests delegation

Check the sidebar to see the executor configuration for each agent!
""",
        placeholder="Ask something to test the multi-provider workflow...",
    )
    
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            workflow=workflow,
            agents=agents,
            config=config,
        )
    st.session_state.chat.run()


if __name__ == "__main__":
    main()
