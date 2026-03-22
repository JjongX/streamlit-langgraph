import os

import streamlit as st
import streamlit_langgraph as slg


def analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis placeholder."""
    # In real implementation, this would use an NLP library
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'poor', 'horrible', 'disappointing']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return f"Sentiment: Positive (score: +{pos_count - neg_count})"
    elif neg_count > pos_count:
        return f"Sentiment: Negative (score: -{neg_count - pos_count})"
    else:
        return "Sentiment: Neutral"


def create_parallel_supervisor_workflow():
    """Create a parallel supervisor workflow for comprehensive product analysis."""
    
    slg.CustomTool.register_tool(
        name="analyze_sentiment",
        description="Perform sentiment analysis on text",
        function=analyze_sentiment
    )
    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/03_supervisor_parallel.yaml")
    agents = slg.Agent(config_path)
    supervisor = agents[0]
    workers = agents[1:]
    
    return supervisor, workers


def main():
    """Parallel supervisor example demonstrating simultaneous multi-agent analysis."""
    
    # Create supervisor and workers
    supervisor, workers = create_parallel_supervisor_workflow()
    # Create a workflow
    builder = slg.WorkflowBuilder()
    parallel_workflow = builder.create_supervisor_workflow(
        supervisor=supervisor,
        workers=workers,
        execution_mode="parallel",
        delegation_mode="handoff"
    )
    
    config = slg.UIConfig(
        title="Parallel Product Analysis Team",
        page_icon="📊",
        welcome_message="""Welcome to the **Parallel Product Analysis Team**!

**Parallel Supervised Workflow**: Our analysis supervisor coordinates specialist agents who work SIMULTANEOUSLY for faster insights.

## 🧠 How It Works:

### 🎯 Team Structure
- 📊 **Analysis Supervisor**: Coordinates parallel analysis and synthesizes results
- 🔍 **Market Analyst**: Market trends and competitive analysis
- ⚙️ **Technical Analyst**: Technical specifications and features
- 👥 **Customer Analyst**: Customer feedback and sentiment analysis

### ⚡ Parallel Workflow
1. **Supervisor** receives your analysis request
2. **All three analysts work SIMULTANEOUSLY** on different aspects
3. **LangGraph automatically waits** for all analysts to complete
4. **Supervisor** receives all results and creates comprehensive report

### ✨ Benefits of Parallel Execution
- ⚡ **Faster**: All analysts work at the same time
- 🎯 **Comprehensive**: Multiple perspectives analyzed simultaneously
- 🔄 **Efficient**: No waiting for sequential handoffs

## ❓ Example Requests:

- *"Analyze the iPhone 15 Pro - market position, technical specs, and customer sentiment"*
- *"Comprehensive analysis of Tesla Model 3"* 
- *"Evaluate the new PlayStation 5 from all angles"*
- *"Full product analysis of Apple Vision Pro"*

**Note**: The supervisor will delegate to ALL analysts in parallel for comprehensive analysis!
""",
        placeholder="What product would you like our team to analyze?"
    )
    
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            workflow=parallel_workflow,
            agents=[supervisor] + workers,
            config=config
        )
    st.session_state.chat.run()

if __name__ == "__main__":
    main()

