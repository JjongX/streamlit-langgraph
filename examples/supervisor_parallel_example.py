import os
import yaml

from streamlit_langgraph import Agent, UIConfig, LangGraphChat, CustomTool
from streamlit_langgraph.workflow import WorkflowBuilder

def analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis placeholder."""
    try:
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
    except Exception as e:
        return f"Analysis error: {str(e)}"

def create_parallel_supervisor_workflow():
    """Create a parallel supervisor workflow for comprehensive product analysis."""
    
    # Register custom tool
    CustomTool.register_tool(
        name="analyze_sentiment",
        description="Perform sentiment analysis on text",
        function=analyze_sentiment
    )
    
    # Load agent configurations from YAML
    config_path = os.path.join(os.path.dirname(__file__), "./configs/supervisor_parallel.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        agent_configs = yaml.safe_load(f)
    
    # Set optional arguments for each agent
    # Order: [supervisor, market_analyst, technical_analyst, customer_analyst]
    optional_args = [
        # Analysis_Supervisor
        dict(allow_web_search=True, temperature=0.0, provider="openai", model="gpt-4.1"),
        # Market_Analyst
        dict(allow_web_search=True, temperature=0.2, provider="openai", model="gpt-4.1"),
        # Technical_Analyst
        dict(allow_code_interpreter=True, temperature=0.2, provider="openai", model="gpt-4.1"),
        # Customer_Analyst
        dict(tools=["analyze_sentiment"], allow_web_search=True, temperature=0.2, provider="openai", model="gpt-4.1"),
    ]
    
    # Create agents from config
    agents = []
    for cfg, opts in zip(agent_configs, optional_args):
        agent = Agent(**cfg, **opts)
        agents.append(agent)
    
    supervisor = agents[0]
    workers = agents[1:]
    
    return supervisor, workers

def main():
    """Parallel supervisor example demonstrating simultaneous multi-agent analysis."""
    
    # Create supervisor and workers
    supervisor, workers = create_parallel_supervisor_workflow()
    
    # Build parallel workflow
    builder = WorkflowBuilder()
    parallel_workflow = builder.create_supervisor_workflow(
        supervisor=supervisor,
        workers=workers,
        execution_mode="parallel"  # Key difference: parallel execution!
    )
    
    config = UIConfig(
        title="Parallel Product Analysis Team",
        page_icon="📊",
        stream=True,
        welcome_message="""Welcome to the **Parallel Product Analysis Team**!

**Parallel Supervised Workflow**: Our analysis supervisor coordinates specialist agents who work SIMULTANEOUSLY for faster insights.

## 🧠 How It Works:

### **🎯 Team Structure**
**📊 Analysis Supervisor**: Coordinates parallel analysis and synthesizes results
🔍 **Market Analyst**: Market trends and competitive analysis  
⚙️ **Technical Analyst**: Technical specifications and features
👥 **Customer Analyst**: Customer feedback and sentiment analysis

### **⚡ Parallel Workflow**
1. **Supervisor** receives your analysis request
2. **All three analysts work SIMULTANEOUSLY** on different aspects
3. **LangGraph automatically waits** for all analysts to complete
4. **Supervisor** receives all results and creates comprehensive report

### **✨ Benefits of Parallel Execution**
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
        enable_file_upload=False,
        placeholder="What product would you like our team to analyze?"
    )
    
    chat = LangGraphChat(
        workflow=parallel_workflow,
        agents=[supervisor] + workers,
        config=config
    )
    chat.run()

if __name__ == "__main__":
    main()

