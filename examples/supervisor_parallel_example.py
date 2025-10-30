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
    
    # Create supervisor agent
    supervisor = Agent(
        name="Analysis_Supervisor",
        role="Product Analysis Coordinator",
        instructions="""You coordinate a team of specialized analysts who work in PARALLEL.

Your team consists of:
- Market_Analyst: Analyzes market trends and competition
- Technical_Analyst: Reviews technical specifications and features
- Customer_Analyst: Examines customer feedback and sentiment

DELEGATION STRATEGY:
1. When user requests a comprehensive product analysis, delegate to ALL analysts simultaneously
   by using delegate_task with worker_name="PARALLEL"
2. After receiving all parallel results, synthesize them into a final comprehensive report
3. You can delegate to PARALLEL multiple times if needed for different aspects

IMPORTANT: To trigger parallel execution, set target_worker to "PARALLEL" in your delegation.""",
        type="response",
        provider="openai",
        model="gpt-4.1",
        temperature=0.0,
        allow_web_search=True
    )
    
    # Create worker agents
    market_analyst = Agent(
        name="Market_Analyst",
        role="Market Research Specialist",
        instructions="""You analyze market trends, competitive landscape, and market positioning.
Focus on:
- Market size and growth potential
- Competitor analysis
- Market positioning strategies
- Pricing analysis
- Market opportunities and threats

Provide structured, data-driven insights.""",
        type="response",
        provider="openai",
        model="gpt-4.1",
        temperature=0.2,
        allow_web_search=True
    )
    
    technical_analyst = Agent(
        name="Technical_Analyst",
        role="Technical Specifications Expert",
        instructions="""You analyze technical aspects, features, and specifications.
Focus on:
- Technical specifications
- Feature analysis
- Technology stack
- Performance metrics
- Innovation assessment
- Technical advantages/disadvantages

Provide detailed technical insights.""",
        type="response",
        provider="openai",
        model="gpt-4.1",
        temperature=0.2,
        allow_code_interpreter=True
    )
    
    customer_analyst = Agent(
        name="Customer_Analyst",
        role="Customer Insights Specialist",
        instructions="""You analyze customer perspectives, reviews, and sentiment.
Focus on:
- Customer satisfaction
- User experience feedback
- Common praise and complaints
- Customer sentiment analysis
- User demographics
- Customer needs and pain points

Provide customer-centric insights.""",
        type="response",
        provider="openai",
        model="gpt-4.1",
        temperature=0.2,
        tools=["analyze_sentiment"],
        allow_web_search=True
    )
    
    workers = [market_analyst, technical_analyst, customer_analyst]
    
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

