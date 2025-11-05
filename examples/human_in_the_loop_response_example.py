from streamlit_langgraph import Agent, UIConfig, LangGraphChat, CustomTool
from streamlit_langgraph.workflow import WorkflowBuilder

# Example tools that will trigger HITL
def analyze_sentiment(text: str, context: str = None) -> str:
    """
    Analyze the sentiment of a given text. This operation processes text to determine emotional tone and requires approval.
    
    Args:
        text: The text content to analyze for sentiment
        context: Optional context about the text (e.g., 'customer_review', 'social_media_post', 'support_ticket')
    
    Returns:
        A detailed sentiment analysis report with scores and classifications
    
    Example:
        analyze_sentiment("I love this product! It works perfectly.", "customer_review")
        analyze_sentiment("This service is terrible and I want a refund.", "support_ticket")
    """
    import time
    
    try:
        # Simulate sentiment analysis
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        analysis_id = f"SENT-{int(time.time())}"
        
        # Simple keyword-based sentiment detection for testing
        text_lower = text.lower()
        positive_words = ['love', 'great', 'excellent', 'good', 'amazing', 'wonderful', 'perfect', 'happy', 'satisfied', 'fantastic']
        negative_words = ['hate', 'terrible', 'awful', 'bad', 'horrible', 'worst', 'disappointed', 'angry', 'frustrated', 'refund']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        if positive_count > negative_count:
            sentiment_label = "POSITIVE"
            sentiment_score = min(0.5 + (positive_count * 0.15), 1.0)
        elif negative_count > positive_count:
            sentiment_label = "NEGATIVE"
            sentiment_score = max(-1.0 + (negative_count * 0.15), -1.0)
        else:
            sentiment_label = "NEUTRAL"
            sentiment_score = 0.0
        
        # Determine emotion intensity
        intensity = abs(sentiment_score)
        if intensity > 0.7:
            intensity_label = "STRONG"
        elif intensity > 0.4:
            intensity_label = "MODERATE"
        else:
            intensity_label = "WEAK"
        
        context_str = f"Context: {context}\n" if context else ""
        
        return f"✓ Sentiment analysis completed\n" \
               f"  Analysis ID: {analysis_id}\n" \
               f"  Timestamp: {timestamp}\n" \
               f"  {context_str}" \
               f"  Sentiment: {sentiment_label}\n" \
               f"  Score: {sentiment_score:.2f} (range: -1.0 to 1.0)\n" \
               f"  Intensity: {intensity_label}\n" \
               f"  Text length: {len(text)} characters\n\n" \
               f"  Analyzed text: {text[:150]}{'...' if len(text) > 150 else ''}"
    
    except Exception as e:
        return f"✗ Error analyzing sentiment: {str(e)}"

def escalate_negative_sentiment(text: str, sentiment_score: float, source: str, urgency: str = "medium") -> str:
    """
    Escalate cases with negative sentiment for review. This operation flags content for human intervention and requires approval.
    
    Args:
        text: The original text that was analyzed
        sentiment_score: The sentiment score from analysis (negative values indicate negative sentiment)
        source: Source of the text (e.g., 'customer_review', 'social_media', 'support_ticket', 'feedback_form')
        urgency: Urgency level - 'low', 'medium', 'high', or 'critical' (default: 'medium')
    
    Returns:
        A status message with escalation details and ticket ID
    
    Example:
        escalate_negative_sentiment("This product is terrible", -0.85, "customer_review", "high")
        escalate_negative_sentiment("I want to cancel my subscription", -0.65, "support_ticket", "medium")
    """
    import time
    import uuid
    
    try:
        # Validate urgency
        valid_urgency = ["low", "medium", "high", "critical"]
        if urgency.lower() not in valid_urgency:
            urgency = "medium"
        
        # Generate escalation ticket
        ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine escalation reason based on score
        if sentiment_score < -0.7:
            reason = "Highly negative sentiment detected - requires immediate attention"
            priority = "HIGH" if urgency == "medium" else urgency.upper()
        elif sentiment_score < -0.4:
            reason = "Negative sentiment detected - review recommended"
            priority = urgency.upper()
        else:
            reason = "Moderate negative sentiment - standard review"
            priority = urgency.upper()
        
        return f"✓ Escalation created successfully\n" \
               f"  Ticket ID: {ticket_id}\n" \
               f"  Priority: {priority}\n" \
               f"  Urgency: {urgency.upper()}\n" \
               f"  Source: {source}\n" \
               f"  Sentiment Score: {sentiment_score:.2f}\n" \
               f"  Reason: {reason}\n" \
               f"  Created: {timestamp}\n" \
               f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}"
    
    except Exception as e:
        return f"✗ Error creating escalation: {str(e)}"

def main():
    CustomTool.register_tool(
        name="analyze_sentiment",
        description=(
            "Analyze the sentiment of text content. Use this tool when you need to: "
            "- Analyze customer reviews or feedback\n"
            "- Evaluate sentiment in social media posts\n"
            "- Assess sentiment in support tickets or emails\n"
            "- Monitor brand sentiment across communications\n\n"
            "Returns sentiment classification (POSITIVE, NEGATIVE, NEUTRAL), sentiment score (-1.0 to 1.0), "
            "and intensity level. This tool requires approval to ensure proper handling of sensitive content. "
            "Example: analyze_sentiment('I love this product!', 'customer_review')"
        ),
        function=analyze_sentiment
    )
    
    CustomTool.register_tool(
        name="escalate_negative_sentiment",
        description=(
            "Escalate cases with negative sentiment for human review. Use this tool when you need to: "
            "- Flag negative customer feedback for follow-up\n"
            "- Escalate support tickets with negative sentiment\n"
            "- Alert teams about negative brand mentions\n"
            "- Create review tickets for negative sentiment cases\n\n"
            "This tool creates escalation tickets with priority levels based on sentiment severity. "
            "Requires approval to ensure appropriate escalation handling. "
            "Example: escalate_negative_sentiment('This product is terrible', -0.85, 'customer_review', 'high')"
        ),
        function=escalate_negative_sentiment
    )
    
    # Using ResponseAPIExecutor (type="response") with HITL enabled
    supervisor = Agent(
        name="supervisor",
        role="Sentiment Analysis Supervisor",
        instructions="You coordinate sentiment analysis tasks and delegate to specialists. You have access to tools for analyzing sentiment, escalating negative sentiment cases, and sending notifications. Coordinate the workflow and ensure all tasks are properly handled.",
        type="response",  # Using ResponseAPIExecutor instead of CreateAgentExecutor
        provider="openai",
        model="gpt-4o-mini",
        tools=["analyze_sentiment", "escalate_negative_sentiment", "send_notification"],
        human_in_loop=True,  # Enable HITL for supervisor
        interrupt_on={
            "analyze_sentiment": {"allowed_decisions": ["approve", "reject"]},
            "escalate_negative_sentiment": {"allowed_decisions": ["approve", "reject", "edit"]},
        },
        hitl_description_prefix="Supervisor action requires approval"
    )
    
    workers = [
        Agent(
            name="researcher",
            role="Information Researcher",
            instructions="You gather comprehensive information on topics and provide research summaries.",
            type="response",  # Using ResponseAPIExecutor
            provider="openai",
            model="gpt-4o-mini",
        ),
        Agent(
            name="analyst",
            role="Sentiment Analyst",
            instructions="You analyze sentiment in customer communications and feedback. You can analyze sentiment using analyze_sentiment and escalate negative cases using escalate_negative_sentiment. Always provide detailed analysis reports.",
            type="response",  # Using ResponseAPIExecutor
            provider="openai",
            model="gpt-4o-mini",
            tools=["analyze_sentiment", "escalate_negative_sentiment"],
            human_in_loop=True,
            interrupt_on={
                "analyze_sentiment": {"allowed_decisions": ["approve", "reject"]},
                "escalate_negative_sentiment": {"allowed_decisions": ["approve", "reject", "edit"]},
            },
            hitl_description_prefix="Analyst action pending approval"
        ),
        Agent(
            name="notifier",
            role="Notification Coordinator",
            instructions="You handle notifications and communications. You can send notifications using send_notification to alert teams about important events or escalated cases.",
            type="response",  # Using ResponseAPIExecutor
            provider="openai",
            model="gpt-4o-mini",
        )
    ]

    # Create a multiagent workflow
    builder = WorkflowBuilder()
    workflow = builder.create_supervisor_workflow(
        supervisor=supervisor,
        workers=workers,
        execution_mode="sequential",
        delegation_mode="handoff"
    )
    
    config = UIConfig(
        title="Human-in-the-Loop Multiagent Workflow (ResponseAPI)",
        welcome_message=(
            "Hello! This is a **multiagent workflow** using **ResponseAPIExecutor** with human-in-the-loop approval enabled.\n\n"
            "**How it works:**\n"
            "- Multiple agents work together in a coordinated workflow\n"
            "- All agents use OpenAI's Response API (ResponseAPIExecutor)\n"
            "- When agents need to execute tools that require approval, execution pauses\n"
            "- You'll see a warning message with the tool details\n"
            "- You can approve, reject, or edit the tool input\n"
            "- Once you make a decision, the workflow continues\n\n"
            "**Key Features:**\n"
            "- Uses ResponseAPIExecutor (type='response') instead of CreateAgentExecutor\n"
            "- Human-in-the-loop approval for tool execution\n"
            "- Multiagent coordination with supervisor and worker agents\n"
            "- Supports approve, reject, and edit decisions\n\n"
            "**Try asking:**\n"
            "- 'Analyze the sentiment of this customer review: I love this product! It works perfectly.'\n"
            "- 'Check the sentiment of this feedback: This service is terrible and needs improvement'\n"
            "- 'Analyze sentiment for this social media post and escalate if negative: I want a refund, this is the worst product ever'\n"
            "- 'Send a notification to customer support about the negative review'\n"
            "- The agents will pause and ask for your approval before executing these actions."
        ),
        stream=False
    )
    
    chat = LangGraphChat(
        workflow=workflow,
        agents=[supervisor] + workers,
        config=config
    )
    chat.run()

if __name__ == "__main__":
    main()

