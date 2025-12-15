import os

import streamlit as st
import streamlit_langgraph as slg


def create_network_workflow_example():
    """Create a network workflow with peer-to-peer agent collaboration."""
    
    config_path = os.path.join(os.path.dirname(__file__), "./configs/network.yaml")
    agents = slg.AgentManager.load_from_yaml(config_path)
    
    return agents


def main():
    """Network workflow example with collaborative peer agents."""
    
    # Create a network of peer agents
    agents = create_network_workflow_example()
    
    # Create the network workflow
    builder = slg.WorkflowBuilder()
    network_workflow = builder.create_network_workflow(agents=agents)
    
    config = slg.UIConfig(
        title="Strategic Consulting Network",
        page_icon="🕸️",
        stream=True,
        welcome_message="""Welcome to the **Strategic Consulting Network**!

This demonstrates a **true network pattern** where specialists collaborate dynamically, handing work back and forth as needed - not just passing it sequentially.

## 🕸️ The Consulting Team:

| Specialist | Focus Area | Hands Off To |
|------------|------------|--------------|
| 🔧 **Tech Strategist** | Architecture, feasibility, tech risks | Business for validation, Risk for security, Delivery for planning |
| 📊 **Business Analyst** | Requirements, ROI, stakeholders | Tech for feasibility, Risk for compliance, Delivery for costs |
| ⚠️ **Risk Strategist** | Security, compliance, mitigation | Tech for design changes, Business for impact, Delivery for timeline |
| 📋 **Delivery Lead** | Planning, resources, roadmap | Back to ANY peer when issues arise |

## 🔄 Why This Creates Back-and-Forth:

Unlike supervisory patterns, these agents have **interdependent concerns**:
- Tech decisions affect Risk → Risk findings change Tech design
- Business requirements drive Tech → Tech constraints change Business case  
- Risk mitigation affects Delivery → Delivery timelines affect Risk tolerance

---

## 🚀 Try This Complex Scenario:

> **"We're a healthcare startup that just received $10M Series A funding. We need to:**
>
> **1. HIPAA Compliance Crisis**: Our current MVP stores patient data in a standard PostgreSQL database with basic encryption. We have 6 months to become HIPAA compliant or lose our hospital partnership worth $2M ARR.
>
> **2. Scale Challenge**: We're growing from 5,000 to 50,000 users. Our monolithic Node.js app on a single EC2 instance won't handle it.
>
> **3. Team Reality**: We have 4 developers (all junior-mid level), no DevOps, no security expertise. Hiring is taking 3+ months per role.
>
> **4. Competing Priorities**: Our investors want us to launch 3 new features for enterprise sales, but our CTO says we need to 'stop and fix the foundation.'
>
> **Budget: $10M total, but investors expect 18-month runway. Help us navigate this."**

---

**Watch how agents hand work BACK to each other** when they identify issues that need re-evaluation!
""",
        placeholder="Describe a complex strategic challenge with competing constraints..."
    )
    
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            workflow=network_workflow,
            agents=agents,
            config=config
        )
    st.session_state.chat.run()


if __name__ == "__main__":
    main()
