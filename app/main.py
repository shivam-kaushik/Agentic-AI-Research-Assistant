"""
Multi-Agent Streamlit Application for Co-Investigator

Features:
- Conversational interface with memory
- Multiple specialized agents
- Dynamic task modification
- Real-time agent visibility
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import logging

from agent.multi_agent import MultiAgentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Co-Investigator | Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .agent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 8px;
    }
    .agent-planner { background-color: #e3f2fd; color: #1565c0; }
    .agent-researcher { background-color: #e8f5e9; color: #2e7d32; }
    .agent-validator { background-color: #fff3e0; color: #ef6c00; }
    .agent-synthesizer { background-color: #f3e5f5; color: #7b1fa2; }
    .agent-orchestrator { background-color: #fce4ec; color: #c2185b; }

    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        color: #000000;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        color: #000000;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)


def get_agent_badge(agent_name: str) -> str:
    """Get HTML badge for agent type."""
    badges = {
        "PLANNER": ("📋", "agent-planner"),
        "RESEARCHER": ("🔍", "agent-researcher"),
        "VALIDATOR": ("⚠️", "agent-validator"),
        "SYNTHESIZER": ("📝", "agent-synthesizer"),
        "CLARIFIER": ("❓", "agent-orchestrator"),
        "ORCHESTRATOR": ("🎯", "agent-orchestrator"),
    }
    icon, css_class = badges.get(agent_name, ("🤖", "agent-orchestrator"))
    return f'<span class="agent-badge {css_class}">{icon} {agent_name}</span>'


def init_session():
    """Initialize session state."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = MultiAgentOrchestrator()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = None


def main():
    init_session()

    # Header
    st.title("🔬 Co-Investigator")
    st.markdown("*Multi-Agent Biomedical Research Assistant*")

    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Agent Status")

        # Show current session info
        st.markdown(f"**Session:** `{st.session_state.orchestrator.session_id[:12]}...`")

        # Show memory status
        memory = st.session_state.orchestrator.memory
        st.markdown(f"**Messages:** {len(memory.messages)}")
        st.markdown(f"**Completed Tasks:** {len(memory.completed_tasks)}")
        st.markdown(f"**Pending Tasks:** {len(memory.pending_tasks)}")

        st.markdown("---")

        st.markdown("### 📊 Available Agents")
        st.markdown("""
        - 📋 **Planner**: Creates research plans
        - 🔍 **Researcher**: Queries databases
        - ⚠️ **Validator**: Checks data quality
        - 📝 **Synthesizer**: Creates reports
        """)

        st.markdown("---")

        st.markdown("### 💾 Data Sources")
        st.markdown("""
        - ClinGen (Gene-Disease)
        - CIViC (Clinical Evidence)
        - Reactome (Pathways)
        - STRING (Protein Interactions)
        - OpenAlex (Researchers)
        - PubMed (Abstracts)
        """)

        st.markdown("---")

        # Reset button
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.orchestrator = MultiAgentOrchestrator()
            st.session_state.messages = []
            st.session_state.current_plan = None
            st.rerun()

    # Main chat interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💬 Research Chat")

        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                agent_badge = get_agent_badge(msg.get("agent", "ORCHESTRATOR"))
                st.markdown(
                    f'<div class="chat-message assistant-message">{agent_badge}<br><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

        # Chat input
        user_input = st.chat_input("Ask a research question or give instructions...")

        if user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            # Process with multi-agent system
            with st.status("🤖 Orchestrating agents...", expanded=True) as status_container:
                def update_status(msg):
                    status_container.write(f"🔄 {msg}")
                    status_container.update(label=f"🤖 {msg}")
                response = st.session_state.orchestrator.process_message(
                    user_input, 
                    status_callback=update_status
                )
                status_container.update(label="✅ Agent processing complete!", state="complete", expanded=False)

            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("message", ""),
                "agent": response.get("agent_used", "ORCHESTRATOR")
            })

            # Update plan if available
            if response.get("plan"):
                st.session_state.current_plan = response["plan"]

            st.rerun()

    with col2:
        # Show current plan
        st.markdown("### 📋 Current Plan")

        if st.session_state.current_plan:
            plan = st.session_state.current_plan

            st.markdown(f"**Goal:** {plan.get('research_goal', 'N/A')}")
            st.markdown(f"**Complexity:** {plan.get('estimated_complexity', 'N/A')}")

            st.markdown("#### Tasks:")
            for task in plan.get("sub_tasks", []):
                task_id = task["task_id"]
                if task_id in memory.completed_tasks:
                    icon = "✅"
                elif task_id in memory.pending_tasks:
                    icon = "⏳"
                else:
                    icon = "📌"

                with st.expander(f"{icon} {task_id}: {task['description'][:30]}..."):
                    st.markdown(f"**Source:** {task['data_source']}")
                    st.markdown(f"**Params:** {task.get('query_params', {})}")

                    if task.get("depends_on"):
                        st.markdown(f"**Depends on:** {task['depends_on']}")
        else:
            st.info("No plan yet. Ask a research question to get started!")

        # Show collected data summary
        st.markdown("### 📊 Collected Data")

        if memory.collected_data:
            for task_id, data in memory.collected_data.items():
                with st.expander(f"📁 {task_id} ({data.get('count', 0)} records)"):
                    st.json(data.get("data", [])[:3])
        else:
            st.info("No data collected yet.")

    # Quick action buttons
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📋 Create Plan", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Please create a research plan based on our conversation"
            })
            with st.status("🤖 Orchestrating agents...", expanded=True) as status_container:
                def update_status(msg):
                    status_container.write(f"🔄 {msg}")
                    status_container.update(label=f"🤖 {msg}")
                response = st.session_state.orchestrator.process_message(
                    "Please create a research plan based on our conversation",
                    status_callback=update_status
                )
                status_container.update(label="✅ Agent processing complete!", state="complete", expanded=False)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("message", ""),
                "agent": response.get("agent_used")
            })
            if response.get("plan"):
                st.session_state.current_plan = response["plan"]
            st.rerun()

    with col2:
        if st.button("🔍 Execute Research", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Execute the pending research tasks"
            })
            with st.status("🤖 Orchestrating agents...", expanded=True) as status_container:
                def update_status(msg):
                    status_container.write(f"🔄 {msg}")
                    status_container.update(label=f"🤖 {msg}")
                response = st.session_state.orchestrator.process_message(
                    "Execute the pending research tasks",
                    status_callback=update_status
                )
                status_container.update(label="✅ Agent processing complete!", state="complete", expanded=False)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("message", ""),
                "agent": response.get("agent_used")
            })
            st.rerun()

    with col3:
        if st.button("⚠️ Validate Data", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Validate the collected data for any issues"
            })
            with st.status("🤖 Orchestrating agents...", expanded=True) as status_container:
                def update_status(msg):
                    status_container.write(f"🔄 {msg}")
                    status_container.update(label=f"🤖 {msg}")
                response = st.session_state.orchestrator.process_message(
                    "Validate the collected data for any issues",
                    status_callback=update_status
                )
                status_container.update(label="✅ Agent processing complete!", state="complete", expanded=False)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("message", ""),
                "agent": response.get("agent_used")
            })
            st.rerun()

    with col4:
        if st.button("📝 Generate Report", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Generate a final research report"
            })
            with st.status("🤖 Orchestrating agents...", expanded=True) as status_container:
                def update_status(msg):
                    status_container.write(f"🔄 {msg}")
                    status_container.update(label=f"🤖 {msg}")
                response = st.session_state.orchestrator.process_message(
                    "Generate a final research report",
                    status_callback=update_status
                )
                status_container.update(label="✅ Agent processing complete!", state="complete", expanded=False)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("message", ""),
                "agent": response.get("agent_used")
            })
            st.rerun()


if __name__ == "__main__":
    main()
