"""
Main Streamlit Application for Co-Investigator Agent

A State-Aware Research Assistant with Human-in-the-Loop capabilities.
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.components.prompt_input import render_prompt_input
from app.components.execution_graph import (
    render_execution_graph,
    render_task_list,
    render_results_summary,
)
from app.components.hitl_panel import render_hitl_panel, render_conflict_summary
from agent.graph import run_agent, resume_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Co-Investigator | Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton button {
        border-radius: 20px;
    }
    .report-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "execution_status" not in st.session_state:
        st.session_state.execution_status = "idle"
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def handle_hitl_response(response: str):
    """Handle user response at HITL checkpoint."""
    st.session_state.hitl_response = response
    st.session_state.execution_status = "resuming"


def main():
    """Main application entry point."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">🔬 Co-Investigator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">State-Aware AI Research Assistant with Human-in-the-Loop</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This AI assistant helps you conduct biomedical research by:

        - 📋 **Planning** - Breaking down complex queries
        - 🔍 **Retrieving** - Querying ClinGen, CIViC, Reactome
        - ⚠️ **Validating** - Detecting data conflicts
        - 👤 **Collaborating** - Seeking your input at key points
        - 🌐 **Enriching** - Adding researcher data from OpenAlex
        - 📝 **Synthesizing** - Generating comprehensive reports
        """)

        st.markdown("---")

        st.markdown("### Session Info")
        if st.session_state.session_id:
            st.code(st.session_state.session_id)
            st.caption(f"Status: {st.session_state.execution_status}")

        st.markdown("---")

        st.markdown("### Data Sources")
        st.markdown("""
        - ClinGen (Gene-Disease)
        - CIViC (Clinical Evidence)
        - Reactome (Pathways)
        - STRING (Protein Interactions)
        - OpenAlex (Researchers)
        - PubMed (Abstracts)
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input (only show if not executing)
        if st.session_state.execution_status == "idle":
            query = render_prompt_input()

            if query:
                # Start execution
                st.session_state.query_history.append(query)
                st.session_state.execution_status = "running"

                with st.spinner("Starting research agent..."):
                    result = run_agent(query)

                    st.session_state.session_id = result["session_id"]
                    st.session_state.agent_state = result.get("state", {})

                    if result["status"] == "paused":
                        st.session_state.execution_status = "hitl_pending"
                    elif result["status"] == "completed":
                        st.session_state.execution_status = "completed"
                    else:
                        st.session_state.execution_status = "error"
                        st.error(f"Error: {result.get('error', 'Unknown error')}")

                st.rerun()

        # Show execution progress
        elif st.session_state.execution_status in ["running", "hitl_pending", "completed"]:
            state = st.session_state.agent_state or {}

            # Execution graph
            render_execution_graph(
                current_node=state.get("current_node"),
                execution_history=state.get("execution_history", []),
            )

            # Task list
            render_task_list(state.get("plan"))

            # Results summary
            if state.get("results"):
                render_results_summary(state.get("results"))

            # HITL panel
            if st.session_state.execution_status == "hitl_pending":
                checkpoint = state.get("hitl_checkpoint", {})
                if checkpoint:
                    render_hitl_panel(checkpoint, handle_hitl_response)

            # Handle HITL response
            if st.session_state.execution_status == "resuming":
                response = st.session_state.get("hitl_response", "")

                with st.spinner("Resuming research..."):
                    result = resume_agent(
                        session_id=st.session_state.session_id,
                        user_feedback=response,
                        current_state=st.session_state.agent_state,
                    )

                    st.session_state.agent_state = result.get("state", {})

                    if result["status"] == "completed":
                        st.session_state.execution_status = "completed"
                    else:
                        st.session_state.execution_status = "error"

                st.rerun()

            # Final report
            if st.session_state.execution_status == "completed":
                final_report = state.get("final_report")
                if final_report:
                    st.markdown("---")
                    st.markdown("## 📄 Research Report")
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)
                    st.markdown(final_report)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Download button
                    st.download_button(
                        label="Download Report (Markdown)",
                        data=final_report,
                        file_name="research_report.md",
                        mime="text/markdown",
                    )

                # New query button
                if st.button("Start New Research", type="primary"):
                    st.session_state.agent_state = None
                    st.session_state.session_id = None
                    st.session_state.execution_status = "idle"
                    st.rerun()

    with col2:
        # Right panel - conflicts and details
        if st.session_state.agent_state:
            state = st.session_state.agent_state

            st.markdown("### Conflict Analysis")
            render_conflict_summary(state.get("conflicts", []))

            if state.get("human_feedback"):
                st.markdown("### Your Feedback")
                st.info(state.get("human_feedback"))

            # Debug info (collapsible)
            with st.expander("Debug Info", expanded=False):
                st.json({
                    "session_id": st.session_state.session_id,
                    "status": st.session_state.execution_status,
                    "current_node": state.get("current_node"),
                    "execution_history": state.get("execution_history"),
                    "error": state.get("error"),
                })


if __name__ == "__main__":
    main()
