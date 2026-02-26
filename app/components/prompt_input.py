"""
Prompt Input Component for Co-Investigator Agent

Provides a natural language input interface for research queries.
"""
import streamlit as st


def render_prompt_input() -> str | None:
    """
    Render the research query input component.

    Returns:
        User's research query or None if not submitted
    """
    st.markdown("### Research Query")

    # Example queries for guidance
    with st.expander("Example queries", expanded=False):
        st.markdown("""
        - "Find experts in IPF progression and identify key therapeutic targets"
        - "What genes are associated with lung cancer and what clinical evidence exists?"
        - "Identify researchers working on TGF-beta pathway in fibrosis"
        - "Find pathogenic variants in BRCA1 and associated clinical trials"
        - "What are the protein interactions for TP53 in cancer?"
        """)

    # Query input
    query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., Find experts in IPF progression and identify key therapeutic targets",
        height=100,
        key="research_query_input",
    )

    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.checkbox(
                "Enable detailed logging",
                value=False,
                key="enable_logging",
            )

        with col2:
            st.checkbox(
                "Auto-approve HITL checkpoints",
                value=False,
                key="auto_approve_hitl",
                help="Skip human-in-the-loop confirmations (not recommended)",
            )

    # Submit button
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        submitted = st.button(
            "Start Research",
            type="primary",
            use_container_width=True,
        )

    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if submitted and query.strip():
        return query.strip()

    return None


def render_query_history():
    """Render previous query history from session state."""
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    if st.session_state.query_history:
        st.markdown("### Recent Queries")
        for i, past_query in enumerate(st.session_state.query_history[-5:]):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(past_query[:80] + "..." if len(past_query) > 80 else past_query)
                with col2:
                    if st.button("Reuse", key=f"reuse_{i}"):
                        st.session_state.research_query_input = past_query
                        st.rerun()
