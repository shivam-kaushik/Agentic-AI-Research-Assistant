"""
Human-in-the-Loop Panel Component

Provides interactive UI for HITL checkpoints where user decisions are required.
"""
import streamlit as st
from typing import Callable


def render_hitl_panel(
    checkpoint: dict,
    on_submit: Callable[[str], None],
) -> str | None:
    """
    Render the HITL checkpoint panel.

    Args:
        checkpoint: HITL checkpoint data
        on_submit: Callback function when user submits response

    Returns:
        User's response or None if not yet submitted
    """
    st.markdown("---")
    st.markdown("## 🛑 Human Review Required")

    checkpoint_id = checkpoint.get("checkpoint_id", "Unknown")
    reason = checkpoint.get("reason", "No reason provided")
    conflicts = checkpoint.get("conflicts", [])
    options = checkpoint.get("options", [])

    # Display reason
    st.warning(f"**Reason:** {reason}")

    # Display conflicts if any
    if conflicts:
        st.markdown("### Detected Issues")

        for i, conflict in enumerate(conflicts):
            conflict_type = conflict.get("type", "unknown")
            description = conflict.get("description", "No description")
            affected = conflict.get("affected_entities", [])
            recommendation = conflict.get("recommendation", "")

            type_icons = {
                "contradiction": "⚔️",
                "outdated": "📅",
                "low_confidence": "❓",
                "missing": "🔍",
                "quality": "⚠️",
            }

            icon = type_icons.get(conflict_type, "❗")

            with st.expander(f"{icon} Issue {i + 1}: {conflict_type.title()}", expanded=True):
                st.markdown(f"**Description:** {description}")

                if affected:
                    st.markdown(f"**Affected Entities:** {', '.join(affected)}")

                if recommendation:
                    st.info(f"**Recommendation:** {recommendation}")

    # Decision options
    st.markdown("### Your Decision")

    # Radio buttons for predefined options
    selected_option = st.radio(
        "Choose an action:",
        options=options,
        key=f"hitl_option_{checkpoint_id}",
    )

    # Custom input option
    custom_input = st.text_area(
        "Or provide custom instructions:",
        placeholder="Enter specific instructions or modifications...",
        key=f"hitl_custom_{checkpoint_id}",
    )

    # Submit button
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Submit Decision", type="primary", key=f"hitl_submit_{checkpoint_id}"):
            response = custom_input.strip() if custom_input.strip() else selected_option

            if response:
                on_submit(response)
                return response

    with col2:
        st.caption("Your decision will guide the next steps of the research process.")

    return None


def render_hitl_history(checkpoints: list[dict]):
    """
    Render history of HITL checkpoints and decisions.

    Args:
        checkpoints: List of past HITL checkpoints
    """
    if not checkpoints:
        return

    st.markdown("### Decision History")

    for checkpoint in checkpoints:
        checkpoint_id = checkpoint.get("checkpoint_id", "Unknown")
        reason = checkpoint.get("reason", "No reason")
        response = checkpoint.get("user_response", "No response")
        responded_at = checkpoint.get("responded_at", "Unknown time")

        with st.expander(f"Checkpoint: {checkpoint_id}", expanded=False):
            st.markdown(f"**Reason:** {reason}")
            st.markdown(f"**Your Decision:** {response}")
            st.caption(f"Responded at: {responded_at}")


def render_conflict_summary(conflicts: list[dict]):
    """
    Render a compact summary of conflicts.

    Args:
        conflicts: List of conflict dictionaries
    """
    if not conflicts:
        st.success("No conflicts detected in the collected data.")
        return

    st.warning(f"Found {len(conflicts)} potential issue(s)")

    # Group by type
    by_type = {}
    for conflict in conflicts:
        c_type = conflict.get("type", "unknown")
        if c_type not in by_type:
            by_type[c_type] = []
        by_type[c_type].append(conflict)

    # Display grouped
    cols = st.columns(len(by_type))
    for col, (c_type, items) in zip(cols, by_type.items()):
        with col:
            type_icons = {
                "contradiction": "⚔️",
                "outdated": "📅",
                "low_confidence": "❓",
                "missing": "🔍",
                "quality": "⚠️",
            }
            icon = type_icons.get(c_type, "❗")
            st.metric(f"{icon} {c_type.title()}", len(items))
