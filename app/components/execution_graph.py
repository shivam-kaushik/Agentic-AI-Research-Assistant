"""
Execution Graph Visualization Component

Displays the current state of the agent's execution pipeline.
"""
import streamlit as st


# Node definitions with display info
NODES = [
    {"id": "planner", "name": "Planner", "icon": "📋", "description": "Decomposing query into sub-tasks"},
    {"id": "internal_retriever", "name": "Internal Retriever", "icon": "🔍", "description": "Querying BigQuery datasets"},
    {"id": "conflict_detector", "name": "Conflict Detector", "icon": "⚠️", "description": "Checking for data conflicts"},
    {"id": "hitl", "name": "Human Review", "icon": "👤", "description": "Awaiting user confirmation"},
    {"id": "external_api_caller", "name": "External APIs", "icon": "🌐", "description": "Fetching from OpenAlex/PubMed"},
    {"id": "synthesizer", "name": "Synthesizer", "icon": "📝", "description": "Generating final report"},
]


def render_execution_graph(current_node: str | None, execution_history: list[str] | None):
    """
    Render the execution graph visualization.

    Args:
        current_node: Currently executing node ID
        execution_history: List of completed node IDs
    """
    st.markdown("### Execution Progress")

    history = execution_history or []

    # Create a horizontal progress view
    cols = st.columns(len(NODES))

    for i, (col, node) in enumerate(zip(cols, NODES)):
        with col:
            node_id = node["id"]

            # Determine node status
            if node_id == current_node:
                status = "current"
                border_color = "#1f77b4"  # Blue
                bg_color = "#e6f2ff"
            elif node_id in history:
                status = "completed"
                border_color = "#2ca02c"  # Green
                bg_color = "#e6ffe6"
            else:
                status = "pending"
                border_color = "#d3d3d3"  # Gray
                bg_color = "#f5f5f5"

            # Status indicator
            if status == "current":
                indicator = "🔄"
            elif status == "completed":
                indicator = "✅"
            else:
                indicator = "⏳"

            # Render node card
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {border_color};
                    border-radius: 10px;
                    padding: 10px;
                    text-align: center;
                    background-color: {bg_color};
                    min-height: 100px;
                ">
                    <div style="font-size: 24px;">{node['icon']}</div>
                    <div style="font-size: 12px; font-weight: bold; margin-top: 5px;">
                        {node['name']}
                    </div>
                    <div style="font-size: 20px; margin-top: 5px;">{indicator}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Show current action description
    if current_node:
        current_node_info = next((n for n in NODES if n["id"] == current_node), None)
        if current_node_info:
            st.info(f"**Current Action:** {current_node_info['description']}")


def render_task_list(plan: dict | None):
    """
    Render the task list from the execution plan.

    Args:
        plan: The execution plan dictionary
    """
    if not plan:
        return

    st.markdown("### Task Breakdown")

    tasks = plan.get("sub_tasks", [])

    for task in tasks:
        task_id = task.get("task_id", "Unknown")
        description = task.get("description", "No description")
        status = task.get("status", "pending")
        data_source = task.get("data_source", "unknown")
        entities = task.get("entities", [])

        # Status styling
        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌",
            "awaiting_hitl": "⏸️",
        }

        status_colors = {
            "pending": "#808080",
            "in_progress": "#1f77b4",
            "completed": "#2ca02c",
            "failed": "#d62728",
            "awaiting_hitl": "#ff7f0e",
        }

        icon = status_icons.get(status, "❓")
        color = status_colors.get(status, "#808080")

        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"""
                    <div style="
                        border-left: 4px solid {color};
                        padding-left: 10px;
                        margin-bottom: 10px;
                    ">
                        <strong>{icon} {task_id}</strong>: {description}<br/>
                        <small style="color: #666;">
                            Source: {data_source} | Entities: {', '.join(entities[:3])}
                            {'...' if len(entities) > 3 else ''}
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.caption(status.replace("_", " ").title())


def render_results_summary(results: dict | None):
    """
    Render a summary of collected results.

    Args:
        results: Dictionary of task results
    """
    if not results:
        return

    st.markdown("### Data Collected")

    total_records = 0
    sources_queried = []

    for task_id, result in results.items():
        if not result:
            continue

        source = result.get("source", "unknown")
        count = result.get("count", result.get("total_count", 0))
        success = result.get("success", False)

        sources_queried.append(source)
        if success:
            total_records += count

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", total_records)

    with col2:
        st.metric("Sources Queried", len(sources_queried))

    with col3:
        success_rate = sum(1 for r in results.values() if r and r.get("success")) / len(results) * 100 if results else 0
        st.metric("Success Rate", f"{success_rate:.0f}%")

    # Detailed breakdown
    with st.expander("View Details", expanded=False):
        for task_id, result in results.items():
            if not result:
                continue

            source = result.get("source", "unknown")
            count = result.get("count", result.get("total_count", 0))
            success = result.get("success", False)

            if success:
                st.success(f"**{task_id}** ({source}): {count} records")
            else:
                st.error(f"**{task_id}** ({source}): Failed - {result.get('error', 'Unknown')}")
