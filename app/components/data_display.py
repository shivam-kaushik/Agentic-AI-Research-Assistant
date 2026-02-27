"""
Data Display Components for Co-Investigator App
Provides interactive components for displaying research data
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Any


def render_researcher_table(researchers: List[Dict[str, Any]]) -> None:
    """Render an interactive researcher table"""
    if not researchers:
        st.info("No researchers found")
        return
    
    # Convert to DataFrame
    df_data = []
    for r in researchers:
        df_data.append({
            "Name": r.get("display_name", r.get("name", "Unknown")),
            "H-Index": r.get("h_index", "N/A"),
            "Citations": r.get("cited_by_count", 0),
            "Institution": r.get("last_known_institution", r.get("affiliation", "Unknown"))
        })
    
    df = pd.DataFrame(df_data)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Researchers", len(df))
    with col2:
        avg_h = df["H-Index"].replace("N/A", 0).astype(float).mean()
        st.metric("Avg H-Index", f"{avg_h:.1f}")
    with col3:
        total_citations = df["Citations"].sum()
        st.metric("Total Citations", f"{total_citations:,}")
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_gene_table(gene_data: List[Dict[str, Any]]) -> None:
    """Render an interactive gene-disease association table"""
    if not gene_data:
        st.info("No gene data found")
        return
    
    # Classification color mapping
    color_map = {
        "Definitive": "🟢",
        "Strong": "🔵",
        "Moderate": "🟡",
        "Limited": "🟠",
        "Disputed": "🔴",
        "No Known Disease Relationship": "⚪"
    }
    
    # Convert to DataFrame
    df_data = []
    for g in gene_data:
        classification = g.get("Classification", "Unknown")
        df_data.append({
            "": color_map.get(classification, "⚪"),
            "Gene": g.get("Gene_Symbol", "Unknown"),
            "Disease": g.get("Disease_Label", "Unknown"),
            "Classification": classification,
            "MOI": g.get("MOI", "Unknown")
        })
    
    df = pd.DataFrame(df_data)
    
    # Display classification summary
    st.markdown("**Classification Summary:**")
    class_counts = df["Classification"].value_counts()
    cols = st.columns(len(class_counts))
    for idx, (classification, count) in enumerate(class_counts.items()):
        with cols[idx]:
            st.metric(classification, count)
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_preprint_cards(preprints: List[Dict[str, Any]]) -> None:
    """Render preprints as interactive cards"""
    if not preprints:
        st.info("No preprints found")
        return
    
    for idx, paper in enumerate(preprints[:10]):  # Show top 10
        with st.expander(f"📄 {paper.get('Title', 'Untitled')[:80]}...", expanded=(idx == 0)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Authors:** {paper.get('Authors', 'Unknown')}")
                abstract = paper.get('Abstract', 'No abstract available')
                if len(abstract) > 300:
                    abstract = abstract[:300] + "..."
                st.markdown(abstract)
            
            with col2:
                st.markdown(f"**Date:** {paper.get('Date', 'Unknown')}")
                st.markdown(f"**Source:** {paper.get('Source', 'Unknown')}")
                if paper.get('DOI'):
                    st.markdown(f"[View Paper]({paper['DOI']})")


def render_knowledge_graph_concepts(concepts: List[str]) -> None:
    """Render knowledge graph concepts as tags"""
    if not concepts:
        st.info("No concepts found")
        return
    
    st.markdown("**Key Scientific Concepts:**")
    
    # Display as columns of tags
    cols_per_row = 2
    for i in range(0, len(concepts), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(concepts):
                with col:
                    # Create a colored badge
                    concept = concepts[idx][:60]
                    st.markdown(
                        f'<div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 5px 0;">'
                        f'🔬 {concept}</div>',
                        unsafe_allow_html=True
                    )


def render_task_timeline(tasks: List[Dict[str, Any]], completed: List[str], pending: List[str]) -> None:
    """Render tasks as a timeline with status indicators"""
    st.markdown("### 📋 Research Task Timeline")
    
    for idx, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{idx}")
        description = task.get("description", "No description")
        
        # Determine status
        if task_id in completed:
            status = "✅ Completed"
            color = "#4caf50"
        elif task_id in pending:
            status = "⏳ Pending"
            color = "#ff9800"
        else:
            status = "📌 Not Started"
            color = "#9e9e9e"
        
        # Render timeline item
        st.markdown(
            f'<div style="border-left: 4px solid {color}; padding-left: 15px; margin: 10px 0;">'
            f'<div style="font-weight: bold; color: {color};">{status}</div>'
            f'<div style="font-size: 14px; margin-top: 5px;"><b>{task_id}:</b> {description}</div>'
            f'<div style="font-size: 12px; color: #666;">Source: {task.get("data_source", "Unknown")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_data_summary_metrics(collected_data: Dict[str, Any]) -> None:
    """Render summary metrics for collected data"""
    if not collected_data:
        st.info("No data collected yet")
        return
    
    total_records = sum(data.get("count", 0) for data in collected_data.values())
    total_sources = len(collected_data)
    successful = sum(1 for data in collected_data.values() if data.get("success", False))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{total_records:,}")
    
    with col2:
        st.metric("Data Sources", total_sources)
    
    with col3:
        success_rate = (successful / total_sources * 100) if total_sources > 0 else 0
        st.metric("Success Rate", f"{success_rate:.0f}%")
