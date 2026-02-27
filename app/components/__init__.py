"""Streamlit UI Components for Co-Investigator Agent."""
from .prompt_input import render_prompt_input
from .execution_graph import render_execution_graph
from .hitl_panel import render_hitl_panel
from .data_display import (
    render_researcher_table,
    render_gene_table,
    render_preprint_cards,
    render_knowledge_graph_concepts,
    render_task_timeline,
    render_data_summary_metrics
)
from .message_parser import (
    parse_agent_message,
    extract_sections,
    simplify_message_for_display
)

__all__ = [
    "render_prompt_input",
    "render_execution_graph",
    "render_hitl_panel",
    "render_researcher_table",
    "render_gene_table",
    "render_preprint_cards",
    "render_knowledge_graph_concepts",
    "render_task_timeline",
    "render_data_summary_metrics",
    "parse_agent_message",
    "extract_sections",
    "simplify_message_for_display",
]
