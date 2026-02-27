"""
Multi-Agent Streamlit Application for Co-Investigator

Features:
- Interactive UI with enhanced data display
- Real-time task state tracking
- Structured data visualization
- Conversational interface with memory
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import logging
import re
import os
from typing import Dict, Any

from agent.multi_agent import MultiAgentOrchestrator
from app.components.data_display import (
    render_researcher_table,
    render_gene_table,
    render_preprint_cards,
    render_knowledge_graph_concepts,
    render_task_timeline,
    render_data_summary_metrics
)
from app.components.message_parser import (
    parse_agent_message,
    extract_sections,
    simplify_message_for_display
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Co-Investigator | Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Agent badges */
    .agent-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        margin-right: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .agent-planner { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .agent-researcher { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .agent-validator { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .agent-synthesizer { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }
    .agent-orchestrator { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }

    /* Task status badges */
    .task-completed {
        color: #4caf50;
        font-weight: bold;
    }
    .task-pending {
        color: #ff9800;
        font-weight: bold;
    }
    .task-not-started {
        color: #9e9e9e;
        font-weight: bold;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: bold;
        margin: 20px 0 10px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Data tables */
    .dataframe {
        font-size: 14px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
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


def render_message_with_images(content: str, parsed_data: Dict[str, Any]) -> None:
    """
    Render message content with interactive components
    """
    # Render graphs prominently if available
    if parsed_data.get('graphs'):
        st.markdown("### 📊 Generated Visualizations")
        cols = st.columns(len(parsed_data['graphs']))
        workspace_root = Path(__file__).parent.parent
        
        for idx, graph in enumerate(parsed_data['graphs']):
            with cols[idx]:
                img_path = graph['path']
                if not img_path.startswith('http'):
                    full_path = workspace_root / img_path
                    if full_path.exists():
                        try:
                            st.image(str(full_path), caption=graph['alt_text'], use_container_width=True)
                        except Exception as e:
                            logger.warning(f"Failed to load image: {e}")
                            st.warning(f"⚠️ {graph['alt_text']}")
    
    # Render metrics if available
    if parsed_data.get('metrics'):
        metrics = parsed_data['metrics']
        
        if metrics.get('step') and metrics.get('total_steps'):
            progress = metrics['step'] / metrics['total_steps']
            st.progress(progress)
            st.caption(f"Step {metrics['step']} of {metrics['total_steps']}: {metrics.get('step_description', '')}")
        
        # Display result metrics
        metric_cols = []
        if 'researchers_count' in metrics:
            metric_cols.append(('👤 Researchers', metrics['researchers_count']))
        if 'genes_count' in metrics:
            metric_cols.append(('🧬 Genes', metrics['genes_count']))
        if 'preprints_count' in metrics:
            metric_cols.append(('📄 Preprints', metrics['preprints_count']))
        if 'knowledge_connections' in metrics:
            metric_cols.append(('🔗 Concepts', metrics['knowledge_connections']))
        
        if metric_cols:
            cols = st.columns(len(metric_cols))
            for idx, (label, value) in enumerate(metric_cols):
                with cols[idx]:
                    st.metric(label, value)
    
    # Render structured data in tabs
    tabs_to_create = []
    if parsed_data.get('researchers'):
        tabs_to_create.append("👤 Researchers")
    if parsed_data.get('genes'):
        tabs_to_create.append("🧬 Genes")
    if parsed_data.get('preprints'):
        tabs_to_create.append("📄 Preprints")
    if parsed_data.get('concepts'):
        tabs_to_create.append("🔬 Concepts")
    
    # Always include raw view
    tabs_to_create.append("📝 Details")
    
    if len(tabs_to_create) > 1:
        tabs = st.tabs(tabs_to_create)
        
        tab_idx = 0
        
        if parsed_data.get('researchers'):
            with tabs[tab_idx]:
                render_researcher_table(parsed_data['researchers'])
            tab_idx += 1
        
        if parsed_data.get('genes'):
            with tabs[tab_idx]:
                render_gene_table(parsed_data['genes'])
            tab_idx += 1
        
        if parsed_data.get('preprints'):
            with tabs[tab_idx]:
                render_preprint_cards(parsed_data['preprints'])
            tab_idx += 1
        
        if parsed_data.get('concepts'):
            with tabs[tab_idx]:
                render_knowledge_graph_concepts(parsed_data['concepts'])
            tab_idx += 1
        
        # Details tab
        with tabs[tab_idx]:
            simplified = simplify_message_for_display(content)
            
            # Remove images from text (already shown above)
            simplified = re.sub(r'!\[([^\]]*)\]\(([^\)]+)\)', '', simplified)
            
            # Show in expanders by section
            sections = extract_sections(simplified)
            if sections:
                for section_name, section_content in sections:
                    if section_content.strip():
                        with st.expander(section_name, expanded=False):
                            st.markdown(section_content)
            else:
                st.markdown(simplified)
    else:
        # No structured data, just show simplified text
        simplified = simplify_message_for_display(content)
        simplified = re.sub(r'!\[([^\]]*)\]\(([^\)]+)\)', '', simplified)
        st.markdown(simplified)


def render_message_with_images_simple(content: str) -> None:
    """
    Simple version for backwards compatibility
    Parses markdown to find image references and displays them using st.image().
    """
    # Pattern to match markdown images: ![alt text](path)
    img_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    
    # Split content by image tags
    parts = re.split(img_pattern, content)
    
    # Process each part
    i = 0
    while i < len(parts):
        # Regular text
        if i % 3 == 0 and parts[i].strip():
            st.markdown(parts[i])
        # Image found (alt text is parts[i], path is parts[i+1])
        elif i % 3 == 1 and i + 1 < len(parts):
            alt_text = parts[i]
            img_path = parts[i + 1]

            
            # Convert relative path to absolute if needed
            if not img_path.startswith('http'):
                # Get workspace root
                workspace_root = Path(__file__).parent.parent
                full_path = workspace_root / img_path
                
                # Check if file exists
                if full_path.exists():
                    try:
                        st.image(str(full_path), caption=alt_text, use_container_width=True)
                    except Exception as e:
                        logger.warning(f"Failed to load image {full_path}: {e}")
                        st.warning(f"⚠️ Image not available: {alt_text}")
                else:
                    st.warning(f"⚠️ Image file not found: {img_path}")
            else:
                # Remote image
                st.image(img_path, caption=alt_text, use_container_width=True)
            
            i += 1  # Skip the path part
        
        i += 1


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
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🔬 Co-Investigator")
        st.caption("Multi-Agent Biomedical Research Assistant")
    with col2:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.orchestrator = MultiAgentOrchestrator()
            st.session_state.messages = []
            st.session_state.current_plan = None
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Session Status")
        
        memory = st.session_state.orchestrator.memory
        
        # Session info in metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(memory.messages))
        with col2:
            st.metric("Tasks", f"{len(memory.completed_tasks)}/{len(memory.completed_tasks) + len(memory.pending_tasks)}")
        
        st.caption(f"Session: `{st.session_state.orchestrator.session_id[:12]}...`")
        
        st.markdown("---")
        
        # Real-time task tracking
        if st.session_state.current_plan:
            plan = st.session_state.current_plan
            st.markdown("### 📋 Active Tasks")
            
            tasks = plan.get("sub_tasks", [])
            completed = memory.completed_tasks
            pending = memory.pending_tasks
            
            for task in tasks:
                task_id = task.get("task_id", "")
                description = task.get("description", "")[:40]
                
                if task_id in completed:
                    st.markdown(f"✅ **{task_id}**")
                    st.caption(description)
                elif task_id in pending:
                    st.markdown(f"⏳ **{task_id}**")
                    st.caption(description)
                else:
                    st.markdown(f"⚪ **{task_id}**")
                    st.caption(description)
                
                st.markdown("")
        
        else:
            st.info("No active research plan")
        
        st.markdown("---")
        
        # Data summary
        if memory.collected_data:
            st.markdown("### 📊 Data Collected")
            render_data_summary_metrics(memory.collected_data)
            
            # Quick view of sources
            with st.expander("View Sources"):
                for task_id, data in memory.collected_data.items():
                    source = data.get("data_source", "Unknown")
                    count = data.get("count", 0)
                    st.markdown(f"**{task_id}** ({source}): {count} records")
        
        st.markdown("---")
        
        # Available agents
        with st.expander("🤖 Available Agents"):
            st.markdown("""
            - 📋 **Planner**: Creates research plans
            - 🔍 **Researcher**: Queries databases
            - ⚠️ **Validator**: Checks data quality
            - 📝 **Synthesizer**: Creates reports
            """)
        
        # Data sources
        with st.expander("💾 Data Sources"):
            st.markdown("""
            - ClinGen (Gene-Disease)
            - bioRxiv/medRxiv (Preprints)
            - ORKG (Knowledge Graph)
            - OpenAlex (Researchers)
            - PubMedQA (Literature Q&A)
            """)

    # Main chat interface
    st.markdown("## 💬 Research Conversation")
    
    # Display chat history with enhanced rendering
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            agent = msg.get("agent", "ORCHESTRATOR")
            with st.chat_message("assistant", avatar="🤖"):
                # Show agent badge
                agent_badge = get_agent_badge(agent)
                st.markdown(agent_badge, unsafe_allow_html=True)
                
                # Parse and render message
                parsed_data = parse_agent_message(msg["content"])
                render_message_with_images(msg["content"], parsed_data)
    
    # Chat input
    user_input = st.chat_input("Ask a research question or give instructions...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Process with multi-agent system - show detailed progress
        status_container = st.empty()
        status_messages = []
        
        def update_status(message: str):
            """Callback to update status in real-time"""
            status_messages.append(message)
            # Display all status messages
            with status_container.container():
                with st.status("🤖 Processing your request...", expanded=True) as status:
                    for msg in status_messages:
                        st.write(msg)
        
        # Call orchestrator with status callback
        response = st.session_state.orchestrator.process_message(user_input, status_callback=update_status)
        
        # Clear status and mark complete
        with status_container.container():
            with st.status("✅ Response ready!", expanded=False, state="complete") as status:
                st.write(f"Processed {len(status_messages)} steps")
        
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
    
    # Quick action buttons at bottom
    if st.session_state.current_plan:
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Continue", use_container_width=True, type="primary"):
                _execute_quick_action("yes")
        
        with col2:
            if st.button("📝 Generate Report", use_container_width=True):
                _execute_quick_action("Generate a comprehensive research report")
        
        with col3:
            if st.button("❓ Explain Plan", use_container_width=True):
                _execute_quick_action("Explain the current research plan and what each step will do")
        
        with col4:
            if st.button("⏭️ Skip Step", use_container_width=True):
                _execute_quick_action("Skip this step and proceed to the next one")


def _execute_quick_action(message: str):
    """Execute a quick action button"""
    st.session_state.messages.append({
        "role": "user",
        "content": message
    })
    
    response = st.session_state.orchestrator.process_message(message)
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.get("message", ""),
        "agent": response.get("agent_used", "ORCHESTRATOR")
    })
    
    if response.get("plan"):
        st.session_state.current_plan = response["plan"]
    
    st.rerun()


if __name__ == "__main__":
    main()
