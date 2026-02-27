"""
Human-in-the-Loop (HITL) Node for Co-Investigator Agent

Pauses execution for human review and decision-making.
Enhanced with SMART Adaptive Checkpoint (QueryQuest v9.0):
- Gemini generates context-aware options based on current results
- Options can modify the search plan for the next step
- Not generic "yes/skip/stop" but real choices like:
  [1] Full literature search for all genes
  [2] Narrow to definitive genes only
  [3] Skip to researcher identification
"""
import json
import uuid
import logging
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import firestore

import sys
sys.path.append("../..")
from config.gcp_config import config
from agent.state import AgentState, HITLCheckpoint, ConflictInfo, SmartOption
from tools import safe_len

logger = logging.getLogger(__name__)


def hitl_node(state: AgentState) -> dict:
    """
    HITL node that pauses execution for human review.

    Enhanced with SMART Adaptive Checkpoint:
    - Generates context-aware options using Gemini
    - Options are specific to the current research context
    - Can modify the search plan based on user choice

    Args:
        state: Current agent state

    Returns:
        Updated state with HITL checkpoint and smart options
    """
    logger.info("HITL node: Creating SMART adaptive checkpoint")

    # Generate checkpoint ID
    checkpoint_id = f"hitl_{uuid.uuid4().hex[:8]}"

    # Build context summary for display
    context_summary = _build_context_summary(state)

    # Generate SMART options using Gemini
    smart_options = _generate_smart_options(state, context_summary)

    # Determine reason for HITL
    conflicts = state.get("conflicts", [])
    if conflicts:
        reason = f"Found {len(conflicts)} potential issue(s) requiring review"
    else:
        reason = "Research checkpoint: Review findings and choose next action"

    # Legacy options (fallback)
    legacy_options = [opt.label for opt in smart_options]

    # Create checkpoint with smart options
    checkpoint = HITLCheckpoint(
        checkpoint_id=checkpoint_id,
        reason=reason,
        conflicts=[ConflictInfo(**c) if isinstance(c, dict) else c for c in conflicts],
        options=legacy_options,
        smart_options=smart_options,
        context_summary=context_summary,
    )

    # Persist to Firestore
    try:
        _persist_checkpoint_to_firestore(state, checkpoint)
        logger.info(f"SMART checkpoint {checkpoint_id} persisted to Firestore")
    except Exception as e:
        logger.error(f"Failed to persist checkpoint: {e}")

    return {
        "hitl_checkpoint": checkpoint.to_dict(),
        "hitl_pending": True,
        "current_node": "hitl",
        "execution_history": state["execution_history"] + ["hitl"],
        "updated_at": datetime.now().isoformat(),
    }


def _build_context_summary(state: AgentState) -> dict:
    """Build a summary of current research context for display."""
    plan = state.get("plan", {})
    results = state.get("results", {})

    # Count results by category
    clingen_count = 0
    definitive_genes = []
    if state.get("clingen_results"):
        cr = state["clingen_results"]
        if isinstance(cr, dict):
            clingen_count = cr.get("total", len(cr.get("all_results", [])))
            definitive_genes = [r.get("Gene_Symbol", "") for r in cr.get("definitive", [])][:5]

    pubmedqa_count = 0
    if state.get("pubmedqa_results"):
        pr = state["pubmedqa_results"]
        if isinstance(pr, dict):
            pubmedqa_count = pr.get("total", len(pr.get("results", [])))

    biorxiv_count = 0
    if state.get("biorxiv_results"):
        br = state["biorxiv_results"]
        if isinstance(br, dict):
            biorxiv_count = br.get("total", len(br.get("results", [])))

    # Get pending tasks
    pending_tasks = []
    for task in plan.get("sub_tasks", []):
        if task.get("status") == "pending":
            pending_tasks.append(task.get("description", "Unknown task"))

    return {
        "query": state.get("user_query", ""),
        "disease_variants": state.get("disease_variants", []),
        "gene_variants": state.get("gene_variants", []),
        "clingen_count": clingen_count,
        "definitive_genes": definitive_genes,
        "pubmedqa_count": pubmedqa_count,
        "biorxiv_count": biorxiv_count,
        "pending_tasks": pending_tasks,
        "completed_tasks": state.get("current_task_index", 0),
    }


def _generate_smart_options(state: AgentState, context: dict) -> list[SmartOption]:
    """
    Generate context-aware HITL options using Gemini.

    NOT generic options like "yes/skip/stop".
    Real choices based on what we've found so far.
    """
    try:
        vertexai.init(project=config.project_id, location=config.location)
        model = GenerativeModel("gemini-2.5-pro")

        # Build the prompt
        prompt = f"""
        You are helping a scientist review research progress.

        Research Query: {context.get('query', 'Unknown')}
        Disease Focus: {', '.join(context.get('disease_variants', []))}
        Genes Found: {', '.join(context.get('gene_variants', []))}

        Results So Far:
        - ClinGen: {context.get('clingen_count', 0)} gene-disease links
        - Definitive Genes: {', '.join(context.get('definitive_genes', [])) or 'none'}
        - PubMedQA: {context.get('pubmedqa_count', 0)} Q&A pairs
        - bioRxiv: {context.get('biorxiv_count', 0)} preprints

        Pending Tasks:
        {chr(10).join(['- ' + t for t in context.get('pending_tasks', [])])}

        Generate 4-5 SPECIFIC actionable options for the scientist.
        NOT generic like "yes/skip/stop".

        Each option should have:
        - label: Short text (max 50 chars)
        - action: Action identifier (e.g., "full_search", "narrow_definitive", "skip_literature")
        - impact: What will happen (1 sentence)
        - modifies_plan: true/false

        Example for gene research checkpoint:
        [
            {{"label": "Full literature search (all {context.get('clingen_count', 0)} genes)", "action": "full_search", "impact": "Search PubMedQA and bioRxiv for all identified genes", "modifies_plan": false}},
            {{"label": "Narrow to definitive genes only", "action": "narrow_definitive", "impact": "Focus search on {len(context.get('definitive_genes', []))} definitive genes for higher confidence", "modifies_plan": true}},
            {{"label": "Skip literature, find researchers", "action": "skip_to_researchers", "impact": "Jump to OpenAlex researcher identification", "modifies_plan": true}},
            {{"label": "Export current findings", "action": "stop_and_export", "impact": "Generate report with current results", "modifies_plan": true}}
        ]

        Return ONLY valid JSON array. No markdown, no explanation.
        """

        response = model.generate_content(prompt)
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        options_data = json.loads(text)

        smart_options = []
        for opt in options_data[:5]:  # Max 5 options
            smart_options.append(SmartOption(
                label=opt.get("label", "Continue")[:50],
                action=opt.get("action", "continue"),
                impact=opt.get("impact", "Proceed to next step"),
                modifies_plan=opt.get("modifies_plan", False),
            ))

        if not smart_options:
            raise ValueError("No options generated")

        return smart_options

    except Exception as e:
        logger.error(f"Failed to generate smart options: {e}")
        # Return fallback options
        return [
            SmartOption(
                label="Continue with full search",
                action="continue",
                impact="Search all remaining datasets",
                modifies_plan=False,
            ),
            SmartOption(
                label="Skip to researcher identification",
                action="skip_to_researchers",
                impact="Jump to finding active researchers",
                modifies_plan=True,
            ),
            SmartOption(
                label="Export current findings",
                action="stop_and_export",
                impact="Generate report with results so far",
                modifies_plan=True,
            ),
            SmartOption(
                label="Abort research",
                action="abort",
                impact="Cancel and start over",
                modifies_plan=True,
            ),
        ]


def should_pause_for_hitl(state: AgentState) -> str:
    """
    Conditional edge function to determine if HITL is needed.

    Args:
        state: Current agent state

    Returns:
        Next node name: "hitl" if HITL needed, "external_api_caller" otherwise
    """
    # Check if HITL is already pending (resuming from checkpoint)
    if state.get("hitl_pending") and state.get("human_feedback"):
        # Human has provided feedback, continue execution
        return "external_api_caller"

    # Check if HITL is already pending (waiting for feedback)
    if state.get("hitl_pending"):
        return "hitl"

    # Check if HITL is required
    if state.get("requires_hitl"):
        return "hitl"

    # No HITL needed, continue to next node
    return "external_api_caller"


def resume_from_hitl(state: AgentState, user_feedback: str, selected_action: str = None) -> dict:
    """
    Resume execution after user provides feedback.

    Enhanced to handle SMART adaptive options that modify the plan.

    Args:
        state: Current agent state
        user_feedback: User's response text
        selected_action: The action identifier from smart options

    Returns:
        Updated state with feedback incorporated and plan potentially modified
    """
    logger.info(f"Resuming from HITL with action: {selected_action}, feedback: {user_feedback}")

    checkpoint = state.get("hitl_checkpoint", {})
    if checkpoint:
        checkpoint["user_response"] = user_feedback
        checkpoint["selected_action"] = selected_action
        checkpoint["responded_at"] = datetime.now().isoformat()

    # Apply the selected action
    plan_updates = _apply_smart_action(state, selected_action)

    return {
        "hitl_checkpoint": checkpoint,
        "hitl_pending": False,
        "human_feedback": user_feedback,
        "current_node": "hitl_resume",
        "execution_history": state["execution_history"] + ["hitl_resume"],
        "updated_at": datetime.now().isoformat(),
        **plan_updates,
    }


def _apply_smart_action(state: AgentState, action: str) -> dict:
    """
    Apply the user's selected action, potentially modifying the plan.

    Returns dict of state updates.
    """
    if not action:
        return {}

    plan = state.get("plan", {})
    updates = {}

    if action == "narrow_definitive":
        # Narrow search to definitive genes only
        clingen_results = state.get("clingen_results", {})
        if isinstance(clingen_results, dict):
            definitive_genes = [
                r.get("Gene_Symbol", "")
                for r in clingen_results.get("definitive", [])
            ]
            if definitive_genes:
                updates["gene_variants"] = definitive_genes
                logger.info(f"Narrowed to {len(definitive_genes)} definitive genes")

    elif action == "skip_to_researchers":
        # Skip remaining literature tasks, go to researcher identification
        sub_tasks = plan.get("sub_tasks", [])
        for task in sub_tasks:
            if task.get("data_source") in ["pubmedqa", "biorxiv", "pubmed"]:
                task["status"] = "skipped"
        updates["plan"] = plan
        logger.info("Skipped literature tasks, proceeding to researchers")

    elif action == "stop_and_export":
        # Mark all pending tasks as skipped, proceed to synthesis
        sub_tasks = plan.get("sub_tasks", [])
        for task in sub_tasks:
            if task.get("status") == "pending":
                task["status"] = "skipped"
        updates["plan"] = plan
        updates["requires_hitl"] = False
        logger.info("Stopping research, proceeding to synthesis")

    elif action == "abort":
        # Set error to trigger abort
        updates["error"] = "Research aborted by user"
        logger.info("Research aborted by user")

    elif action == "full_search" or action == "continue":
        # Continue as planned
        logger.info("Continuing with full search")

    return updates


def _persist_checkpoint_to_firestore(state: AgentState, checkpoint: HITLCheckpoint):
    """
    Persist the current state and checkpoint to Firestore.

    Args:
        state: Current agent state
        checkpoint: HITL checkpoint data
    """
    db = firestore.Client(project=config.project_id)

    # Persist to hitl_checkpoints collection
    doc_ref = db.collection(config.firestore_collection_hitl).document(
        checkpoint.checkpoint_id
    )

    doc_data = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "session_id": state["session_id"],
        "reason": checkpoint.reason,
        "conflicts": [c.to_dict() if hasattr(c, 'to_dict') else c for c in checkpoint.conflicts],
        "options": checkpoint.options,
        "user_response": None,
        "created_at": datetime.now(),
        "responded_at": None,
        "state_snapshot": {
            "user_query": state["user_query"],
            "plan": state.get("plan"),
            "results": state.get("results"),
            "current_task_index": state.get("current_task_index"),
        },
    }

    doc_ref.set(doc_data)

    # Also update the session document
    session_ref = db.collection(config.firestore_collection_sessions).document(
        state["session_id"]
    )

    session_ref.set(
        {
            "session_id": state["session_id"],
            "user_query": state["user_query"],
            "current_checkpoint": checkpoint.checkpoint_id,
            "status": "awaiting_hitl",
            "updated_at": datetime.now(),
        },
        merge=True,
    )


def load_checkpoint_from_firestore(checkpoint_id: str) -> dict | None:
    """
    Load a checkpoint from Firestore.

    Args:
        checkpoint_id: The checkpoint ID to load

    Returns:
        Checkpoint data or None if not found
    """
    db = firestore.Client(project=config.project_id)

    doc_ref = db.collection(config.firestore_collection_hitl).document(checkpoint_id)
    doc = doc_ref.get()

    if doc.exists:
        return doc.to_dict()

    return None


def update_checkpoint_response(checkpoint_id: str, user_response: str):
    """
    Update a checkpoint with user's response.

    Args:
        checkpoint_id: The checkpoint ID
        user_response: User's feedback/decision
    """
    db = firestore.Client(project=config.project_id)

    doc_ref = db.collection(config.firestore_collection_hitl).document(checkpoint_id)

    doc_ref.update(
        {
            "user_response": user_response,
            "responded_at": datetime.now(),
        }
    )
