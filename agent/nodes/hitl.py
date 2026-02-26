"""
Human-in-the-Loop (HITL) Node for Co-Investigator Agent

Pauses execution for human review and decision-making.
Persists state to Firestore and waits for user feedback.
"""
import uuid
import logging
from datetime import datetime

from google.cloud import firestore

import sys
sys.path.append("../..")
from config.gcp_config import config
from agent.state import AgentState, HITLCheckpoint, ConflictInfo

logger = logging.getLogger(__name__)


def hitl_node(state: AgentState) -> dict:
    """
    HITL node that pauses execution for human review.

    Creates a checkpoint, persists state to Firestore, and sets
    hitl_pending flag to pause the graph execution.

    Args:
        state: Current agent state

    Returns:
        Updated state with HITL checkpoint
    """
    logger.info("HITL node: Creating checkpoint for human review")

    # Generate checkpoint ID
    checkpoint_id = f"hitl_{uuid.uuid4().hex[:8]}"

    # Determine reason for HITL
    conflicts = state.get("conflicts", [])
    if conflicts:
        reason = f"Found {len(conflicts)} potential issue(s) requiring review"
        options = [
            "Proceed with current results",
            "Skip conflicting data and continue",
            "Modify search parameters",
            "Abort and start over",
        ]
    else:
        reason = "Checkpoint reached as per execution plan"
        options = [
            "Continue to next phase",
            "Review results before continuing",
            "Modify the execution plan",
            "Abort",
        ]

    # Create checkpoint
    checkpoint = HITLCheckpoint(
        checkpoint_id=checkpoint_id,
        reason=reason,
        conflicts=[ConflictInfo(**c) if isinstance(c, dict) else c for c in conflicts],
        options=options,
    )

    # Persist to Firestore
    try:
        _persist_checkpoint_to_firestore(state, checkpoint)
        logger.info(f"Checkpoint {checkpoint_id} persisted to Firestore")
    except Exception as e:
        logger.error(f"Failed to persist checkpoint: {e}")

    return {
        "hitl_checkpoint": checkpoint.to_dict(),
        "hitl_pending": True,
        "current_node": "hitl",
        "execution_history": state["execution_history"] + ["hitl"],
        "updated_at": datetime.now().isoformat(),
    }


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


def resume_from_hitl(state: AgentState, user_feedback: str) -> dict:
    """
    Resume execution after user provides feedback.

    Args:
        state: Current agent state
        user_feedback: User's response at the checkpoint

    Returns:
        Updated state with feedback incorporated
    """
    logger.info(f"Resuming from HITL with feedback: {user_feedback}")

    checkpoint = state.get("hitl_checkpoint", {})
    if checkpoint:
        checkpoint["user_response"] = user_feedback
        checkpoint["responded_at"] = datetime.now().isoformat()

    return {
        "hitl_checkpoint": checkpoint,
        "hitl_pending": False,
        "human_feedback": user_feedback,
        "current_node": "hitl_resume",
        "execution_history": state["execution_history"] + ["hitl_resume"],
        "updated_at": datetime.now().isoformat(),
    }


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
