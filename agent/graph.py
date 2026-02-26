"""
LangGraph State Machine for Co-Investigator Agent

Defines the execution graph with cyclical execution model,
dynamic fact-checking, and plan-correction loops.
"""
import uuid
import logging
from datetime import datetime
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState, create_initial_state
from agent.nodes.planner import planner_node
from agent.nodes.internal_retriever import internal_retriever_node
from agent.nodes.conflict_detector import conflict_detector_node
from agent.nodes.hitl import hitl_node, should_pause_for_hitl, resume_from_hitl
from agent.nodes.external_api_caller import external_api_caller_node
from agent.nodes.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def route_after_conflict_detector(state: AgentState) -> Literal["hitl", "external_api_caller"]:
    """
    Conditional routing after conflict detection.

    Determines whether to pause for HITL or continue to external APIs.
    """
    return should_pause_for_hitl(state)


def route_after_hitl(state: AgentState) -> Literal["external_api_caller", "hitl", "end"]:
    """
    Conditional routing after HITL node.

    If still pending (no feedback), stay at HITL.
    If feedback provided, check the decision.
    """
    if state.get("hitl_pending") and not state.get("human_feedback"):
        # Still waiting for feedback - this shouldn't happen in normal flow
        # as the graph pauses at HITL
        return "hitl"

    feedback = state.get("human_feedback", "").lower()

    if "abort" in feedback or "cancel" in feedback:
        return "end"

    return "external_api_caller"


def check_for_errors(state: AgentState) -> Literal["synthesizer", "end"]:
    """
    Check if there are critical errors that should end execution.
    """
    if state.get("error") and "critical" in state.get("error", "").lower():
        return "end"
    return "synthesizer"


def create_graph() -> StateGraph:
    """
    Create the LangGraph state machine for the Co-Investigator agent.

    Graph Structure:
    ┌─────────────┐
    │   START     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Planner   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────┐
    │ Internal Retriever  │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Conflict Detector   │
    └──────────┬──────────┘
               │
        ┌──────┴──────┐
        │  requires   │
        │   HITL?     │
        └──────┬──────┘
         yes/  │  \no
            ▼      ▼
    ┌───────────┐   │
    │   HITL    │   │
    └─────┬─────┘   │
          │         │
          ▼         │
    ┌─────────────────────┐
    │ External API Caller │◄─┘
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │    Synthesizer      │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────┐
    │     END     │
    └─────────────┘

    Returns:
        Compiled StateGraph
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("internal_retriever", internal_retriever_node)
    workflow.add_node("conflict_detector", conflict_detector_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("external_api_caller", external_api_caller_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Add edges
    workflow.add_edge("planner", "internal_retriever")
    workflow.add_edge("internal_retriever", "conflict_detector")

    # Conditional edge after conflict detector
    workflow.add_conditional_edges(
        "conflict_detector",
        route_after_conflict_detector,
        {
            "hitl": "hitl",
            "external_api_caller": "external_api_caller",
        },
    )

    # Conditional edge after HITL
    workflow.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "external_api_caller": "external_api_caller",
            "hitl": "hitl",
            "end": END,
        },
    )

    # Continue to synthesizer
    workflow.add_edge("external_api_caller", "synthesizer")

    # End after synthesis
    workflow.add_edge("synthesizer", END)

    return workflow


def compile_graph(checkpointer=None):
    """
    Compile the graph with optional checkpointing.

    Args:
        checkpointer: LangGraph checkpointer for state persistence

    Returns:
        Compiled graph ready for execution
    """
    workflow = create_graph()

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


def run_agent(
    user_query: str,
    session_id: str | None = None,
    use_checkpointing: bool = True,
) -> dict:
    """
    Run the agent on a user query.

    Args:
        user_query: The research question from the user
        session_id: Optional session ID (generated if not provided)
        use_checkpointing: Whether to use memory checkpointing

    Returns:
        Final agent state with results
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:12]}"

    logger.info(f"Starting agent session {session_id} for query: {user_query}")

    # Create initial state
    initial_state = create_initial_state(session_id, user_query)

    # Compile graph
    checkpointer = MemorySaver() if use_checkpointing else None
    graph = compile_graph(checkpointer)

    # Configuration for the run
    config = {"configurable": {"thread_id": session_id}}

    # Run the graph
    try:
        # Stream execution for visibility
        final_state = None
        for event in graph.stream(initial_state, config):
            node_name = list(event.keys())[0]
            node_output = event[node_name]

            logger.info(f"Executed node: {node_name}")

            # Check for HITL pause
            if node_output.get("hitl_pending") and not node_output.get("human_feedback"):
                logger.info("Agent paused for HITL - awaiting user feedback")
                final_state = node_output
                break

            final_state = node_output

        return {
            "session_id": session_id,
            "status": "paused" if final_state.get("hitl_pending") else "completed",
            "state": final_state,
        }

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "error": str(e),
            "state": None,
        }


def resume_agent(
    session_id: str,
    user_feedback: str,
    current_state: dict,
) -> dict:
    """
    Resume agent execution after HITL feedback.

    Args:
        session_id: The session ID to resume
        user_feedback: User's response at the checkpoint
        current_state: The current agent state

    Returns:
        Final agent state after resumption
    """
    logger.info(f"Resuming session {session_id} with feedback: {user_feedback}")

    # Update state with user feedback
    updated_state = resume_from_hitl(current_state, user_feedback)

    # Merge with current state
    for key, value in updated_state.items():
        current_state[key] = value

    # Compile graph with fresh checkpointer
    checkpointer = MemorySaver()
    graph = compile_graph(checkpointer)

    config = {"configurable": {"thread_id": session_id}}

    try:
        # Continue from external_api_caller since HITL is complete
        final_state = None

        # Run remaining nodes
        for event in graph.stream(current_state, config):
            node_name = list(event.keys())[0]
            node_output = event[node_name]

            logger.info(f"Executed node: {node_name}")
            final_state = node_output

        return {
            "session_id": session_id,
            "status": "completed",
            "state": final_state,
        }

    except Exception as e:
        logger.error(f"Agent resumption failed: {e}")
        return {
            "session_id": session_id,
            "status": "error",
            "error": str(e),
            "state": current_state,
        }


# For direct testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    query = sys.argv[1] if len(sys.argv) > 1 else "Find experts in IPF progression"

    print(f"Running agent with query: {query}")
    result = run_agent(query)

    print(f"\nSession ID: {result['session_id']}")
    print(f"Status: {result['status']}")

    if result.get("state"):
        state = result["state"]
        if state.get("final_report"):
            print("\n" + "=" * 50)
            print("FINAL REPORT")
            print("=" * 50)
            print(state["final_report"])
        elif state.get("hitl_pending"):
            checkpoint = state.get("hitl_checkpoint", {})
            print("\n" + "=" * 50)
            print("HITL CHECKPOINT")
            print("=" * 50)
            print(f"Reason: {checkpoint.get('reason')}")
            print(f"Options: {checkpoint.get('options')}")
