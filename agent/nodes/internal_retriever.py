"""
Internal Retriever Node for Co-Investigator Agent

Executes BigQuery queries against internal datasets (ClinGen, CIViC, Reactome, STRING).
"""
import logging
from datetime import datetime

import sys
sys.path.append("../..")
from agent.state import AgentState, SubTask, TaskStatus, ResearchPlan
from tools.query_bigquery import execute_bigquery_tool

logger = logging.getLogger(__name__)

# Data sources handled by this node
INTERNAL_DATA_SOURCES = {"clingen", "civic", "reactome", "string"}


def internal_retriever_node(state: AgentState) -> dict:
    """
    Internal retriever node that executes BigQuery queries.

    Processes all pending tasks that use internal data sources.

    Args:
        state: Current agent state

    Returns:
        Updated state with query results
    """
    logger.info("Internal retriever node executing")

    plan_dict = state.get("plan")
    if not plan_dict:
        logger.error("No plan found in state")
        return {
            "error": "No execution plan found",
            "current_node": "internal_retriever",
            "execution_history": state["execution_history"] + ["internal_retriever"],
            "updated_at": datetime.now().isoformat(),
        }

    plan = ResearchPlan.from_dict(plan_dict)
    current_results = state.get("results", {})
    updated_tasks = []
    new_results = {}

    for task in plan.sub_tasks:
        # Skip tasks that are not for internal sources
        if task.data_source not in INTERNAL_DATA_SOURCES:
            updated_tasks.append(task)
            continue

        # Skip completed or failed tasks
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            updated_tasks.append(task)
            continue

        # Check dependencies
        if not _dependencies_met(task, plan.sub_tasks):
            logger.info(f"Task {task.task_id} waiting on dependencies")
            updated_tasks.append(task)
            continue

        # Execute the task
        logger.info(f"Executing task {task.task_id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            result = execute_bigquery_tool(
                data_source=task.data_source,
                query_type=task.query_type,
                entities=task.entities,
            )

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Store result with task_id as key
            new_results[task.task_id] = result

            logger.info(
                f"Task {task.task_id} completed: {result.get('total_count', 0)} results"
            )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

        updated_tasks.append(task)

    # Update plan with task statuses
    plan.sub_tasks = updated_tasks
    updated_plan = plan.to_dict()

    # Merge results
    merged_results = {**current_results, **new_results}

    # Check if we need HITL checkpoint
    requires_hitl = _should_trigger_hitl(plan, updated_tasks)

    return {
        "plan": updated_plan,
        "results": merged_results,
        "requires_hitl": requires_hitl,
        "current_node": "internal_retriever",
        "execution_history": state["execution_history"] + ["internal_retriever"],
        "updated_at": datetime.now().isoformat(),
    }


def _dependencies_met(task: SubTask, all_tasks: list[SubTask]) -> bool:
    """Check if all dependencies for a task are completed."""
    if not task.depends_on:
        return True

    task_status_map = {t.task_id: t.status for t in all_tasks}

    for dep_id in task.depends_on:
        if dep_id not in task_status_map:
            continue
        if task_status_map[dep_id] != TaskStatus.COMPLETED:
            return False

    return True


def _should_trigger_hitl(plan: ResearchPlan, tasks: list[SubTask]) -> bool:
    """Determine if HITL checkpoint should be triggered."""
    if not plan.hitl_checkpoint_after:
        return False

    # Find the checkpoint task
    for task in tasks:
        if task.task_id == plan.hitl_checkpoint_after:
            return task.status == TaskStatus.COMPLETED

    return False


def get_internal_results_summary(results: dict) -> str:
    """
    Generate a summary of internal retrieval results.

    Args:
        results: Dictionary of task results

    Returns:
        Formatted summary string
    """
    summary_parts = []

    for task_id, result in results.items():
        if not result:
            continue

        source = result.get("source", "unknown")
        count = result.get("total_count", result.get("count", 0))
        success = result.get("success", False)

        if success:
            summary_parts.append(f"- {task_id} ({source}): {count} results")
        else:
            error = result.get("error", "Unknown error")
            summary_parts.append(f"- {task_id} ({source}): Failed - {error}")

    return "\n".join(summary_parts) if summary_parts else "No results collected"
