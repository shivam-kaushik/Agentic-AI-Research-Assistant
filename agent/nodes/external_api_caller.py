"""
External API Caller Node for Co-Investigator Agent

Executes queries against external APIs (OpenAlex, PubMed) to enrich
the research findings with live researcher and publication data.
"""
import logging
from datetime import datetime

import sys
sys.path.append("../..")
from agent.state import AgentState, SubTask, TaskStatus, ResearchPlan
from tools.search_openalex import search_researchers, search_works
from tools.pubmed_entrez import search_and_fetch_pubmed

logger = logging.getLogger(__name__)

# Data sources handled by this node
EXTERNAL_DATA_SOURCES = {"openalex", "pubmed"}


def external_api_caller_node(state: AgentState) -> dict:
    """
    External API caller node that queries OpenAlex and PubMed.

    Args:
        state: Current agent state

    Returns:
        Updated state with external API results
    """
    logger.info("External API caller node executing")

    plan_dict = state.get("plan")
    if not plan_dict:
        logger.error("No plan found in state")
        return {
            "error": "No execution plan found",
            "current_node": "external_api_caller",
            "execution_history": state["execution_history"] + ["external_api_caller"],
            "updated_at": datetime.now().isoformat(),
        }

    plan = ResearchPlan.from_dict(plan_dict)
    current_results = state.get("results", {})
    updated_tasks = []
    new_results = {}

    # Get entities discovered from internal sources for enrichment
    discovered_entities = _extract_discovered_entities(current_results)

    for task in plan.sub_tasks:
        # Skip tasks that are not for external sources
        if task.data_source not in EXTERNAL_DATA_SOURCES:
            updated_tasks.append(task)
            continue

        # Skip completed or failed tasks
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            updated_tasks.append(task)
            continue

        # Execute the task
        logger.info(f"Executing external task {task.task_id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            # Merge task entities with discovered entities
            search_entities = list(set(task.entities + discovered_entities.get(task.query_type, [])))

            result = _execute_external_query(
                data_source=task.data_source,
                query_type=task.query_type,
                entities=search_entities,
            )

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            new_results[task.task_id] = result

            logger.info(
                f"External task {task.task_id} completed: {result.get('count', 0)} results"
            )

        except Exception as e:
            logger.error(f"External task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

        updated_tasks.append(task)

    # Update plan
    plan.sub_tasks = updated_tasks
    updated_plan = plan.to_dict()

    # Merge results
    merged_results = {**current_results, **new_results}

    return {
        "plan": updated_plan,
        "results": merged_results,
        "current_node": "external_api_caller",
        "execution_history": state["execution_history"] + ["external_api_caller"],
        "updated_at": datetime.now().isoformat(),
    }


def _execute_external_query(
    data_source: str,
    query_type: str,
    entities: list[str],
) -> dict:
    """
    Execute query against external APIs.

    Args:
        data_source: openalex or pubmed
        query_type: researcher, abstract, works
        entities: List of entities to search

    Returns:
        Combined results from API queries
    """
    all_results = []
    total_count = 0

    for entity in entities[:5]:  # Limit to prevent rate limiting
        if data_source == "openalex":
            if query_type == "researcher":
                result = search_researchers(query=entity, concept=entity)
            else:
                result = search_works(query=entity, concept=entity, from_year=2022)

        elif data_source == "pubmed":
            result = search_and_fetch_pubmed(query=entity, max_results=10)

        else:
            continue

        if result.get("success"):
            all_results.extend(result.get("data", []))
            total_count += result.get("count", 0)

    # Deduplicate results
    seen_ids = set()
    unique_results = []
    for item in all_results:
        item_id = item.get("id") or item.get("pmid") or item.get("doi")
        if item_id and item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_results.append(item)

    return {
        "success": True,
        "source": f"{data_source}_{query_type}",
        "entities": entities,
        "count": len(unique_results),
        "total_found": total_count,
        "data": unique_results,
    }


def _extract_discovered_entities(results: dict) -> dict[str, list[str]]:
    """
    Extract entities discovered from internal sources.

    Maps query types to discovered entities for enrichment.

    Args:
        results: Dictionary of internal query results

    Returns:
        Dictionary mapping query types to entity lists
    """
    discovered = {
        "researcher": [],
        "gene": [],
        "disease": [],
        "pathway": [],
    }

    for task_id, result in results.items():
        if not result or not result.get("success"):
            continue

        data = result.get("data", [])
        source = result.get("source", "")

        for record in data[:20]:  # Limit processing
            # Extract genes
            gene = record.get("gene") or record.get("gene_symbol")
            if gene and gene not in discovered["gene"]:
                discovered["gene"].append(gene)

            # Extract diseases
            disease = record.get("disease") or record.get("disease_name")
            if disease and disease not in discovered["disease"]:
                discovered["disease"].append(disease)

            # Extract pathways
            pathway = record.get("pathway_name") or record.get("pathway")
            if pathway and pathway not in discovered["pathway"]:
                discovered["pathway"].append(pathway)

    # Build researcher search terms from genes and diseases
    discovered["researcher"] = discovered["gene"][:5] + discovered["disease"][:3]

    return discovered
