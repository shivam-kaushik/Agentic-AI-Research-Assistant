"""
Conflict Detector Node for Co-Investigator Agent

Analyzes collected data for conflicts, contradictions, and quality issues.
Uses Gemini to identify potential problems that may require human review.
"""
import json
import logging
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

import sys
sys.path.append("../..")
from config.gcp_config import config
from config.prompts import CONFLICT_DETECTOR_PROMPT
from agent.state import AgentState, ConflictInfo

logger = logging.getLogger(__name__)


def conflict_detector_node(state: AgentState) -> dict:
    """
    Conflict detector node that analyzes results for issues.

    Args:
        state: Current agent state with collected results

    Returns:
        Updated state with conflict analysis
    """
    logger.info("Conflict detector node analyzing results")

    results = state.get("results", {})

    if not results:
        logger.info("No results to analyze")
        return {
            "conflicts": [],
            "requires_hitl": False,
            "current_node": "conflict_detector",
            "execution_history": state["execution_history"] + ["conflict_detector"],
            "updated_at": datetime.now().isoformat(),
        }

    try:
        # Initialize Vertex AI
        vertexai.init(project=config.project_id, location=config.location)

        # Create model instance
        model = GenerativeModel(config.planner_model)

        # Configure for JSON output
        generation_config = GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        # Prepare data summary for analysis
        data_summary = _prepare_data_summary(results)

        # Generate conflict analysis
        prompt = CONFLICT_DETECTOR_PROMPT.format(data=data_summary)
        response = model.generate_content(prompt, generation_config=generation_config)

        # Parse response
        analysis = json.loads(response.text)

        conflicts = [
            ConflictInfo(
                conflict_type=c.get("type", "unknown"),
                description=c.get("description", ""),
                affected_entities=c.get("affected_entities", []),
                recommendation=c.get("recommendation", ""),
            ).to_dict()
            for c in analysis.get("conflicts", [])
        ]

        requires_hitl = analysis.get("requires_human_review", False)
        review_reason = analysis.get("review_reason", "")

        if requires_hitl:
            logger.info(f"HITL required: {review_reason}")

        return {
            "conflicts": conflicts,
            "requires_hitl": requires_hitl or state.get("requires_hitl", False),
            "current_node": "conflict_detector",
            "execution_history": state["execution_history"] + ["conflict_detector"],
            "updated_at": datetime.now().isoformat(),
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse conflict analysis: {e}")
        # Use rule-based fallback
        conflicts, requires_hitl = _rule_based_conflict_detection(results)
        return {
            "conflicts": conflicts,
            "requires_hitl": requires_hitl or state.get("requires_hitl", False),
            "current_node": "conflict_detector",
            "execution_history": state["execution_history"] + ["conflict_detector"],
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Conflict detector error: {e}")
        return {
            "conflicts": [],
            "requires_hitl": state.get("requires_hitl", False),
            "error": f"Conflict detection failed: {str(e)}",
            "current_node": "conflict_detector",
            "execution_history": state["execution_history"] + ["conflict_detector"],
            "updated_at": datetime.now().isoformat(),
        }


def _prepare_data_summary(results: dict) -> str:
    """Prepare a summary of results for conflict analysis."""
    summary_parts = []

    for task_id, result in results.items():
        if not result or not result.get("success"):
            continue

        source = result.get("source", "unknown")
        data = result.get("data", [])

        # Limit data for context window
        sample_data = data[:5] if len(data) > 5 else data

        summary_parts.append(f"""
### {task_id} (Source: {source})
Total records: {len(data)}
Sample data:
```json
{json.dumps(sample_data, indent=2, default=str)[:2000]}
```
""")

    return "\n".join(summary_parts)


def _rule_based_conflict_detection(results: dict) -> tuple[list[dict], bool]:
    """
    Fallback rule-based conflict detection.

    Checks for common issues without using LLM.
    """
    conflicts = []
    requires_hitl = False

    # Check for empty results
    for task_id, result in results.items():
        if not result:
            continue

        if result.get("success") and result.get("total_count", result.get("count", 0)) == 0:
            conflicts.append(
                ConflictInfo(
                    conflict_type="missing",
                    description=f"No data found for {task_id}",
                    affected_entities=result.get("entities", []),
                    recommendation="Consider broadening search criteria or using alternative data sources",
                ).to_dict()
            )

        # Check for errors
        if not result.get("success"):
            conflicts.append(
                ConflictInfo(
                    conflict_type="quality",
                    description=f"Query failed for {task_id}: {result.get('error', 'Unknown error')}",
                    affected_entities=[],
                    recommendation="Review query parameters and retry",
                ).to_dict()
            )
            requires_hitl = True

    # Check for contradictory evidence levels (CIViC specific)
    civic_results = [r for r in results.values() if r and r.get("source") == "civic_evidence"]
    if civic_results:
        for result in civic_results:
            data = result.get("data", [])
            # Check if same gene has conflicting evidence
            gene_evidence = {}
            for record in data:
                gene = record.get("gene")
                if gene:
                    if gene not in gene_evidence:
                        gene_evidence[gene] = set()
                    evidence_level = record.get("evidence_level")
                    if evidence_level:
                        gene_evidence[gene].add(evidence_level)

            for gene, levels in gene_evidence.items():
                if len(levels) > 2:
                    conflicts.append(
                        ConflictInfo(
                            conflict_type="contradiction",
                            description=f"Gene {gene} has multiple evidence levels: {', '.join(levels)}",
                            affected_entities=[gene],
                            recommendation="Review individual evidence items for context",
                        ).to_dict()
                    )

    return conflicts, requires_hitl
