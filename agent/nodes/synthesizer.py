"""
Synthesizer Node for Co-Investigator Agent

Compiles all collected research findings into a structured markdown report.
"""
import json
import logging
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

import sys
sys.path.append("../..")
from config.gcp_config import config
from config.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_PROMPT
from agent.state import AgentState, ResearchPlan

logger = logging.getLogger(__name__)


def synthesizer_node(state: AgentState) -> dict:
    """
    Synthesizer node that generates the final research report.

    Args:
        state: Current agent state with all collected data

    Returns:
        Updated state with final report
    """
    logger.info("Synthesizer node generating report")

    try:
        # Initialize Vertex AI
        vertexai.init(project=config.project_id, location=config.location)

        # Create model instance
        model = GenerativeModel(
            config.synthesizer_model,
            system_instruction=SYNTHESIZER_SYSTEM_PROMPT,
        )

        # Configure generation
        generation_config = GenerationConfig(
            temperature=0.3,
            max_output_tokens=4096,
        )

        # Prepare inputs
        plan_dict = state.get("plan", {})
        plan = ResearchPlan.from_dict(plan_dict) if plan_dict else None

        execution_plan = _format_execution_plan(plan) if plan else "No plan available"
        collected_data = _format_collected_data(state.get("results", {}))
        steps_completed = _format_steps_completed(plan) if plan else "No steps recorded"

        # Generate report
        prompt = SYNTHESIZER_USER_PROMPT.format(
            research_query=state["user_query"],
            execution_plan=execution_plan,
            collected_data=collected_data,
            steps_completed=steps_completed,
            human_feedback=state.get("human_feedback", "None provided"),
        )

        response = model.generate_content(prompt, generation_config=generation_config)

        final_report = response.text

        logger.info("Report generated successfully")

        return {
            "final_report": final_report,
            "current_node": "synthesizer",
            "execution_history": state["execution_history"] + ["synthesizer"],
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Synthesizer error: {e}")
        # Generate fallback report
        fallback_report = _generate_fallback_report(state)
        return {
            "final_report": fallback_report,
            "error": f"Report generation partially failed: {str(e)}",
            "current_node": "synthesizer",
            "execution_history": state["execution_history"] + ["synthesizer"],
            "updated_at": datetime.now().isoformat(),
        }


def _format_execution_plan(plan: ResearchPlan) -> str:
    """Format execution plan for the prompt."""
    lines = [f"**Research Goal:** {plan.research_goal}", "", "**Sub-tasks:**"]

    for task in plan.sub_tasks:
        status_emoji = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌",
            "awaiting_hitl": "⏸️",
        }.get(task.status.value, "❓")

        lines.append(
            f"- {status_emoji} {task.task_id}: {task.description} "
            f"(Source: {task.data_source}, Entities: {', '.join(task.entities[:3])})"
        )

    return "\n".join(lines)


def _format_collected_data(results: dict) -> str:
    """Format collected data for the prompt."""
    if not results:
        return "No data collected"

    sections = []

    for task_id, result in results.items():
        if not result:
            continue

        source = result.get("source", "unknown")
        success = result.get("success", False)
        count = result.get("count", result.get("total_count", 0))

        section = [f"### {task_id} (Source: {source})"]

        if not success:
            section.append(f"**Error:** {result.get('error', 'Unknown error')}")
        else:
            section.append(f"**Records found:** {count}")

            # Add sample data
            data = result.get("data", [])
            if data:
                section.append("\n**Sample records:**")
                for item in data[:3]:
                    # Format based on data type
                    if "gene" in item or "gene_symbol" in item:
                        gene = item.get("gene") or item.get("gene_symbol")
                        disease = item.get("disease") or item.get("disease_name", "N/A")
                        section.append(f"- Gene: {gene}, Disease: {disease}")
                    elif "title" in item:
                        title = item.get("title", "N/A")[:100]
                        authors = item.get("authors", [])[:2]
                        section.append(f"- \"{title}\" by {', '.join(str(a) if isinstance(a, str) else a.get('name', 'Unknown') for a in authors)}")
                    elif "display_name" in item:
                        name = item.get("display_name")
                        inst = item.get("last_known_institution", "N/A")
                        section.append(f"- {name} ({inst})")
                    elif "pathway_name" in item:
                        pathway = item.get("pathway_name")
                        section.append(f"- Pathway: {pathway}")
                    else:
                        # Generic formatting
                        section.append(f"- {json.dumps(item, default=str)[:200]}")

        sections.append("\n".join(section))

    return "\n\n".join(sections)


def _format_steps_completed(plan: ResearchPlan) -> str:
    """Format completed steps for the prompt."""
    completed = []
    for task in plan.sub_tasks:
        if task.status.value == "completed":
            duration = ""
            if task.started_at and task.completed_at:
                delta = task.completed_at - task.started_at
                duration = f" ({delta.total_seconds():.1f}s)"
            completed.append(f"- {task.task_id}: {task.description}{duration}")

    return "\n".join(completed) if completed else "No steps completed"


def _generate_fallback_report(state: AgentState) -> str:
    """Generate a basic report when LLM synthesis fails."""
    results = state.get("results", {})
    plan_dict = state.get("plan", {})

    report = f"""# Research Report

## Research Question
{state['user_query']}

## Summary
This report was generated using automated data retrieval from multiple biomedical databases.

## Methodology
"""

    if plan_dict:
        plan = ResearchPlan.from_dict(plan_dict)
        report += f"Research goal: {plan.research_goal}\n\n"
        report += "Tasks executed:\n"
        for task in plan.sub_tasks:
            report += f"- {task.description} (Status: {task.status.value})\n"

    report += "\n## Findings\n\n"

    for task_id, result in results.items():
        if not result:
            continue

        source = result.get("source", "unknown")
        count = result.get("count", result.get("total_count", 0))
        success = result.get("success", False)

        report += f"### {source.replace('_', ' ').title()}\n"

        if success:
            report += f"Found {count} records.\n\n"
            data = result.get("data", [])
            if data:
                for item in data[:5]:
                    if "gene" in item or "gene_symbol" in item:
                        gene = item.get("gene") or item.get("gene_symbol")
                        disease = item.get("disease", "N/A")
                        report += f"- **{gene}**: {disease}\n"
                    elif "title" in item:
                        report += f"- {item.get('title', 'N/A')}\n"
                    elif "display_name" in item:
                        report += f"- {item.get('display_name', 'N/A')}\n"
        else:
            report += f"Error: {result.get('error', 'Unknown error')}\n"

        report += "\n"

    if state.get("human_feedback"):
        report += f"\n## Human Feedback\n{state['human_feedback']}\n"

    report += f"""
## Data Sources
- ClinGen (Gene-Disease Validity)
- CIViC (Clinical Evidence)
- Reactome (Biological Pathways)
- STRING (Protein Interactions)
- OpenAlex (Researcher Information)
- PubMed (Research Abstracts)

---
*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return report
