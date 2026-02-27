"""
Synthesizer Node for Co-Investigator Agent

Compiles all collected research findings into a structured markdown report.
Enhanced with QueryQuest v9.0 features:
- Export to markdown file automatically
- Comprehensive research brief format
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

import sys
sys.path.append("../..")
from config.gcp_config import config
from config.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_PROMPT
from agent.state import AgentState, ResearchPlan

logger = logging.getLogger(__name__)

# Output directory for exported reports
OUTPUT_DIR = Path("outputs")


# Enhanced synthesizer prompt for QueryQuest-style reports
QUERYQUEST_SYNTHESIZER_PROMPT = """You are a biomedical research analyst writing a comprehensive research brief.

Write a well-structured markdown research report that includes:

1. **Executive Summary** (3-4 sentences)
   - Key findings at a glance
   - Most important genes/pathways identified
   - Research significance

2. **Research Methodology**
   - Datasets queried (ClinGen, PubMedQA, bioRxiv/medRxiv, ORKG, OpenAlex)
   - Search terms used
   - Filtering criteria

3. **Gene-Disease Findings** (from ClinGen)
   - Definitive associations (list genes)
   - Strong associations
   - Classification breakdown

4. **Literature Insights** (from PubMedQA, bioRxiv)
   - Key Q&A findings
   - Recent preprint themes
   - Emerging research directions

5. **Key Researchers** (from OpenAlex)
   - Top researchers by H-index
   - Institutional affiliations
   - Collaboration opportunities

6. **Knowledge Graph Connections** (from ORKG)
   - Relevant concepts and papers
   - Cross-references

7. **Recommendations**
   - Suggested next steps
   - Potential collaboration targets
   - Research gaps identified

8. **Data Sources & Citations**

Be specific. Use actual data values. Cite sources."""


def synthesizer_node(state: AgentState) -> dict:
    """
    Synthesizer node that generates the final research report.

    Enhanced with QueryQuest v9.0 features:
    - Comprehensive research brief format
    - Automatic export to markdown file

    Args:
        state: Current agent state with all collected data

    Returns:
        Updated state with final report and export path
    """
    logger.info("Synthesizer node generating report")

    try:
        # Initialize Vertex AI
        vertexai.init(project=config.project_id, location=config.location)

        # Create model instance with QueryQuest-style prompt
        model = GenerativeModel(
            config.synthesizer_model,
            system_instruction=QUERYQUEST_SYNTHESIZER_PROMPT,
        )

        # Configure generation
        generation_config = GenerationConfig(
            temperature=0.3,
            max_output_tokens=4096,
        )

        # Format the collected data for the prompt
        collected_data = _format_queryquest_data(state)

        # Build the prompt
        prompt = f"""Generate a comprehensive research brief for this query:

## Research Query
{state['user_query']}

## Disease Focus
{', '.join(state.get('disease_variants', [])) or 'Not specified'}

## Genes Identified
{', '.join(state.get('gene_variants', [])) or 'None specified'}

## Topic Keywords
{', '.join(state.get('topic_keywords', [])) or 'None specified'}

## Collected Data

{collected_data}

## Human Feedback
{state.get('human_feedback', 'None provided')}

Generate the research brief now:"""

        response = model.generate_content(prompt, generation_config=generation_config)
        final_report = response.text

        # Export to markdown file
        export_path = _export_to_file(final_report, state["session_id"])

        logger.info(f"Report generated and exported to {export_path}")

        return {
            "final_report": final_report,
            "export_path": export_path,
            "current_node": "synthesizer",
            "execution_history": state["execution_history"] + ["synthesizer"],
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Synthesizer error: {e}")
        # Generate fallback report
        fallback_report = _generate_fallback_report(state)
        export_path = _export_to_file(fallback_report, state["session_id"])

        return {
            "final_report": fallback_report,
            "export_path": export_path,
            "error": f"Report generation partially failed: {str(e)}",
            "current_node": "synthesizer",
            "execution_history": state["execution_history"] + ["synthesizer"],
            "updated_at": datetime.now().isoformat(),
        }


def _export_to_file(report: str, session_id: str) -> str:
    """Export report to markdown file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_brief_{session_id}_{timestamp}.md"
    filepath = OUTPUT_DIR / filename

    filepath.write_text(report, encoding="utf-8")

    logger.info(f"Report exported to {filepath}")
    return str(filepath)


def _format_queryquest_data(state: AgentState) -> str:
    """Format collected data in QueryQuest style."""
    sections = []

    # ClinGen results
    clingen = state.get("clingen_results")
    if clingen and isinstance(clingen, dict):
        section = ["### ClinGen Gene-Disease Associations"]
        total = clingen.get("total", 0)
        section.append(f"Total: {total} gene-disease links")

        definitive = clingen.get("definitive", [])
        if definitive:
            genes = [r.get("Gene_Symbol", "") for r in definitive[:10]]
            section.append(f"- Definitive ({len(definitive)}): {', '.join(genes)}")

        strong = clingen.get("strong", [])
        if strong:
            genes = [r.get("Gene_Symbol", "") for r in strong[:10]]
            section.append(f"- Strong ({len(strong)}): {', '.join(genes)}")

        sections.append("\n".join(section))

    # PubMedQA results
    pubmedqa = state.get("pubmedqa_results")
    if pubmedqa and isinstance(pubmedqa, dict):
        section = ["### PubMedQA Q&A Pairs"]
        total = pubmedqa.get("total", 0)
        yes_count = pubmedqa.get("yes_count", 0)
        no_count = pubmedqa.get("no_count", 0)
        section.append(f"Total: {total} Q&A pairs (YES: {yes_count}, NO: {no_count})")

        results = pubmedqa.get("results", [])
        for qa in results[:5]:
            q = qa.get("Question", "")[:100]
            a = qa.get("Answer", "")
            section.append(f"- Q: {q}... A: {a}")

        sections.append("\n".join(section))

    # bioRxiv results
    biorxiv = state.get("biorxiv_results")
    if biorxiv and isinstance(biorxiv, dict):
        section = ["### bioRxiv/medRxiv Preprints"]
        total = biorxiv.get("total", 0)
        bx = biorxiv.get("biorxiv_count", 0)
        mx = biorxiv.get("medrxiv_count", 0)
        section.append(f"Total: {total} preprints (bioRxiv: {bx}, medRxiv: {mx})")

        results = biorxiv.get("results", [])
        for paper in results[:5]:
            title = paper.get("Title", "")[:80]
            authors = paper.get("Authors", "")[:50]
            date = paper.get("Date", "")
            section.append(f"- \"{title}\" ({authors}, {date})")

        sections.append("\n".join(section))

    # ORKG results
    orkg = state.get("orkg_results")
    if orkg and isinstance(orkg, dict):
        section = ["### ORKG Knowledge Graph"]
        total = orkg.get("total", 0)
        section.append(f"Total: {total} knowledge entries")

        results = orkg.get("results", [])
        for entry in results[:5]:
            obj = entry.get("object", "")[:100]
            section.append(f"- {obj}")

        sections.append("\n".join(section))

    # Researcher results
    researchers = state.get("researcher_results")
    if researchers:
        if isinstance(researchers, dict):
            researchers = researchers.get("researchers", [])

        section = ["### OpenAlex Researchers"]
        section.append(f"Total: {len(researchers)} researchers identified")

        for r in researchers[:5]:
            name = r.get("name", "Unknown")
            h_index = r.get("h_index", "N/A")
            citations = r.get("cited_by_count", 0)
            inst = r.get("affiliation", "N/A")
            section.append(f"- {name} (H-index: {h_index}, Citations: {citations:,}, {inst})")

        sections.append("\n".join(section))

    return "\n\n".join(sections) if sections else "No data collected"


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
    disease_name = state.get("disease_variants", ["Disease"])[0] if state.get("disease_variants") else "Disease"

    report = f"""# Research Brief: {disease_name}

## Research Question
{state['user_query']}

## Executive Summary
This research brief was generated using automated data retrieval from multiple biomedical databases including ClinGen, PubMedQA, bioRxiv/medRxiv, ORKG, and OpenAlex.

## Research Methodology
**Disease Focus:** {', '.join(state.get('disease_variants', ['Not specified']))}
**Genes Identified:** {', '.join(state.get('gene_variants', [])) or 'None specified'}
**Topic Keywords:** {', '.join(state.get('topic_keywords', [])) or 'None specified'}

## Gene-Disease Findings
"""

    # ClinGen results
    clingen = state.get("clingen_results")
    if clingen and isinstance(clingen, dict):
        total = clingen.get("total", 0)
        report += f"Found {total} gene-disease associations in ClinGen.\n\n"

        definitive = clingen.get("definitive", [])
        if definitive:
            genes = [r.get("Gene_Symbol", "") for r in definitive[:10]]
            report += f"**Definitive genes ({len(definitive)}):** {', '.join(genes)}\n\n"

        strong = clingen.get("strong", [])
        if strong:
            genes = [r.get("Gene_Symbol", "") for r in strong[:10]]
            report += f"**Strong genes ({len(strong)}):** {', '.join(genes)}\n\n"
    else:
        report += "No ClinGen results available.\n\n"

    report += "## Literature Insights\n"

    # PubMedQA results
    pubmedqa = state.get("pubmedqa_results")
    if pubmedqa and isinstance(pubmedqa, dict):
        total = pubmedqa.get("total", 0)
        report += f"Found {total} relevant Q&A pairs from PubMedQA.\n\n"
    else:
        report += "No PubMedQA results available.\n\n"

    # bioRxiv results
    biorxiv = state.get("biorxiv_results")
    if biorxiv and isinstance(biorxiv, dict):
        total = biorxiv.get("total", 0)
        report += f"Found {total} relevant preprints from bioRxiv/medRxiv.\n\n"
    else:
        report += "No bioRxiv results available.\n\n"

    report += "## Key Researchers\n"

    # Researcher results
    researchers = state.get("researcher_results")
    if researchers:
        if isinstance(researchers, dict):
            researchers = researchers.get("researchers", [])
        report += f"Identified {len(researchers)} researchers in this field.\n\n"
        for r in researchers[:5]:
            name = r.get("name", "Unknown")
            h_index = r.get("h_index", "N/A")
            report += f"- **{name}** (H-index: {h_index})\n"
    else:
        report += "No researcher data available.\n"

    if state.get("human_feedback"):
        report += f"\n## Human Feedback\n{state['human_feedback']}\n"

    report += f"""
---

## Data Sources
- ClinGen (Gene-Disease Validity)
- PubMedQA (Biomedical Q&A)
- bioRxiv/medRxiv (Preprints)
- ORKG (Open Research Knowledge Graph)
- OpenAlex (Researcher Information)

---
*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Session ID: {state.get('session_id', 'N/A')}*
"""

    return report
