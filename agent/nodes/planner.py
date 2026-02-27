"""
Planner Node for Co-Investigator Agent

Decomposes user research requests into executable sub-tasks using Gemini.
Enhanced with QueryQuest v9.0 structured extraction features.
"""
import json
import logging
import re
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

import sys
sys.path.append("../..")
from config.gcp_config import config
from agent.state import AgentState, SubTask, ResearchPlan, TaskStatus

logger = logging.getLogger(__name__)

# Abbreviations that should NOT be used as disease_variants (need full names)
BLOCKED_ABBREVIATIONS = {
    "AD", "PD", "MS", "ALS", "HD", "CF", "IBD", "IBS", "RA", "SLE",
    "AR", "XL", "SD", "MT", "UD", "DAT", "LOAD", "SDAT", "CAD", "SMA",
    "IPF", "COPD", "CKD", "CHF", "MI", "DVT", "PE", "TB", "HIV", "AIDS",
}

# QueryQuest-style planner prompt
PLANNER_SYSTEM_PROMPT_V2 = """You are a biomedical research planning expert.
You help scientists decompose complex research questions into executable sub-tasks.

Your output must be valid JSON with these exact fields:
{
    "research_goal": "one sentence describing what the scientist wants",
    "disease_variants": ["Full Disease Name", "Alternative Name"],
    "gene_variants": ["GENE1", "GENE2"],
    "topic_keywords": ["keyword1", "keyword2", "keyword3"],
    "researcher_search_query": "disease name only",
    "disease_category": "genetic|complex|neurological|cancer|infectious|other",
    "sub_tasks": [...],
    "hitl_checkpoint_after": "task_id or null"
}

CRITICAL RULES:
1. disease_variants: Full disease names only (minimum 8 characters).
   NO abbreviations like AD, PD, MS, ALS, HD, CF, RA, IPF, COPD.
   Include synonyms and alternate names.

2. gene_variants: Only if genes are explicitly mentioned in the query.
   Use official gene symbols (e.g., BRCA1, TP53, TERT).

3. topic_keywords: 3-5 molecular/biological keywords (minimum 6 characters each).
   Focus on mechanisms, pathways, treatments.

4. researcher_search_query: 2-3 word disease name ONLY.
   NO words like: expert, researcher, top, leading, working, best, who, find.
   Example: "pulmonary fibrosis" NOT "pulmonary fibrosis experts"

5. disease_category: Choose ONE from:
   - genetic: Single-gene disorders
   - complex: Polygenic/multifactorial
   - neurological: Brain/nervous system
   - cancer: Malignancies
   - infectious: Pathogen-caused
   - other: Default

6. sub_tasks: Create exactly 3 tasks:
   - Task 1: Gene & Disease Biology (ClinGen)
   - Task 2: Research Literature (PubMedQA, bioRxiv)
   - Task 3: People & Knowledge (OpenAlex, ORKG)
"""


def planner_node(state: AgentState) -> dict:
    """
    Planner node that decomposes user query into sub-tasks.

    Enhanced with QueryQuest v9.0 structured extraction:
    - disease_variants, gene_variants, topic_keywords
    - researcher_search_query, disease_category

    Args:
        state: Current agent state

    Returns:
        Updated state with execution plan and structured extraction fields
    """
    logger.info(f"Planner node processing query: {state['user_query']}")

    try:
        # Initialize Vertex AI
        vertexai.init(project=config.project_id, location=config.location)

        # Create model instance with QueryQuest-style prompt
        model = GenerativeModel(
            config.planner_model,
            system_instruction=PLANNER_SYSTEM_PROMPT_V2,
        )

        # Configure for JSON output
        generation_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        # Generate the plan
        user_prompt = f"""Analyze this research query and create an execution plan:

"{state["user_query"]}"

Return a JSON object with the exact structure specified in your instructions.
Include disease_variants, gene_variants, topic_keywords, researcher_search_query,
disease_category, and exactly 3 sub_tasks."""

        response = model.generate_content(user_prompt, generation_config=generation_config)

        # Parse the JSON response
        plan_json = json.loads(response.text)

        # Clean and validate the structured extraction fields
        disease_variants = _clean_disease_variants(plan_json.get("disease_variants", []))
        gene_variants = plan_json.get("gene_variants", [])
        topic_keywords = plan_json.get("topic_keywords", [])
        researcher_search_query = _clean_researcher_query(
            plan_json.get("researcher_search_query", "")
        )
        disease_category = plan_json.get("disease_category", "other")

        # Convert sub_tasks to SubTask objects
        sub_tasks = []
        for task in plan_json.get("sub_tasks", []):
            sub_tasks.append(SubTask(
                task_id=task.get("task_id", f"task_{len(sub_tasks)+1}"),
                description=task.get("description", ""),
                data_source=task.get("data_source", "clingen"),
                query_type=task.get("query_type", "gene_disease"),
                entities=task.get("entities", disease_variants + gene_variants),
                depends_on=task.get("depends_on", []),
                status=TaskStatus.PENDING,
            ))

        # Create ResearchPlan with all QueryQuest fields
        plan = ResearchPlan(
            research_goal=plan_json.get("research_goal", state["user_query"]),
            sub_tasks=sub_tasks,
            disease_variants=disease_variants,
            gene_variants=gene_variants,
            topic_keywords=topic_keywords,
            researcher_search_query=researcher_search_query,
            disease_category=disease_category,
            hitl_checkpoint_after=plan_json.get("hitl_checkpoint_after"),
        )

        logger.info(
            f"Created plan with {len(sub_tasks)} sub-tasks, "
            f"{len(disease_variants)} disease variants, "
            f"{len(gene_variants)} genes"
        )

        return {
            "plan": plan.to_dict(),
            "current_task_index": 0,
            # Copy structured fields to state for easy access
            "disease_variants": disease_variants,
            "gene_variants": gene_variants,
            "topic_keywords": topic_keywords,
            "researcher_search_query": researcher_search_query,
            "disease_category": disease_category,
            # Tracking
            "current_node": "planner",
            "execution_history": state["execution_history"] + ["planner"],
            "updated_at": datetime.now().isoformat(),
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse planner response as JSON: {e}")
        # Create a fallback simple plan
        fallback_plan = create_fallback_plan(state["user_query"])
        return {
            "plan": fallback_plan.to_dict(),
            "current_task_index": 0,
            "disease_variants": fallback_plan.disease_variants,
            "gene_variants": fallback_plan.gene_variants,
            "topic_keywords": fallback_plan.topic_keywords,
            "researcher_search_query": fallback_plan.researcher_search_query,
            "disease_category": fallback_plan.disease_category,
            "current_node": "planner",
            "execution_history": state["execution_history"] + ["planner"],
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Planner node error: {e}")
        return {
            "error": f"Planner failed: {str(e)}",
            "current_node": "planner",
            "execution_history": state["execution_history"] + ["planner"],
            "updated_at": datetime.now().isoformat(),
        }


def _clean_disease_variants(variants: list) -> list[str]:
    """Clean and validate disease variants (min 8 chars, no abbreviations)."""
    cleaned = []
    for v in variants:
        if not isinstance(v, str):
            continue
        v = v.strip()
        # Must be at least 8 characters
        if len(v) < 8:
            continue
        # Must not be a blocked abbreviation
        if v.upper() in BLOCKED_ABBREVIATIONS:
            continue
        cleaned.append(v)
    return cleaned


def _clean_researcher_query(query: str) -> str:
    """Clean researcher search query (remove expert/researcher/top words)."""
    if not query:
        return ""

    # Words to remove from researcher queries
    bad_words = [
        "expert", "researcher", "top", "leading", "best", "working",
        "research", "find", "who", "are", "the", "list", "active",
        "scientists", "doctors", "professors", "investigators"
    ]

    query_lower = query.lower()
    for bad in bad_words:
        query_lower = query_lower.replace(bad, "").strip()

    # Clean up extra spaces
    return " ".join(query_lower.split())


def create_fallback_plan(user_query: str) -> ResearchPlan:
    """
    Create a simple fallback plan when LLM planning fails.

    Extracts keywords and creates basic QueryQuest-style search tasks.
    """
    # Extract potential disease/topic from query
    # Look for phrases after "on", "about", "for"
    disease = user_query
    for marker in [" on ", " about ", " for ", " in "]:
        if marker in user_query.lower():
            parts = user_query.lower().split(marker)
            if len(parts) > 1:
                disease = parts[-1].strip()
                break

    # Clean up the disease name
    disease = disease.strip("?.,!").strip()
    if len(disease) < 8:
        disease = user_query

    # Extract keywords (words > 3 chars)
    keywords = [w for w in user_query.split() if len(w) > 5 and w.isalpha()][:5]

    # Create QueryQuest-style 3-category tasks
    sub_tasks = [
        SubTask(
            task_id="task_1",
            description="Retrieve validated gene-disease associations",
            data_source="clingen",
            query_type="gene_disease",
            entities=[disease] + keywords[:2],
            status=TaskStatus.PENDING,
        ),
        SubTask(
            task_id="task_2",
            description="Scan recent literature and preprints",
            data_source="biorxiv",  # Changed from civic
            query_type="literature",
            entities=[disease] + keywords[:2],
            depends_on=["task_1"],
            status=TaskStatus.PENDING,
        ),
        SubTask(
            task_id="task_3",
            description="Identify active researchers and knowledge connections",
            data_source="openalex",
            query_type="researcher",
            entities=[disease] + keywords[:2],
            depends_on=["task_2"],
            status=TaskStatus.PENDING,
        ),
    ]

    return ResearchPlan(
        research_goal=user_query,
        sub_tasks=sub_tasks,
        disease_variants=[disease] if len(disease) >= 8 else [],
        gene_variants=[],
        topic_keywords=keywords,
        researcher_search_query=disease[:30],
        disease_category="other",
        hitl_checkpoint_after="task_1",  # Checkpoint after gene search
    )
