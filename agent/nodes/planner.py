"""
Planner Node for Co-Investigator Agent

Decomposes user research requests into executable sub-tasks using Gemini 1.5 Pro.
"""
import json
import logging
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

import sys
sys.path.append("../..")
from config.gcp_config import config
from config.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from agent.state import AgentState, SubTask, ResearchPlan, TaskStatus

logger = logging.getLogger(__name__)


def planner_node(state: AgentState) -> dict:
    """
    Planner node that decomposes user query into sub-tasks.

    Args:
        state: Current agent state

    Returns:
        Updated state with execution plan
    """
    logger.info(f"Planner node processing query: {state['user_query']}")

    try:
        # Initialize Vertex AI
        vertexai.init(project=config.project_id, location=config.location)

        # Create model instance
        model = GenerativeModel(
            config.planner_model,
            system_instruction=PLANNER_SYSTEM_PROMPT,
        )

        # Configure for JSON output
        generation_config = GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

        # Generate the plan
        prompt = PLANNER_USER_PROMPT.format(user_query=state["user_query"])
        response = model.generate_content(prompt, generation_config=generation_config)

        # Parse the JSON response
        plan_json = json.loads(response.text)

        # Convert to ResearchPlan
        sub_tasks = [
            SubTask(
                task_id=task["task_id"],
                description=task["description"],
                data_source=task["data_source"],
                query_type=task["query_type"],
                entities=task["entities"],
                depends_on=task.get("depends_on", []),
                status=TaskStatus.PENDING,
            )
            for task in plan_json.get("sub_tasks", [])
        ]

        plan = ResearchPlan(
            research_goal=plan_json.get("research_goal", state["user_query"]),
            sub_tasks=sub_tasks,
            hitl_checkpoint_after=plan_json.get("hitl_checkpoint_after"),
        )

        logger.info(f"Created plan with {len(sub_tasks)} sub-tasks")

        return {
            "plan": plan.to_dict(),
            "current_task_index": 0,
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


def create_fallback_plan(user_query: str) -> ResearchPlan:
    """
    Create a simple fallback plan when LLM planning fails.

    Extracts keywords and creates basic search tasks.
    """
    # Simple keyword extraction
    keywords = [w for w in user_query.split() if len(w) > 3 and w.isalpha()][:3]

    if not keywords:
        keywords = ["disease", "gene"]

    sub_tasks = [
        SubTask(
            task_id="task_1",
            description="Search ClinGen for gene-disease associations",
            data_source="clingen",
            query_type="gene_disease",
            entities=keywords,
            status=TaskStatus.PENDING,
        ),
        SubTask(
            task_id="task_2",
            description="Search CIViC for clinical evidence",
            data_source="civic",
            query_type="gene_disease",
            entities=keywords,
            depends_on=["task_1"],
            status=TaskStatus.PENDING,
        ),
        SubTask(
            task_id="task_3",
            description="Find active researchers in the field",
            data_source="openalex",
            query_type="researcher",
            entities=keywords,
            depends_on=["task_2"],
            status=TaskStatus.PENDING,
        ),
    ]

    return ResearchPlan(
        research_goal=user_query,
        sub_tasks=sub_tasks,
        hitl_checkpoint_after="task_2",
    )
