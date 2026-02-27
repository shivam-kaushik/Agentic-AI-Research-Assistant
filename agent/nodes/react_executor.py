"""
ReAct Executor Node for Co-Investigator Agent

Implements the Reason→Act→Observe→Decide pattern from QueryQuest v9.0.
Executes research tasks against datasets with explicit reasoning steps.
"""
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel

from config.gcp_config import config
from agent.state import AgentState, TaskStatus, add_task_history
from tools import (
    clingen_loader,
    pubmedqa_loader,
    biorxiv_loader,
    orkg_loader,
    smart_search,
    gemini_filter,
    safe_len,
    search_researchers,
)

logger = logging.getLogger(__name__)


class ReActExecutor:
    """
    Implements the Reason→Act→Observe→Decide pattern for research tasks.

    Each step goes through:
    1. REASON: Generate reasoning about what data we need
    2. ACT: Execute queries against appropriate datasets
    3. OBSERVE: Analyze results, generate observation text
    4. DECIDE: Determine if step is complete or needs refinement
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        vertexai.init(project=config.project_id, location=config.location)
        self.model = GenerativeModel(model_name)

    def execute_step(self, step: dict, state: AgentState) -> dict:
        """
        Execute a single research step with ReAct pattern.

        Returns dict with: reasoning, results, observation, decision
        """
        task_id = step.get("task_id", "unknown")
        logger.info(f"ReAct executing step: {task_id}")

        # REASON
        reasoning = self._reason(step, state)
        logger.info(f"[REASON] {reasoning[:100]}...")

        # ACT
        results = self._act(step, state)
        logger.info(f"[ACT] Retrieved {safe_len(results)} results")

        # OBSERVE
        observation = self._observe(results, step, state)
        logger.info(f"[OBSERVE] {observation[:100]}...")

        # DECIDE
        decision = self._decide(observation, step, state)
        logger.info(f"[DECIDE] {decision}")

        return {
            "task_id": task_id,
            "reasoning": reasoning,
            "results": results,
            "observation": observation,
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
        }

    def _reason(self, step: dict, state: AgentState) -> str:
        """Generate reasoning about what data we need."""
        disease = state.get("disease_variants", [""])[0] if state.get("disease_variants") else ""
        genes = state.get("gene_variants", [])
        data_source = step.get("data_source", "unknown")

        prompt = f"""
        Research query: {state['user_query']}
        Current step: {step.get('description', '')}
        Data source: {data_source}
        Disease focus: {disease}
        Genes identified: {genes}

        Generate a brief (2-3 sentences) reasoning about:
        1. What specific data we need from {data_source}
        2. Why this data is relevant to the research question
        3. What we expect to find

        Be concise and specific.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return f"Searching {data_source} for {disease} related data."

    def _act(self, step: dict, state: AgentState) -> dict:
        """Execute queries against the appropriate dataset."""
        data_source = step.get("data_source", "").lower()
        disease_variants = state.get("disease_variants", [])
        gene_variants = state.get("gene_variants", [])
        topic_keywords = state.get("topic_keywords", [])
        primary_term = disease_variants[0] if disease_variants else state["user_query"]

        all_terms = disease_variants + gene_variants + topic_keywords

        results = {}

        if data_source == "clingen":
            results = self._search_clingen(disease_variants, gene_variants, primary_term)

        elif data_source in ["pubmedqa", "pubmed"]:
            results = self._search_pubmedqa(all_terms, primary_term)

        elif data_source in ["biorxiv", "medrxiv", "preprints"]:
            results = self._search_biorxiv(all_terms, primary_term)

        elif data_source == "orkg":
            results = self._search_orkg(disease_variants, topic_keywords, gene_variants)

        elif data_source == "openalex":
            results = self._search_openalex(state.get("researcher_search_query", primary_term))

        else:
            logger.warning(f"Unknown data source: {data_source}")
            results = {"error": f"Unknown data source: {data_source}"}

        return results

    def _search_clingen(
        self,
        disease_variants: list[str],
        gene_variants: list[str],
        primary_term: str
    ) -> dict:
        """Search ClinGen for gene-disease associations."""
        try:
            df = clingen_loader.load_all()

            # Search by disease
            results = smart_search(df, "Disease_Label", disease_variants, threshold=85)

            # Also search by genes if provided
            if gene_variants:
                gene_hits = smart_search(df, "Gene_Symbol", gene_variants, threshold=95)
                results = pd.concat([results, gene_hits]).drop_duplicates()

            # Filter with Gemini for relevance
            if not results.empty:
                results = gemini_filter(results, "Disease_Label", primary_term, max_results=15)

            # Categorize by classification
            definitive = results[results["Classification"] == "Definitive"] if not results.empty else pd.DataFrame()
            strong = results[results["Classification"] == "Strong"] if not results.empty else pd.DataFrame()
            moderate = results[results["Classification"] == "Moderate"] if not results.empty else pd.DataFrame()

            return {
                "total": len(results),
                "definitive": definitive.to_dict("records") if not definitive.empty else [],
                "strong": strong.to_dict("records") if not strong.empty else [],
                "moderate": moderate.to_dict("records") if not moderate.empty else [],
                "all_results": results.to_dict("records") if not results.empty else [],
            }

        except Exception as e:
            logger.error(f"ClinGen search failed: {e}")
            return {"error": str(e), "total": 0}

    def _search_pubmedqa(self, terms: list[str], primary_term: str) -> dict:
        """Search PubMedQA for Q&A pairs."""
        try:
            print("\n❓ Searching PubMedQA (labelled entries)...")
            df = pubmedqa_loader.load_searchable()

            # Fix: Only search for primary term/disease variants to avoid unrelated results
            search_terms = [primary_term]
            
            raw_q = smart_search(df, "Question", search_terms, threshold=80)
            raw_c = smart_search(df, "Context", search_terms, threshold=80)
            results = pd.concat([raw_q, raw_c]).drop_duplicates()

            print(f"   Raw: {len(results)} → Gemini filtering...")

            if not results.empty:
                results = gemini_filter(results, "Question", primary_term, max_results=10)

            yes_count = len(results[results["Answer"] == "YES"]) if not results.empty else 0
            no_count = len(results[results["Answer"] == "NO"]) if not results.empty else 0
            
            print(f"   ✅ {len(results)} relevant Q&A | YES:{yes_count} NO:{no_count}")
            if not results.empty:
                print(results[["Question", "Answer"]].to_string(index=False))

            return {
                "total": len(results),
                "yes_count": yes_count,
                "no_count": no_count,
                "results": results.to_dict("records") if not results.empty else [],
            }

        except Exception as e:
            logger.error(f"PubMedQA search failed: {e}")
            return {"error": str(e), "total": 0}

    def _search_biorxiv(self, terms: list[str], primary_term: str) -> dict:
        """Search bioRxiv/medRxiv for preprints."""
        try:
            print("\n📰 Searching bioRxiv/medRxiv...")
            df = biorxiv_loader.load_all()

            # Fix: Only search for primary term/disease variants to avoid unrelated results
            search_terms = [primary_term]

            raw_t = smart_search(df, "Title", search_terms, threshold=85)
            raw_a = smart_search(df, "Abstract", search_terms, threshold=85)
            results = pd.concat([raw_t, raw_a]).drop_duplicates()
            
            print(f"   Raw: {len(results)} → Gemini filtering...")

            if not results.empty:
                results = gemini_filter(results, "Title", primary_term, max_results=10)

            biorxiv_count = len(results[results["source"] == "biorxiv"]) if not results.empty else 0
            medrxiv_count = len(results[results["source"] == "medrxiv"]) if not results.empty else 0
            
            print(f"   ✅ {len(results)} preprints | bioRxiv:{biorxiv_count} medRxiv:{medrxiv_count}")
            if not results.empty:
                cols_to_print = ["Title", "Authors", "Date", "source"]
                # Only keep columns that actually exist in the df
                cols_to_print = [c for c in cols_to_print if c in results.columns]
                print(results[cols_to_print].to_string(index=False))

            return {
                "total": len(results),
                "biorxiv_count": biorxiv_count,
                "medrxiv_count": medrxiv_count,
                "results": results.to_dict("records") if not results.empty else [],
            }

        except Exception as e:
            logger.error(f"bioRxiv search failed: {e}")
            return {"error": str(e), "total": 0}

    def _search_orkg(
        self,
        disease_variants: list[str],
        topic_keywords: list[str],
        gene_variants: list[str]
    ) -> dict:
        """Search ORKG for knowledge graph triples."""
        try:
            results = orkg_loader.multi_search(
                disease_variants=disease_variants,
                topic_keywords=topic_keywords,
                gene_variants=gene_variants,
            )

            return {
                "total": len(results),
                "results": results.head(20).to_dict("records") if not results.empty else [],
            }

        except Exception as e:
            logger.error(f"ORKG search failed: {e}")
            return {"error": str(e), "total": 0}

    def _search_openalex(self, query: str) -> dict:
        """Search OpenAlex for researchers."""
        try:
            researchers = search_researchers(query, max_results=10)

            return {
                "total": len(researchers),
                "researchers": researchers,
            }

        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return {"error": str(e), "total": 0}

    def _observe(self, results: dict, step: dict, state: AgentState) -> str:
        """Analyze results and generate observation text."""
        data_source = step.get("data_source", "unknown")
        total = results.get("total", 0)
        error = results.get("error")

        if error:
            return f"Error searching {data_source}: {error}"

        if total == 0:
            disease_category = state.get("disease_category", "other")
            if data_source == "clingen" and disease_category in ["complex", "polygenic"]:
                return (
                    f"No single-gene entries in ClinGen. Expected for "
                    f"'{disease_category}' diseases (polygenic). "
                    f"Genetic signals will surface from bioRxiv/literature."
                )
            return f"No results found in {data_source}."

        # Generate observation based on data source
        if data_source == "clingen":
            definitive = len(results.get("definitive", []))
            strong = len(results.get("strong", []))
            moderate = len(results.get("moderate", []))
            top_genes = [r.get("Gene_Symbol", "") for r in results.get("definitive", [])[:5]]
            return (
                f"Found {total} gene-disease links: {definitive} definitive, "
                f"{strong} strong, {moderate} moderate. "
                f"Top genes: {', '.join(top_genes) if top_genes else 'none definitive'}."
            )

        elif data_source in ["pubmedqa", "pubmed"]:
            yes_count = results.get("yes_count", 0)
            no_count = results.get("no_count", 0)
            return f"Found {total} Q&A pairs: {yes_count} YES, {no_count} NO answers."

        elif data_source in ["biorxiv", "preprints"]:
            bx = results.get("biorxiv_count", 0)
            mx = results.get("medrxiv_count", 0)
            titles = [r.get("Title", "")[:50] for r in results.get("results", [])[:3]]
            return (
                f"Found {total} preprints: {bx} bioRxiv, {mx} medRxiv. "
                f"Key themes: {'; '.join(titles)}"
            )

        elif data_source == "orkg":
            return f"Found {total} knowledge graph entries related to the query."

        elif data_source == "openalex":
            researchers = results.get("researchers", [])
            if researchers:
                top = researchers[0]
                return (
                    f"Found {total} researchers. Top: {top.get('name', 'Unknown')} "
                    f"(H-index: {top.get('h_index', 'N/A')}, "
                    f"Citations: {top.get('cited_by_count', 'N/A'):,})."
                )
            return f"Found {total} researchers."

        return f"Retrieved {total} results from {data_source}."

    def _decide(self, observation: str, step: dict, state: AgentState) -> str:
        """Determine if step is complete or needs refinement."""
        # Check for errors
        if "Error" in observation or "failed" in observation.lower():
            return "retry_or_skip"

        # Check for no results
        if "No results" in observation or "No single-gene" in observation:
            return "continue"  # Expected for some disease types

        # Check for successful results
        if "Found" in observation:
            return "complete"

        return "continue"


def react_executor_node(state: AgentState) -> dict:
    """
    LangGraph node that executes the current task using ReAct pattern.

    Args:
        state: Current agent state

    Returns:
        Updated state with execution results
    """
    plan = state.get("plan")
    if not plan:
        return {
            "error": "No plan available for execution",
            "current_node": "react_executor",
            "updated_at": datetime.now().isoformat(),
        }

    current_index = state.get("current_task_index", 0)
    sub_tasks = plan.get("sub_tasks", [])

    if current_index >= len(sub_tasks):
        return {
            "current_node": "react_executor",
            "execution_history": state["execution_history"] + ["react_executor"],
            "updated_at": datetime.now().isoformat(),
        }

    current_task = sub_tasks[current_index]
    executor = ReActExecutor()

    # Execute with ReAct pattern
    result = executor.execute_step(current_task, state)

    # Update task status
    current_task["status"] = TaskStatus.COMPLETED.value
    current_task["result"] = result
    current_task["observation"] = result.get("observation", "")

    # Store results by category
    data_source = current_task.get("data_source", "").lower()
    category_results = {}

    if data_source == "clingen":
        category_results["clingen_results"] = result.get("results", result)
    elif data_source in ["pubmedqa", "pubmed"]:
        category_results["pubmedqa_results"] = result.get("results", result)
    elif data_source in ["biorxiv", "preprints"]:
        category_results["biorxiv_results"] = result.get("results", result)
    elif data_source == "orkg":
        category_results["orkg_results"] = result.get("results", result)
    elif data_source == "openalex":
        category_results["researcher_results"] = result.get("researchers", result)

    # Add to task history
    task_history = add_task_history(
        state,
        task_id=current_task.get("task_id", "unknown"),
        action="completed",
        observation=result.get("observation", ""),
    )

    # Check if HITL checkpoint is needed
    hitl_after = plan.get("hitl_checkpoint_after")
    requires_hitl = hitl_after == current_task.get("task_id")

    return {
        "plan": plan,
        "current_task_index": current_index + 1,
        "results": {current_task.get("task_id"): result},
        **category_results,
        "task_history": task_history,
        "requires_hitl": requires_hitl,
        "current_node": "react_executor",
        "execution_history": state["execution_history"] + ["react_executor"],
        "updated_at": datetime.now().isoformat(),
    }
