"""
Multi-Agent Architecture for Co-Investigator

This implements a true agentic system with:
- Orchestrator Agent: Central decision maker with memory
- Specialized Sub-Agents: Planner, Researcher, Validator, Synthesizer
- Conversation Memory: Maintains context across queries
- Dynamic Routing: Agents trigger based on context and user intent
"""
import uuid
import json
import logging
from datetime import datetime
from typing import Literal, Any
from dataclasses import dataclass, field

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
from google.cloud import firestore

from config.gcp_config import config

logger = logging.getLogger(__name__)

# Initialize Vertex AI
vertexai.init(project=config.project_id, location=config.location)


@dataclass
class ConversationMemory:
    """Maintains conversation history and context."""
    session_id: str
    messages: list = field(default_factory=list)
    current_plan: dict = None
    collected_data: dict = field(default_factory=dict)
    pending_tasks: list = field(default_factory=list)
    completed_tasks: list = field(default_factory=list)
    user_preferences: dict = field(default_factory=dict)
    # Step-by-step execution tracking
    current_step_index: int = 0  # Which step we're on (0-indexed)
    awaiting_step_confirmation: bool = False  # Waiting for user to confirm next step
    last_step_result: dict = None  # Result from the last executed step

    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_summary(self) -> str:
        """Get a summary of current context for agents."""
        # Determine execution state
        has_plan = self.current_plan is not None
        has_pending = len(self.pending_tasks) > 0
        has_completed = len(self.completed_tasks) > 0
        has_data = len(self.collected_data) > 0

        if not has_plan:
            state_description = "NO_PLAN - Need to create a research plan first"
        elif self.awaiting_step_confirmation:
            state_description = "AWAITING_CONFIRMATION - Waiting for user to confirm next step"
        elif has_completed and not has_pending:
            state_description = "ALL_COMPLETE - All tasks done, ready for synthesis"
        elif has_completed and has_pending:
            state_description = "IN_PROGRESS - Some tasks complete, more pending"
        elif has_pending:
            state_description = "PLANNED - Plan created, ready to execute"
        else:
            state_description = "UNKNOWN"

        # Build plan summary
        plan_summary = "No plan yet"
        if self.current_plan:
            plan_summary = f"""
Research Goal: {self.current_plan.get('research_goal', 'Unknown')}
Disease Focus: {self.current_plan.get('disease_variants', [])}
Complexity: {self.current_plan.get('estimated_complexity', 'unknown')}
Tasks: {[t.get('task_id') for t in self.current_plan.get('sub_tasks', [])]}
"""

        summary = f"""
## Current Session Context

**Session ID:** {self.session_id}
**Execution State:** {state_description}
**Awaiting Step Confirmation:** {self.awaiting_step_confirmation}
**Current Step Index:** {self.current_step_index}

**Plan Summary:**
{plan_summary}

**Pending Tasks:** {self.pending_tasks}
**Completed Tasks:** {self.completed_tasks}
**Data Collected:** {list(self.collected_data.keys())}

**Recent Messages (last 5):**
"""
        for msg in self.messages[-5:]:
            summary += f"\n- [{msg['role']}]: {msg['content'][:200]}..."

        return summary

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "current_plan": self.current_plan,
            "collected_data": self.collected_data,
            "pending_tasks": self.pending_tasks,
            "completed_tasks": self.completed_tasks,
            "user_preferences": self.user_preferences,
            "current_step_index": self.current_step_index,
            "awaiting_step_confirmation": self.awaiting_step_confirmation,
            "last_step_result": self.last_step_result,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        memory = cls(
            session_id=data.get("session_id", ""),
            messages=data.get("messages", []),
            current_plan=data.get("current_plan"),
            collected_data=data.get("collected_data", {}),
            pending_tasks=data.get("pending_tasks", []),
            completed_tasks=data.get("completed_tasks", []),
            user_preferences=data.get("user_preferences", {}),
        )
        memory.current_step_index = data.get("current_step_index", 0)
        memory.awaiting_step_confirmation = data.get("awaiting_step_confirmation", False)
        memory.last_step_result = data.get("last_step_result")
        return memory


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str, model_name: str = "gemini-2.5-pro"):
        self.name = name
        self.model = GenerativeModel(model_name)

    def _generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate response from LLM."""
        config = GenerationConfig(temperature=temperature, max_output_tokens=4096)
        response = self.model.generate_content(prompt, generation_config=config)
        return response.text


class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator that decides which agent to invoke.
    Maintains context and routes user requests appropriately.
    """

    SYSTEM_PROMPT = """You are the Orchestrator Agent for a biomedical research assistant.

Your role is to:
1. Understand user intent from their message and conversation context
2. Decide which specialized agent should handle the request
3. Coordinate multi-step research workflows
4. Maintain conversation context and ensure continuity
5. RECOGNIZE FOLLOW-UP QUESTIONS and maintain state

CRITICAL ROUTING RULES (in priority order):

1. **EXECUTION COMMANDS** (HIGHEST PRIORITY - Route to RESEARCHER):
   - User wants to RUN, EXECUTE, PROCEED, CONTINUE, or START
   - Keywords: "yes", "proceed", "continue", "execute", "run", "start", "go", "ok", "sure", "go ahead", "do it", "next"
   - Even if user says "proceed with step 2" or "run step 1" → Route to RESEARCHER
   - Even after asking a clarifying question, if user then says "proceed" → Route to RESEARCHER
   - The RESEARCHER will handle dependency checking internally
   - Examples: "yes", "proceed", "Now proceed", "proceed with step 2", "run the plan", "execute step 1", "continue", "next step"

2. **AWAITING_CONFIRMATION state**:
   - If "Awaiting Step Confirmation: True" in context AND user confirms
   - Same keywords as above → Route to RESEARCHER

3. **REPORT/SYNTHESIS requests** (Route to SYNTHESIZER):
   - Keywords: "report", "synthesize", "summary", "summarize", "brief", "generate report"
   - Examples: "Generate report", "Summarize findings", "Create a brief"

4. **VALIDATION requests** (Route to VALIDATOR):
   - Keywords: "validate", "check", "verify", "quality"
   - Examples: "Validate data", "Check for conflicts"

5. **CLARIFYING QUESTIONS** (Route to CLARIFIER):
   - User asks "what is", "why", "explain", "how does", "what does X mean"
   - Questions about the plan structure or data sources
   - NOT requests to proceed or execute
   - Examples: "What is Alzheimer's?", "Why ClinGen?", "What does step 2 do?"

6. **NEW RESEARCH QUESTION** (Route to PLANNER):
   - User asks a completely new research question unrelated to current plan
   - Examples: "Find researchers in Parkinson's", "What genes cause diabetes?"

7. **MODIFY PLAN requests** (Route to PLANNER):
   - User wants to change, add, or modify the current plan
   - Examples: "Add more sources", "Focus only on definitive genes", "Skip step 2"

IMPORTANT: When in doubt between CLARIFIER and RESEARCHER:
- If the message contains action words (proceed, run, execute, continue, yes, ok, go) → RESEARCHER
- If the message is a question asking for explanation → CLARIFIER

Available Agents:
- PLANNER: Creates/modifies research plans, breaks down complex queries
- RESEARCHER: Queries datasets and EXECUTES steps (handles dependencies internally)
- VALIDATOR: Checks data for conflicts, contradictions, quality issues
- SYNTHESIZER: Creates final reports, summaries, insights
- CLARIFIER: Answers questions about current state, explains plan steps (NOT for execution)

Based on the user message and context, respond with a JSON object:
{
    "intent": "description of what user wants",
    "is_followup": true/false,
    "next_agent": "PLANNER|RESEARCHER|VALIDATOR|SYNTHESIZER|CLARIFIER",
    "agent_instructions": "specific instructions for the chosen agent",
    "preserve_state": true/false,
    "requires_user_input": true/false,
    "user_prompt": "question to ask user if clarification needed"
}
"""

    def __init__(self):
        super().__init__("Orchestrator")

    def route(self, user_message: str, memory: ConversationMemory) -> dict:
        """Determine which agent should handle the request."""

        # DETERMINISTIC PRE-CHECK: Execution commands ALWAYS go to RESEARCHER
        # This bypasses LLM for clear action keywords to prevent misrouting
        msg_lower = user_message.lower().strip()
        execution_keywords = [
            "yes", "proceed", "continue", "execute", "run", "start", "go ahead",
            "do it", "next", "ok", "sure", "go", "yeah", "yep", "y",
            "proceed with step", "run step", "execute step", "next step",
            "proceed to step", "move to step", "go to step"
        ]

        # Check if message starts with or is primarily an execution command
        is_execution_command = any(
            msg_lower == kw or
            msg_lower.startswith(kw + " ") or
            msg_lower.startswith("now " + kw) or
            msg_lower.startswith("please " + kw)
            for kw in execution_keywords
        )

        # Also check for "step N" patterns with action words
        import re
        step_pattern = re.search(r'(proceed|run|execute|go|continue|start)\s*(with\s*)?(to\s*)?(step\s*\d+|next|the\s*plan)', msg_lower)

        if is_execution_command or step_pattern:
            # Deterministically route to RESEARCHER
            return {
                "intent": "Execute research step",
                "is_followup": True,
                "next_agent": "RESEARCHER",
                "agent_instructions": "Execute the next pending task in the research plan",
                "preserve_state": True,
                "requires_user_input": False,
                "user_prompt": None
            }

        # For non-execution messages, use LLM routing
        prompt = f"""{self.SYSTEM_PROMPT}

## Current Context
{memory.get_context_summary()}

## User Message
{user_message}

Respond with JSON only:"""

        response = self._generate(prompt, temperature=0.1)

        try:
            # Extract JSON from response
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "intent": "unclear",
                "next_agent": "CLARIFIER",
                "agent_instructions": "Ask user to clarify their request",
                "requires_user_input": True,
                "user_prompt": "I'm not sure what you'd like me to do. Could you please clarify your research question?"
            }


class PlannerAgent(BaseAgent):
    """Creates and modifies research plans using QueryQuest 3-category approach."""

    # Block common abbreviations that return too many false positives
    BLOCKED_ABBREVIATIONS = {
        "AD", "PD", "MS", "ALS", "HD", "CF", "DMD", "SMA", "FA",
        "RP", "LCA", "MD", "CM", "DCM", "HCM", "ARVC",
        "LQTS", "BrS", "CPVT", "SQTS",
        "ASD", "ID", "DD",
        "CKD", "FSGS", "PKD",
    }

    SYSTEM_PROMPT = """You are the Planner Agent for biomedical research using QueryQuest architecture.

Your role is to:
1. Extract structured information from the research query
2. Generate topic_keywords that capture molecular mechanisms, pathways, and biological processes
3. Output EXACTLY 3 fixed research sub-tasks

Available Data Sources (QueryQuest datasets):
- clingen: Gene-disease validity associations with classifications (Definitive, Strong, Moderate, Limited)
- pubmedqa: Biomedical question-answer pairs from PubMed abstracts
- biorxiv: Recent preprints from bioRxiv and medRxiv with abstracts
- orkg: Open Research Knowledge Graph with scientific concepts and papers
- openalex: Researcher information, H-index, citations, publications (live API)

CRITICAL EXTRACTION RULES:

1. disease_variants (REQUIRED):
   - Use FULL disease names, minimum 8 characters
   - NEVER use abbreviations (AD, PD, MS, ALS, IPF, COPD, etc.)
   - Include synonyms: ["Idiopathic Pulmonary Fibrosis", "Idiopathic interstitial pneumonia", "Cryptogenic fibrosing alveolitis"]

2. gene_variants:
   - Only include genes if EXPLICITLY mentioned in the query
   - Use official HGNC symbols (e.g., TERT, MUC5B, SFTPC)

3. topic_keywords (REQUIRED, 4-6 keywords):
   - Focus on MOLECULAR/BIOLOGICAL mechanisms related to the disease
   - Examples for pulmonary fibrosis: ["fibrosis process", "extracellular matrix", "transforming growth", "cytokine signaling", "epithelial injury"]
   - Examples for Alzheimer's: ["amyloid beta", "tau protein", "neuroinflammation", "synaptic dysfunction"]
   - Each keyword should be 2-3 words describing a process, pathway, or molecular concept

4. researcher_search_query:
   - 2-3 word disease name ONLY
   - NO words like: expert, researcher, top, leading, working, best, who, find
   - Example: "pulmonary fibrosis" NOT "pulmonary fibrosis experts"

5. disease_category: Choose ONE from:
   - genetic: Single-gene disorders (cystic fibrosis, Huntington's)
   - complex: Polygenic/multifactorial (diabetes, heart disease, IPF)
   - neurological: Brain/nervous system (Alzheimer's, Parkinson's)
   - cancer: Malignancies
   - infectious: Pathogen-caused
   - other: Default

You MUST return a JSON research plan with EXACTLY these 3 sub_tasks:

{
    "research_goal": "clear statement copied from user query",
    "disease_variants": ["Full Disease Name", "Alternative Name", "Another Synonym"],
    "gene_variants": [],
    "topic_keywords": ["mechanism keyword 1", "pathway keyword 2", "process keyword 3", "biological term 4"],
    "researcher_search_query": "disease name only",
    "disease_category": "complex",
    "sub_tasks": [
        {
            "task_id": "task_1",
            "description": "Retrieve validated gene-disease associations",
            "data_source": "clingen",
            "category": "Gene & Disease Biology",
            "query_params": {"disease": "Full Disease Name"},
            "depends_on": [],
            "priority": 1
        },
        {
            "task_id": "task_2",
            "description": "Scan recent literature and preprints",
            "data_source": "biorxiv",
            "category": "Research Literature",
            "query_params": {"terms": ["topic_keyword1", "topic_keyword2"]},
            "depends_on": ["task_1"],
            "priority": 2
        },
        {
            "task_id": "task_3",
            "description": "Identify active researchers and knowledge connections",
            "data_source": "openalex",
            "category": "People & Knowledge",
            "query_params": {"query": "disease name"},
            "depends_on": ["task_1"],
            "priority": 2
        }
    ],
    "suggested_hitl_after": "task_1",
    "estimated_complexity": "low|medium|high"
}
"""

    def __init__(self):
        super().__init__("Planner")

    def create_plan(self, instructions: str, memory: ConversationMemory) -> dict:
        """Create a research plan based on instructions."""
        prompt = f"""{self.SYSTEM_PROMPT}

## Context
{memory.get_context_summary()}

## Instructions
{instructions}

Create a detailed research plan. Respond with JSON only:"""

        response = self._generate(prompt)

        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            plan = json.loads(json_str)
            memory.current_plan = plan
            memory.pending_tasks = [t["task_id"] for t in plan.get("sub_tasks", [])]
            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan: {e}")
            return {"error": "Failed to create plan", "raw_response": response}

    def modify_plan(self, modification: str, memory: ConversationMemory) -> dict:
        """Modify existing plan based on user feedback."""
        if not memory.current_plan:
            return self.create_plan(modification, memory)

        prompt = f"""{self.SYSTEM_PROMPT}

## Current Plan
{json.dumps(memory.current_plan, indent=2)}

## Modification Request
{modification}

## Context
Completed tasks: {memory.completed_tasks}
Pending tasks: {memory.pending_tasks}

Provide the modified plan. Respond with JSON only:"""

        response = self._generate(prompt)

        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]

            plan = json.loads(json_str)
            memory.current_plan = plan
            memory.pending_tasks = [t["task_id"] for t in plan.get("sub_tasks", [])
                                    if t["task_id"] not in memory.completed_tasks]
            return plan

        except json.JSONDecodeError:
            return {"error": "Failed to modify plan"}


class ResearcherAgent(BaseAgent):
    """Executes research queries against QueryQuest data sources."""

    def __init__(self):
        super().__init__("Researcher")
        # Import GCS data loaders
        from tools.clingen_loader import ClinGenLoader
        from tools.pubmedqa_loader import PubMedQALoader
        from tools.biorxiv_loader import BioRxivLoader
        from tools.orkg_loader import ORKGLoader

        self.clingen_loader = ClinGenLoader()
        self.pubmedqa_loader = PubMedQALoader()
        self.biorxiv_loader = BioRxivLoader()
        self.orkg_loader = ORKGLoader()

    def execute_task(self, task: dict, memory: ConversationMemory) -> dict:
        """Execute a single research task against QueryQuest datasets."""
        task_id = task.get("task_id", "unknown")
        data_source = task.get("data_source", "")
        query_params = task.get("query_params", {})
        
        # Determine current disease topic to enforce content relevance
        primary_disease = memory.current_plan.get("disease_variants", [""])[0] if memory.current_plan and memory.current_plan.get("disease_variants") else ""

        logger.info(f"Researcher executing task: {task_id} on {data_source}")

        result = {
            "task_id": task_id,
            "data_source": data_source,
            "success": False,
            "data": [],
            "count": 0,
        }

        try:
            if data_source == "clingen":
                # Query ClinGen via GCS loader
                disease = query_params.get("disease", primary_disease)
                gene = query_params.get("gene", "")
                terms = [disease] if disease else []
                if gene:
                    terms.append(gene)

                df = self.clingen_loader.load_all()
                if not df.empty and terms:
                    from tools.search_utils import smart_search, gemini_filter
                    matches = smart_search(df, "Disease_Label", terms)
                    if matches.empty and gene:
                        matches = smart_search(df, "Gene_Symbol", [gene])
                    
                    if not matches.empty and primary_disease:
                        matches = gemini_filter(matches, "Disease_Label", primary_disease, max_results=30)

                    result["data"] = matches.to_dict("records")[:50]
                    result["count"] = len(matches)
                    result["success"] = True

                    # Categorize by classification
                    if not matches.empty:
                        result["definitive"] = matches[matches["Classification"] == "Definitive"].to_dict("records")
                        result["strong"] = matches[matches["Classification"] == "Strong"].to_dict("records")
                        result["total"] = len(matches)

            elif data_source == "pubmedqa":
                # Query PubMedQA via GCS loader
                terms = query_params.get("terms", [])
                if not terms:
                    terms = [query_params.get("disease", ""), query_params.get("query", "")]
                if primary_disease and primary_disease not in terms:
                    terms.append(primary_disease)
                terms = [t for t in terms if t]

                df = self.pubmedqa_loader.load_all()
                if not df.empty and terms:
                    from tools.search_utils import smart_search, gemini_filter
                    matches = smart_search(df, "Question", terms, threshold=80)
                    if matches.empty:
                        matches = smart_search(df, "Context", terms, threshold=80)
                        
                    if not matches.empty and primary_disease:
                        matches = gemini_filter(matches, "Question", primary_disease, max_results=30)

                    result["data"] = matches.to_dict("records")[:30]
                    result["count"] = len(matches)
                    result["success"] = True

                    # Count answer types
                    if not matches.empty and "Answer" in matches.columns:
                        result["yes_count"] = len(matches[matches["Answer"] == "yes"])
                        result["no_count"] = len(matches[matches["Answer"] == "no"])
                        result["total"] = len(matches)

            elif data_source == "biorxiv":
                # Query bioRxiv/medRxiv via GCS loader
                terms = query_params.get("terms", [])
                if not terms:
                    terms = [query_params.get("disease", ""), query_params.get("query", "")]
                if primary_disease and primary_disease not in terms:
                    terms.append(primary_disease)
                terms = [t for t in terms if t]

                df = self.biorxiv_loader.load_all()
                if not df.empty and terms:
                    from tools.search_utils import smart_search, gemini_filter
                    matches = smart_search(df, "Title", [primary_disease], threshold=85)
                    if matches.empty:
                        matches = smart_search(df, "Title", terms, threshold=80)
                    
                    if not matches.empty and primary_disease:
                        matches = gemini_filter(matches, "Title", primary_disease, max_results=30)

                    result["data"] = matches.to_dict("records")[:30]
                    result["count"] = len(matches)
                    result["success"] = True

                    # Count by source
                    if not matches.empty and "source" in matches.columns:
                        result["biorxiv_count"] = len(matches[matches["source"] == "biorxiv"])
                        result["medrxiv_count"] = len(matches[matches["source"] == "medrxiv"])
                        result["total"] = len(matches)

            elif data_source == "orkg":
                # Query ORKG via GCS loader
                terms = query_params.get("terms", [])
                if not terms:
                    terms = [query_params.get("disease", ""), query_params.get("query", "")]
                if primary_disease and primary_disease not in terms:
                    terms.append(primary_disease)
                terms = [t for t in terms if t]

                if terms:
                    matches = self.orkg_loader.multi_search(
                        disease_variants=[primary_disease] if primary_disease else terms,
                        topic_keywords=terms,
                        gene_variants=[]
                    )

                    result["data"] = matches.to_dict("records")[:50]
                    result["count"] = len(matches)
                    result["success"] = True
                    result["total"] = len(matches)

            elif data_source == "openalex":
                # Query OpenAlex for researchers
                from tools.search_openalex import search_researchers
                query = query_params.get("query", primary_disease)

                if query:
                    query_result = search_researchers(query=query)
                    result["success"] = query_result.get("success", False)
                    result["data"] = query_result.get("data", [])
                    result["count"] = len(result["data"])
                    result["researchers"] = result["data"]
                    result["openalex_query"] = query

                    # Also query ORKG for knowledge connections
                    # Use gene_variants and topic_keywords from plan for better results
                    try:
                        plan_gene_variants = memory.current_plan.get("gene_variants", []) if memory.current_plan else []
                        plan_topic_keywords = memory.current_plan.get("topic_keywords", []) if memory.current_plan else []
                        plan_disease_variants = memory.current_plan.get("disease_variants", [query]) if memory.current_plan else [query]

                        # Get genes found from ClinGen results (task_1)
                        clingen_genes = []
                        if "task_1" in memory.collected_data:
                            clingen_data = memory.collected_data["task_1"]
                            for gene_record in clingen_data.get("data", []):
                                gene_sym = gene_record.get("Gene_Symbol", "")
                                if gene_sym and gene_sym not in clingen_genes:
                                    clingen_genes.append(gene_sym)

                        # Combine gene variants
                        all_genes = list(set(plan_gene_variants + clingen_genes))

                        logger.info(f"ORKG search: diseases={plan_disease_variants}, topics={plan_topic_keywords[:3]}, genes={all_genes}")

                        orkg_matches = self.orkg_loader.multi_search(
                            disease_variants=plan_disease_variants,
                            topic_keywords=plan_topic_keywords,
                            gene_variants=all_genes
                        )

                        result["orkg_raw_count"] = len(orkg_matches)
                        result["orkg_search_params"] = {
                            "disease_variants": plan_disease_variants,
                            "topic_keywords": plan_topic_keywords[:5],
                            "gene_variants": all_genes
                        }

                        # Store RAW data before filtering (for visualization)
                        raw_orkg_data = orkg_matches.to_dict("records")[:30] if not orkg_matches.empty else []

                        # Try to filter with Gemini, but fallback to raw if empty
                        filtered_orkg = orkg_matches
                        if not orkg_matches.empty:
                            from tools.search_utils import gemini_filter
                            try:
                                filtered_orkg = gemini_filter(orkg_matches, "object", primary_disease, max_results=30)
                                # If filter returns empty but we had raw data, use raw data
                                if filtered_orkg.empty and len(raw_orkg_data) > 0:
                                    logger.info(f"Gemini filter returned empty, using raw ORKG data ({len(raw_orkg_data)} records)")
                                    result["orkg_data"] = raw_orkg_data[:20]
                                    result["orkg_count"] = len(raw_orkg_data)
                                    result["orkg_filter_fallback"] = True
                                else:
                                    result["orkg_data"] = filtered_orkg.to_dict("records")[:20]
                                    result["orkg_count"] = len(filtered_orkg)
                            except Exception as filter_err:
                                logger.warning(f"Gemini filter failed: {filter_err}, using raw data")
                                result["orkg_data"] = raw_orkg_data[:20]
                                result["orkg_count"] = len(raw_orkg_data)
                        else:
                            result["orkg_data"] = []
                            result["orkg_count"] = 0

                        # Generate Knowledge Graph visualizations using RAW data (more comprehensive)
                        try:
                            from tools.knowledge_graph_viz import create_knowledge_graph, create_gene_disease_graph

                            # Use raw data for visualization (more nodes = better graph)
                            viz_data = raw_orkg_data if raw_orkg_data else result["orkg_data"]

                            # Create ORKG Knowledge Graph
                            kg_result = create_knowledge_graph(
                                orkg_data=viz_data,
                                disease_name=primary_disease,
                                gene_symbols=all_genes,
                                output_dir="outputs",
                                session_id=memory.session_id,
                            )
                            result["knowledge_graph"] = kg_result

                            # Create Gene-Disease Graph from ClinGen data
                            if "task_1" in memory.collected_data:
                                clingen_records = memory.collected_data["task_1"].get("data", [])
                                gd_result = create_gene_disease_graph(
                                    clingen_data=clingen_records,
                                    disease_name=primary_disease,
                                    output_dir="outputs",
                                    session_id=memory.session_id,
                                )
                                result["gene_disease_graph"] = gd_result

                            logger.info(f"Knowledge graphs generated: KG={kg_result.get('success')}, GD={gd_result.get('success') if 'gd_result' in dir() else 'N/A'}")

                        except Exception as viz_error:
                            logger.warning(f"Knowledge graph visualization failed: {viz_error}")
                            result["knowledge_graph"] = {"success": False, "error": str(viz_error)}

                    except Exception as e:
                        logger.warning(f"ORKG query failed: {e}")
                        result["orkg_data"] = []
                        result["orkg_count"] = 0

            # Store in memory
            memory.collected_data[task_id] = result
            if task_id in memory.pending_tasks:
                memory.pending_tasks.remove(task_id)
            memory.completed_tasks.append(task_id)

        except Exception as e:
            logger.error(f"Research task failed: {e}")
            result["error"] = str(e)

        return result

    def execute_next_task(self, memory: ConversationMemory) -> dict | None:
        """Execute the next pending task (step-by-step execution)."""
        if not memory.current_plan:
            return {"error": "No plan to execute"}

        tasks = memory.current_plan.get("sub_tasks", [])

        # Find the next task to execute
        for task in tasks:
            task_id = task.get("task_id")
            if task_id in memory.pending_tasks:
                # Check dependencies
                depends_on = task.get("depends_on", [])
                if all(dep in memory.completed_tasks for dep in depends_on):
                    result = self.execute_task(task, memory)
                    memory.last_step_result = result
                    memory.current_step_index = memory.completed_tasks.index(task_id) if task_id in memory.completed_tasks else len(memory.completed_tasks) - 1
                    return result

        return None  # No more tasks to execute

    def execute_all_pending(self, memory: ConversationMemory) -> list:
        """Execute all pending tasks in order."""
        results = []

        if not memory.current_plan:
            return [{"error": "No plan to execute"}]

        tasks = memory.current_plan.get("sub_tasks", [])

        for task in tasks:
            task_id = task.get("task_id")
            if task_id in memory.pending_tasks:
                # Check dependencies
                depends_on = task.get("depends_on", [])
                if all(dep in memory.completed_tasks for dep in depends_on):
                    result = self.execute_task(task, memory)
                    results.append(result)

        return results


class ValidatorAgent(BaseAgent):
    """Validates collected data for conflicts and issues."""

    SYSTEM_PROMPT = """You are the Validator Agent for biomedical research.

Your role is to:
1. Check collected data for contradictions between sources
2. Identify low-confidence or uncertain findings
3. Flag missing critical information
4. Recommend whether human review is needed

Analyze the data and respond with JSON:
{
    "validation_passed": true/false,
    "issues": [
        {
            "type": "contradiction|low_confidence|missing|outdated",
            "severity": "high|medium|low",
            "description": "what the issue is",
            "affected_data": ["task_id1", "task_id2"],
            "recommendation": "suggested action"
        }
    ],
    "requires_human_review": true/false,
    "review_reason": "why human review is needed",
    "summary": "brief summary of data quality"
}
"""

    def __init__(self):
        super().__init__("Validator")

    def validate(self, memory: ConversationMemory) -> dict:
        """Validate collected data."""
        if not memory.collected_data:
            return {
                "validation_passed": True,
                "issues": [],
                "requires_human_review": False,
                "summary": "No data to validate yet"
            }

        # Prepare data summary for LLM analysis
        data_summary = {}
        for task_id, result in memory.collected_data.items():
            data_summary[task_id] = {
                "source": result.get("data_source"),
                "count": result.get("count", 0),
                "success": result.get("success"),
                "sample": result.get("data", [])[:3]  # First 3 records
            }

        prompt = f"""{self.SYSTEM_PROMPT}

## Collected Data Summary
{json.dumps(data_summary, indent=2, default=str)}

## Research Goal
{memory.current_plan.get('research_goal', 'Unknown') if memory.current_plan else 'Unknown'}

Analyze and respond with JSON only:"""

        response = self._generate(prompt)

        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]

            return json.loads(json_str)

        except json.JSONDecodeError:
            # Rule-based fallback
            issues = []
            for task_id, result in memory.collected_data.items():
                if result.get("count", 0) == 0:
                    issues.append({
                        "type": "missing",
                        "severity": "medium",
                        "description": f"No data returned for {task_id}",
                        "affected_data": [task_id],
                        "recommendation": "Consider broadening search criteria"
                    })

            return {
                "validation_passed": len(issues) == 0,
                "issues": issues,
                "requires_human_review": len(issues) > 0,
                "summary": f"Found {len(issues)} potential issues"
            }


class ClarifierAgent(BaseAgent):
    """Answers questions about current state, plan, and progress."""

    SYSTEM_PROMPT = """You are the Clarifier Agent for a biomedical research assistant.

Your role is to:
1. Answer questions about the current research plan and its status
2. Explain what each step does and why it's included
3. Provide status updates on completed vs pending tasks
4. Help the user understand the data sources being used

When explaining the plan, be specific:
- ClinGen: Curated gene-disease associations with evidence levels (Definitive/Strong/Moderate/Limited)
- PubMedQA: Question-answer pairs from biomedical literature
- bioRxiv/medRxiv: Recent preprint papers with cutting-edge research
- ORKG: Knowledge graph connecting scientific concepts
- OpenAlex: Researcher profiles, citations, H-index, institutions

Always reference the actual data in the context when answering.
"""

    def __init__(self):
        super().__init__("Clarifier")

    def answer(self, question: str, memory: ConversationMemory) -> str:
        """Answer a question about the current state or plan."""
        plan_info = json.dumps(memory.current_plan, indent=2) if memory.current_plan else "No plan created yet"

        completed = memory.completed_tasks or []
        pending = memory.pending_tasks or []
        collected_summary = {k: {"source": v.get("data_source"), "count": v.get("count", 0)}
                           for k, v in memory.collected_data.items()} if memory.collected_data else {}

        prompt = f"""{self.SYSTEM_PROMPT}

## Current Research Plan
{plan_info}

## Task Status
- Completed tasks: {completed}
- Pending tasks: {pending}

## Collected Data Summary
{json.dumps(collected_summary, indent=2)}

## User Question
{question}

Provide a clear, helpful answer:"""

        return self._generate(prompt, temperature=0.3)


class SynthesizerAgent(BaseAgent):
    """Creates final reports and summaries."""

    SYSTEM_PROMPT = """You are a biomedical research analyst creating a concise research synthesis.

Write a focused markdown report with EXACTLY these 5 sections:

## 🎯 Direct Answer
2-3 sentences directly answering the user's research query. Mention specific researchers, genes, or findings. Be direct and actionable.

## 🧬 Gene & Biological Context
Summarize what ClinGen found about gene-disease associations. Explain the genetic basis (or lack thereof) for the disease. Reference specific genes and their classification status (Definitive/Strong/Moderate/Limited/No Known Relationship).

## 📚 Current Research Landscape
Synthesize findings from preprints and Q&A pairs. Mention:
- Key research themes from recent papers
- Author groups doing active work (e.g., "Leavy et al.", "Jenkins group")
- Specific findings that relate to treatments or mechanisms

## 🔬 Knowledge Graph Connections
Analyze the ORKG (Open Research Knowledge Graph) findings:
- List key scientific concepts connected to this disease
- Highlight semantic relationships between genes, pathways, and the disease
- Identify research directions suggested by the knowledge graph
- If no ORKG data, explain the significance of this (emerging field, limited semantic coverage, etc.)

## 👤 Key Researchers to Follow
Numbered list of top researchers with this EXACT format:
1. **Name** — H-index: X | Citations: Y | Institution: Z
2. **Name** — H-index: X | Citations: Y | Institution: Z
3. **Name** — H-index: X | Citations: Y | Institution: Z

Also mention any author names from recent preprints who appear active in the field.

RULES:
- Be SPECIFIC - use actual data values from the context
- Keep it CONCISE - no filler text
- Be ACTIONABLE - tell the researcher what to do next
- DO NOT make up data - only use what's in the context
- For Knowledge Graph section, explain what the connections MEAN for research directions
"""

    def __init__(self):
        super().__init__("Synthesizer")

    def synthesize(self, memory: ConversationMemory) -> dict:
        """Create a comprehensive synthesis report with ALL data from Steps 1-3."""
        import os
        from datetime import datetime

        # Get research context
        research_goal = memory.current_plan.get("research_goal", "Research query") if memory.current_plan else "Research query"
        disease_variants = memory.current_plan.get("disease_variants", []) if memory.current_plan else []
        primary_disease = disease_variants[0] if disease_variants else "the target disease"
        topic_keywords = memory.current_plan.get("topic_keywords", []) if memory.current_plan else []
        completed_steps = len(memory.completed_tasks)

        # ══════════════════════════════════════════════════════════════
        # EXTRACT ALL DATA FROM STEPS 1-3
        # ══════════════════════════════════════════════════════════════

        # Step 1: ClinGen Gene-Disease Data
        clingen_data = []
        clingen_stats = {"total": 0, "definitive": 0, "strong": 0, "moderate": 0, "limited": 0}
        if "task_1" in memory.collected_data:
            task1 = memory.collected_data["task_1"]
            clingen_data = task1.get("data", [])
            clingen_stats["total"] = len(clingen_data)
            clingen_stats["definitive"] = len(task1.get("definitive", []))
            clingen_stats["strong"] = len(task1.get("strong", []))
            clingen_stats["moderate"] = clingen_stats["total"] - clingen_stats["definitive"] - clingen_stats["strong"]

        # Step 2: Literature/Preprint Data
        preprint_data = []
        preprint_stats = {"total": 0, "biorxiv": 0, "medrxiv": 0}
        if "task_2" in memory.collected_data:
            task2 = memory.collected_data["task_2"]
            preprint_data = task2.get("data", [])
            preprint_stats["total"] = len(preprint_data)
            preprint_stats["biorxiv"] = task2.get("biorxiv_count", 0)
            preprint_stats["medrxiv"] = task2.get("medrxiv_count", 0)

        # Step 3: Researcher & ORKG Data
        researcher_data = []
        orkg_data = []
        orkg_concepts = []
        knowledge_graph_path = None
        gene_disease_graph_path = None
        if "task_3" in memory.collected_data:
            task3 = memory.collected_data["task_3"]
            researcher_data = task3.get("researchers", task3.get("data", []))
            orkg_data = task3.get("orkg_data", [])
            for item in orkg_data[:15]:
                obj = item.get("object", "")
                if isinstance(obj, str) and len(obj) > 10:
                    orkg_concepts.append(obj)
            # Get graph paths
            kg_info = task3.get("knowledge_graph", {})
            if kg_info.get("success"):
                knowledge_graph_path = kg_info.get("graph_path")
            gd_info = task3.get("gene_disease_graph", {})
            if gd_info.get("success"):
                gene_disease_graph_path = gd_info.get("graph_path")

        # ══════════════════════════════════════════════════════════════
        # BUILD COMPREHENSIVE REPORT
        # ══════════════════════════════════════════════════════════════
        lines = []

        # Header
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("📝  **LAYER 5 — COMPREHENSIVE RESEARCH SYNTHESIS**")
        lines.append(f"    Synthesizing {completed_steps} completed step(s)...")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")
        lines.append(f"🔬 **Research Query:** {research_goal}")
        lines.append(f"🦠 **Disease Focus:** {primary_disease}")
        lines.append(f"🔑 **Topic Keywords:** {', '.join(topic_keywords[:5])}")
        lines.append(f"📅 **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # SECTION 1: GENE-DISEASE ASSOCIATIONS (ClinGen)
        # ══════════════════════════════════════════════════════════════
        lines.append("═" * 62)
        lines.append("## 🧬 SECTION 1: Gene-Disease Associations (ClinGen)")
        lines.append("═" * 62)
        lines.append("")
        lines.append(f"**Summary:** {clingen_stats['total']} gene-disease associations found")
        lines.append(f"  • Definitive: {clingen_stats['definitive']}")
        lines.append(f"  • Strong: {clingen_stats['strong']}")
        lines.append(f"  • Moderate/Limited: {clingen_stats['moderate']}")
        lines.append("")

        if clingen_data:
            lines.append("**Complete Gene List:**")
            lines.append("```")
            lines.append(f"{'Gene':<15} {'Disease':<45} {'MOI':<5} {'Classification':<15}")
            lines.append("-" * 80)
            for gene in clingen_data[:20]:
                g = str(gene.get("Gene_Symbol", ""))[:15]
                d = str(gene.get("Disease_Label", ""))[:45]
                m = str(gene.get("MOI", ""))[:5]
                c = str(gene.get("Classification", ""))[:15]
                lines.append(f"{g:<15} {d:<45} {m:<5} {c:<15}")
            lines.append("```")
        else:
            lines.append("_No validated gene-disease associations found in ClinGen._")
        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # SECTION 2: LITERATURE & PREPRINTS
        # ══════════════════════════════════════════════════════════════
        lines.append("═" * 62)
        lines.append("## 📚 SECTION 2: Recent Literature & Preprints")
        lines.append("═" * 62)
        lines.append("")
        lines.append(f"**Summary:** {preprint_stats['total']} preprints found")
        lines.append(f"  • bioRxiv: {preprint_stats['biorxiv']}")
        lines.append(f"  • medRxiv: {preprint_stats['medrxiv']}")
        lines.append("")

        if preprint_data:
            lines.append("**Complete Preprint List:**")
            lines.append("")
            for i, paper in enumerate(preprint_data[:15], 1):
                title = paper.get("Title", paper.get("title", "Untitled"))
                authors = paper.get("Authors", paper.get("authors", "Unknown"))
                date = paper.get("Date", paper.get("date", "N/A"))
                source = paper.get("source", "preprint")
                lines.append(f"**{i}. {title}**")
                lines.append(f"   _Authors:_ {authors}")
                lines.append(f"   _Date:_ {date} | _Source:_ {source}")
                lines.append("")
        else:
            lines.append("_No recent preprints found matching the search criteria._")
        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # SECTION 3: KNOWLEDGE GRAPH ANALYSIS
        # ══════════════════════════════════════════════════════════════
        lines.append("═" * 62)
        lines.append("## 🔬 SECTION 3: Knowledge Graph Analysis (ORKG)")
        lines.append("═" * 62)
        lines.append("")
        lines.append(f"**Summary:** {len(orkg_data)} knowledge connections found")
        lines.append("")

        if orkg_concepts:
            lines.append("**Key Scientific Concepts:**")
            for i, concept in enumerate(orkg_concepts[:10], 1):
                lines.append(f"   {i}. {concept}")
            lines.append("")

            lines.append("**Complete Knowledge Graph Triples:**")
            lines.append("```")
            lines.append(f"{'Subject':<40} {'Object (Concept)':<50}")
            lines.append("-" * 90)
            for item in orkg_data[:15]:
                subj = str(item.get("subject", ""))
                if "/" in subj:
                    subj = subj.split("/")[-1]
                subj = subj[:40]
                obj = str(item.get("object", ""))[:50]
                lines.append(f"{subj:<40} {obj:<50}")
            lines.append("```")
        else:
            lines.append("_No ORKG knowledge connections found for this disease._")
        lines.append("")

        # Knowledge Graph Visualizations
        if knowledge_graph_path or gene_disease_graph_path:
            lines.append("**Generated Visualizations:**")
            if knowledge_graph_path:
                lines.append(f"   📊 ORKG Semantic Network: `{knowledge_graph_path}`")
            if gene_disease_graph_path:
                lines.append(f"   🧬 Gene-Disease Graph: `{gene_disease_graph_path}`")
            lines.append("")
        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # SECTION 4: ACTIVE RESEARCHERS
        # ══════════════════════════════════════════════════════════════
        lines.append("═" * 62)
        lines.append("## 👤 SECTION 4: Active Researchers (OpenAlex)")
        lines.append("═" * 62)
        lines.append("")
        lines.append(f"**Summary:** {len(researcher_data)} researchers found")
        lines.append("")

        if researcher_data:
            lines.append("**Complete Researcher List:**")
            lines.append("```")
            lines.append(f"{'Name':<35} {'H-Index':<10} {'Citations':<12} {'Institution':<35}")
            lines.append("-" * 92)
            for r in researcher_data[:15]:
                name = str(r.get("display_name", r.get("name", "Unknown")))[:35]
                h = str(r.get("h_index", "N/A"))[:10]
                c = str(r.get("cited_by_count", "N/A"))[:12]
                inst = str(r.get("last_known_institution", r.get("institution", "Unknown")))[:35]
                lines.append(f"{name:<35} {h:<10} {c:<12} {inst:<35}")
            lines.append("```")
        else:
            lines.append("_No researchers found matching the criteria._")
        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # SECTION 5: AI ANALYSIS & RECOMMENDATIONS
        # ══════════════════════════════════════════════════════════════
        lines.append("═" * 62)
        lines.append("## 🤖 SECTION 5: AI Analysis & Recommendations")
        lines.append("═" * 62)
        lines.append("")

        # Generate AI analysis using all collected data
        analysis_prompt = f"""{self.SYSTEM_PROMPT}

## Research Query
{research_goal}

## Disease Focus
{primary_disease}

## ClinGen Gene Data ({clingen_stats['total']} genes)
{json.dumps(clingen_data[:10], indent=2, default=str)}

## Preprint Data ({preprint_stats['total']} papers)
{json.dumps([{"title": p.get("Title", p.get("title")), "authors": p.get("Authors", p.get("authors"))} for p in preprint_data[:10]], indent=2, default=str)}

## ORKG Concepts ({len(orkg_concepts)} concepts)
{json.dumps(orkg_concepts[:10], indent=2)}

## Researcher Data ({len(researcher_data)} researchers)
{json.dumps([{"name": r.get("display_name"), "h_index": r.get("h_index"), "citations": r.get("cited_by_count")} for r in researcher_data[:10]], indent=2, default=str)}

Write a research synthesis with these exact 5 sections. Be SPECIFIC and reference actual data values."""

        ai_analysis = self._generate(analysis_prompt, temperature=0.3)
        lines.append(ai_analysis)
        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # FOOTER
        # ══════════════════════════════════════════════════════════════
        lines.append("═" * 62)
        lines.append("## 📋 Report Metadata")
        lines.append("═" * 62)
        lines.append(f"• **Session ID:** {memory.session_id}")
        lines.append(f"• **Steps Completed:** {completed_steps}")
        lines.append(f"• **Total Data Points:** {clingen_stats['total'] + preprint_stats['total'] + len(orkg_data) + len(researcher_data)}")
        if knowledge_graph_path:
            lines.append(f"• **ORKG Graph:** {knowledge_graph_path}")
        if gene_disease_graph_path:
            lines.append(f"• **Gene-Disease Graph:** {gene_disease_graph_path}")
        lines.append("")
        lines.append("_Generated by QueryQuest Co-Investigator v9.0_")

        formatted_report = "\n".join(lines)

        # Export to markdown
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_brief_{memory.session_id}_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted_report)

        return {
            "report": formatted_report,
            "export_path": filepath,
            "knowledge_graph_path": knowledge_graph_path,
            "gene_disease_graph_path": gene_disease_graph_path,
        }



class MultiAgentOrchestrator:
    """
    Main entry point for the multi-agent system.
    Coordinates agents and maintains session state.
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self.memory = ConversationMemory(session_id=self.session_id)

        # Initialize agents
        self.orchestrator = OrchestratorAgent()
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.validator = ValidatorAgent()
        self.synthesizer = SynthesizerAgent()
        self.clarifier = ClarifierAgent()

        # Firestore for persistence (optional)
        try:
            self.db = firestore.Client(project=config.project_id)
        except Exception:
            self.db = None
            logger.warning("Firestore not available, using in-memory state only")

    def process_message(self, user_message: str, status_callback=None) -> dict:
        """
        Process a user message and return agent response.

        This is the main entry point for user interactions.
        """
        if status_callback: status_callback("Analyzing user intent and routing to the appropriate agent...")
        
        # Add user message to memory
        self.memory.add_message("user", user_message)

        # Route to appropriate agent
        routing = self.orchestrator.route(user_message, self.memory)
        logger.info(f"Orchestrator routing: {routing}")

        next_agent = routing.get("next_agent", "CLARIFIER")
        instructions = routing.get("agent_instructions", "")

        if status_callback: status_callback(f"Routing task to **{next_agent}** agent...")

        response = {
            "session_id": self.session_id,
            "intent": routing.get("intent"),
            "agent_used": next_agent,
            "requires_input": routing.get("requires_user_input", False),
        }

        # Execute appropriate agent
        if next_agent == "PLANNER":
            if status_callback: status_callback("Generating or modifying the research plan based on the request...")
            if self.memory.current_plan:
                plan = self.planner.modify_plan(instructions, self.memory)
            else:
                plan = self.planner.create_plan(instructions, self.memory)

            response["plan"] = plan
            response["message"] = self._format_plan_message(plan)

        elif next_agent == "RESEARCHER":
            # Step-by-step execution with HITL checkpoints
            # Reset confirmation flag when continuing
            self.memory.awaiting_step_confirmation = False

            if status_callback: status_callback("Executing next research step...")

            result = self.researcher.execute_next_task(self.memory)

            if result is None:
                # All tasks complete
                response["message"] = self._format_all_complete_message()
                response["all_tasks_complete"] = True
                self.memory.awaiting_step_confirmation = False
            elif result.get("error"):
                response["message"] = f"Error executing task: {result['error']}"
                response["results"] = [result]
                self.memory.awaiting_step_confirmation = False
            else:
                # Format single step result with HITL checkpoint
                response["results"] = [result]
                response["message"] = self._format_single_step_result(result)
                response["requires_input"] = True  # Wait for user confirmation
                self.memory.awaiting_step_confirmation = True

        elif next_agent == "VALIDATOR":
            if status_callback: status_callback("Validating aggregated evidence for conflicts and correctness...")
            validation = self.validator.validate(self.memory)
            response["validation"] = validation
            response["message"] = self._format_validation_message(validation)
            response["requires_input"] = validation.get("requires_human_review", False)

        elif next_agent == "SYNTHESIZER":
            if status_callback: status_callback("Synthesizing findings into a structured report...")
            synth_result = self.synthesizer.synthesize(self.memory)
            response["report"] = synth_result["report"]
            response["message"] = synth_result["report"]
            response["export_path"] = synth_result["export_path"]
            response["research_state"] = self.memory.collected_data

        elif next_agent == "CLARIFIER":
            is_followup = routing.get("is_followup", False)

            if is_followup and self.memory.current_plan:
                # User is asking about the existing plan/state
                if status_callback: status_callback("Answering question about current research state...")
                answer = self.clarifier.answer(user_message, self.memory)
                response["message"] = answer
                response["requires_input"] = False
            else:
                # Need actual clarification from user
                if status_callback: status_callback("Intent unclear. Preparing a clarification prompt...")
                response["message"] = routing.get("user_prompt", "Could you please clarify?")
                response["requires_input"] = True

        if status_callback: status_callback("Saving session and logging metrics...")
        
        # Add assistant message to memory
        self.memory.add_message("assistant", response.get("message", ""))

        # Persist state
        self._save_state()

        if status_callback: status_callback("Response ready.")
        return response

    def _format_plan_message(self, plan: dict) -> str:
        """Format plan as a detailed, layered message with QueryQuest extraction."""
        if "error" in plan:
            return f"I encountered an issue creating the plan: {plan['error']}"

        from datetime import datetime

        lines = []

        # ══════════════════════════════════════════════════════════════
        # LAYER 1 — PLANNING
        # ══════════════════════════════════════════════════════════════
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("🧠  **LAYER 1 — PLANNING**")
        lines.append("    Gemini decomposing your query into research steps...")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")

        # Extraction details block
        lines.append("```")
        lines.append(f"   Intent           : {plan.get('research_goal', 'Research')}")

        disease_variants = plan.get('disease_variants', [])
        lines.append(f"   Disease variants : {disease_variants}")

        gene_variants = plan.get('gene_variants', [])
        lines.append(f"   Gene variants    : {gene_variants}")

        topic_keywords = plan.get('topic_keywords', [])
        lines.append(f"   Topic keywords   : {topic_keywords}")

        researcher_query = plan.get('researcher_search_query', '')
        lines.append(f"   Researcher query : {researcher_query}")

        disease_category = plan.get('disease_category', 'other')
        lines.append(f"   Disease category : {disease_category}")
        lines.append("```")
        lines.append("")

        # Data source descriptions for "Why" explanations
        source_descriptions = {
            "clingen": lambda d: f"ClinGen provides authoritative gene curation for {d}.",
            "pubmedqa": lambda d: f"PubMedQA reveals answered research questions related to {d}.",
            "biorxiv": lambda d: f"Literature reveals active research themes and potential treatments for {d}.",
            "orkg": lambda d: f"ORKG connects scientific concepts and papers related to {d}.",
            "openalex": lambda d: f"OpenAlex ranks researchers by citations and H-index related to {d}.",
        }

        # Data source friendly names
        source_names = {
            "clingen": "ClinGen",
            "pubmedqa": "PubMed, PubMedQA",
            "biorxiv": "PubMed, bioRxiv, medRxiv",
            "orkg": "OpenAlex, CORD-19, ORKG",
            "openalex": "OpenAlex, CORD-19",
        }

        primary_disease = disease_variants[0] if disease_variants else "the target disease"
        sub_tasks = plan.get("sub_tasks", [])

        # Research plan steps
        lines.append(f"📋 **Research plan** ({len(sub_tasks)} steps):")

        for i, task in enumerate(sub_tasks, 1):
            description = task.get("description", "Execute task")
            data_source = task.get("data_source", "unknown")

            source_display = source_names.get(data_source, data_source.upper())
            why_fn = source_descriptions.get(data_source, lambda d: f"Provides data related to {d}.")
            why_text = why_fn(primary_disease)

            lines.append(f"   **Step {i}:** {description}")
            lines.append(f"           Uses   : {source_display}")
            lines.append(f"           Why    : {why_text}")

        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # LAYER 2 — TASK STATE INITIALISATION
        # ══════════════════════════════════════════════════════════════
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("📋  **LAYER 2 — TASK STATE INITIALISATION**")
        lines.append("    Registering all steps as PENDING.")
        lines.append("    Agent will update status after each execution.")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")

        lines.append("```")
        lines.append(f"   Query   : {plan.get('research_goal', 'Research query')}")
        lines.append(f"   Steps   : {len(sub_tasks)} registered")
        lines.append(f"   Created : {datetime.now().isoformat()}")
        lines.append("```")
        lines.append("")

        lines.append("📋 **Task State:**")

        for i, task in enumerate(sub_tasks, 1):
            task_id = task.get("task_id", f"task_{i}")
            description = task.get("description", "Execute task")

            if task_id in self.memory.completed_tasks:
                status_icon = "✅"
                status_text = "COMPLETED"
            else:
                status_icon = "⏳"
                status_text = "PENDING"

            lines.append(f"   {status_icon} Step {i}: {description} [{status_text}]")

        lines.append("")
        lines.append("──────────────────────────────────────────────────────────────")
        lines.append("")
        lines.append("**Would you like me to proceed with executing this plan?**")
        lines.append("_(Type 'yes', 'proceed', or 'execute' to start the research)_")

        return "\n".join(lines)

    def _format_research_message(self, results: list) -> str:
        """Format research results as Layer 3 REACT agentic loop output."""
        lines = []

        # ══════════════════════════════════════════════════════════════
        # LAYER 3 — REACT AGENTIC LOOP
        # ══════════════════════════════════════════════════════════════
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("🔄  **LAYER 3 — REACT AGENTIC LOOP**")
        lines.append("    Reason → Act → Observe → [Layer 4: Smart Checkpoint] → Decide")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")

        # Data source friendly names and descriptions
        source_names = {
            "clingen": "ClinGen",
            "pubmedqa": "PubMedQA",
            "biorxiv": "bioRxiv/medRxiv",
            "orkg": "ORKG",
            "openalex": "OpenAlex",
        }

        source_whys = {
            "clingen": "ClinGen provides authoritative gene curation for the target disease.",
            "pubmedqa": "PubMedQA reveals answered research questions on the topic.",
            "biorxiv": "Literature reveals active research themes and potential treatments.",
            "orkg": "ORKG connects scientific concepts and papers.",
            "openalex": "OpenAlex ranks researchers by citations and H-index.",
        }

        total_tasks = len(results)

        for idx, result in enumerate(results, 1):
            task_id = result.get("task_id", f"task_{idx}")
            data_source = result.get("data_source", "unknown")
            success = result.get("success", False)
            count = result.get("count", 0)

            source_display = source_names.get(data_source, data_source.upper())
            why_text = source_whys.get(data_source, "Provides relevant data.")

            # Get task description from plan
            description = "Execute research task"
            if self.memory.current_plan:
                for task in self.memory.current_plan.get("sub_tasks", []):
                    if task.get("task_id") == task_id:
                        description = task.get("description", description)
                        break

            # ──────────────────────────────────────────────────────────────
            # REASON
            # ──────────────────────────────────────────────────────────────
            lines.append("──────────────────────────────────────────────────────────────")
            lines.append(f"🔎  **REASON:** Step {idx}/{total_tasks} — {description}")
            lines.append(f"    Datasets: {source_display}")
            lines.append(f"    Why     : {why_text}")
            lines.append("──────────────────────────────────────────────────────────────")
            lines.append("")

            # ⚡ ACT
            lines.append(f"⚡  **ACT:** Executing Step {idx}...")
            lines.append("")

            # Format results based on data source
            if data_source == "clingen":
                definitive = len(result.get("definitive", []))
                strong = len(result.get("strong", []))
                moderate = result.get("total", count) - definitive - strong
                disputed = 0

                lines.append(f"📊 **Gene Results:** {count} total | {definitive} definitive | {strong} strong | {moderate} moderate | {disputed} disputed")
                lines.append("")

                # Show top gene hits if available
                if result.get("data"):
                    lines.append("🏆 **Top Gene Hits:**")
                    lines.append("```")
                    lines.append(f"{'Gene_Symbol':<20} {'Disease_Label':<40} {'MOI':<5} {'Classification':<25}")
                    for gene in result["data"][:5]:
                        gene_sym = str(gene.get("Gene_Symbol", ""))[:20]
                        disease = str(gene.get("Disease_Label", ""))[:40]
                        moi = str(gene.get("MOI", ""))[:5]
                        classification = str(gene.get("Classification", ""))[:25]
                        lines.append(f"{gene_sym:<20} {disease:<40} {moi:<5} {classification:<25}")
                    lines.append("```")
                    lines.append("")

                # 👁️ OBSERVE
                top_genes = ", ".join([g.get("Gene_Symbol", "") for g in result.get("definitive", [])[:3]]) or "none definitive"
                lines.append(f"👁️  **OBSERVE:** Found {count} gene-disease links: {definitive} definitive, {strong} strong, {moderate} moderate. Top: {top_genes}.")

            elif data_source == "biorxiv":
                biorxiv_count = result.get("biorxiv_count", 0)
                medrxiv_count = result.get("medrxiv_count", 0)

                lines.append(f"📰 **Preprint Results:** {count} total | {biorxiv_count} bioRxiv | {medrxiv_count} medRxiv")
                lines.append("")

                if result.get("data"):
                    lines.append("📄 **Recent Preprints:**")
                    for paper in result["data"][:3]:
                        title = paper.get("Title", paper.get("title", "Untitled"))[:70]
                        lines.append(f"   • {title}...")
                    lines.append("")

                lines.append(f"👁️  **OBSERVE:** Found {count} recent preprints related to the research topic.")

            elif data_source == "pubmedqa":
                yes_count = result.get("yes_count", 0)
                no_count = result.get("no_count", 0)

                lines.append(f"❓ **Q&A Results:** {count} total | {yes_count} 'yes' answers | {no_count} 'no' answers")
                lines.append("")

                if result.get("data"):
                    lines.append("💬 **Sample Questions:**")
                    for qa in result["data"][:3]:
                        question = qa.get("Question", qa.get("question", ""))[:80]
                        lines.append(f"   • {question}...")
                    lines.append("")

                lines.append(f"👁️  **OBSERVE:** Found {count} Q&A pairs from PubMed literature.")

            elif data_source == "openalex":
                researchers = result.get("researchers", result.get("data", []))
                orkg_data = result.get("orkg_data", [])
                orkg_count = result.get("orkg_count", 0)

                lines.append(f"🔍 **OpenAlex query:** '{result.get('openalex_query', 'disease')}'")
                lines.append(f"   ✅ {count} researchers (H≥10)")
                lines.append("")

                if researchers:
                    lines.append("🏆 **Top Researchers:**")
                    lines.append("```")
                    lines.append(f"{'Name':<30} {'Citations':<12} {'H_Index':<10} {'Institution':<30}")
                    for researcher in researchers[:5]:
                        name = str(researcher.get("name", "Unknown"))[:30]
                        h_index = str(researcher.get("h_index", "N/A"))[:10]
                        citations = str(researcher.get("cited_by_count", "N/A"))[:12]
                        inst = str(researcher.get("institution", "Unknown"))[:30]
                        lines.append(f"{name:<30} {citations:<12} {h_index:<10} {inst:<30}")
                    lines.append("```")
                    lines.append("")

                # ORKG section
                if orkg_data or orkg_count > 0:
                    lines.append(f"🔬 **ORKG Knowledge Connections:** {orkg_count} found")
                    if orkg_data:
                        lines.append("```")
                        for item in orkg_data[:5]:
                            obj = item.get("object", item.get("label", str(item)[:80]))
                            if isinstance(obj, str):
                                lines.append(f"   • {obj[:80]}")
                        lines.append("```")
                    lines.append("")

                top_name = researchers[0].get("name", "None") if researchers else "None"
                top_h = researchers[0].get("h_index", "N/A") if researchers else "N/A"
                lines.append(f"👁️  **OBSERVE:** Found {count} researchers (H≥10). Top: {top_name} (H:{top_h}). Found {orkg_count} ORKG connections.")

            elif data_source == "orkg":
                lines.append(f"🔬 **Knowledge Graph Results:** {count} concepts/papers found")
                lines.append("")
                lines.append(f"👁️  **OBSERVE:** Found {count} knowledge graph connections.")

            else:
                lines.append(f"📊 **Results:** {count} records found")
                lines.append("")
                lines.append(f"👁️  **OBSERVE:** Task completed with {count} results.")

            if result.get("error"):
                lines.append(f"⚠️  **ERROR:** {result['error']}")

            lines.append("")

        # Summary
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("✅  **EXECUTION COMPLETE**")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")
        lines.append("**Would you like me to:**")
        lines.append("• **Validate** the collected data for conflicts?")
        lines.append("• **Synthesize** findings into a research report?")

        return "\n".join(lines)

    def _format_single_step_result(self, result: dict) -> str:
        """Format a single step result with HITL checkpoint for step-by-step execution."""
        lines = []

        task_id = result.get("task_id", "unknown")
        data_source = result.get("data_source", "unknown")
        count = result.get("count", 0)

        # Get task info from plan
        step_num = 1
        total_steps = 3
        description = "Execute task"
        next_step_description = ""

        if self.memory.current_plan:
            tasks = self.memory.current_plan.get("sub_tasks", [])
            total_steps = len(tasks)
            for i, task in enumerate(tasks):
                if task.get("task_id") == task_id:
                    step_num = i + 1
                    description = task.get("description", description)
                    # Get next step description
                    if i + 1 < len(tasks):
                        next_step_description = tasks[i + 1].get("description", "")
                    break

        # Data source friendly names
        source_names = {
            "clingen": "ClinGen",
            "pubmedqa": "PubMedQA",
            "biorxiv": "bioRxiv/medRxiv",
            "orkg": "ORKG",
            "openalex": "OpenAlex",
        }
        source_whys = {
            "clingen": "ClinGen provides authoritative gene curation for the target disease.",
            "pubmedqa": "PubMedQA reveals answered research questions on the topic.",
            "biorxiv": "Literature reveals active research themes and potential treatments.",
            "orkg": "ORKG connects scientific concepts and papers.",
            "openalex": "OpenAlex ranks researchers by citations and H-index.",
        }

        source_display = source_names.get(data_source, data_source.upper())
        why_text = source_whys.get(data_source, "Provides relevant data.")

        # Primary disease for context
        primary_disease = "the target disease"
        if self.memory.current_plan:
            disease_variants = self.memory.current_plan.get("disease_variants", [])
            if disease_variants:
                primary_disease = disease_variants[0]

        # ══════════════════════════════════════════════════════════════
        # LAYER 3 — REACT AGENTIC LOOP (Single Step)
        # ══════════════════════════════════════════════════════════════
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("🔄  **LAYER 3 — REACT AGENTIC LOOP**")
        lines.append("    Reason → Act → Observe → [Layer 4: Smart Checkpoint] → Decide")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")

        # REASON
        lines.append("──────────────────────────────────────────────────────────────")
        lines.append(f"🔎  **REASON:** Step {step_num}/{total_steps} — {description}")
        lines.append(f"    Datasets: {source_display}")
        lines.append(f"    Why     : {why_text}")
        lines.append("──────────────────────────────────────────────────────────────")
        lines.append("")

        # ACT
        lines.append(f"⚡  **ACT:** Executing Step {step_num}...")
        lines.append("")

        # Format results based on data source
        if data_source == "clingen":
            definitive = len(result.get("definitive", []))
            strong = len(result.get("strong", []))
            moderate = count - definitive - strong
            disputed = 0

            lines.append(f"📊 **Gene Results:** {count} total | {definitive} definitive | {strong} strong | {moderate} moderate | {disputed} disputed")
            lines.append("")

            if result.get("data"):
                lines.append("🏆 **Top Gene Hits:**")
                lines.append("```")
                lines.append(f"{'Gene_Symbol':<20} {'Disease_Label':<40} {'MOI':<5} {'Classification':<25}")
                for gene in result["data"][:5]:
                    gene_sym = str(gene.get("Gene_Symbol", ""))[:20]
                    disease = str(gene.get("Disease_Label", ""))[:40]
                    moi = str(gene.get("MOI", ""))[:5]
                    classification = str(gene.get("Classification", ""))[:25]
                    lines.append(f"{gene_sym:<20} {disease:<40} {moi:<5} {classification:<25}")
                lines.append("```")
                lines.append("")

            # OBSERVE
            top_genes = ", ".join([g.get("Gene_Symbol", "") for g in result.get("definitive", [])[:3]]) or "none definitive"
            lines.append(f"👁️  **OBSERVE:** Found {count} gene-disease links: {definitive} definitive, {strong} strong, {moderate} moderate. Top: {top_genes}.")

        elif data_source == "biorxiv":
            biorxiv_count = result.get("biorxiv_count", 0)
            medrxiv_count = result.get("medrxiv_count", 0)

            lines.append(f"📰 **Preprint Results:** {count} total | {biorxiv_count} bioRxiv | {medrxiv_count} medRxiv")
            lines.append("")

            if result.get("data"):
                lines.append("📄 **Recent Preprints:**")
                lines.append("```")
                lines.append(f"{'Title':<80} {'Authors':<40} {'Date':<12} {'Source':<10}")
                for paper in result["data"][:5]:
                    title = str(paper.get("Title", paper.get("title", "Untitled")))[:80]
                    authors = str(paper.get("Authors", paper.get("authors", "")))[:40]
                    date = str(paper.get("Date", paper.get("date", "")))[:12]
                    source = str(paper.get("source", ""))[:10]
                    lines.append(f"{title:<80}")
                    lines.append(f"  {authors:<40} {date:<12} {source:<10}")
                lines.append("```")
                lines.append("")

            # Extract themes from titles
            themes = []
            for paper in result.get("data", [])[:3]:
                title = paper.get("Title", paper.get("title", ""))
                if title:
                    themes.append(title[:45])
            themes_str = "; ".join(themes) if themes else "various topics"

            lines.append(f"👁️  **OBSERVE:** Found {count} preprints. Key themes: {themes_str}")

        elif data_source == "pubmedqa":
            yes_count = result.get("yes_count", 0)
            no_count = result.get("no_count", 0)

            lines.append(f"❓ **Q&A Results:** {count} total | {yes_count} 'yes' answers | {no_count} 'no' answers")
            lines.append("")

            if result.get("data"):
                lines.append("💬 **Sample Questions:**")
                lines.append("```")
                lines.append(f"{'Question':<80} {'Answer':<10}")
                for qa in result["data"][:5]:
                    question = str(qa.get("Question", qa.get("question", "")))[:80]
                    answer = str(qa.get("Answer", qa.get("answer", "")))[:10]
                    lines.append(f"{question:<80} {answer:<10}")
                lines.append("```")
                lines.append("")

            lines.append(f"👁️  **OBSERVE:** Found {count} Q&A pairs from PubMed literature.")

        elif data_source == "openalex":
            researchers = result.get("researchers", result.get("data", []))
            openalex_query = result.get("openalex_query", primary_disease)
            orkg_data = result.get("orkg_data", [])
            orkg_count = result.get("orkg_count", 0)

            # OpenAlex section
            lines.append(f"🔍 **OpenAlex query:** '{openalex_query}'")
            lines.append(f"   ✅ {count} researchers (H≥10) via '{openalex_query}'")
            lines.append("")

            if researchers:
                lines.append("🏆 **Top Researchers:**")
                lines.append("```")
                lines.append(f"{'Name':<30} {'Citations':<12} {'H_Index':<10} {'Institution':<30}")
                for researcher in researchers[:5]:
                    name = str(researcher.get("display_name", "Unknown"))[:30]
                    h_index = str(researcher.get("h_index", "N/A"))[:10]
                    citations = str(researcher.get("cited_by_count", "N/A"))[:12]
                    inst = str(researcher.get("last_known_institution", "Unknown"))[:30]
                    lines.append(f"{name:<30} {citations:<12} {h_index:<10} {inst:<30}")
                lines.append("```")
                lines.append("")

            # ══════════════════════════════════════════════════════════════
            # ORKG KNOWLEDGE GRAPH SECTION
            # ══════════════════════════════════════════════════════════════
            lines.append("")
            lines.append("══════════════════════════════════════════════════════════════")
            lines.append("🔬  **KNOWLEDGE GRAPH ANALYSIS (ORKG)**")
            lines.append("══════════════════════════════════════════════════════════════")
            lines.append("")

            # Show search parameters
            orkg_params = result.get("orkg_search_params", {})
            lines.append("**Search Strategy:**")
            lines.append(f"   • Disease terms: {orkg_params.get('disease_variants', [primary_disease])}")
            lines.append(f"   • Topic keywords: {orkg_params.get('topic_keywords', [])[:3]}")
            lines.append(f"   • Gene symbols: {orkg_params.get('gene_variants', [])}")
            lines.append("")

            raw_count = result.get("orkg_raw_count", orkg_count)
            lines.append(f"📊 **Raw ORKG matches:** {raw_count}")

            if orkg_data:
                relevant_count = len(orkg_data)
                lines.append(f"✅ **Filtered relevant connections:** {relevant_count}")
                lines.append("")
                lines.append("📚 **Knowledge Graph Triples:**")
                lines.append("```")
                lines.append(f"{'Subject (URI)':<50} {'Object (Label)':<60}")
                lines.append("-" * 110)
                for item in orkg_data[:10]:
                    # Extract subject and object
                    subj = str(item.get("subject", ""))
                    # Clean up subject URI to show meaningful part
                    if "/" in subj:
                        subj = subj.split("/")[-1]
                    subj = subj[:50]
                    obj = str(item.get("object", item.get("label", "")))[:60]
                    lines.append(f"{subj:<50} {obj:<60}")
                lines.append("```")
                lines.append("")

                # Extract key concepts from ORKG
                concepts = []
                for item in orkg_data[:15]:
                    obj = item.get("object", "")
                    if isinstance(obj, str) and len(obj) > 10:
                        concepts.append(obj[:50])

                if concepts:
                    lines.append("🧠 **Key Scientific Concepts:**")
                    for i, concept in enumerate(concepts[:5], 1):
                        lines.append(f"   {i}. {concept}")
                    lines.append("")

            else:
                lines.append("⚠️ No relevant ORKG connections found after filtering.")
                lines.append(f"   _Raw matches: {raw_count} — using raw data for visualization._")
                lines.append("")

            # ══════════════════════════════════════════════════════════════
            # KNOWLEDGE GRAPH VISUALIZATIONS (always show, even with raw data)
            # ══════════════════════════════════════════════════════════════
            lines.append("══════════════════════════════════════════════════════════════")
            lines.append("📊  **KNOWLEDGE GRAPH VISUALIZATIONS**")
            lines.append("══════════════════════════════════════════════════════════════")
            lines.append("")

            # Show fallback notice if using raw data
            if result.get("orkg_filter_fallback"):
                lines.append("ℹ️  _Using raw ORKG data for visualization (filter returned empty)_")
                lines.append("")

            # ORKG Knowledge Graph
            kg_info = result.get("knowledge_graph", {})
            if kg_info.get("success"):
                import os
                kg_path = os.path.abspath(kg_info.get("graph_path")).replace("\\", "/")
                lines.append("✅ **ORKG Semantic Network Graph Generated:**")
                lines.append(f"![ORKG Semantic Network Graph](file:///{kg_path})")
                lines.append(f"   📍 Nodes: {kg_info.get('node_count', 0)} | Edges: {kg_info.get('edge_count', 0)}")
                
                concepts = kg_info.get('concept_nodes', [])
                if concepts:
                    lines.append(f"   🧬 Concepts: {', '.join(concepts[:5])}")
                    
                    # Generate dynamic analysis
                    if len(concepts) > 0:
                        analysis_prompt = f"Briefly analyze these semantic concepts related to {primary_disease} found in a knowledge graph: {', '.join(concepts)}. What do they suggest about potential research directions or mechanisms? Keep it to 2-3 short sentences."
                        try:
                            analysis = self.synthesizer._generate(analysis_prompt)
                            lines.append("\n**📈 Graph Analysis:**")
                            lines.append(analysis.strip())
                        except Exception as e:
                            logger.warning(f"Failed to generate graph analysis: {e}")
                
                lines.append("")
            else:
                kg_error = kg_info.get("error", "No data available")
                lines.append(f"⚠️ ORKG Graph: {kg_error}")
                lines.append("")
            # Gene-Disease Graph
            gd_info = result.get("gene_disease_graph", {})
            if gd_info.get("success"):
                import os
                gd_path = os.path.abspath(gd_info.get("graph_path")).replace("\\", "/")
                lines.append("✅ **Gene-Disease Relationship Graph Generated:**")
                lines.append(f"![Gene-Disease Relationship Graph](file:///{gd_path})")
                lines.append(f"   📍 Nodes: {gd_info.get('node_count', 0)} | Edges: {gd_info.get('edge_count', 0)}")
                
                # Generate dynamic analysis
                clingen_data = []
                if "task_1" in self.memory.collected_data:
                    clingen_data = self.memory.collected_data["task_1"].get("data", [])
                    
                if clingen_data:
                    top_genes = [g.get('Gene_Symbol') for g in clingen_data[:5] if g.get('Gene_Symbol')]
                    analysis_prompt = f"Briefly analyze this gene-disease relationship graph for {primary_disease}. The primary genes identified are: {', '.join(top_genes)}. What does this suggest about the genetic basis of the disease? Keep it to 2-3 short sentences."
                    try:
                        analysis = self.synthesizer._generate(analysis_prompt)
                        lines.append("\n**📈 Graph Analysis:**")
                        lines.append(analysis.strip())
                    except Exception as e:
                        logger.warning(f"Failed to generate graph analysis: {e}")
                
                lines.append("")
            elif gd_info:
                gd_error = gd_info.get("error", "No ClinGen data")
                lines.append(f"⚠️ Gene-Disease Graph: {gd_error}")
                lines.append("")

            # OBSERVE
            top_researcher = researchers[0] if researchers else None
            top_name = top_researcher.get("display_name", top_researcher.get("name", "None")) if top_researcher else "None"
            top_h = top_researcher.get("h_index", "N/A") if top_researcher else "N/A"
            orkg_relevant = len(orkg_data) if orkg_data else 0

            lines.append(f"👁️  **OBSERVE:** Found {count} researchers. Top: {top_name} (H:{top_h}). ORKG: {orkg_relevant} knowledge connections.")
            lines.append("")
            lines.append("✅ **All steps completed** — proceeding to synthesis.")

        else:
            lines.append(f"📊 **Results:** {count} records found")
            lines.append("")
            lines.append(f"👁️  **OBSERVE:** Task completed with {count} results.")

        lines.append("")

        # ══════════════════════════════════════════════════════════════
        # LAYER 4 — CONVERSATIONAL HITL CHECKPOINT
        # ══════════════════════════════════════════════════════════════
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("🤖  **LAYER 4 — CONVERSATIONAL HITL CHECKPOINT**")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")

        # Generate contextual HITL prompt based on current step and results
        hitl_prompt = self._generate_hitl_prompt(result, step_num, total_steps, next_step_description, primary_disease)
        lines.append(hitl_prompt)

        return "\n".join(lines)

    def _generate_hitl_prompt(self, result: dict, step_num: int, total_steps: int, next_step_desc: str, disease: str) -> str:
        """Generate a contextual HITL prompt based on current results."""
        data_source = result.get("data_source", "")
        count = result.get("count", 0)

        if data_source == "clingen":
            definitive = len(result.get("definitive", []))
            if count == 0:
                return f"🤖 **QueryQuest:** I couldn't find any gene-disease associations for {disease} in ClinGen. Would you like me to proceed with scanning recent literature and preprints to look for emerging genetic evidence?"
            elif definitive == 0:
                return f"🤖 **QueryQuest:** I found {count} gene-disease association(s) for {disease}, but no definitively validated genes. Would you like me to proceed with scanning recent literature and preprints to look for any new evidence on potential genetic factors?\n\n_(Type 'yes', 'proceed', or ask a question about genes)_"
            else:
                genes = ", ".join([g.get("Gene_Symbol", "") for g in result.get("definitive", [])[:3]])
                return f"🤖 **QueryQuest:** Great news! I found {definitive} definitively validated gene(s) for {disease}: {genes}. Would you like me to search for recent literature and preprints focusing on these genes?\n\n_(Type 'yes' to proceed, or specify which genes to focus on)_"

        elif data_source == "biorxiv" or data_source == "pubmedqa":
            # Get themes from papers
            themes = []
            for paper in result.get("data", [])[:3]:
                title = paper.get("Title", paper.get("title", paper.get("Question", "")))
                if title:
                    themes.append(title[:40])

            if count == 0:
                return f"🤖 **QueryQuest:** I didn't find any recent preprints or Q&A pairs matching {disease}. Would you like me to broaden the search terms, or shall I proceed to identify active researchers?\n\n_(Type 'yes' to find researchers, or suggest alternative search terms)_"
            else:
                themes_str = ", ".join(themes[:3]) if themes else "various topics"
                return f"🤖 **QueryQuest:** I've scanned recent literature and preprints for {disease} and found {count} results including topics on: {themes_str}. Shall I now identify active researchers and knowledge connections related to these specific findings or {disease} in general?\n\n_(Type 'yes' to proceed, or ask about specific findings)_"

        elif data_source == "openalex":
            researchers = result.get("researchers", result.get("data", []))
            orkg_count = result.get("orkg_count", 0)
            orkg_data = result.get("orkg_data", [])

            if count == 0 and orkg_count == 0:
                return f"🤖 **QueryQuest:** I couldn't find researchers or knowledge connections for {disease}. Would you like me to broaden the search, or shall we proceed to generate a summary report with the data we have?\n\n_(Type 'yes' for report, or suggest alternative search terms)_"
            else:
                top_names = ", ".join([r.get("display_name", "Unknown")[:20] for r in researchers[:3]]) if researchers else "None found"

                # Extract key concepts from ORKG
                key_concepts = []
                for item in orkg_data[:5]:
                    obj = item.get("object", "")
                    if isinstance(obj, str) and len(obj) > 10:
                        key_concepts.append(obj[:40])

                # This is the final step - show synthesis prompt
                lines = []
                lines.append("══════════════════════════════════════════════════════════════")
                lines.append("📝  **LAYER 5 — SYNTHESIS READY**")
                lines.append("══════════════════════════════════════════════════════════════")
                lines.append("")
                lines.append("📊 **Research Summary:**")
                lines.append(f"   • **Researchers:** {count} found (Top: {top_names})")
                lines.append(f"   • **Knowledge Graph:** {orkg_count} ORKG connections")
                if key_concepts:
                    lines.append("")
                    lines.append("🧠 **Key Concepts from Knowledge Graph:**")
                    for i, concept in enumerate(key_concepts[:3], 1):
                        lines.append(f"   {i}. {concept}")
                lines.append("")
                lines.append("🤖 **QueryQuest:** All research steps are complete!")
                lines.append("")
                lines.append("Would you like me to **generate a comprehensive research report**")
                lines.append("including Knowledge Graph analysis?")
                lines.append("")
                lines.append("_(Type 'yes', 'report', or 'synthesize' to generate the report)_")
                return "\n".join(lines)

        # Default prompt
        if step_num < total_steps:
            return f"🤖 **QueryQuest:** Step {step_num} complete with {count} results. Would you like me to proceed with the next step: {next_step_desc}?\n\n_(Type 'yes' to proceed, or ask a question)_"
        else:
            return f"🤖 **QueryQuest:** All {total_steps} research steps are complete! Would you like me to generate a comprehensive research report?\n\n_(Type 'yes' for report, or 'validate' to check data quality)_"

    def _format_all_complete_message(self) -> str:
        """Format message when all tasks are complete."""
        lines = []
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("✅  **ALL RESEARCH STEPS COMPLETE**")
        lines.append("══════════════════════════════════════════════════════════════")
        lines.append("")

        # Summary of collected data
        lines.append("📊 **Data Summary:**")
        for task_id, data in self.memory.collected_data.items():
            source = data.get("data_source", "unknown")
            count = data.get("count", 0)
            lines.append(f"   • {task_id}: {count} records from {source}")

        lines.append("")
        lines.append("🤖 **QueryQuest:** All research tasks have been completed successfully!")
        lines.append("")
        lines.append("**What would you like to do next?**")
        lines.append("• Type **'report'** or **'synthesize'** to generate a comprehensive research brief")
        lines.append("• Type **'validate'** to check the data for any conflicts or quality issues")
        lines.append("• Ask any follow-up questions about the findings")

        return "\n".join(lines)

    def _format_validation_message(self, validation: dict) -> str:
        """Format validation results as a message."""
        msg = "## Data Validation\n\n"
        msg += f"**Status:** {'✅ Passed' if validation.get('validation_passed') else '⚠️ Issues Found'}\n\n"

        if validation.get("issues"):
            msg += "### Issues Detected:\n"
            for issue in validation["issues"]:
                msg += f"- **{issue['type'].title()}** ({issue['severity']}): {issue['description']}\n"
                msg += f"  Recommendation: {issue['recommendation']}\n"

        if validation.get("requires_human_review"):
            msg += f"\n**Human Review Required:** {validation.get('review_reason', 'Please review the issues above')}\n"
            msg += "\nHow would you like to proceed?"

        return msg

    def _save_state(self):
        """Save session state to Firestore."""
        if self.db:
            try:
                doc_ref = self.db.collection("agent_sessions").document(self.session_id)
                doc_ref.set(self.memory.to_dict())
            except Exception as e:
                logger.warning(f"Failed to save state: {e}")

    def load_state(self, session_id: str) -> bool:
        """Load session state from Firestore."""
        if self.db:
            try:
                doc_ref = self.db.collection("agent_sessions").document(session_id)
                doc = doc_ref.get()
                if doc.exists:
                    self.memory = ConversationMemory.from_dict(doc.to_dict())
                    self.session_id = session_id
                    return True
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

        return False


# Convenience function for quick testing
def chat(message: str, session_id: str = None) -> dict:
    """Quick chat function for testing."""
    orchestrator = MultiAgentOrchestrator(session_id)
    return orchestrator.process_message(message)


if __name__ == "__main__":
    # Test the multi-agent system
    import sys
    logging.basicConfig(level=logging.INFO)

    query = sys.argv[1] if len(sys.argv) > 1 else "Find experts in IPF progression"

    print(f"\n🔬 Multi-Agent Co-Investigator\n")
    print(f"Query: {query}\n")

    orchestrator = MultiAgentOrchestrator()
    response = orchestrator.process_message(query)

    print(f"Agent Used: {response['agent_used']}")
    print(f"\n{response['message']}")
