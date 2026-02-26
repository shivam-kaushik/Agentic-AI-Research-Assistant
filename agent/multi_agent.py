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

    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_summary(self) -> str:
        """Get a summary of current context for agents."""
        summary = f"""
## Current Session Context

**Session ID:** {self.session_id}

**Current Plan:** {json.dumps(self.current_plan, indent=2) if self.current_plan else 'No plan yet'}

**Pending Tasks:** {self.pending_tasks}

**Completed Tasks:** {self.completed_tasks}

**Data Collected:** {list(self.collected_data.keys())}

**Recent Messages:**
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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        return cls(**data)


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

Available Agents:
- PLANNER: Creates/modifies research plans, breaks down complex queries
- RESEARCHER: Queries BigQuery databases (ClinGen, CIViC, Reactome, STRING)
- VALIDATOR: Checks data for conflicts, contradictions, quality issues
- SYNTHESIZER: Creates final reports, summaries, insights
- CLARIFIER: Asks user for clarification when intent is unclear

Based on the user message and context, respond with a JSON object:
{
    "intent": "description of what user wants",
    "next_agent": "PLANNER|RESEARCHER|VALIDATOR|SYNTHESIZER|CLARIFIER",
    "agent_instructions": "specific instructions for the chosen agent",
    "requires_user_input": true/false,
    "user_prompt": "question to ask user if clarification needed"
}
"""

    def __init__(self):
        super().__init__("Orchestrator")

    def route(self, user_message: str, memory: ConversationMemory) -> dict:
        """Determine which agent should handle the request."""
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
    """Creates and modifies research plans."""

    SYSTEM_PROMPT = """You are the Planner Agent for biomedical research.

Your role is to:
1. Decompose complex research queries into 2-4 executable sub-tasks
2. Determine data sources needed for each task
3. Identify dependencies between tasks
4. Suggest HITL checkpoints where human review is valuable

Available Data Sources:
- clingen: Gene-disease associations, variant pathogenicity
- civic: Clinical evidence for cancer variants
- reactome: Biological pathways, protein mappings
- string: Protein-protein interactions
- openalex: Researcher information, publications (external API)
- pubmed: Latest research abstracts (external API)

Respond with a JSON research plan:
{
    "research_goal": "clear statement of objective",
    "sub_tasks": [
        {
            "task_id": "task_1",
            "description": "what this task does",
            "data_source": "clingen|civic|reactome|string|openalex|pubmed",
            "query_params": {"disease": "...", "gene": "..."},
            "depends_on": [],
            "priority": 1
        }
    ],
    "suggested_hitl_after": "task_id",
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
    """Executes research queries against data sources."""

    def __init__(self):
        super().__init__("Researcher")
        from tools.query_bigquery import (
            query_clingen, query_civic, query_reactome, query_string
        )
        self.query_functions = {
            "clingen": query_clingen,
            "civic": query_civic,
            "reactome": query_reactome,
            "string": query_string,
        }

    def execute_task(self, task: dict, memory: ConversationMemory) -> dict:
        """Execute a single research task."""
        task_id = task.get("task_id", "unknown")
        data_source = task.get("data_source", "")
        query_params = task.get("query_params", {})

        logger.info(f"Researcher executing task: {task_id} on {data_source}")

        result = {
            "task_id": task_id,
            "data_source": data_source,
            "success": False,
            "data": [],
            "count": 0,
        }

        if data_source in self.query_functions:
            try:
                query_result = self.query_functions[data_source](**query_params)
                result["success"] = query_result.get("success", False)
                result["data"] = query_result.get("data", [])
                result["count"] = len(result["data"])

                # Store in memory
                memory.collected_data[task_id] = result
                if task_id in memory.pending_tasks:
                    memory.pending_tasks.remove(task_id)
                memory.completed_tasks.append(task_id)

            except Exception as e:
                logger.error(f"Research task failed: {e}")
                result["error"] = str(e)

        elif data_source == "openalex":
            from tools.search_openalex import search_researchers, search_works
            try:
                if query_params.get("search_type") == "researchers":
                    query_result = search_researchers(**query_params)
                else:
                    query_result = search_works(**query_params)

                result["success"] = query_result.get("success", False)
                result["data"] = query_result.get("data", [])
                result["count"] = len(result["data"])
                memory.collected_data[task_id] = result

            except Exception as e:
                result["error"] = str(e)

        elif data_source == "pubmed":
            from tools.pubmed_entrez import search_and_fetch_pubmed
            try:
                query_result = search_and_fetch_pubmed(
                    query=query_params.get("query", ""),
                    max_results=query_params.get("max_results", 10)
                )
                result["success"] = query_result.get("success", False)
                result["data"] = query_result.get("data", [])
                result["count"] = len(result["data"])
                memory.collected_data[task_id] = result

            except Exception as e:
                result["error"] = str(e)

        return result

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


class SynthesizerAgent(BaseAgent):
    """Creates final reports and summaries."""

    SYSTEM_PROMPT = """You are the Synthesizer Agent for biomedical research.

Your role is to:
1. Compile research findings into clear, structured reports
2. Highlight key discoveries and insights
3. Provide proper citations to data sources
4. Suggest next steps for further research

Create a comprehensive markdown report with:
- Executive Summary
- Research Question
- Methodology (data sources queried)
- Key Findings (organized by topic)
- Data Quality Notes
- Recommendations
- References
"""

    def __init__(self):
        super().__init__("Synthesizer")

    def synthesize(self, memory: ConversationMemory) -> str:
        """Create a synthesis report."""
        prompt = f"""{self.SYSTEM_PROMPT}

## Research Context
{memory.get_context_summary()}

## Research Plan
{json.dumps(memory.current_plan, indent=2) if memory.current_plan else 'No formal plan'}

## Collected Data
{json.dumps({k: {"source": v.get("data_source"), "count": v.get("count"), "sample": v.get("data", [])[:5]} for k, v in memory.collected_data.items()}, indent=2, default=str)}

## Conversation History
{chr(10).join([f"- {m['role']}: {m['content'][:200]}" for m in memory.messages[-10:]])}

Create a comprehensive research report in markdown format:"""

        return self._generate(prompt, temperature=0.4)


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
            if status_callback: status_callback("Executing database queries across connected sources (ClinGen, Reactome, etc.)...")
            results = self.researcher.execute_all_pending(self.memory)
            if status_callback: status_callback(f"Completed {len(results)} queries. Synthesizing data...")
            response["results"] = results
            response["message"] = self._format_research_message(results)

        elif next_agent == "VALIDATOR":
            if status_callback: status_callback("Validating aggregated evidence for conflicts and correctness...")
            validation = self.validator.validate(self.memory)
            response["validation"] = validation
            response["message"] = self._format_validation_message(validation)
            response["requires_input"] = validation.get("requires_human_review", False)

        elif next_agent == "SYNTHESIZER":
            if status_callback: status_callback("Synthesizing findings into a structured report...")
            report = self.synthesizer.synthesize(self.memory)
            response["report"] = report
            response["message"] = report

        elif next_agent == "CLARIFIER":
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
        """Format plan as a user-friendly message."""
        if "error" in plan:
            return f"I encountered an issue creating the plan: {plan['error']}"

        msg = f"## Research Plan: {plan.get('research_goal', 'Research')}\n\n"
        msg += "### Sub-tasks:\n"

        for task in plan.get("sub_tasks", []):
            status = "✅" if task["task_id"] in self.memory.completed_tasks else "⏳"
            msg += f"{status} **{task['task_id']}**: {task['description']}\n"
            msg += f"   - Source: {task['data_source']}\n"

        msg += f"\n**Complexity:** {plan.get('estimated_complexity', 'unknown')}\n"
        msg += "\nWould you like me to proceed with executing this plan?"

        return msg

    def _format_research_message(self, results: list) -> str:
        """Format research results as a message."""
        msg = "## Research Results\n\n"

        for result in results:
            status = "✅" if result.get("success") else "❌"
            msg += f"{status} **{result['task_id']}** ({result['data_source']}): "
            msg += f"{result.get('count', 0)} records found\n"

            if result.get("error"):
                msg += f"   Error: {result['error']}\n"

        msg += "\nWould you like me to validate this data or proceed to synthesis?"

        return msg

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
