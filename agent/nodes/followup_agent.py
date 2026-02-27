"""
Follow-up Q&A Agent for Co-Investigator Agent

Provides post-brief conversational Q&A session (QueryQuest v9.0 Layer 7).
Answers questions about the generated research brief from collected data.
"""
import logging
from datetime import datetime
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from config.gcp_config import config
from agent.state import AgentState

logger = logging.getLogger(__name__)


class FollowUpAgent:
    """
    Post-brief conversational Q&A agent.

    Answers questions about the research brief using collected data.
    Maintains conversation history for context.
    """

    SYSTEM_PROMPT = """You are a research assistant helping with follow-up questions.

You have access to:
- The complete research brief that was just generated
- All collected data from the research session
- The execution history showing what was done

IMPORTANT RULES:
1. Answer questions ONLY based on the collected data
2. If information was not collected, say so clearly
3. Cite specific data sources when making claims
4. Be concise but comprehensive
5. Use actual numbers and gene names from the data
6. Never make up data that wasn't in the research

If asked about something not covered by the research, explain what was searched
and suggest how to find that information."""

    EXIT_COMMANDS = {"exit", "done", "quit", "stop", "bye", "end", "finish"}

    def __init__(self, state: AgentState, model_name: str = "gemini-2.5-pro"):
        """
        Initialize the follow-up agent with research state.

        Args:
            state: The agent state with all collected data
            model_name: Gemini model to use
        """
        vertexai.init(project=config.project_id, location=config.location)
        self.model = GenerativeModel(model_name, system_instruction=self.SYSTEM_PROMPT)
        self.state = state
        self.research_brief = state.get("final_report", "")
        self.qa_history: list[tuple[str, str]] = []

    def answer_question(self, question: str) -> str:
        """
        Answer a follow-up question about the research.

        Args:
            question: User's question

        Returns:
            Answer based on collected data
        """
        if self.is_exit_command(question):
            return "Q&A session ended. Thank you for using the research assistant!"

        # Build the prompt with context
        prompt = self._build_prompt(question)

        try:
            generation_config = GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024,
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            answer = response.text.strip()

            # Record in history
            self.qa_history.append((question, answer))

            return answer

        except Exception as e:
            logger.error(f"Follow-up agent error: {e}")
            return f"I encountered an error processing your question. Please try rephrasing it. Error: {str(e)[:100]}"

    def _build_prompt(self, question: str) -> str:
        """Build the full prompt with context."""
        data_summary = self._summarize_collected_data()
        qa_history = self._format_qa_history()

        return f"""## Research Brief
{self.research_brief}

## Collected Data Summary
{data_summary}

## Previous Q&A in This Session
{qa_history}

## User's Current Question
{question}

Provide a clear, accurate answer based ONLY on the collected data. Cite sources."""

    def _summarize_collected_data(self) -> str:
        """Summarize the collected data for context."""
        lines = []

        # ClinGen summary
        clingen = self.state.get("clingen_results")
        if clingen and isinstance(clingen, dict):
            total = clingen.get("total", 0)
            definitive = len(clingen.get("definitive", []))
            strong = len(clingen.get("strong", []))
            genes = [r.get("Gene_Symbol", "") for r in clingen.get("definitive", [])[:10]]
            lines.append(f"- ClinGen: {total} gene-disease links ({definitive} definitive, {strong} strong)")
            if genes:
                lines.append(f"  Definitive genes: {', '.join(genes)}")

        # PubMedQA summary
        pubmedqa = self.state.get("pubmedqa_results")
        if pubmedqa and isinstance(pubmedqa, dict):
            total = pubmedqa.get("total", 0)
            yes_count = pubmedqa.get("yes_count", 0)
            no_count = pubmedqa.get("no_count", 0)
            lines.append(f"- PubMedQA: {total} Q&A pairs (YES: {yes_count}, NO: {no_count})")

        # bioRxiv summary
        biorxiv = self.state.get("biorxiv_results")
        if biorxiv and isinstance(biorxiv, dict):
            total = biorxiv.get("total", 0)
            lines.append(f"- bioRxiv/medRxiv: {total} preprints")

        # ORKG summary
        orkg = self.state.get("orkg_results")
        if orkg and isinstance(orkg, dict):
            total = orkg.get("total", 0)
            lines.append(f"- ORKG: {total} knowledge entries")

        # Researcher summary
        researchers = self.state.get("researcher_results")
        if researchers:
            if isinstance(researchers, dict):
                researchers = researchers.get("researchers", [])
            lines.append(f"- OpenAlex: {len(researchers)} researchers identified")
            if researchers:
                top = researchers[0]
                lines.append(f"  Top: {top.get('name', 'Unknown')} (H-index: {top.get('h_index', 'N/A')})")

        return "\n".join(lines) if lines else "No data was collected."

    def _format_qa_history(self) -> str:
        """Format previous Q&A for context."""
        if not self.qa_history:
            return "None yet."

        lines = []
        for q, a in self.qa_history[-5:]:  # Last 5 exchanges
            lines.append(f"Q: {q[:100]}...")
            lines.append(f"A: {a[:200]}...")
            lines.append("")

        return "\n".join(lines)

    def is_exit_command(self, text: str) -> bool:
        """Check if the user wants to exit the Q&A session."""
        return text.lower().strip() in self.EXIT_COMMANDS

    def get_session_summary(self) -> str:
        """Get a summary of the Q&A session."""
        if not self.qa_history:
            return "No questions were asked in this session."

        return f"Answered {len(self.qa_history)} questions in this session."


def create_followup_agent(state: AgentState) -> FollowUpAgent:
    """
    Create a follow-up agent from the current state.

    Args:
        state: Agent state with collected data and research brief

    Returns:
        FollowUpAgent instance ready for Q&A
    """
    return FollowUpAgent(state)


def followup_qa_loop(state: AgentState) -> None:
    """
    Run an interactive Q&A loop (for console/notebook usage).

    Args:
        state: Agent state with collected data
    """
    agent = FollowUpAgent(state)

    print("\n" + "=" * 60)
    print("  FOLLOW-UP Q&A SESSION")
    print("  Ask questions about the research findings.")
    print("  Type 'exit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Your question: ").strip()

            if not question:
                continue

            if agent.is_exit_command(question):
                print("\nQ&A session ended. Thank you!")
                break

            answer = agent.answer_question(question)
            print(f"\nAnswer: {answer}\n")

        except KeyboardInterrupt:
            print("\n\nQ&A session interrupted.")
            break
        except EOFError:
            print("\n\nQ&A session ended.")
            break

    print(f"\n{agent.get_session_summary()}")
