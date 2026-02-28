# BenchSci Co-Investigator: Technical Deep Dive

**A Comprehensive Analysis of Architecture, Design Decisions, and Implementation Rationale**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Multi-Agent Orchestration Design](#3-multi-agent-orchestration-design)
4. [Agent Composition & Rationale](#4-agent-composition--rationale)
5. [LangGraph Workflow Design](#5-langgraph-workflow-design)
6. [Memory & State Management](#6-memory--state-management)
7. [RAG Implementation](#7-rag-implementation)
8. [Tool & Data Source Design](#8-tool--data-source-design)
9. [HITL (Human-in-the-Loop) Design](#9-hitl-human-in-the-loop-design)
10. [GCP Service Selection Rationale](#10-gcp-service-selection-rationale)
11. [Search & Filtering Strategy](#11-search--filtering-strategy)
12. [Performance & Scalability Considerations](#12-performance--scalability-considerations)
13. [Technical Questions & Trade-offs](#13-technical-questions--trade-offs)

---

## 1. Executive Summary

### What is Co-Investigator?

Co-Investigator is an **agentic AI research assistant** that helps biomedical scientists discover gene-disease associations, find relevant literature, and identify key researchers. It uses a **multi-agent architecture** with **7 specialized agents** orchestrated via LangGraph, integrated with **5 biomedical data sources**, and features **RAG-based cross-questioning** capabilities.

### Key Technical Innovations

| Innovation | Description | Why It Matters |
|------------|-------------|----------------|
| **Cyclical Execution Model** | Not linear - supports dynamic fact-checking and plan-correction loops | Allows mid-execution questions without losing state |
| **SMART Adaptive Checkpoints** | AI-generated context-aware options (not generic yes/no) | Better user decisions based on current research context |
| **QueryQuest Structured Extraction** | Enforces disease_variants, gene_variants, topic_keywords | Prevents false positives from abbreviations, improves search quality |
| **Dual Search Strategy** | Fuzzy matching + Gemini filtering | Handles biomedical terminology variations |
| **Vector-based Context Retrieval** | Semantic search on conversation + findings | Enables contextual Q&A during execution |

---

## 2. System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                 │
│  ┌────────────────┐ ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐    │
│  │ Prompt Input   │ │ HITL Panel  │ │ Data Display │ │ Execution Graph │    │
│  │ (Streamlit)    │ │ (Options)   │ │ (Results)    │ │ (Progress)      │    │
│  └────────────────┘ └─────────────┘ └──────────────┘ └─────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER (multi_agent.py)                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    ORCHESTRATOR AGENT                                │    │
│  │  • Deterministic pre-routing (execution commands)                    │    │
│  │  • Question detection (14 indicators)                                │    │
│  │  • LLM-based intent classification                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                       │                                      │
│           ┌───────────────────────────┼───────────────────────────┐         │
│           ▼                           ▼                           ▼         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   PLANNER   │  │ RESEARCHER  │  │  VALIDATOR  │  │ SYNTHESIZER │        │
│  │ (Planning)  │  │ (Execution) │  │ (Conflicts) │  │  (Reports)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│           │                           │                                      │
│           ▼                           ▼                                      │
│  ┌─────────────┐  ┌─────────────┐                                           │
│  │  CLARIFIER  │  │  FOLLOW-UP  │  ← RAG-enabled Q&A                        │
│  │ (Mid-exec)  │  │ (Post-brief)│                                           │
│  └─────────────┘  └─────────────┘                                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEMORY & CONTEXT LAYER                                │
│  ┌────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐   │
│  │ ConversationMemory │  │ VectorStoreManager   │  │ AgentState         │   │
│  │ • Session context  │  │ • Embeddings         │  │ • Execution state  │   │
│  │ • Plan tracking    │  │ • Semantic search    │  │ • Task results     │   │
│  │ • Pause/Resume     │  │ • RAG context        │  │ • HITL checkpoints │   │
│  └────────────────────┘  └──────────────────────┘  └────────────────────┘   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOOLS LAYER                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ DATA LOADERS (GCS)                                                    │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │   │
│  │ │ ClinGen     │ │ PubMedQA    │ │ BioRxiv     │ │ ORKG        │      │   │
│  │ │ (CSV/~10K)  │ │ (JSON/~1K)  │ │ (JSON/~50K) │ │ (NT/~50K)   │      │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ API CLIENTS                                                           │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                      │   │
│  │ │ OpenAlex    │ │ PubMed      │ │ BigQuery    │                      │   │
│  │ │ (REST API)  │ │ (Entrez)    │ │ (SQL)       │                      │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ SEARCH UTILITIES                                                      │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                      │   │
│  │ │smart_search │ │gemini_filter│ │ Fuzzy Match │                      │   │
│  │ │(thefuzz)    │ │ (LLM)       │ │ (Threshold) │                      │   │
│  │ └─────────────┘ └─────────────┘ └─────────────┘                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PERSISTENCE LAYER (GCP)                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Firestore       │  │ Cloud Storage   │  │ BigQuery        │              │
│  │ • Sessions      │  │ • ClinGen CSV   │  │ • Structured    │              │
│  │ • HITL states   │  │ • PubMedQA JSON │  │   queries       │              │
│  │ • Embeddings    │  │ • bioRxiv JSON  │  │ • Analytics     │              │
│  │ • Findings      │  │ • ORKG N-Triples│  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Layer | Design Choice | Rationale |
|-------|---------------|-----------|
| **Presentation** | Streamlit | Rapid prototyping, built-in state management, easy deployment to Cloud Run |
| **Orchestration** | Class-based agents + LangGraph | Flexibility for complex routing + state machine benefits |
| **Memory** | Dual memory (in-process + vector) | Fast access for routing + semantic search for RAG |
| **Tools** | Pandas-based loaders | Familiar data science patterns, efficient for tabular biomedical data |
| **Persistence** | Firestore + GCS | Serverless, auto-scaling, native GCP integration |

---

## 3. Multi-Agent Orchestration Design

### Why Multi-Agent Over Single Agent?

**Problem**: A single agent handling planning, execution, validation, and synthesis becomes:
- Prompt bloat (too many responsibilities in one prompt)
- Inconsistent behavior (competing objectives)
- Hard to debug (unclear which "mode" caused an issue)

**Solution**: Specialized agents with single responsibilities

```python
# Each agent has ONE job
class PlannerAgent:      # Creates research plans
class ResearcherAgent:   # Executes queries against datasets
class ValidatorAgent:    # Checks data quality
class SynthesizerAgent:  # Generates reports
class ClarifierAgent:    # Answers mid-execution questions
class FollowUpAgent:     # Post-brief Q&A
```

### Orchestrator Routing Strategy

The Orchestrator uses a **three-tier routing strategy**:

```
Tier 1: DETERMINISTIC PRE-ROUTING
├── Execution keywords ("yes", "proceed", "continue") → RESEARCHER
├── Question during active execution → CLARIFIER (pause execution)
└── Resume commands after question → RESEARCHER

Tier 2: LLM-BASED CLASSIFICATION
├── New research queries → PLANNER
├── Validation requests → VALIDATOR
├── Report requests → SYNTHESIZER
└── Plan modification → PLANNER

Tier 3: FALLBACK
└── Unclear intent → CLARIFIER (ask for clarification)
```

**Why Deterministic First?**

```python
# Deterministic routing for execution commands (multi_agent.py:293-327)
execution_keywords = [
    "yes", "proceed", "continue", "execute", "run", "start", "go ahead",
    "do it", "next", "ok", "sure", "go", "yeah", "yep", "y"
]

# This bypasses LLM for clear action keywords
# Reason: LLM sometimes misroutes "proceed" as a new query about procedures
# Cost: ~0ms vs ~500ms for LLM routing
# Accuracy: 100% vs ~95% for common commands
```

### Cross-Questioning Detection

**Problem**: User asks a question mid-execution, but system continues executing

**Solution**: Question detection with execution pause

```python
# 14 question indicators (multi_agent.py:262-268)
question_indicators = [
    "what", "why", "how", "when", "who", "where", "which", "explain",
    "tell me", "can you", "could you", "show me", "describe", "clarify"
]

# Check if execution is in progress
has_active_execution = (
    memory.current_plan is not None and
    len(memory.pending_tasks) > 0 and
    not memory.execution_paused
)

# Pause and route to CLARIFIER
if is_question and has_active_execution:
    memory.pause_execution()  # Save state
    return {"next_agent": "CLARIFIER"}
```

**Why This Matters**: Scientists often need clarification during research. Without this, they'd have to restart the entire workflow.

---

## 4. Agent Composition & Rationale

### Agent 1: Orchestrator

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-flash | Fast response for routing decisions |
| **Temperature** | 0.1 | Low creativity needed; deterministic routing preferred |
| **Memory** | ConversationMemory | Needs full session context for routing |
| **Tools** | None | Pure routing logic; no data access |

**Design Decision**: Use deterministic pre-routing before LLM

```python
# Why? LLM can misinterpret "proceed" as asking about procedures
# Result: 3x faster for common commands, 100% accuracy
```

---

### Agent 2: Planner

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-flash | Good at structured JSON output |
| **Temperature** | 0.2 | Slightly creative for query decomposition |
| **Memory** | Writes to memory.current_plan | Central plan storage |
| **Tools** | None | Pure planning logic |
| **Output** | JSON with exactly 3 tasks | QueryQuest architecture requirement |

**Design Decision**: Fixed 3-task structure

```python
# Always 3 tasks:
# 1. Gene & Disease Biology (ClinGen)
# 2. Research Literature (PubMedQA, bioRxiv)
# 3. People & Knowledge (OpenAlex, ORKG)

# Why fixed structure?
# - Predictable execution flow
# - Consistent HITL checkpoint placement
# - Easier progress tracking in UI
# - Matches biomedical research workflow
```

**Design Decision**: Blocked abbreviations

```python
BLOCKED_ABBREVIATIONS = {
    "AD", "PD", "MS", "ALS", "HD", "CF", "DMD", "SMA", "FA",
    "IPF", "COPD", "CKD", "CHF", "MI", "DVT", ...
}

# Why?
# - "AD" matches 1000s of records (too generic)
# - "Alzheimer's Disease" matches ~50 specific records
# - Minimum 8 characters for disease names
```

**QueryQuest Structured Extraction**:

```json
{
  "disease_variants": ["Idiopathic Pulmonary Fibrosis", "IPF"],
  "gene_variants": ["TERT", "TERC", "MUC5B"],
  "topic_keywords": ["fibrosis process", "extracellular matrix"],
  "researcher_search_query": "pulmonary fibrosis",
  "disease_category": "complex"
}
```

| Field | Rules | Rationale |
|-------|-------|-----------|
| disease_variants | Min 8 chars, no abbreviations | Prevents false positives |
| gene_variants | Only if explicitly mentioned | Avoids hallucinated genes |
| topic_keywords | Molecular/biological terms | Better literature search |
| researcher_search_query | 2-3 words, no "expert/researcher" | OpenAlex API optimization |
| disease_category | genetic/complex/neurological/cancer/other | Dataset selection hint |

---

### Agent 3: Researcher

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-flash | Balance of speed and quality |
| **Temperature** | 0.3 | Some creativity for search strategies |
| **Memory** | Updates memory.collected_data, stores to VectorStore | Results need persistence |
| **Tools** | ClinGenLoader, PubMedQALoader, BioRxivLoader, ORKGLoader, OpenAlexClient |

**Design Decision**: Step-by-step execution with HITL

```python
def execute_next_task(self, memory, status_callback, vector_store):
    # Execute ONE task at a time
    # Wait for user confirmation before next

    # Why?
    # 1. Scientists want to review intermediate results
    # 2. Allows plan modification based on findings
    # 3. Prevents wasted computation on wrong path
    # 4. Builds trust through transparency
```

**Design Decision**: Dual search strategy (Fuzzy + Gemini)

```python
# Step 1: Fuzzy matching with thefuzz
matches = smart_search(df, "Disease_Label", terms, threshold=85)

# Step 2: Gemini filtering for relevance
if not matches.empty and primary_disease:
    matches = gemini_filter(matches, "Disease_Label", primary_disease, max_results=30)

# Why both?
# - Fuzzy alone: catches variations but includes false positives
# - Gemini alone: accurate but expensive ($0.0001/call)
# - Combined: fuzzy reduces dataset, Gemini ensures relevance
```

---

### Agent 4: Validator (Conflict Detector)

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-flash | Consistent with other agents |
| **Temperature** | 0.1 | Deterministic conflict detection |
| **Memory** | Reads results, writes conflicts | Analysis only |
| **Tools** | None | Pure analysis logic |

**Conflict Types**:

| Type | Detection Method | Example |
|------|------------------|---------|
| contradiction | LLM comparison across sources | Gene classified as both pathogenic and benign |
| outdated | Date field analysis | Classification from 2015 when newer exists |
| low_confidence | Evidence level checking | "Limited" classifications flagged |
| missing | Expected field validation | No mode of inheritance specified |
| quality | Data integrity rules | Invalid gene symbols |

**Design Decision**: Rule-based fallback

```python
def _rule_based_conflict_detection(results):
    # Used when LLM parsing fails
    # Checks: empty results, query failures, evidence level conflicts

    # Why?
    # - LLM output is JSON; parsing can fail
    # - Basic issues can be caught programmatically
    # - Ensures HITL triggers even on LLM failure
```

---

### Agent 5: HITL Node

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-pro | Higher quality for option generation |
| **Temperature** | 0.3 | Creative but grounded options |
| **Memory** | Persists to Firestore | Checkpoint recovery |
| **Tools** | Firestore persistence | State durability |

**Design Decision**: SMART Adaptive Options (not generic yes/no)

```python
# Generic options (BAD):
# [Yes] [No] [Skip]

# SMART options (GOOD):
# [1] Full literature search (all 4 genes)
# [2] Narrow to definitive genes only (TERT, TERC)
# [3] Skip literature, find researchers
# [4] Export current findings

# Why SMART?
# - Scientists make better decisions with context
# - Options reflect actual data found
# - Can modify plan based on selection
```

**Design Decision**: Use gemini-2.5-pro for options

```python
# Why Pro instead of Flash?
# - Options shown directly to users
# - Quality matters more than speed
# - Generated once per checkpoint (~5 per session)
# - Cost difference: ~$0.01 vs ~$0.001 per checkpoint
```

---

### Agent 6: Synthesizer

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-flash | Good at long-form generation |
| **Temperature** | 0.3 | Readable but factual reports |
| **Memory** | Reads all *_results from state | Comprehensive synthesis |
| **Tools** | File export (outputs/) | Deliverable artifact |

**Report Structure**:

```markdown
1. Executive Summary (3-4 sentences)
2. Research Methodology (datasets, search terms)
3. Gene-Disease Findings (ClinGen classifications)
4. Literature Insights (PubMedQA, bioRxiv)
5. Key Researchers (OpenAlex H-index, affiliations)
6. Knowledge Graph Connections (ORKG)
7. Recommendations (next steps)
8. Data Sources & Citations
```

**Design Decision**: Automatic markdown export

```python
def _export_to_file(report, session_id):
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_brief_{session_id}_{timestamp}.md"

    # Why auto-export?
    # - Scientists need shareable artifacts
    # - Markdown renders well in most tools
    # - Timestamped for version tracking
```

---

### Agent 7: Clarifier

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-flash | Fast responses for Q&A |
| **Temperature** | 0.3 | Conversational but accurate |
| **Memory** | VectorStoreManager (semantic search) | RAG context retrieval |
| **Tools** | semantic_search_messages, semantic_search_findings |

**RAG Pipeline**:

```
User Question
     │
     ▼
┌─────────────────────┐
│ Generate Embedding  │ ← textembedding-gecko@003
│ for user question   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Semantic Search     │
│ • Messages (top 3)  │ ← Cosine similarity
│ • Findings (top 5)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Format Context      │ ← Max 2000 tokens
│ (markdown sections) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ LLM Answer          │ ← "Answer ONLY from context"
│ + Context           │
└─────────────────────┘
```

---

### Agent 8: FollowUp

| Component | Value | Rationale |
|-----------|-------|-----------|
| **Model** | gemini-2.5-flash | Consistent with Clarifier |
| **Temperature** | 0.2 | More conservative for post-brief Q&A |
| **Memory** | qa_history (last 5 exchanges) | Conversational context |
| **Tools** | None (uses pre-collected state) |

**Design Decision**: Lower temperature than Clarifier

```python
# Clarifier: 0.3 (during execution, more exploratory)
# FollowUp: 0.2 (after report, more factual)

# Why?
# - Post-brief questions should reference the report
# - Less room for speculation
# - Scientists expect consistent answers
```

---

## 5. LangGraph Workflow Design

### State Machine Flow

```
                    START
                      │
                      ▼
              ┌─────────────┐
              │   Planner   │ ── Creates 3-task plan
              └──────┬──────┘
                     │
                     ▼
          ┌───────────────────┐
          │ Internal Retriever│ ── Queries ClinGen (GCS/BQ)
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Conflict Detector │ ── Analyzes for issues
          └─────────┬─────────┘
                    │
              ┌─────┴─────┐
              │ requires  │
              │   HITL?   │
              └─────┬─────┘
              yes/  │  \no
                 ▼      ▼
          ┌───────────┐   │
          │   HITL    │◄──┘ ── SMART adaptive checkpoint
          │  (pause)  │
          └─────┬─────┘
                │ user feedback
                ▼
          ┌───────────────────┐
          │ External API Call │ ── OpenAlex, PubMed APIs
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │    Synthesizer    │ ── Generates report
          └─────────┬─────────┘
                    │
                    ▼
                   END
                    │
                    ▼
          ┌───────────────────┐
          │  FollowUp Q&A     │ ── Post-brief conversation
          └───────────────────┘
```

### AgentState Schema

```python
class AgentState(TypedDict):
    # Session
    session_id: str
    user_query: str

    # Planning
    plan: dict                    # ResearchPlan
    current_task_index: int

    # QueryQuest Structured Extraction
    disease_variants: list[str]   # Full disease names
    gene_variants: list[str]      # HGNC symbols
    topic_keywords: list[str]     # Molecular terms
    researcher_search_query: str  # Clean 2-3 words
    disease_category: str         # genetic/complex/etc.

    # Category-based Results
    clingen_results: dict         # Gene-disease associations
    pubmedqa_results: dict        # Q&A pairs
    biorxiv_results: dict         # Preprints
    orkg_results: dict            # Knowledge graph
    researcher_results: dict      # OpenAlex researchers

    # HITL State
    hitl_checkpoint: dict         # Checkpoint data
    hitl_pending: bool            # Waiting for user
    human_feedback: str           # User's response

    # Output
    final_report: str             # Markdown report
    export_path: str              # File path

    # Tracking
    execution_history: list[str]  # Node traversal
    task_history: list[dict]      # For rollback
    error: str                    # Error message
```

### Why LangGraph?

| Feature | Benefit | Alternative Considered |
|---------|---------|----------------------|
| State machine | Explicit execution flow | Custom state machine (more code) |
| Checkpointing | Built-in state persistence | Manual Firestore writes |
| Conditional edges | Dynamic routing | if/else chains |
| Streaming | Real-time node execution | Batch processing |
| Memory saver | In-memory checkpoints | Always Firestore (slow) |

---

## 6. Memory & State Management

### Three-Tier Memory Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ TIER 1: In-Process Memory (ConversationMemory)                 │
│ • Fast read/write for routing decisions                        │
│ • Session context, plan tracking, pause/resume                 │
│ • Lost on server restart                                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ TIER 2: Vector Memory (VectorStoreManager)                     │
│ • Semantic search for RAG                                      │
│ • Embeddings stored in Firestore                               │
│ • conversation_history + research_findings collections         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ TIER 3: Persistent Memory (Firestore)                          │
│ • HITL checkpoint recovery                                     │
│ • Session state snapshots                                      │
│ • Cross-restart durability                                     │
└────────────────────────────────────────────────────────────────┘
```

### ConversationMemory Design

```python
@dataclass
class ConversationMemory:
    session_id: str
    messages: list                    # Chat history
    current_plan: dict                # Active research plan
    collected_data: dict              # Results by task_id
    pending_tasks: list               # Tasks to execute
    completed_tasks: list             # Finished tasks

    # Step-by-step execution
    current_step_index: int           # Which step we're on
    awaiting_step_confirmation: bool  # Waiting for user
    last_step_result: dict            # Previous step's result

    # Cross-questioning support
    execution_paused: bool            # Paused for question
    execution_context: dict           # State before pause
    question_mode: bool               # Answering a question
```

**Why separate `execution_paused` and `question_mode`?**

```python
# execution_paused: True when user asks question mid-execution
# question_mode: True while actively answering the question

# Scenario:
# 1. User asks "What is IPF?" during task 2
# 2. execution_paused = True, question_mode = True
# 3. Clarifier answers the question
# 4. question_mode = False (answered)
# 5. execution_paused = True (still paused, waiting for "continue")
# 6. User says "continue"
# 7. execution_paused = False, resume task 2
```

### VectorStoreManager Design

```python
class VectorStoreManager:
    def __init__(self, session_id: str):
        self.embedding_model = TextEmbeddingModel.from_pretrained(
            "textembedding-gecko@003"
        )
        self.db = firestore.Client()

    def store_message(self, role, content, metadata):
        embedding = self.generate_embedding(content)
        doc_data = {
            "session_id": self.session_id,
            "role": role,
            "content": content,
            "embedding": embedding,  # 768-dim vector
            "timestamp": datetime.now()
        }
        self.db.collection("conversation_history").add(doc_data)

    def semantic_search_messages(self, query, top_k=5):
        query_embedding = self.generate_embedding(query)
        # Load all session messages, compute cosine similarity
        # Return top_k most similar
```

**Why Firestore for embeddings instead of Vertex AI Vector Search?**

| Factor | Firestore | Vertex AI Vector Search |
|--------|-----------|------------------------|
| Setup | Zero config | Index creation, endpoint deployment |
| Cost | Pay per operation | Fixed endpoint cost (~$50/month) |
| Scale | 100s of embeddings | Millions of embeddings |
| Latency | ~100ms | ~50ms |
| Use case | Prototype/hackathon | Production at scale |

**Decision**: Firestore for hackathon scope; Vector Search for production.

---

## 7. RAG Implementation

### Embedding Model Selection

| Model | Dimensions | Cost | Rationale |
|-------|------------|------|-----------|
| textembedding-gecko@003 | 768 | $0.025/1K tokens | GCP native, good for biomedical text |

**Why not OpenAI embeddings?**
- GCP-native (same project)
- No API key management
- Consistent billing

### Context Retrieval Strategy

```python
def get_relevant_context(self, query, max_tokens=2000):
    # Search both messages and findings
    relevant_messages = self.semantic_search_messages(query, top_k=3)
    relevant_findings = self.semantic_search_findings(query, top_k=5)

    # Format context
    context = """
    ## Relevant Conversation History
    {messages}

    ## Relevant Research Findings
    {findings}
    """

    # Truncate if needed
    if len(context) > max_tokens * 4:  # ~4 chars per token
        context = context[:max_tokens * 4] + "\n...[truncated]"

    return context
```

**Why top_k=3 for messages, top_k=5 for findings?**

| Collection | top_k | Rationale |
|------------|-------|-----------|
| Messages | 3 | Recent context matters most; older messages less relevant |
| Findings | 5 | Research data is dense; need more context for accurate answers |

### Cosine Similarity Implementation

```python
def _cosine_similarity(self, vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)
```

**Why not use numpy?**
- Single-threaded Streamlit environment
- Small vectors (768 dims)
- Pure Python sufficient for 100s of comparisons

---

## 8. Tool & Data Source Design

### Data Source Selection Rationale

| Source | Data Type | Size | Why Selected |
|--------|-----------|------|--------------|
| **ClinGen** | Gene-disease validity | ~10K records | Gold standard for genetic associations |
| **PubMedQA** | Q&A pairs | ~1K records | Pre-answered biomedical questions |
| **bioRxiv/medRxiv** | Preprints | ~50K records | Cutting-edge research not yet published |
| **ORKG** | Knowledge graph | ~50K triples | Scientific concept mapping |
| **OpenAlex** | Researcher info | API | Real-time researcher discovery |

### ClinGen Loader Design

```python
class ClinGenLoader:
    _df_clingen: Optional[pd.DataFrame] = None  # Class-level cache

    def load_all(self, force_reload=False):
        if self._df_clingen is not None and not force_reload:
            return self._df_clingen.copy()  # Return cached copy

        # Load from GCS
        files = gcs_loader.list_files(self.PREFIX, extension=".csv")
        df_parts = []
        for filepath in files:
            df = self._load_single_file(filepath)
            df_parts.append(df)

        # Combine and deduplicate
        df_combined = pd.concat(df_parts, ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["Gene_Symbol", "Disease_Label"]
        )

        self._df_clingen = df_combined
        return self._df_clingen.copy()
```

**Why class-level caching?**
- ClinGen data is static (~10K records)
- Loading from GCS takes ~5 seconds
- Cache persists across requests in same process
- `.copy()` prevents accidental mutation

**Why load into memory instead of BigQuery?**

| Approach | Latency | Cost | Complexity |
|----------|---------|------|------------|
| BigQuery | ~2s/query | $5/TB scanned | SQL required |
| Pandas in-memory | ~10ms/filter | GCS egress only | Simple Python |

**Decision**: Data is small enough (~10MB) for in-memory; faster and cheaper.

### OpenAlex Client Design

```python
class OpenAlexClient:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, endpoint, params):
        # Retry with exponential backoff
        # Handles rate limits (429) and transient failures
```

**Why exponential backoff?**
- OpenAlex rate limit: 10 req/sec
- Bursts can trigger 429 errors
- Backoff: 2s → 4s → 8s (max 10s)

**Why polite pool?**

```python
if self.email:
    params["mailto"] = self.email
    self.session.headers["User-Agent"] = f"CoInvestigator/1.0 (mailto:{email})"
```
- Polite pool gets higher rate limits
- Email identifies the application
- Required for production use

---

## 9. HITL (Human-in-the-Loop) Design

### SMART Adaptive Checkpoint Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HITL CHECKPOINT                               │
├─────────────────────────────────────────────────────────────────┤
│ Reason: "Found 4 gene-disease associations, review before       │
│          literature search"                                      │
├─────────────────────────────────────────────────────────────────┤
│ Context Summary:                                                 │
│ • Disease: Idiopathic Pulmonary Fibrosis                        │
│ • ClinGen: 4 genes (2 definitive, 2 strong)                     │
│ • Definitive: TERT, TERC                                        │
│ • Pending: Literature search, Researcher identification         │
├─────────────────────────────────────────────────────────────────┤
│ SMART OPTIONS (generated by Gemini Pro):                        │
│                                                                  │
│ [1] Full literature search (all 4 genes)                        │
│     → Search PubMedQA and bioRxiv for all identified genes      │
│                                                                  │
│ [2] Narrow to definitive genes only                             │
│     → Focus search on TERT, TERC for higher confidence          │
│     ⚡ Modifies plan                                             │
│                                                                  │
│ [3] Skip literature, find researchers                           │
│     → Jump to OpenAlex researcher identification                │
│     ⚡ Modifies plan                                             │
│                                                                  │
│ [4] Export current findings                                     │
│     → Generate report with current results                      │
│     ⚡ Modifies plan                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Option Generation Prompt

```python
prompt = f"""
You are helping a scientist review research progress.

Research Query: {context.get('query')}
Disease Focus: {context.get('disease_variants')}

Results So Far:
- ClinGen: {clingen_count} gene-disease links
- Definitive Genes: {definitive_genes}
- PubMedQA: {pubmedqa_count} Q&A pairs
- bioRxiv: {biorxiv_count} preprints

Pending Tasks:
{pending_tasks}

Generate 4-5 SPECIFIC actionable options for the scientist.
NOT generic like "yes/skip/stop".

Return JSON array with: label, action, impact, modifies_plan
"""
```

### Action Application

```python
def _apply_smart_action(state, action):
    if action == "narrow_definitive":
        # Filter to definitive genes only
        definitive_genes = [g for g in clingen_results if g["Classification"] == "Definitive"]
        state["gene_variants"] = [g["Gene_Symbol"] for g in definitive_genes]

    elif action == "skip_to_researchers":
        # Skip literature tasks
        for task in state["plan"]["sub_tasks"]:
            if task["data_source"] in ["pubmedqa", "biorxiv"]:
                task["status"] = "skipped"

    elif action == "stop_and_export":
        # Skip all pending, go to synthesis
        for task in state["plan"]["sub_tasks"]:
            if task["status"] == "pending":
                task["status"] = "skipped"
```

### Firestore Checkpoint Persistence

```python
def _persist_checkpoint_to_firestore(state, checkpoint):
    doc_data = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "session_id": state["session_id"],
        "reason": checkpoint.reason,
        "smart_options": [o.to_dict() for o in checkpoint.smart_options],
        "state_snapshot": {
            "user_query": state["user_query"],
            "plan": state["plan"],
            "results": state["results"],
            "current_task_index": state["current_task_index"],
        },
        "created_at": datetime.now(),
        "user_response": None,  # Filled when user responds
    }

    db.collection("hitl_checkpoints").document(checkpoint.checkpoint_id).set(doc_data)
```

**Why persist state_snapshot?**
- Server may restart between checkpoint and response
- Full state recovery without re-execution
- Audit trail for research reproducibility

---

## 10. GCP Service Selection Rationale

### Service Overview

| Service | Purpose | Why Selected |
|---------|---------|--------------|
| **Vertex AI** | LLM inference (Gemini) | Native GCP, managed endpoints, auto-scaling |
| **Firestore** | State persistence, embeddings | Serverless, NoSQL, real-time listeners |
| **Cloud Storage** | Dataset storage | Cost-effective, high durability |
| **Cloud Run** | Streamlit hosting | Serverless, auto-scaling, HTTPS |
| **BigQuery** | Structured queries (optional) | SQL interface for complex analytics |

### Vertex AI Model Selection

| Model | Use Case | Temperature | Max Tokens |
|-------|----------|-------------|------------|
| gemini-2.5-flash | Most agents | 0.1-0.3 | 2048-4096 |
| gemini-2.5-pro | HITL option generation | 0.3 | 2048 |
| textembedding-gecko@003 | RAG embeddings | N/A | N/A |

**Why Gemini over Claude/GPT?**

| Factor | Gemini | Claude | GPT-4 |
|--------|--------|--------|-------|
| GCP Native | ✅ | ❌ | ❌ |
| Billing | Same account | Separate | Separate |
| Latency | ~200ms | ~500ms | ~300ms |
| JSON Mode | Native | Prompting | Native |
| Cost/1K tokens | $0.0001 (Flash) | $0.003 | $0.03 |

### Firestore Schema

```
firestore/
├── agent_sessions/
│   └── {session_id}
│       ├── session_id: string
│       ├── user_query: string
│       ├── status: "active" | "paused" | "completed"
│       ├── current_checkpoint: string (optional)
│       └── updated_at: timestamp
│
├── hitl_checkpoints/
│   └── {checkpoint_id}
│       ├── checkpoint_id: string
│       ├── session_id: string
│       ├── reason: string
│       ├── smart_options: array
│       ├── state_snapshot: map
│       ├── user_response: string (optional)
│       ├── created_at: timestamp
│       └── responded_at: timestamp (optional)
│
├── conversation_history/
│   └── {doc_id}
│       ├── session_id: string
│       ├── role: "user" | "assistant"
│       ├── content: string
│       ├── embedding: array (768 floats)
│       └── timestamp: timestamp
│
└── research_findings/
    └── {doc_id}
        ├── session_id: string
        ├── task_id: string
        ├── data_source: string
        ├── content: string
        ├── structured_data: map
        ├── embedding: array (768 floats)
        └── timestamp: timestamp
```

---

## 11. Search & Filtering Strategy

### Two-Stage Search Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: FUZZY MATCHING (smart_search)                          │
│                                                                  │
│ Input: DataFrame, search terms                                  │
│ Method: thefuzz.partial_ratio                                   │
│ Threshold: 80-95 (configurable)                                 │
│ Output: Candidate matches (100s of records)                     │
│                                                                  │
│ Why fuzzy?                                                       │
│ • "Pulmonary Fibrosis" matches "pulmonary fibrotic disease"    │
│ • "TERT" matches "TERT gene" and "TERT-related"                │
│ • Handles typos and variations                                  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: GEMINI FILTERING (gemini_filter)                       │
│                                                                  │
│ Input: Candidate matches (from Stage 1)                         │
│ Method: LLM relevance judgment                                  │
│ Max results: 10-30                                              │
│ Output: Relevant records only                                   │
│                                                                  │
│ Why Gemini?                                                      │
│ • Understands biomedical context                                │
│ • "IPF" and "Idiopathic Pulmonary Fibrosis" are same           │
│ • Filters out false positives from fuzzy matching              │
└─────────────────────────────────────────────────────────────────┘
```

### Fuzzy Search Implementation

```python
def smart_search(df, column, terms, threshold=90, min_len=8):
    matched_indices = set()

    for term in terms:
        if len(term) < min_len:
            continue  # Skip short terms

        term_lower = term.lower()

        for idx, val in df[column].items():
            val_str = str(val).lower()

            # Exact substring match (fast path)
            if term_lower in val_str:
                matched_indices.add(idx)
                continue

            # Fuzzy partial match (handles variations)
            if fuzz.partial_ratio(term_lower, val_str) >= threshold:
                matched_indices.add(idx)

    return df.loc[list(matched_indices)]
```

**Threshold Tuning**:

| Dataset | Threshold | Rationale |
|---------|-----------|-----------|
| ClinGen Disease | 85 | Disease names have variations |
| ClinGen Gene | 95 | Gene symbols must be exact |
| PubMedQA | 80 | Abstracts have diverse phrasing |
| bioRxiv | 85 | Titles are more standardized |

### Gemini Filter Implementation

```python
def gemini_filter(df, column, topic, max_results=10):
    # Sample first 40 candidates
    sample_texts = [f"Index {i}: {row}" for i, row in enumerate(df[column].head(40))]

    prompt = f"""
    Topic: "{topic}"

    Below is a list of candidate text segments.
    Identify which segments are relevant to the topic.

    {sample_texts}

    Return ONLY a JSON array of relevant indices.
    """

    response = model.generate_content(prompt)
    indices = json.loads(response.text)

    return df.iloc[indices[:max_results]]
```

**Why sample 40?**
- Context window efficiency
- Most relevant records are in top results anyway
- Cost optimization (~$0.0001 per filter call)

---

## 12. Performance & Scalability Considerations

### Current Performance Profile

| Operation | Latency | Bottleneck |
|-----------|---------|------------|
| Orchestrator routing | ~500ms | LLM inference |
| Plan generation | ~1s | LLM inference |
| ClinGen load (first time) | ~5s | GCS download |
| ClinGen load (cached) | ~10ms | Memory |
| Fuzzy search (10K records) | ~200ms | CPU |
| Gemini filter | ~800ms | LLM inference |
| Embedding generation | ~300ms | API call |
| Semantic search (100 docs) | ~50ms | Cosine computation |
| HITL checkpoint persist | ~200ms | Firestore write |

### Scalability Considerations

**Current Limitations**:

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| In-memory datasets | ~100MB max | Use BigQuery for larger |
| Firestore vector search | ~1000 docs/session | Use Vertex AI Vector Search |
| Single-threaded Streamlit | One request at a time | Cloud Run scales horizontally |
| Embedding generation | $0.025/1K tokens | Cache common embeddings |

**Production Recommendations**:

```
1. Move to Vertex AI Vector Search
   - Handles millions of embeddings
   - Sub-10ms latency
   - ~$50/month for basic endpoint

2. Use BigQuery for large datasets
   - Scale to billions of records
   - SQL interface for complex queries
   - ~$5/TB scanned

3. Add Redis caching
   - Cache frequent LLM responses
   - Cache embeddings
   - ~$10/month for basic instance

4. Enable Streamlit server clustering
   - Multiple workers per instance
   - Load balancer in front
   - ~$30/month for 3 instances
```

---

## 13. Technical Questions & Trade-offs

### Architecture Questions

| # | Question | Current Choice | Trade-off |
|---|----------|----------------|-----------|
| 1 | Why two orchestration patterns (LangGraph + class-based)? | Both coexist | Flexibility vs. complexity |
| 2 | Why synchronous Firestore? | Simpler code | Latency vs. complexity |
| 3 | Why Firestore for embeddings? | Zero config | Scale vs. simplicity |
| 4 | Why fixed 3-task planning? | Predictable UX | Flexibility vs. consistency |

### Agent Design Questions

| # | Question | Current Choice | Trade-off |
|---|----------|----------------|-----------|
| 5 | Why deterministic pre-routing? | Speed + accuracy | Maintenance vs. LLM cost |
| 6 | Why gemini-pro for HITL only? | Quality where visible | Cost vs. quality |
| 7 | Why different temperatures? | Task-specific | Tuning vs. simplicity |
| 8 | Why blocked abbreviations? | Prevent false positives | Coverage vs. precision |

### Memory & RAG Questions

| # | Question | Current Choice | Trade-off |
|---|----------|----------------|-----------|
| 9 | Why top_k=3 messages, 5 findings? | Balance relevance/noise | Recall vs. precision |
| 10 | Why 2000 token context limit? | LLM context efficiency | Completeness vs. cost |
| 11 | Why 5-exchange FollowUp history? | Memory efficiency | Context vs. memory |

### Data & Search Questions

| # | Question | Current Choice | Trade-off |
|---|----------|----------------|-----------|
| 12 | Why load entire GCS files? | Simplicity | Memory vs. streaming |
| 13 | Why dual search strategy? | Best of both | Complexity vs. accuracy |
| 14 | Why not BigQuery for all? | Latency for small data | Consistency vs. performance |

---

## Appendix A: Configuration Reference

### Model Configuration

```python
# config/gcp_config.py
@dataclass
class GCPConfig:
    project_id: str = "queryquest-1771952465"
    location: str = "us-central1"
    planner_model: str = "gemini-2.5-flash"
    synthesizer_model: str = "gemini-2.5-flash"
    data_bucket: str = "gs://benchspark-data-1771447466-datasets"
```

### Agent Temperature Settings

| Agent | Temperature | Use Case |
|-------|-------------|----------|
| Orchestrator | 0.1 | Deterministic routing |
| Planner | 0.2 | Structured planning |
| Researcher | 0.3 | Creative search strategies |
| Validator | 0.1 | Deterministic conflict detection |
| HITL | 0.3 | Creative option generation |
| Synthesizer | 0.3 | Readable report generation |
| Clarifier | 0.3 | Conversational Q&A |
| FollowUp | 0.2 | Factual post-brief Q&A |

### Search Thresholds

| Dataset | Column | Threshold | min_len |
|---------|--------|-----------|---------|
| ClinGen | Disease_Label | 85 | 8 |
| ClinGen | Gene_Symbol | 95 | 2 |
| PubMedQA | Question | 80 | 8 |
| PubMedQA | Context | 80 | 8 |
| bioRxiv | Title | 85 | 8 |
| bioRxiv | Abstract | 85 | 8 |

---

## Appendix B: Key Code Locations

| Component | File | Line Range |
|-----------|------|------------|
| Orchestrator Agent | `agent/multi_agent.py` | 178-359 |
| Planner Agent | `agent/multi_agent.py` | 361-527 |
| Researcher Agent | `agent/multi_agent.py` | 529-888 |
| HITL Node | `agent/nodes/hitl.py` | 1-445 |
| Synthesizer Node | `agent/nodes/synthesizer.py` | 1-448 |
| VectorStoreManager | `config/vector_store.py` | 1-314 |
| ClinGenLoader | `tools/clingen_loader.py` | 1-178 |
| OpenAlexClient | `tools/search_openalex.py` | 1-378 |
| Search Utilities | `tools/search_utils.py` | 1-282 |

---

*Document Version: 1.0*
*Generated: 2026-02-27*
*Project: BenchSci Co-Investigator*
