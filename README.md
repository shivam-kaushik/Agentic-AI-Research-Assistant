# BenchSci Co-Investigator

**A Multi-Agent Biomedical Research Assistant with RAG-based Cross-Questioning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI-4285F4.svg)](https://cloud.google.com/vertex-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Co-Investigator is an intelligent biomedical research assistant that uses a multi-agent architecture to help scientists discover gene-disease associations, find relevant literature, and identify key researchers in their field. Built on Google Cloud's Vertex AI with Gemini models, it provides a conversational interface with Human-in-the-Loop (HITL) checkpoints and semantic search capabilities.

### Key Highlights

- **7 Specialized AI Agents** - Orchestrator, Planner, Researcher, Validator, Synthesizer, Clarifier, and FollowUp
- **5 Biomedical Data Sources** - ClinGen, PubMedQA, bioRxiv/medRxiv, ORKG, and OpenAlex
- **RAG-based Cross-Questioning** - Ask questions mid-execution without losing progress
- **Vector Search** - Semantic context retrieval using Vertex AI embeddings
- **Step-by-Step Execution** - Human-in-the-Loop checkpoints for research oversight
- **Automated Reports** - Generate comprehensive markdown research briefs

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Agents](#agents)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Multi-Agent Orchestration

```
User Query --> Orchestrator --> [Planner | Researcher | Validator | Synthesizer | Clarifier]
```

The system intelligently routes user requests to specialized agents:

| Input Type | Route To |
|------------|----------|
| New research queries | Planner Agent |
| "yes", "proceed", "continue" | Researcher Agent |
| "validate", "check" | Validator Agent |
| "generate report", "summarize" | Synthesizer Agent |
| Questions during execution | Clarifier Agent (with RAG) |

### Cross-Questioning with RAG

Ask questions at any point during research execution:

```
User: "Find experts in Idiopathic Pulmonary Fibrosis"
[Task 1 executes...]

User: "What is IPF?"  <-- Question detected!
[Execution pauses, retrieves relevant context via vector search]

Agent: "IPF is a chronic lung disease. Based on our ClinGen search,
        TERT and TERC are definitively associated genes..."

User: "continue"  <-- Resume execution
[Task 2 continues...]
```

### Step-by-Step Execution

Every research task requires user confirmation before proceeding:

```
[Checkmark] Task 1: Query ClinGen for gene-disease associations
   --> Found 4 genes (TERT, TERC, SFTPC, MUC5B)
   --> "Proceed to Task 2?"

[Pending] Task 2: Search bioRxiv for recent preprints
[Pending] Task 3: Query OpenAlex for researchers
```

### Comprehensive Research Reports

Automatically generate structured markdown reports with:
- Executive Summary
- Research Methodology
- Gene-Disease Findings (with classifications)
- Literature Insights
- Key Researchers (with H-index, affiliations)
- Knowledge Graph Connections
- Recommendations
- Data Sources & Citations

---

## Architecture

### System Architecture Diagram

```
+---------------------------------------------------------------------+
|                        STREAMLIT UI LAYER                           |
|   [Prompt Input] [HITL Panel] [Data Display] [Execution Graph]      |
+---------------------------------------------------------------------+
                                    |
                                    v
+---------------------------------------------------------------------+
|                    MULTI-AGENT ORCHESTRATOR                         |
|  +----------+ +-----------+ +-----------+ +------------+            |
|  | PLANNER  | | RESEARCHER| | VALIDATOR | | SYNTHESIZER|            |
|  +----------+ +-----------+ +-----------+ +------------+            |
|  +----------+ +----------+                                          |
|  | CLARIFIER| | FOLLOWUP |  <-- RAG-enabled Q&A                     |
|  +----------+ +----------+                                          |
+---------------------------------------------------------------------+
                                    |
                                    v
+---------------------------------------------------------------------+
|                    MEMORY & VECTOR LAYER                            |
|  [ConversationMemory]  [VectorStoreManager]  [AgentState]           |
+---------------------------------------------------------------------+
                                    |
                                    v
+---------------------------------------------------------------------+
|                         TOOLS LAYER                                 |
|  [ClinGen] [PubMedQA] [bioRxiv] [ORKG] [OpenAlex] [PubMed]         |
+---------------------------------------------------------------------+
                                    |
                                    v
+---------------------------------------------------------------------+
|                      PERSISTENCE LAYER                              |
|  [Firestore Sessions] [Vector Collections] [GCS Datasets]           |
+---------------------------------------------------------------------+
```

### Data Flow

1. **User Input** - User enters research query or command
2. **Orchestrator** - Detects intent and routes to appropriate agent
3. **Agent Execution** - Agent uses tools to execute task
4. **Storage** - Results stored in memory and vector store
5. **Response** - Agent returns response to user
6. **HITL Checkpoint** - Wait for user confirmation before next step

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Google Cloud Platform account with billing enabled
- GCP APIs enabled:
  - Vertex AI API
  - Firestore API
  - Cloud Storage API
  - BigQuery API (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/benchsci-co-investigator.git
cd benchsci-co-investigator
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure GCP Authentication

```bash
# Option 1: Application Default Credentials (recommended for development)
gcloud auth application-default login

# Option 2: Service Account Key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Step 5: Set Environment Variables

Create a `.env` file in the project root:

```env
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_CLOUD_QUOTA_PROJECT=your-project-id

# Optional: OpenAlex API (for polite pool - faster rate limits)
OPENALEX_EMAIL=your-email@example.com
```

### Step 6: Run the Application

```bash
# Run Streamlit UI
streamlit run app/main_v2.py

# Or run with custom port
streamlit run app/main_v2.py --server.port 8501
```

---

## Configuration

### GCP Configuration

Edit `config/gcp_config.py`:

```python
@dataclass
class GCPConfig:
    project_id: str = "your-project-id"
    location: str = "us-central1"
    bigquery_dataset: str = "coinvestigator"
    gcs_bucket: str = "benchspark-data-your-bucket"
```

### Model Configuration

All agents use Gemini 2.5 Flash by default. Model settings:

| Agent | Model | Temperature | Max Tokens |
|-------|-------|-------------|------------|
| Orchestrator | gemini-2.5-flash | 0.1 | 4096 |
| Planner | gemini-2.5-flash | 0.2 | 2048 |
| Researcher | gemini-2.5-flash | 0.3 | 4096 |
| Validator | gemini-2.5-flash | 0.1 | 4096 |
| Synthesizer | gemini-2.5-flash | 0.3 | 4096 |
| Clarifier | gemini-2.5-flash | 0.3 | 4096 |
| FollowUp | gemini-2.5-flash | 0.2 | 1024 |
| HITL Options | gemini-2.5-pro | 0.3 | 2048 |

### Vector Store Configuration

Edit `config/vector_store.py`:

```python
# Embedding model
self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# Context retrieval parameters
message_top_k = 3      # Top 3 relevant messages
findings_top_k = 5     # Top 5 relevant findings
max_tokens = 2000      # Maximum context length
```

---

## Usage

### Basic Research Query

```
User: Find researchers working on Idiopathic Pulmonary Fibrosis

Agent: I'll create a research plan with 3 tasks:
       1. Query ClinGen for gene-disease associations
       2. Search bioRxiv for recent preprints
       3. Query OpenAlex for key researchers

       Would you like to proceed with Step 1?

User: yes

Agent: Found 4 gene-disease associations:
       - TERT (Definitive)
       - TERC (Strong)
       - SFTPC (Moderate)
       - MUC5B (Limited)

       Proceed to Step 2?

User: proceed

[... continues through all tasks ...]

User: generate report

Agent: [Generates comprehensive markdown report]
       Report saved to: outputs/research_brief_session123_20260227.md
```

### Cross-Questioning During Execution

```
User: Research Alzheimer's disease treatments

[Task 1 executing...]

User: What databases are you searching?

Agent: [Execution paused]
       I'm searching the following databases:
       - ClinGen: Gene-disease validity associations
       - bioRxiv: Recent preprint literature
       - PubMedQA: Biomedical Q&A pairs
       - ORKG: Knowledge graph connections
       - OpenAlex: Researcher information

       Type "continue" to resume execution.

User: continue

[Execution resumes...]
```

### Post-Brief Q&A

After report generation, enter conversational Q&A mode:

```
Agent: Report generated! You can now ask follow-up questions.
       Type "exit" when done.

User: Which genes have the strongest evidence?

Agent: Based on the ClinGen data collected:
       - TERT has Definitive classification
       - TERC has Strong classification
       These genes have the most robust evidence for disease association.

User: exit

Agent: Session ended. Thank you for using Co-Investigator!
```

---

## Agents

### 1. Orchestrator Agent

**Purpose:** Central router and decision maker

**Model:** Gemini 2.5 Flash (temperature: 0.1)

**Capabilities:**
- Intent detection from user messages
- Question detection (14 indicators: what, why, how, when, who, where, which, explain, tell me, can you, could you, show me, describe, clarify)
- Agent routing based on context
- Execution pause/resume for cross-questioning

### 2. Planner Agent

**Purpose:** Creates structured research plans

**Model:** Gemini 2.5 Flash (temperature: 0.2)

**Output:** 3-task research plan with:
- disease_variants (full names, min 8 chars)
- gene_variants (HGNC symbols)
- topic_keywords (molecular/biological terms)
- disease_category (genetic/complex/neurological/cancer/other)

**Rules:**
- Always generates exactly 3 tasks
- Blocks common abbreviations (AD, PD, MS, ALS, etc.)

### 3. Researcher Agent

**Purpose:** Executes research queries against datasets

**Model:** Gemini 2.5 Flash (temperature: 0.3)

**Tools:** ClinGenLoader, PubMedQALoader, BioRxivLoader, ORKGLoader, OpenAlexClient, smart_search, gemini_filter

**Features:**
- Step-by-step execution with HITL
- Stores findings in vector store
- Dependency checking between tasks

### 4. Validator Agent

**Purpose:** Validates data quality and detects conflicts

**Model:** Gemini 2.5 Flash (temperature: 0.1)

**Checks:**
- Contradictions between sources
- Low confidence findings
- Missing critical information
- Outdated data
- Data quality issues

### 5. Synthesizer Agent

**Purpose:** Generates comprehensive research reports

**Model:** Gemini 2.5 Flash (temperature: 0.3)

**Report Sections:**
1. Executive Summary
2. Research Methodology
3. Gene-Disease Findings (ClinGen)
4. Literature Insights (bioRxiv, PubMedQA)
5. Key Researchers (OpenAlex)
6. Knowledge Graph Connections (ORKG)
7. Recommendations
8. Data Sources & Citations

### 6. Clarifier Agent

**Purpose:** Context-aware Q&A with RAG

**Model:** Gemini 2.5 Flash (temperature: 0.3)

**Features:**
- Retrieves relevant context from vector store
- Semantic search across conversation history
- Semantic search across research findings
- Answers only from available data

### 7. FollowUp Agent

**Purpose:** Post-brief conversational Q&A

**Model:** Gemini 2.5 Flash (temperature: 0.2)

**Features:**
- Maintains conversation history
- Answers from collected research data
- Cites specific sources
- Exit commands: exit, done, quit, stop, bye

---

## Data Sources

### ClinGen (Gene-Disease Validity)

| Field | Description |
|-------|-------------|
| Gene_Symbol | HGNC gene symbol (e.g., TERT) |
| Disease_Label | Full disease name |
| MOI | Mode of inheritance |
| Classification | Definitive, Strong, Moderate, Limited |

**Source:** `gs://benchspark-data-*/clingen/`
**Records:** ~10,000 gene-disease associations

### PubMedQA (Biomedical Q&A)

| Field | Description |
|-------|-------------|
| ID | PubMed question ID |
| Question | Biomedical question |
| Answer | YES/NO/MAYBE |
| Context | Supporting abstract text |

**Source:** `gs://benchspark-data-*/pubmedqa/`
**Records:** ~1,000 Q&A pairs

### bioRxiv/medRxiv (Preprints)

| Field | Description |
|-------|-------------|
| Title | Preprint title |
| Authors | Author list |
| Date | Publication date |
| Abstract | Full abstract |
| DOI | Digital Object Identifier |
| Source | bioRxiv or medRxiv |

**Source:** `gs://benchspark-data-*/biorxiv-medrxiv/`
**Records:** ~50,000 preprints

### ORKG (Knowledge Graph)

| Field | Description |
|-------|-------------|
| Subject | RDF subject URI |
| Predicate | RDF predicate (rdfs:label) |
| Object | Human-readable label |

**Source:** `gs://benchspark-data-*/orkg/orkg-dump.nt`
**Format:** N-Triples (RDF)
**Records:** ~50,000 searchable labels

### OpenAlex (Researchers)

| Field | Description |
|-------|-------------|
| display_name | Researcher name |
| h_index | H-index score |
| cited_by_count | Total citations |
| works_count | Number of publications |
| last_known_institution | Current affiliation |

**API:** https://api.openalex.org
**Rate Limit:** 10 req/sec (polite pool with email)

---

## Tools

### Data Loaders

```python
from tools.clingen_loader import ClinGenLoader
from tools.pubmedqa_loader import PubMedQALoader
from tools.biorxiv_loader import BioRxivLoader
from tools.orkg_loader import ORKGLoader

# Load ClinGen data
clingen = ClinGenLoader()
results = clingen.search("pulmonary fibrosis")

# Load PubMedQA
pubmedqa = PubMedQALoader()
results = pubmedqa.search("lung disease treatment")
```

### API Clients

```python
from tools.search_openalex import OpenAlexClient
from tools.pubmed_entrez import PubMedClient

# Search researchers
openalex = OpenAlexClient()
researchers = openalex.search_researchers("pulmonary fibrosis", min_h_index=20)

# Search PubMed
pubmed = PubMedClient()
articles = pubmed.search_pubmed("IPF treatment", max_results=50)
```

### Search Utilities

```python
from tools.search_utils import smart_search, gemini_filter

# Fuzzy search
results = smart_search(data, "pulmonary fibrosis", threshold=60)

# AI-powered filtering
filtered = gemini_filter(results, "genes related to lung fibrosis")
```

### Vector Store

```python
from config.vector_store import VectorStoreManager

# Initialize
vector_store = VectorStoreManager(session_id="session_123")

# Store message
vector_store.store_message("user", "Find IPF researchers")

# Store research finding
vector_store.store_research_finding(
    task_id="task_1",
    data_source="clingen",
    content="Found 4 gene-disease associations",
    structured_data={"count": 4}
)

# Semantic search
context = vector_store.get_relevant_context("What genes cause IPF?")
```

---

## API Reference

### MultiAgentOrchestrator

```python
from agent.multi_agent import MultiAgentOrchestrator

# Initialize
orchestrator = MultiAgentOrchestrator(session_id="optional_id")

# Process message
response = orchestrator.process_message(
    user_message="Find researchers in Alzheimer's",
    status_callback=lambda msg: print(msg)  # Optional progress callback
)

# Access memory
memory = orchestrator.memory
print(memory.current_plan)
print(memory.collected_data)
print(memory.pending_tasks)
```

### ConversationMemory

```python
from agent.multi_agent import ConversationMemory

memory = ConversationMemory(session_id="session_123")

# Add message
memory.add_message("user", "Hello")

# Pause execution (for cross-questioning)
memory.pause_execution()

# Resume execution
memory.resume_execution()

# Get context summary
summary = memory.get_context_summary()

# Serialize/deserialize
data = memory.to_dict()
memory = ConversationMemory.from_dict(data)
```

### VectorStoreManager

```python
from config.vector_store import VectorStoreManager

vs = VectorStoreManager(session_id="session_123")

# Store conversation message
doc_id = vs.store_message(
    role="user",
    content="Find IPF researchers",
    metadata={"intent": "research_query"}
)

# Store research finding
doc_id = vs.store_research_finding(
    task_id="task_1",
    data_source="clingen",
    content="Found 4 genes",
    structured_data={"count": 4, "genes": ["TERT", "TERC"]}
)

# Semantic search messages
messages = vs.semantic_search_messages("IPF genes", top_k=5)

# Semantic search findings
findings = vs.semantic_search_findings("gene disease", top_k=10)

# Get formatted context for RAG
context = vs.get_relevant_context("What causes IPF?", max_tokens=2000)

# Clear session data
vs.clear_session_data()
```

---

## Project Structure

```
BenchSci/
|-- agent/                          # Core agent implementations
|   |-- __init__.py
|   |-- multi_agent.py              # Primary multi-agent orchestrator
|   |-- graph.py                    # LangGraph state machine (alternative)
|   |-- state.py                    # State definitions & data classes
|   +-- nodes/                      # LangGraph node implementations
|       |-- __init__.py
|       |-- planner.py
|       |-- internal_retriever.py
|       |-- conflict_detector.py
|       |-- hitl.py
|       |-- external_api_caller.py
|       |-- synthesizer.py
|       |-- followup_agent.py
|       +-- react_executor.py
|
|-- app/                            # Streamlit UI application
|   |-- __init__.py
|   |-- main.py                     # Main Streamlit app
|   |-- main_v2.py                  # Enhanced UI with pause indicator
|   +-- components/                 # UI components
|       |-- __init__.py
|       |-- prompt_input.py
|       |-- execution_graph.py
|       |-- hitl_panel.py
|       |-- message_parser.py
|       +-- data_display.py
|
|-- config/                         # Configuration files
|   |-- __init__.py
|   |-- gcp_config.py               # GCP settings
|   |-- prompts.py                  # LLM prompt templates
|   +-- vector_store.py             # Vector store manager
|
|-- tools/                          # Data loaders & API clients
|   |-- __init__.py
|   |-- clingen_loader.py           # ClinGen gene-disease data
|   |-- pubmedqa_loader.py          # PubMedQA dataset
|   |-- biorxiv_loader.py           # bioRxiv/medRxiv preprints
|   |-- orkg_loader.py              # ORKG knowledge graph
|   |-- gcs_data_loader.py          # GCS base loader
|   |-- search_openalex.py          # OpenAlex API client
|   |-- pubmed_entrez.py            # PubMed Entrez API
|   |-- query_bigquery.py           # BigQuery client
|   |-- search_utils.py             # Search utilities
|   +-- knowledge_graph_viz.py      # Graph visualization
|
|-- visualization/                  # Visualization components
|   |-- __init__.py
|   +-- chart_engine.py             # Chart generation with Gemini
|
|-- local_storage/                  # Local persistence
|   |-- hitl_checkpoints/
|   +-- sessions/
|
|-- outputs/                        # Generated research reports
|
|-- .env                            # Environment variables
|-- Dockerfile                      # Docker configuration
|-- requirements.txt                # Python dependencies
+-- README.md                       # This file
```

---

## Firestore Collections

### agent_sessions
Stores session state snapshots.

### hitl_checkpoints
Stores HITL checkpoint data.

### conversation_history
Stores conversation messages with embeddings for semantic search.

### research_findings
Stores research findings with embeddings for semantic search.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID |
| `GOOGLE_CLOUD_REGION` | No | GCP region (default: us-central1) |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | No | Quota project for API calls |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Path to service account key |
| `OPENALEX_EMAIL` | No | Email for OpenAlex polite pool |

---

## Troubleshooting

### Vector Store Not Available

**Error:** "Vector store unavailable (will use memory-only mode)"

**Solutions:**
1. Enable Vertex AI API: `gcloud services enable aiplatform.googleapis.com`
2. Enable Firestore API: `gcloud services enable firestore.googleapis.com`
3. Check authentication: `gcloud auth application-default login`
4. Verify project ID in `config/gcp_config.py`

### Questions Not Pausing Execution

**Issue:** Questions during execution don't trigger pause

**Solutions:**
1. Ensure question starts with one of 14 indicators (what, why, how, etc.)
2. Check if `memory.pending_tasks` is not empty
3. Verify not already in confirmation mode

### Slow Vector Search

**Issue:** Context retrieval is slow

**Solutions:**
1. Reduce `top_k` values in `get_relevant_context()`
2. Consider migrating to Vertex AI Vector Search for production
3. Add Firestore composite indexes

### API Rate Limits

**Issue:** OpenAlex or PubMed returning 429 errors

**Solutions:**
1. Set `OPENALEX_EMAIL` for polite pool access
2. Implement exponential backoff (already included)
3. Reduce concurrent requests

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit: `git commit -m "Add amazing feature"`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings for public methods
- Keep functions focused and small

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Google Vertex AI](https://cloud.google.com/vertex-ai) - LLM and embedding models
- [ClinGen](https://clinicalgenome.org/) - Gene-disease validity data
- [OpenAlex](https://openalex.org/) - Research publication data
- [bioRxiv](https://www.biorxiv.org/) - Preprint server
- [ORKG](https://orkg.org/) - Open Research Knowledge Graph
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration framework

---

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Built by the BenchSci Team**
