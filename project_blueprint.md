# BenchSpark Hackathon: Project "State-Aware Co-Investigator"

## 1. Challenge Instructions: Challenge 7 - Co-Investigator

**Challenge Statement:**
Build an agentic AI Research Assistant that operates like a high-level research intern. It must decompose complex research requests into multi-step, event-driven workflows, track task state, and interact with users to confirm next steps before proceeding.

**Core Tasks:**
- Accept natural language research requests (e.g., "Find experts in IPF progression").
- Decompose requests into 2-3 executable sub-tasks using a planner pattern.
- Query pre-loaded disease/research datasets in BigQuery.
- Maintain basic task state (completed vs. pending steps).
- Implement at least one Human-in-the-Loop (HITL) checkpoint that pauses for user confirmation.
- Return a structured summary of findings.

**Acceptance Criteria:**
- **Submit a request:** Agent handles a complex multi-step scenario.
- **Verify task decomposition:** Agent breaks down the task visibly.
- **Confirm HITL checkpoint:** Agent pauses to ask for user input before proceeding.
- **Review final output:** Agent returns a structured summary with retrieved data and steps taken.

**Stretch Goals:**
- Integrate external APIs (OpenAlex or PubMed) for researcher identification.
- Add richer task-tracking with full history and rollback capability.
- Generate formatted PDF or markdown research reports.

---

## 2. Our Approach: State-Aware Co-Investigator

**Executive Summary:**
Our solution shifts from linear execution to a cyclical execution model with "Dynamic Fact-Checking" and "Plan-Correction" loops using a LangGraph-based state machine orchestrated via Vertex AI. The core differentiator is its Transparent Execution Graph: the agent explicitly pauses upon identifying data conflicts, updates its state, and prompts the human user for a strategic decision.

**Technical Architecture & Google Cloud Integration:**
- **Orchestration Layer (Vertex AI & LangGraph):** 
  - Uses `LangGraph` inside Vertex AI Workbench to act as a cyclical state machine.
  - **Models:** CLaRA (7B-Instruct) for biomedical reasoning; Gemini 1.5 Pro for task planning and large-scale context synthesis.
- **Memory & Data Layer (BigQuery & Firestore):**
  - **BigQuery:** Executes parameterized SQL for multi-hop entity mapping and traversing the Knowledge Graph.
  - **Firestore (State Tracker):** Serializes the agent's state when hitting a HITL checkpoint. Once the user provides feedback, Firestore triggers the LangGraph process from the paused node.
- **Frontend (Streamlit on Cloud Run):** 
  - Provides a natural language prompt bar, a live execution graph for visibility, an interactive window for HITL prompts, and a final Markdown ledger with explicit citations.

**LangGraph/Agentic Node Workflow:**
1. **The Planner:** Translates the user prompt into a structured JSON execution plan composed of 3 fixed subtasks:
    - Retrieve validated Gene disease association related to the topic.
    - Scan recent literature and pre-prints focusing on the topic.
    - Identify active researchers and knowledge connections.
2. **The Data Executors:** Specialized agents for each dataset (e.g., ClinGen, ORKG) that execute queries and retrieve data.
3. **The Conflict_Detector:** Validates data across sources for recent retractions or toxicity tags.
4. **Smart Adaptive Checkpoint (HITL):** Pauses execution, generating context-aware options based on execution results, and updates the plan based on user choice.
5. **The Synthesizer:** Compiles the aggregated internal and external evidence into a final Markdown report.
6. **The Visualizer:** Generates charts and provides concise, scientific insights for each visualization.
7. **Conversational Follow-up Agent:** Engages with the user in a Q&A session after the brief is generated, allowing for clarifying questions.

**Execution Roadmap (24-Hour Sprint):**
- **Phase 1 (Hours 1-3):** Initialize Vertex AI Workbench, create GCP Service Account, config Cloud Run CI/CD.
- **Phase 2 (Hours 3-7):** Write Python tools (`query_bigquery`, `search_openalex`, `normalize_entity_pubtator`).
- **Phase 3 (Hours 7-14):** Build the LangGraph State Machine, config graph nodes/edges, set up Firestore persistence.
- **Phase 4 (Hours 14-18):** Connect Streamlit UI to the Firestore checkpointer.
- **Phase 5 (Hours 18-24):** Edge-case testing, record the 3-minute demo video, finalize README.md.

---

## 3. Datasets & APIs We Will Use

**GCS Bucket:** `gs://benchspark-data-1771447466-datasets/`

To maximize our "Data Mastery" score, datasets are structured into three intelligence layers:

### Layer A: The Genetic Foundation (Ground Truth)
Goal: Establish what is biologically validated before looking at literature.

- **ClinGen** (GCS: `clingen/*.csv`)
  - Gene-disease validity with classifications (Definitive, Strong, Moderate, Limited, Disputed)
  - Schema: Gene_Symbol, Disease_Label, MOI, Classification, Disease_ID_MONDO
  - First source queried to establish definitive genetic associations
  - Approx. ~10,000 gene-disease entries

### Layer B: Research Literature (Active Research)
Goal: Find the latest research and preprints discussing these genes and diseases.

- **PubMedQA** (GCS: `pubmedqa/*.json`)
  - Labelled Q&A from PubMed abstracts (YES/NO/MAYBE answers)
  - Schema: ID, Question, Answer, Context, Type
  - Enables question-answering capability for biomedical queries
  - Approx. ~1,000 searchable Q&A pairs

- **bioRxiv/medRxiv** (GCS: `biorxiv-medrxiv/*.json`)
  - Recent preprints with Title, Authors, Date, Abstract, DOI
  - Reveals cutting-edge research not yet peer-reviewed
  - Trend analysis over time
  - Approx. ~50,000 preprints

### Layer C: Knowledge Graph & Researchers
Goal: Map scientific concepts and find the humans behind the science.

- **ORKG** (GCS: `orkg/orkg-dump.nt`)
  - Open Research Knowledge Graph RDF triples (N-Triples format)
  - Searchable label text (rdfs:label predicates)
  - Maps scientific concepts to papers
  - Approx. ~50,000 searchable label triples

- **OpenAlex** (Live API)
  - Researcher identification by citations and H-index
  - Institutional affiliations and publication records
  - Real-time queries via API
  - Used for finding active researchers in a field
