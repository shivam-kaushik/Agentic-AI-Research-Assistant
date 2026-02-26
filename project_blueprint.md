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

**LangGraph Node Workflow:**
1. **The Planner:** Translates the user prompt into a structured JSON execution plan of sub-tasks.
2. **The Internal_Retriever:** Executes parameterized BigQuery SQL against datasets (e.g., OpenTargets) and stores results in the state.
3. **The Conflict_Detector:** Validates BigQuery data via API sets (e.g., PubTator) for recent retractions or toxicity tags.
4. **The Human_in_the_Loop (HITL):** Pauses execution, pushes state to Firestore, and waits for human feedback via the Streamlit UI.
5. **The External_API_Caller:** Retrieves dynamic data from external APIs (e.g., OpenAlex for researcher info) on remaining valid targets.
6. **The Synthesizer:** Compiles the aggregated internal and external evidence into a final Markdown report.

**Execution Roadmap (24-Hour Sprint):**
- **Phase 1 (Hours 1-3):** Initialize Vertex AI Workbench, create GCP Service Account, config Cloud Run CI/CD.
- **Phase 2 (Hours 3-7):** Write Python tools (`query_bigquery`, `search_openalex`, `normalize_entity_pubtator`).
- **Phase 3 (Hours 7-14):** Build the LangGraph State Machine, config graph nodes/edges, set up Firestore persistence.
- **Phase 4 (Hours 14-18):** Connect Streamlit UI to the Firestore checkpointer.
- **Phase 5 (Hours 18-24):** Edge-case testing, record the 3-minute demo video, finalize README.md.

---

## 3. Datasets & APIs We Will Use

To maximize our "Data Mastery" score, datasets are structured into three intelligence layers:

### Layer A: The Genetic & Clinical Foundation (The "Ground Truth")
Goal: Establish what is biologically and clinically true before looking at literature.
- **ClinGen (The Anchor):** When a user asks about a disease (e.g., IPF), the agent first hits ClinGen to find definitively linked genes and pathogenic variants. It ignores speculative biology.
- **GTEx v8 (The Safety Check):** The agent verifies where these genes are expressed in the human body to anticipate off-target toxicity.
- **CIViC (The Oncology Specialized Check):** If the query is cancer-related, the agent uses CIViC to check the clinical evidence level of specific variants.

### Layer B: The Knowledge Graph & Mechanism Engine (The "Multi-Hop" Layer)
Goal: Map how these genes interact and find the exact research papers discussing these connections.
- **Pathway Commons & Reactome (The Biological Graph):** The agent takes the genes from Layer A and traverses the biological network. *Example: "ClinGen gave me Gene X. Pathway Commons shows Gene X physically interacts with Protein Y in the TGF-beta pathway."*
- **ORKG - Open Research Knowledge Graph (The Concept Graph):** This is your massive differentiator. The agent queries the N-Triples graph to find how scientific concepts map to papers. *Example: "Find all papers where the Research Problem is 'IPF' and the Method involves 'TGF-beta inhibition'."*
- **PubTator 3.0 (The entity-to-PMID Bridge):** The agent cross-references the targets with PubTator to extract the exact PubMed IDs (PMIDs) of the most heavily annotated literature.

### Layer C: The External Validation APIs (The "Live Researcher" Check)
Goal: Fulfill the MVP API requirement by finding the humans behind the science.
- **OpenAlex API (Researcher Velocity):** The agent takes the PMIDs and DOIs discovered via ORKG and PubTator (Layer B) and pings OpenAlex. It calculates which authors are currently active (publishing post-2023) and gathers their institutional affiliations.
- **PubMed (Entrez API):** Used to pull the absolute latest abstracts that might not yet be indexed in the static BigQuery datasets.
