# Cross-Questioning Feature Guide

## Overview
The Co-Investigator now supports **mid-execution cross-questioning**, allowing users to ask clarifying questions during active research without disrupting the execution flow. This feature uses **vector-based semantic search** for context-aware answers.

---

## Key Features

### 1. **Question Detection**
The Orchestrator automatically detects when a user input is a question versus a command:
- **Question indicators**: what, why, how, when, who, where, which, explain, tell me, can you, could you, show me, describe, clarify
- **Active execution check**: Only pauses if there are pending tasks
- **No interruption during confirmation**: Questions during step confirmation are handled normally

### 2. **Execution Pause/Resume**
When a question is detected during active execution:
- ✅ Execution state is **paused** (tasks remain in pending)
- ✅ Current step index and confirmation state are **saved**
- ✅ Question mode is **activated**
- ✅ User can **resume** by typing "continue", "proceed", or clicking the Resume button

**ConversationMemory attributes**:
```python
execution_paused: bool           # True when paused
question_mode: bool              # True during questioning
execution_context: dict          # Saved state snapshot
```

### 3. **Vector Store Integration**
All conversations and research findings are stored with embeddings for semantic search:

**Storage**:
- **User messages**: Stored with timestamp in `conversation_history` collection
- **Assistant messages**: Stored with agent metadata (PLANNER, RESEARCHER, etc.)
- **Research findings**: Stored in `research_findings` collection with task_id, data_source, count

**Retrieval**:
- **Semantic search**: Uses `textembedding-gecko@003` model for embeddings
- **Cosine similarity**: Ranks relevant messages/findings by similarity score
- **Context window**: Returns top 3 messages + top 5 findings (max 2000 tokens)

### 4. **Context-Aware Answers**
The Clarifier agent retrieves relevant context before answering:
```python
answer = clarifier.answer(question, memory, vector_store=vector_store)
```

**Answer structure**:
1. **Current Research Plan**: JSON representation of active plan
2. **Task Status**: Completed and pending tasks
3. **Collected Data Summary**: Data sources and counts
4. **Relevant Context**: Semantically retrieved conversation history
5. **User Question**: The actual question
6. **AI-Generated Answer**: Gemini-2.5-Flash response with all context

---

## Architecture

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    MultiAgentOrchestrator                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ VectorStoreManager (session_id: str)                  │  │
│  │  - store_message(role, content, metadata)             │  │
│  │  - store_research_finding(task_id, source, content)   │  │
│  │  - semantic_search_messages(query, top_k)             │  │
│  │  - get_relevant_context(query, max_tokens)            │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ OrchestratorAgent                                      │  │
│  │  - Detects questions vs commands                       │  │
│  │  - Pauses execution when question detected             │  │
│  │  - Routes to CLARIFIER agent                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ ClarifierAgent                                         │  │
│  │  - Calls vector_store.get_relevant_context()           │  │
│  │  - Builds comprehensive prompt with context            │  │
│  │  - Generates context-aware answer                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ ResearcherAgent                                        │  │
│  │  - Calls vector_store.store_research_finding()         │  │
│  │  - Stores after each successful task execution         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
User Question → Orchestrator.route()
                    ↓
              Is question? ✓
              Has active execution? ✓
                    ↓
              memory.pause_execution()
                    ↓
              Route to CLARIFIER
                    ↓
        ClarifierAgent.answer(vector_store=...)
                    ↓
        vector_store.get_relevant_context()
                    ↓
              semantic_search_messages() [top 3]
              semantic_search_findings() [top 5]
                    ↓
              Build prompt with:
                - Plan info
                - Task status
                - Relevant context
                - User question
                    ↓
              Generate answer via Gemini
                    ↓
              Return to user
                    ↓
        User types "continue" → memory.resume_execution()
```

---

## File Changes

### New Files
1. **`config/vector_store.py`** (~350 lines)
   - `VectorStoreManager` class
   - Firestore integration
   - Embedding generation with `textembedding-gecko@003`
   - Semantic search with cosine similarity

### Modified Files
1. **`agent/multi_agent.py`**
   - **ConversationMemory** (lines 30-89):
     - Added `execution_paused`, `question_mode`, `execution_context`
     - Added `pause_execution()` and `resume_execution()` methods
   
   - **OrchestratorAgent.route()** (lines 254-303):
     - Added question detection with 14 indicators
     - Added pause logic when question detected during active execution
   
   - **ClarifierAgent.answer()** (lines 1014-1050):
     - Added `vector_store` parameter
     - Retrieves relevant context before answering
     - Injects context into prompt
   
   - **ResearcherAgent.execute_task()** (lines 545-842):
     - Added `vector_store` parameter
     - Stores research findings after successful data retrieval
   
   - **MultiAgentOrchestrator.__init__()** (lines 1367-1390):
     - Initializes `self.vector_store = VectorStoreManager(session_id)`
   
   - **MultiAgentOrchestrator.process_message()** (lines 1392-1493):
     - Stores user messages in vector store
     - Stores assistant messages with agent metadata
     - Passes `vector_store` to `ClarifierAgent.answer()`
     - Passes `vector_store` to `ResearcherAgent.execute_next_task()`

2. **`app/main_v2.py`**
   - **Sidebar** (lines 320-329):
     - Added pause indicator when `memory.execution_paused == True`
     - Shows warning: "⏸️ **Execution Paused**"
     - Displays resume button: "▶️ Resume Execution"

---

## Usage Examples

### Example 1: Ask about disease mid-execution
```
User: Find researchers working on Idiopathic Pulmonary Fibrosis
[Planner creates plan with 4 tasks]
[Task 1 starts executing...]

User: What is Idiopathic Pulmonary Fibrosis?
[Execution pauses, Clarifier retrieves context]

Agent: Idiopathic Pulmonary Fibrosis (IPF) is a chronic, progressive lung disease...
Based on the current research plan, we're investigating:
- Gene-disease associations from ClinGen
- Recent literature from bioRxiv
- Clinical studies from PubMedQA

[User sees "⏸️ Execution Paused" in sidebar]

User: continue
[Execution resumes from task 1]
```

### Example 2: Check progress without pausing
```
User: Find genes related to lung cancer
[Task 1 completed, Task 2 in progress...]

User: how many tasks are left?
[NOT paused - only has pending tasks, no confirmation]

Agent: Currently, 2 tasks are completed and 2 tasks are pending:
✅ task_1: Query ClinGen for lung cancer
✅ task_2: Query bioRxiv for lung cancer preprints
⏳ task_3: Query PubMedQA
⏳ task_4: Query ORKG

[Execution continues automatically]
```

### Example 3: Clarification during planning
```
User: Research Alzheimer's disease
[Planner creates plan]

Agent: I've created a research plan. Should I proceed with execution?

User: what databases will you search?
[No pause - awaiting_step_confirmation is True]

Agent: I'll search the following databases:
- ClinGen: Gene-disease associations
- bioRxiv: Preprint literature
- PubMedQA: Clinical Q&A datasets
- ORKG: Knowledge graph concepts

User: yes, proceed
[Execution starts normally]
```

---

## Configuration

### Firestore Collections
Two collections are used for vector storage:

**1. `conversation_history`**
```json
{
  "session_id": "session_abc123",
  "role": "user" | "assistant",
  "content": "Find researchers in IPF",
  "timestamp": "2025-02-27T10:30:00",
  "embedding": [0.123, -0.456, ...],  // 768 dimensions
  "metadata": {
    "agent": "PLANNER",
    "intent": "research_query"
  }
}
```

**2. `research_findings`**
```json
{
  "session_id": "session_abc123",
  "task_id": "task_1",
  "data_source": "clingen",
  "content": "Task task_1: Retrieved 15 records from clingen",
  "timestamp": "2025-02-27T10:32:00",
  "embedding": [0.789, -0.234, ...],
  "structured_data": {
    "count": 15,
    "query_params": {"disease": "Idiopathic Pulmonary Fibrosis"},
    "success": true
  }
}
```

### Vector Store Parameters
```python
# In VectorStoreManager.__init__()
self.model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
self.project_id = config.project_id
self.session_id = session_id

# In get_relevant_context()
max_tokens = 2000         # Maximum context length
message_top_k = 3         # Top 3 relevant messages
findings_top_k = 5        # Top 5 relevant findings
similarity_threshold = 0.3  # Minimum cosine similarity
```

---

## Testing

### Manual Testing Steps
1. **Start a research query**:
   ```
   Find researchers in Idiopathic Pulmonary Fibrosis
   ```

2. **Wait for first task to execute**

3. **Ask a question mid-execution**:
   ```
   What is Idiopathic Pulmonary Fibrosis?
   ```

4. **Verify**:
   - ✅ Sidebar shows "⏸️ Execution Paused"
   - ✅ Agent provides context-aware answer
   - ✅ Relevant conversation history is included

5. **Resume execution**:
   ```
   continue
   ```
   Or click "▶️ Resume Execution" button

6. **Verify**:
   - ✅ Sidebar no longer shows pause indicator
   - ✅ Next task executes normally

### Unit Test Coverage
Areas to test:
- **Question detection**: All 14 indicators trigger CLARIFIER
- **Pause logic**: Only pauses when pending_tasks > 0 AND not awaiting_confirmation
- **Vector storage**: Messages and findings stored with valid embeddings
- **Semantic search**: Returns relevant results with similarity > 0.3
- **Context injection**: Clarifier prompt includes relevant_context section
- **Resume behavior**: Execution continues from saved step_index

---

## Performance Considerations

### Embedding Generation
- **Model**: `textembedding-gecko@003` (768 dimensions)
- **Latency**: ~200-300ms per embedding
- **Batching**: Not currently implemented (could optimize for bulk storage)

### Firestore Queries
- **Read latency**: ~50-100ms per query
- **Vector search**: O(n) linear scan (could optimize with Vertex AI Vector Search)
- **Indexes**: Composite index on `session_id` + `timestamp`

### Context Window
- **Max tokens**: 2000 tokens (~8000 characters)
- **Message limit**: Top 3 messages (~600 tokens)
- **Finding limit**: Top 5 findings (~1400 tokens)

---

## Troubleshooting

### Issue: "Vector store not available"
**Cause**: Firestore or Vertex AI initialization failed

**Solution**:
1. Check GCP credentials: `gcloud auth application-default login`
2. Verify Firestore is enabled: `gcloud services enable firestore.googleapis.com`
3. Check logs: Look for errors in orchestrator initialization

### Issue: Questions not pausing execution
**Cause**: Question indicator not detected

**Solution**:
1. Ensure question starts with one of 14 indicators
2. Check if `memory.pending_tasks` is empty (won't pause if no active work)
3. Verify not in confirmation mode (`memory.awaiting_step_confirmation == False`)

### Issue: Context not appearing in answers
**Cause**: Vector search returned no results

**Solution**:
1. Check if messages are being stored: Look at Firestore `conversation_history` collection
2. Verify embeddings are generated: Check `embedding` field exists
3. Lower similarity threshold in `semantic_search_messages()` (currently 0.3)

### Issue: Execution won't resume
**Cause**: Resume logic not triggered

**Solution**:
1. Type explicit resume commands: "continue", "proceed", "yes"
2. Click sidebar resume button
3. Check logs for `memory.resume_execution()` call

---

## Future Enhancements

### 1. **Enhanced Vector Search**
- Migrate to Vertex AI Vector Search for faster retrieval
- Implement approximate nearest neighbor (ANN) indexing
- Add hybrid search (semantic + keyword)

### 2. **Context Ranking**
- Use LLM to re-rank retrieved context by relevance
- Filter out redundant information
- Summarize long context sections

### 3. **Multi-turn Questioning**
- Support follow-up questions in question mode
- Maintain question context across turns
- Auto-resume after N questions

### 4. **Planner Context Awareness**
- Update PlannerAgent to retrieve context when modifying plans
- Show execution progress in plan modifications
- Avoid duplicate tasks based on completed work

### 5. **UI Enhancements**
- Show "Searching knowledge base..." status during retrieval
- Display matched context snippets in expander
- Add "Ask a question" text input in sidebar during execution

---

## API Reference

### VectorStoreManager

#### `__init__(session_id: str)`
Initialize vector store for a session.

#### `store_message(role: str, content: str, metadata: dict = None) -> str`
Store a conversation message with embedding.
- **role**: "user" or "assistant"
- **content**: Message text
- **metadata**: Optional metadata (agent, intent, etc.)
- **Returns**: Document ID

#### `store_research_finding(task_id: str, data_source: str, content: str, structured_data: dict = None) -> str`
Store a research finding with embedding.
- **task_id**: Task identifier (e.g., "task_1")
- **data_source**: Database name (e.g., "clingen")
- **content**: Summary text
- **structured_data**: Structured metadata
- **Returns**: Document ID

#### `semantic_search_messages(query: str, top_k: int = 5) -> List[dict]`
Search conversation history by semantic similarity.
- **query**: Search query
- **top_k**: Number of results to return
- **Returns**: List of messages sorted by similarity

#### `get_relevant_context(query: str, max_tokens: int = 2000) -> str`
Get formatted context from messages and findings.
- **query**: User question
- **max_tokens**: Maximum context length
- **Returns**: Formatted context string

### ConversationMemory

#### `pause_execution()`
Pause execution and enter question mode. Saves current state.

#### `resume_execution()`
Resume execution from paused state. Restores saved state.

---

## License
Part of the Co-Investigator Multi-Agent System  
© 2025 BenchSci
