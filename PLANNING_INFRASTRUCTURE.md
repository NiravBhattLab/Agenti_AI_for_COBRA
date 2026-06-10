# Planning Infrastructure: Complete Technical Report

---

## 1. Big Picture

The planning layer sits **on top of** the existing agent infrastructure, not inside it. The existing app had a ReAct agent that answers free-form questions and calls tools reactively. The planning layer adds a separate mode where, instead of reacting, the system:

1. Searches a pre-built knowledge base of published metabolic modeling workflows
2. Asks the LLM to assemble a structured, ordered, step-by-step plan from what it found
3. Shows the plan to the user interactively for editing and annotation
4. Iterates the plan through the LLM until the user is happy
5. Executes the approved steps one at a time by calling tool functions directly

The two modes — chat and plan — are completely independent. Toggling "Plan Mode" in the header just switches which UI function renders. The underlying FastAPI backend, model manager, and LLM are shared between both.

---

## 2. The Knowledge Base: ChromaDB

**Location:** `chroma_db/` at the root of the codebase.

**Contents:** A single ChromaDB persistent collection named `gsm_workflows`. It was built from a set of published metabolic modeling papers whose workflows were manually annotated into a hierarchical JSON structure (stored elsewhere as `*_with_objectives.json` files). The `build_index.py` script processed those JSONs and embedded them into four tiers of granularity.

**Physical files:**
- `chroma.sqlite3` — SQLite database that stores all metadata, document text, and collection bookkeeping
- Three UUID-named directories — each is an HNSW (Hierarchical Navigable Small World) graph index storing the actual floating-point embedding vectors for fast approximate nearest-neighbor lookup

**The four tiers — what is stored at each level:**

| Tier | Level | What is embedded | What metadata is stored |
|------|-------|-----------------|------------------------|
| 0 | Paper | `relevance + research_question` (whole-paper summary) | `paper_id`, `tier=0` |
| 1 | Category (pipeline stage) | `relevance` text per category | `paper_id`, `step_category`, `tier=1` |
| 2 | Sub-workflow | `t2_label + objective_paragraph` (a cluster of steps with a shared goal) | `paper_id`, `step_category`, `step_ids` (JSON list of all steps in the cluster), `tier=2` |
| 3 | Individual step | `objective_paragraph` per step | `paper_id`, `step_ids` (single-element JSON list), `technique`, `tier=3` |

The tiered structure is the key design insight: tier 2 hits give you **whole sub-workflows** (e.g., "steps S1-S5 from this paper"), while tier 3 hits give you **individual atomic steps** from any paper. This is what allows the planner to reconstruct complete workflows even when not every step in the workflow directly matched the query.

**Embedding model:** `BAAI/bge-large-en-v1.5`, a 1024-dimensional asymmetric retrieval model. "Asymmetric" means documents were embedded with the prefix `"Represent this document for retrieval: "` and queries must be embedded with `"Represent this query for retrieval: "`. Using the wrong prefix silently degrades retrieval quality. The distance metric is cosine, with L2-normalized embeddings, so `similarity = 1.0 - cosine_distance`.

---

## 3. The Retrieval System: `planner/retrieve.py`

**Class:** `GSMRetriever`

**Initialization** (`__init__`):
- Loads the `BAAI/bge-large-en-v1.5` SentenceTransformer model into memory (happens once, on first call to `_get_planner()` in `main.py`)
- Opens the ChromaDB persistent client pointing at `CHROMA_DB_PATH`
- Connects to the `gsm_workflows` collection

**How a query flows through retrieval** (`retrieve(query)`):

```
query string
    ↓
_embed_query()
    Prepend BGE_QUERY_PREFIX: "Represent this query for retrieval: " + query
    Encode with SentenceTransformer → 1024-dimensional float vector
    normalize_embeddings=True → unit vector (required for cosine distance)
    ↓
Phase 1 — Coarse (two independent ChromaDB queries)
    ├── tier0_hits = _search(vec, tiers=[0], n=5)   ← paper-level matches
    └── tier1_hits = _search(vec, tiers=[1], n=5)   ← category-level matches

    shortlisted_paper_ids = union of all paper_ids from tier0 + tier1 hits
    (used only if filter_by_phase1=True; default is False)
    ↓
Phase 2 — Fine (two more independent ChromaDB queries)
    ├── tier2_hits = _search(vec, tiers=[2], n=5)   ← sub-workflow matches
    └── tier3_hits = _search(vec, tiers=[3], n=15)  ← individual step matches
    ↓
Return RetrievalResult dict
```

**Inside `_search()`:**
1. Guards: if `paper_ids=[]` returns immediately (empty list crashes ChromaDB's `$in` filter); if collection is empty returns `[]`
2. Calls `_build_where()` to construct a ChromaDB metadata filter dict
3. Calls `collection.query(query_embeddings=..., n_results=..., where=..., include=[documents, metadatas, distances])`
4. Passes raw output to `_parse_results()` which converts cosine distance → similarity score (`1.0 - distance`), decodes `step_ids` from stored JSON strings back to Python lists, and sorts descending by score

**The `_build_where()` logic** — ChromaDB requires specific filter syntax:
- Single condition: `{"tier": 2}` — passed directly
- Two or more conditions: must be wrapped in `{"$and": [...]}` — ChromaDB rejects bare multi-condition dicts
- Paper ID filter: `{"paper_id": "xyz"}` for single, `{"paper_id": {"$in": [...]}}` for multiple
- Technique filter: `{"technique": "fluxVariability"}` — only meaningful for tier 3

**What a Hit looks like** (the core unit coming out of retrieval):
```python
{
    "paper_id":      "1-s2.0-S0278691521001186-main",   # which paper
    "entry_id":      "T2_Configure_Model",               # tier+category label
    "score":         0.8734,                             # similarity 0-1
    "document":      "Configure the model objectives...", # embedded text
    "step_ids":      ["S1", "S2", "S3"],                 # steps in this entry
    "step_category": "Model Reconstruction",             # pipeline stage
    "technique":     "",                                 # non-empty only for tier 3
    "tier":          2
}
```

**Why two phases?** Phase 1 establishes which papers and categories are topically relevant. Phase 2 finds the granular sub-workflows and steps. The optional `filter_by_phase1=True` mode restricts Phase 2 to only papers found in Phase 1 — higher precision but lower recall. Default is off (unfiltered) for maximum coverage.

---

## 4. The Planner: `planner/planner_core.py`

**Class:** `GSMPlanner`

**Constructor parameters:**
- `llm_fn`: a callable `(prompt: str, system_prompt: str) -> str` — fully decoupled from any LLM provider
- `collection`: the ChromaDB Collection object (same one `GSMRetriever` opened — reused, not re-opened)
- `top_subworkflows=3`: how many tier 2 hits to use as plan seeds
- `max_retries=3`: retry count for LLM call failures

### 4a. Candidate Building (`_build_candidates`)

This runs before the LLM is called. It takes the raw retrieval result and organizes it into a structured set of candidates that will be formatted into the LLM prompt.

**Step 1 — Primary seeds:** Take the top 3 tier 2 hits (by score). For each one, the tier 2 entry only carries a list of `step_ids` (e.g., `["S1", "S2", "S3"]`). Those step IDs are used to **fetch full tier 3 detail** via `_fetch_step_details()`:

```python
entry_ids = [f"T3_{sid}" for sid in step_ids]   # e.g., ["T3_S1", "T3_S2", "T3_S3"]
collection.get(where={"$and": [
    {"paper_id": paper_id},
    {"tier": 3},
    {"entry_id": {"$in": entry_ids}}
]}, include=["documents", "metadatas"])
```

This is a direct lookup by ID (not a semantic search), so it retrieves all steps in the sub-workflow regardless of whether they individually scored well in the query. This is important: a sub-workflow about FVA might have intermediate setup steps (like `set_bounds`) that wouldn't rank highly against the query "run FVA", but are still needed for the workflow to be correct. The tier 2 hit guarantees the whole group is fetched.

Each step detail is immediately mapped through `_map_tool(technique)`, which looks up the technique string (e.g., `"fluxVariability"`) in `TECHNIQUE_TO_TOOL` to produce the actual COBRApy tool name (e.g., `"run_fva"`), a `requires_user_input` flag, and a deep copy of the default `user_inputs` template for that tool.

**Step 2 — Supplementary:** Any tier 3 hits that weren't already covered by a tier 2 group (tracked via a `covered` set of `(paper_id, step_id)` pairs) are added as supplementary candidates. These are individual steps from other papers that might be relevant but weren't part of any top-3 sub-workflow.

**Result structure:**
```python
{
    "subworkflows": [
        {
            "score": 0.87, "paper_id": "...", "step_ids": ["S1","S2","S3"],
            "objective": "Configure model and run FBA...",
            "step_category": "Simulation",
            "steps": [
                {"step_id": "S1", "tool": "load_model", "technique": "COBRApy",
                 "user_inputs": {"model_id": {...}}, "description": "..."},
                {"step_id": "S2", "tool": "run_fba", "technique": "FBA", ...},
                ...
            ]
        },
        ...  # up to 3 subworkflows
    ],
    "supplementary": [
        {"step_id": "S4", "tool": "run_fva", "score": 0.71, ...},
        ...
    ]
}
```

### 4b. The Planning Prompt (`_build_planning_prompt`)

The LLM is given a single structured prompt containing:

1. **User query** — verbatim
2. **Candidate sub-workflows** — for each: similarity score, paper ID, category, step IDs, overview text (≤300 chars), and for each step: step ID, tool name, technique, description (≤200 chars)
3. **Supplementary individual steps** — any tier 3 hits not in any sub-workflow
4. **Available tools** — exact tool names and one-line descriptions (the LLM must use these exact strings)
5. **8 ordering/selection instructions** — e.g., "load_model must always be first", "use EXACT tool names"
6. **JSON output schema** — the exact structure the LLM must produce

**System prompt** tells the LLM: you are a planning assistant, output ONLY valid JSON, no preamble, no markdown fences, select only from candidates provided.

The LLM is called at temperature 0.0 (deterministic) with up to 3 retries on failure.

### 4c. JSON Parsing and Validation (`_parse_and_validate`)

The raw LLM string goes through several repair steps:
1. **Markdown fence stripping** — if the LLM wrapped output in ` ```json ``` `, extract the JSON block between first `{` and last `}`
2. **JSON parse** — `json.loads()`, returns empty plan on failure
3. **Tool normalization** — if `step["tool"]` isn't in `TOOL_USER_INPUTS`, try to recover it via `TECHNIQUE_TO_TOOL.get(step["technique"], "manual")`
4. **user_inputs injection** — if the LLM produced an empty `user_inputs` dict for a known tool, replace it with the default template from `TOOL_USER_INPUTS`
5. **Step renumbering** — step numbers are reassigned 1..N regardless of what the LLM wrote
6. **requires_user_input recomputation** — flag is set to True only if there's at least one `required=True` param with `value=None`

**The final plan dict — schema every step conforms to:**
```python
{
    "query": "the original user query",
    "summary": "1-2 sentence description",
    "plan": [
        {
            "step_number": 1,
            "step_name": "Load E. coli core model",
            "description": "...",
            "tool": "load_model",
            "requires_user_input": True,
            "user_inputs": {
                "model_id": {
                    "description": "Model identifier or SBML path",
                    "value": None,      # None = user must fill before execution
                    "required": True
                }
            },
            "source_paper": "1-s2.0-S0278...",
            "source_step_id": "S1",
            "included": True,
            "rationale": "Why this step is in the plan for this query"
        },
        ...
    ]
}
```

### 4d. Plan Revision (`revise_plan`)

Takes the current plan dict, serializes it as `json.dumps(current_plan, indent=2)`, and builds a revision prompt containing:
- Original query
- Full current plan JSON
- User feedback string (natural language)
- Instructions: make only the requested changes, set `included: false` rather than deleting steps, preserve unchanged steps, renumber if inserting

The LLM returns a new JSON plan in the same schema, which goes through the same `_parse_and_validate()` pipeline.

---

## 5. The LLM Bridge: `_make_planner_llm_fn()` in `main.py`

The planner requires a `llm_fn(prompt, system_prompt) -> str` callable. The app already has a Llama-Index LLM instance (the same one the ReAct agent uses). The bridge is:

```python
def _make_planner_llm_fn():
    from llama_index.core.llms import ChatMessage
    from agent import llm as _llm          # the current Llama-Index LLM global

    def llm_fn(prompt: str, system_prompt: str) -> str:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt),
        ]
        response = _llm.chat(messages)     # synchronous .chat(), not async
        return response.message.content

    return llm_fn
```

**Why sync and not async:** The planner is pure Python with no async/await. The `/plan/generate` and `/plan/revise` endpoints are sync `def` functions (not `async def`), so FastAPI automatically runs them in a thread pool executor — no event loop conflict with the existing async `/chat/` endpoint. The Llama-Index LLM's sync `.chat()` method works in a thread pool context.

**LLM reset on provider change:** When the user changes the LLM provider via `/set_llm/`, `_planner` is set to `None`. On the next planning call, `_get_planner()` rebuilds the planner with a fresh `_make_planner_llm_fn()` that captures the new LLM instance. The retriever (`_retriever`) is not reset — it doesn't depend on the LLM and embedding model loading is expensive.

---

## 6. The Executor: `planner/executor.py`

**Purpose:** Execute a single plan step by calling the actual Python tool function directly from `tools.py`, bypassing the ReAct agent entirely.

**`TOOL_FN_MAP`** maps tool name strings to function objects:
```python
{
    "run_fba":                      run_fba,
    "run_fva":                      run_fva,
    "gene_knockout_simulation":     gene_knockout_simulation,
    "reaction_knockout_simulation": reaction_knockout_simulation,
    "sample_metabolic_model":       sample_metabolic_model,
    "load_model":                   load_model,
    "set_model_objective":          set_model_objective,
}
```

**`_coerce(value)`:** All parameter values coming from `st.text_input` are strings. This helper tries `int()` first (so `"100"` → `100`), then `float()` (so `"1.0"` → `1.0`), and falls back to the original string if neither works (e.g., gene names, model IDs). Without this, passing `"1.0"` as `fraction_of_optimum` to `flux_variability_analysis` raises "can't multiply sequence by non-int of type 'float'".

**`execute_step(step: dict) -> dict`:**
1. Check if `tool == "manual"` → return status message, no execution
2. Look up function in `TOOL_FN_MAP`, return error dict if not found
3. Build `kwargs` from `step["user_inputs"]` — only entries where `value is not None` are forwarded, each value passed through `_coerce()`
4. Call `fn(**kwargs)` in a try/except — returns error dict on exception
5. Normalize the return type to a JSON-serializable dict:
   - `str` result → `{"output": result}`
   - `dict` result → returned as-is
   - `pandas.DataFrame` → converted via `.to_dict(orient="records")`

The tools in `tools.py` all operate on the global `model_manager` object. So executing via the executor has exactly the same effect on model state as executing through the agent — it's the same functions.

---

## 7. The FastAPI Backend: New Endpoints in `main.py`

### Lazy Initialization (`_get_planner()`)

```python
_retriever: GSMRetriever | None = None
_planner:   GSMPlanner   | None = None

def _get_planner():
    global _retriever, _planner
    if _retriever is None:
        _retriever = GSMRetriever()          # loads BGE model + opens ChromaDB
    if _planner is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        _planner = GSMPlanner(llm_fn=_make_planner_llm_fn(), collection=collection)
    return _retriever, _planner
```

Both objects are `None` at startup. The first planning call triggers initialization. Loading the BGE model on first use means the server starts instantly regardless of whether plan mode is ever used.

Note: ChromaDB opens the collection twice — once inside `GSMRetriever.__init__()` for query operations, and once here to get the `collection` object for `GSMPlanner` (which uses it for direct `.get()` lookups in `_fetch_step_details`). Both point at the same physical files; ChromaDB handles concurrent access to the same persistent client safely.

### Three New Endpoints

**`POST /plan/generate`** — accepts `{"query": str}`
1. Calls `_get_planner()` (initializing lazily if first call)
2. Calls `retriever.retrieve(query)` — runs the full two-phase semantic search
3. Calls `planner.plan(query, retrieval_result)` — builds candidates, calls LLM, validates JSON
4. Returns `{"plan": plan_dict}`

**`POST /plan/revise`** — accepts `{"query": str, "current_plan": dict, "feedback": str}`
1. Calls `_get_planner()`
2. Calls `planner.revise_plan(query, current_plan, feedback)` — revision prompt → LLM → validated JSON
3. Returns `{"plan": revised_plan_dict}`

**`POST /plan/execute_step`** — accepts `{"step": dict}`
1. Calls `execute_step(req.step)` from `executor.py`
2. Returns `{"result": result_dict}`

All three are sync `def` endpoints — FastAPI runs them in its default thread pool, so they don't block the async event loop.

---

## 8. The Streamlit Frontend: `app.py`

### Session State Variables for Planning

```
plan_mode:    bool       # True when the toggle is on
current_plan: dict|None # The active plan from the LLM (or None before first generation)
exec_cursor:  int        # -1 = editing phase; 0+ = index of next step to execute
exec_results: list       # Accumulated results from executed steps so far
exec_steps:   list       # The finalized list of included steps (set at "Execute Plan" click)
```

Per-step widget state (created dynamically for each step `i`):
```
plan_include_{i}       bool  — whether this step is included
plan_input_{i}_{param} str   — user-typed value for each tool parameter
plan_comment_{i}       str   — feedback text from the popover
plan_insert_open_{i}   bool  — whether the insert input field is showing
plan_insert_desc_{i}   str   — text describing the step to insert after step i
```

### The Plan Mode Toggle

A `st.toggle` widget in the 5th header column. When its value changes from the stored `session_state.plan_mode`, it updates session state and calls `st.rerun()`, switching the main render path between `plan_mode_ui()` and the regular chat UI.

### `plan_mode_ui()` — Three Phases

---

**Phase A: Query Input**

Renders a text input and "Generate Plan" button side by side. When clicked with a non-empty query:
1. POSTs to `/plan/generate`
2. On success: calls `_clear_plan_widget_state()` (removes all `plan_include_*`, `plan_input_*`, etc. keys so a fresh plan doesn't inherit stale widget values from the previous plan)
3. Stores the returned plan in `session_state.current_plan`
4. Resets `exec_cursor = -1`, `exec_results = []`, `exec_steps = []`
5. Calls `st.rerun()`

---

**Phase B: Interactive Plan Editor**

Rendered when `current_plan is not None` and `exec_cursor == -1`.

**Between every pair of steps**, `_insert_divider(i)` renders a centered `＋` button. Clicking it sets `plan_insert_open_{i} = True` and shows a text input for describing the step to insert, plus a `✕` cancel button.

**For each step**, widget state is initialized once:
```python
if f"plan_include_{i}" not in st.session_state:
    st.session_state[f"plan_include_{i}"] = step.get("included", True)
```
The `if key not in st.session_state` guard ensures user edits survive Streamlit reruns without being overwritten.

If `plan_include_{i}` is False: render a compact struck-through line with a `↩` restore button.

If included: render `st.container(border=True)` with:
- **Header row** (`[10, 1, 1]` columns): step name | `💬` popover | `✕` delete
  - `st.popover("💬")` opens a floating panel containing the feedback `st.text_area`. The value persists in session state even when the popover is closed.
  - `✕` immediately sets `plan_include_{i} = False` and reruns, collapsing the card.
- **Body** (`[1, 1]` two equal columns):
  - Left: description, rationale in italic caption
  - Right: `st.code(tool_name)`, source info, user input fields

**`_collect_feedback(plan)`** assembles the feedback string by reading all per-step session state:
```
Step 1 (Load model): Delete this step.         ← plan_include_0 == False
Step 2 (Run FBA): use pFBA instead             ← plan_comment_1 has text
[Insert after step 2]: add a gene knockout     ← plan_insert_desc_1 has text
```

**"Re-plan with feedback"**: POSTs to `/plan/revise`, then calls `_clear_plan_widget_state()` and stores the new plan.

**"Execute Plan"**: iterates included steps, merges all `plan_input_{i}_{param}` values back into deep-copied step dicts, stores in `exec_steps`, sets `exec_cursor = 0`, reruns → triggers Phase C.

---

**Phase C: Step-by-Step Execution**

Uses a cursor + results list state machine:

```
exec_cursor = k, len(exec_results) = k     → execute step k (spinner)
exec_cursor = k, len(exec_results) = k+1   → show result + Continue/Stop buttons
exec_cursor = k+1 (after Continue click)   → next iteration
exec_cursor = len(exec_steps)              → all done screen
```

On each rerun:
1. Show all previously completed steps as collapsed green containers
2. If current step hasn't run: POST to `/plan/execute_step`, append result, rerun immediately
3. If current step already ran: show result (red on error, JSON on success) + two buttons
4. "Continue" increments cursor and reruns; "Stop" jumps cursor to end
5. Final screen appends execution summary to `chat_history` and offers "Back to plan editor"

---

## 9. End-to-End Data Flow: One Complete Run

```
User types: "simulate gene knockout and check growth impact"
User clicks: "Generate Plan"
    ↓
app.py → POST /plan/generate → main.py:plan_generate()
    ↓
_get_planner()  [first call: loads BGE model, opens ChromaDB]
    ↓
retriever.retrieve(query)
    embed query → [1024-dim vector]
    phase1: search tier0 (5 paper hits), tier1 (5 category hits)
    phase2: search tier2 (5 sub-workflow hits), tier3 (15 step hits)
    → RetrievalResult dict
    ↓
planner.plan(query, retrieval_result)
    _build_candidates():
        top 3 tier2 hits → fetch all their tier3 steps via collection.get()
        supplementary: tier3 hits not already covered by any tier2 group
        → candidates dict
    _build_planning_prompt():
        format sub-workflows + supplementary + tool list + instructions + JSON schema
        → ~3000-token prompt string
    _call_llm_with_retry(prompt, system_prompt):
        llm_fn(prompt, system_prompt)
            ChatMessage(system=..., user=...) → llm.chat(messages) → response string
        → raw JSON string from LLM
    _parse_and_validate(raw):
        strip markdown fences → json.loads → normalize tools → inject user_inputs templates
        → validated plan dict
    ↓
return {"plan": {...}} → app.py stores in session_state.current_plan
    ↓
Streamlit reruns → Phase B renders
    [Step 1: Load model]  [💬] [✕]
    | description         | tool: load_model       |
    | rationale           | model_id: [text input] |
    ↓
User fills in model_id: "e_coli_core"
User clicks 💬 on Step 3, types "use double knockout"
User clicks "Re-plan with feedback"
    ↓
_collect_feedback() → "Step 3 (Gene Knockout Simulation): use double knockout"
POST /plan/revise → planner.revise_plan(query, current_plan, feedback)
    _build_revision_prompt → LLM → _parse_and_validate
    → revised plan (step 3 now has type="double")
    ↓
session_state.current_plan = revised plan → Phase B re-renders
    ↓
User is happy, clicks "Execute Plan"
    merge session state inputs into step dicts
    session_state.exec_steps = [step1, step2, step3]
    exec_cursor = 0 → st.rerun() → Phase C renders
    ↓
exec_cursor=0, len(exec_results)=0 → spinner → POST /plan/execute_step {step1}
    executor.execute_step(step1):
        fn = TOOL_FN_MAP["load_model"] = load_model from tools.py
        kwargs = {"model_id": "e_coli_core"}
        result = load_model(model_id="e_coli_core")
        → {"output": "Model e_coli_core loaded successfully."}
    exec_results = [result] → st.rerun()
    ↓
exec_cursor=0, len(exec_results)=1 → show result + [Continue] [Stop]
User clicks Continue → exec_cursor=1 → st.rerun()
    ↓
exec_cursor=1, len(exec_results)=1 → spinner → POST /plan/execute_step {step2}
    run_fba() → {"Objective value": "0.874", "status": "optimal"}
    exec_results = [result1, result2] → st.rerun()
    ↓
... and so on for step 3 (gene_knockout_simulation with type="double")
    ↓
exec_cursor=3 = len(exec_steps) → "All steps executed" screen
    Appends summary to chat_history
```

---

## 10. Key Constraints and Design Decisions

**The planner only selects from retrieved candidates.** The system prompt explicitly forbids the LLM from inventing steps not present in the retrieved candidates. All plan steps are grounded in real, published methodology. This becomes more important when validation rules are added later.

**`step_ids` are the join key.** Tier 2 entries store `step_ids` as a JSON-encoded string. The planner decodes them and re-queries ChromaDB using `T3_S1`, `T3_S2`, etc. as IDs to fetch full step details. This lookup-by-ID pattern ensures the complete sub-workflow is always reconstructed, even if some steps didn't score well against the query individually.

**`plan_include_{i}` is the single source of truth for inclusion.** The `included` field in the plan dict is only used to initialize the widget state. After that, only `session_state[f"plan_include_{i}"]` matters — for feedback collection, for execution filtering, and for re-planning. The plan dict itself is never mutated by the UI; it only changes when a new LLM call produces a new plan.

**Execution bypasses the agent deliberately.** The plan specifies exactly which tool to call with which parameters. Using the ReAct agent for execution would add an extra LLM call, possible misinterpretation, and non-determinism. Direct function calls in `executor.py` are deterministic and faster.

**The BGE embedding model loads once.** `GSMRetriever.__init__()` loads the SentenceTransformer model. Since `_retriever` is a module-level singleton that is never reset (unlike `_planner`, which resets when the LLM changes), the BGE model is loaded at most once per server process regardless of how many planning calls are made.

**All UI parameter values are strings.** Streamlit's `st.text_input` always returns strings. The `_coerce()` helper in `executor.py` converts them to `int` or `float` where possible before passing to tool functions. Without this, numeric parameters like `fraction_of_optimum="1.0"` or `reaction_count="500"` would cause type errors inside COBRApy.
