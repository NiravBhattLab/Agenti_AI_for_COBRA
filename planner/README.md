# `planner/` — the MetaPlan plan-and-execute system

MetaPlan retrieves relevant published GSM (genome-scale metabolic) modelling
methodology for a user's research question, drafts a structured multi-step
plan from it, lets the user review/edit/approve the plan, then executes each
step against the tools in `tools.py` and interprets the results. This folder
is the implementation of that pipeline, independent of the chat (MetaInteract)
ReAct agent in `agent.py`.

## Scripts

- **`retrieve.py`** — `GSMRetriever`. Two-phase semantic retrieval over a
  ChromaDB collection of workflow steps extracted from published GSM
  modelling papers, indexed at four tiers (0 = paper, 1 = pipeline category,
  2 = sub-workflow, 3 = individual step).
  - **Phase 1 (coarse)**: embeds the query and searches tier 0 + tier 1 to
    shortlist which papers/pipeline stages are relevant.
  - **Phase 2 (fine)**: searches tier 2 (sub-workflow hits — the primary plan
    seeds, each carrying a full `step_ids` list for a validated workflow) and
    tier 3 (individual step hits, supplementary, optionally filterable by
    `technique`). Phase 2 can optionally be restricted to the phase-1
    shortlist (`filter_by_phase1`) to trade recall for precision; unfiltered
    is the default and recommended setting.
  - Uses the `BAAI/bge-large-en-v1.5` embedding model with BGE's asymmetric
    query/document prompt prefixes (`BGE_QUERY_PREFIX` at query time — the
    index was built with `BGE_DOC_PREFIX`, so this must not change without
    re-indexing).
  - `retrieve(query, ...)` returns a `RetrievalResult` dict (see the
    docstring in the file for the exact schema) consumed by
    `planner_core.py`.
  - Runnable standalone for debugging: `python -m planner.retrieve --query "..."`.

- **`planner_core.py`** — `GSMPlanner`. Turns a `RetrievalResult` into a
  concrete, structured, user-editable plan:
  1. Fetches full step details from ChromaDB for the top-scored tier 2
     sub-workflows (retrieval only returns top-k steps, not necessarily every
     step in a selected sub-workflow).
  2. Builds a candidate-step set for the LLM.
  3. Makes one LLM call (via the injected `llm_fn`) to select, order, and
     justify plan steps — constrained to only the retrieved candidates and
     the tool names in `AVAILABLE_TOOLS_TEXT`.
  4. Parses/validates the LLM output into the typed plan dict shape described
     in the module docstring (step name, description, tool, required user
     inputs with descriptions/defaults, source paper/step, rationale).
  5. Maps a small set of opaque technique aliases (e.g. `fluxVariability`,
     `ACHR`) to tool names via `TECHNIQUE_TO_TOOL`, where the LLM can't
     reasonably infer the mapping from the tool description alone.
  - `plan(query, retrieval_result)` produces the initial plan;
    `revise_plan(query, current_plan, feedback)` regenerates it incorporating
    free-text user feedback (e.g. "also add a gene knockout after FVA").
  - The planner is LLM-provider-agnostic: it just needs a callable matching
    `llm_fn(prompt: str, system_prompt: str) -> str`. `main.py` supplies a
    bridge (`_make_planner_llm_fn`) that wraps whatever LlamaIndex LLM is
    currently configured for the chat agent.

- **`tool_registry.py`** — Central registry bridging `tools.py` and the
  planner. Reads the `_planner_meta` dict attached to each planner-usable
  function in `tools.py` (name, description, and per-parameter
  description/default/required/UI hints) and builds:
  - `TOOL_FN_MAP` — `{tool_name: function}`, used by `executor.py` to actually
    call a tool.
  - `TOOL_USER_INPUTS` — `{tool_name: {param: {description, value, required,
    ...}}}`, used by `planner_core.py` for plan validation and by the
    Streamlit plan editor to render input fields (including file-upload
    widgets, via the optional `input_type`/`file_types` hints).
  - `AVAILABLE_TOOLS_TEXT` — a formatted `name — description` listing injected
    directly into the planner's LLM prompt so it knows what it's allowed to
    use.
  - A `"manual"` pseudo-tool is registered separately for plan steps that
    have no automated equivalent and require the user to act outside the app.
  - Adding a new planner-usable tool requires only: define it in `tools.py`,
    attach a `_planner_meta` dict, and add it to `_TOOL_FUNCTIONS` here.

- **`executor.py`** — `execute_step(step)`. Executes a single plan step by
  calling the mapped tool function **directly** — this bypasses the
  MetaInteract ReAct agent entirely, since by execution time the user has
  already supplied/approved every parameter through the plan UI. Coerces
  string-typed `user_inputs` values to `int`/`float`/comma-split `list` where
  they parse cleanly, calls the tool, and normalizes the return value (str,
  dict, or `pandas.DataFrame`) into a JSON-serializable dict for the
  frontend. The `"manual"` tool short-circuits with a fixed "requires manual
  execution" response.

- **`config.py`** — Shared constants: `CHROMA_DB_PATH` (defaults to
  `<repo_root>/gsm_procedural_knowledge_base`), `COLLECTION_NAME` (defaults to
  `gsm_workflows_v2`), and `EMBEDDING_MODEL` (`BAAI/bge-large-en-v1.5`). The
  first two are overridable via the `CHROMA_DB_PATH` / `CHROMA_COLLECTION`
  environment variables.

- **`__init__.py`** — Re-exports `GSMRetriever`, `GSMPlanner`, and
  `execute_step` so callers can `from planner import GSMRetriever` etc.
  instead of reaching into individual submodules.

## How a MetaPlan request flows through these scripts

1. `main.py`'s `/plan/generate` calls `GSMRetriever.retrieve(query)`
   (`retrieve.py`), then `GSMPlanner.plan(query, retrieval_result)`
   (`planner_core.py`), which internally consults `AVAILABLE_TOOLS_TEXT` from
   `tool_registry.py`.
2. The user reviews/edits the returned plan in the Streamlit UI (`app.py`);
   edits and re-generation feed back through `/plan/revise` →
   `GSMPlanner.revise_plan`.
3. Approving a step calls `/plan/execute_step` → `executor.execute_step`,
   which looks the tool up in `TOOL_FN_MAP` and calls it with the user's
   filled-in `user_inputs`.
4. `main.py` separately interprets each raw tool result and, once all steps
   are done, summarizes the whole run — both via direct LLM calls, not
   through this package.
