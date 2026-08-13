# Codebase Documentation

A brief, per-script breakdown of what everything in this repository does. For a
feature-level overview (what the app does, how to install/run it), see the
[main README](../README.md).

## Backend (root)

- **`main.py`** — FastAPI backend and entrypoint. Defines every HTTP endpoint
  the frontend calls: session management, model/CSV/FASTA upload, objective
  setting, LLM provider configuration, the free-form chat endpoint
  (`/chat/`), and the MetaPlan endpoints (`/plan/generate`, `/plan/revise`,
  `/plan/execute_step`, `/plan/retry_step`, `/plan/summarize`). Also
  auto-downloads the procedural knowledge base from Hugging Face Hub on first
  run if it isn't present locally.
- **`agent.py`** — Builds the MetaInteract chat agent: a LlamaIndex
  `ReActAgent` wired up with every tool in `tools.py` and the prompts from
  `prompts.py`. `agent_query()` runs a user message through the agent, and if
  the agent decided it needs a file upload or missing tool parameters instead
  of running a tool, intercepts that signal and returns it to the frontend as
  a structured request rather than plain text. Otherwise the raw tool result
  is passed through a second LLM call (`reformat_tool_result`) to turn it into
  a readable, scientific-toned response.
- **`tools.py`** — The tool library. Every function the agent (or the
  planner, via `planner/tool_registry.py`) can call: loading/searching models
  (BiGG, BioModels, or upload), inspecting reactions/metabolites/genes,
  running simulations (FBA, pFBA, geometric FBA, FVA, gene/reaction
  knockouts, MOMA, ROOM, flux sampling), editing models (add reaction, set
  bounds, remove genes, prune unused reactions/metabolites, gap-filling),
  quality control (MEMOTE report, consistency/mass-balance checks, blocked
  reactions, essential gene/reaction finding), reconstruction
  (`build_model_with_carveme`, `build_model_with_mackinac`,
  `build_context_model_with_corda`), Escher visualization, and the two
  signal-only tools (`request_file_upload`, `request_tool_inputs`) the agent
  uses to ask the frontend for missing input instead of guessing. Tool
  functions destined for the planner also carry a `_planner_meta` dict
  describing their parameters for the plan-editor UI.
- **`models.py`** — `ModelManager`: holds the currently loaded COBRApy
  model(s) in memory, keyed by model ID. `load_model_by_id` resolves a
  user-given ID/organism name against BiGG and BioModels (exact match first,
  then fuzzy search with organism-name aliasing), `load_sbml` loads an
  uploaded file, and the manager tracks the active model, uploaded bounds
  data, and the flux-sampling method to use.
- **`llm_factory.py`** — `get_llm(provider, model, api_key)`: a small factory
  that returns the right LlamaIndex LLM wrapper for the selected provider
  (Groq, OpenAI, Gemini, Hugging Face Inference, Ollama, or a local
  llama.cpp GGUF model).
- **`prompts.py`** — All system/context prompt strings used by the chat
  agent: the main system prompt, tool-usage context, the reformatting prompt
  for the second LLM pass, and three protocol prompts that tell the agent how
  to ask for a missing file upload, missing tool parameters, or PATRIC
  credentials (for Mackinac) instead of fabricating them.
- **`ptypes.py`** — A single tiny pydantic model (`LoadModelInput`); mostly a
  placeholder for future typed-input expansion.
- **`app.py`** — The Streamlit frontend. Renders the welcome page, the
  MetaInteract chat UI and the MetaPlan plan-review/execute UI, session
  naming, LLM configuration, file-attach, and model-management dialogs. Talks
  to `main.py` purely over HTTP.

## `planner/` — the MetaPlan plan-and-execute system

See [`planner/README.md`](../planner/README.md) for full details on each
script. Summary:

- **`retrieve.py`** — `GSMRetriever`: two-phase semantic search over a
  ChromaDB collection of steps extracted from published GSM modelling papers,
  used to find relevant prior workflows for a user's query.
- **`planner_core.py`** — `GSMPlanner`: turns retrieval results into a
  structured, editable, executable step-by-step plan via an LLM call, and can
  revise a plan from user feedback.
- **`tool_registry.py`** — Builds the tool lookup tables (`TOOL_FN_MAP`,
  `TOOL_USER_INPUTS`, `AVAILABLE_TOOLS_TEXT`) that `planner_core.py` and
  `executor.py` rely on, sourced from `_planner_meta` on functions in
  `tools.py`.
- **`executor.py`** — `execute_step()`: runs one plan step by calling its
  tool function directly (no LLM/agent in the loop), coercing user-supplied
  string inputs to the right type.
- **`config.py`** — Shared config constants (ChromaDB path, collection name,
  embedding model), overridable via environment variables.
- **`__init__.py`** — Re-exports `GSMRetriever`, `GSMPlanner`, and
  `execute_step` for convenient importing.

## Environment / setup

- **`environment-linux.yml`** — Conda environment file for Linux/macOS.
  Includes `prodigal` and `glpk` as conda (bioconda) packages, since bioconda
  ships native builds for those platforms.
- **`environment-windows.yml`** — Conda environment file for Windows. Omits
  `prodigal` (bioconda has no win-64 build) and pins
  `llama-cpp-python==0.3.19` from a prebuilt-wheel index to avoid needing a
  C/C++ compiler. Meant to be used via `setup_windows.ps1`, not directly.
- **`setup_windows.ps1`** — Creates the `agentic_cobra` conda environment
  from `environment-windows.yml`, then downloads the official Windows
  binaries for `prodigal` and `diamond` (both required by the CarveMe/Mackinac
  reconstruction tools, neither shipped by bioconda for Windows) into the
  environment's `Library\bin`. Idempotent — safe to re-run.

## Data / artifacts

- **`gene_expression_data/`** — Example/working input files for
  `build_context_model_with_corda`. See
  [`gene_expression_data/README.md`](../gene_expression_data/README.md) for
  the accepted CSV formats.
- **`uploads/`** — Sample SBML models and bounds CSVs used for local testing.
- **`artifacts/`** — Per-session working directory (uploads and generated
  outputs — FVA tables, knockout summaries, sampling matrices, CarveMe/CORDA
  models) created at runtime by `main.py`'s session endpoints.
- **`gsm_procedural_knowledge_base/`** — The ChromaDB vector store that
  `planner/retrieve.py` queries. Auto-downloaded from the Hugging Face Hub
  dataset `sistasaathvik/gsm_procedural_knowledge_base` on first backend run
  if not already present.
