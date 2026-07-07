# MetaPilot

An AI-enabled web application that lets domain-focused researchers — including experimental biologists with no systems-biology programming background — design and run genome-scale metabolic (GSM) modelling workflows using natural language.

***This repository is under active development. Please pull periodically.***

## Parts of MetaPilot

- **MetaInteract** — a free-form, ReAct-style agentic chat interface. You describe what you want in plain English and the agent decides which COBRApy tools to call, in what order, to answer you.
- **MetaPlan** — a plan-and-execute interface. Give it a high-level research objective and it retrieves relevant published methodology, drafts a structured multi-step analysis plan, lets you review/edit/approve it, then executes and interprets each step.
- **Shared backend** — a single FastAPI service holds the active model (`ModelManager`), the LLM instance, and the full COBRApy tool suite, so both modes operate on the same live model and session state.
- **Procedural Knowledge Base (PKB)** — a vector database of workflow steps extracted from thousands of published GSM papers, which grounds MetaPlan's plan generation in real methodology rather than free invention.

## Project Goal

GSM modelling is a powerful framework for analyzing cellular metabolism, but it demands programming proficiency and fluency with a fragmented, fast-moving software ecosystem — putting it out of reach for most domain-focused biologists. MetaPilot removes that barrier by offering unified, natural-language-driven access to the full GSM modelling lifecycle — reconstruction, curation, constraint definition, simulation, perturbation analysis, and visualization — without writing a single line of code.

## Features

- Load SBML models from BiGG, BioModels, or upload your own
- Reconstruct new models with **CarveMe**, build context-specific models with **CORDA**, or use **Mackinac**/ModelSEED
- Curate and QC models: MEMOTE quality reports, consistency and mass-balance checks, gap-filling, pruning unused reactions/metabolites, blocked-reaction detection
- Configure models: set objectives (with direction), edit reaction bounds, add reactions, remove genes, find minimal medium
- Run simulations:
  - Flux Balance Analysis (FBA), parsimonious FBA, geometric FBA
  - Flux Variability Analysis (FVA), production envelopes
  - Flux Sampling
  - Gene/reaction knockouts (single/double), essential gene/reaction finding, MOMA, ROOM
- Visualize fluxes on Escher maps
- Export results as CSV for large queries
- Named sessions with persistent artifacts (FVA tables, knockout summaries, sampling matrices) that can be saved or discarded on exit
- Supports natural language querying for reactions, metabolites, gene info, and simulation results
- Multi-backend LLM support, switchable at runtime without restarting: **Groq**, **OpenAI**, **Gemini**, **Ollama** (local), **Hugging Face Inference**, and **llama.cpp** (local GGUF models)
  - **Note:** Use Groq/OpenAI/Gemini for best results

## Getting Started

1. Installation

```bash
# Clone repository (MetaPilot branch)
$ git clone https://github.com/NiravBhattLab/Agenti_AI_for_COBRA.git -b MetaPilot
$ cd Agenti_AI_for_COBRA

# Option A: virtual environment
$ python -m venv venv && ./venv/Scripts/activate
$ pip install -r requirements.txt

# Option B: conda environment (recommended — pulls in non-pip deps like
# prodigal/glpk needed by the CarveMe/CORDA reconstruction tools)
$ conda env create -f environment.yml
$ conda activate agentic_cobra
```

2. **Get an API key or use a local provider**: supported services are Groq, OpenAI, Gemini, Hugging Face, or a local model via Ollama/llama.cpp.
3. **Start the FastAPI backend**: `uvicorn main:app --reload`
   > On first run, the Procedural Knowledge Base is automatically downloaded from the [Hugging Face Hub dataset `sistasaathvik/gsm_procedural_knowledge_base`](https://huggingface.co/datasets/sistasaathvik/gsm_procedural_knowledge_base) into a local ChromaDB directory. This is a one-time download that MetaPlan's retriever queries for every planning request.
4. **Start the Streamlit frontend**: `streamlit run app.py`
5. Name a session, load or upload a model, then either chat freely (MetaInteract) or toggle **Plan Mode** to draft a multi-step plan (MetaPlan).

### Example Queries

- "What is the metadata of the loaded model?"
- "What are the first ten reactions in the model?"
- "Run Flux Balance Analysis on the model."
- "Build a context-specific model for a breast cancer cell line using CORDA."

## Future Work

- Expand the tool suite: MATLAB COBRA Toolbox-only methods (strain design, transcriptomic integration, thermodynamic FBA), plus live literature search (PubMed/Semantic Scholar) to ground answers beyond the static PKB
- Fine-tune the retrieval embedding model and the agent LLMs on GSM-specific literature to improve retrieval quality and plan/tool-call accuracy
- Move beyond the current ReAct architecture toward planning frameworks with stronger long-horizon reasoning, better error recovery, and less reliance on manual parameter input

---

Built for enabling scientific discovery with AI, by [Saathvik Sista](https://github.com/saathviksista), [Aadhitya Sriram](https://github.com/aadhitya-sriram) and [Pavan Kumar](https://github.com/pavan-kumar-s).
