# AI Systems Overview — Agenti AI for COBRA

> Overview of the AI and LLM component in this project.

---

## What Does This Application Do?

This app lets a biologist type a question in plain English — like *"What happens to E. coli growth if I knock out gene b0351?"* — and get a real scientific answer, without writing a single line of code.

Under the hood, it uses an **AI agent** that understands the question, picks the right scientific tool, runs a metabolic simulation, and then rewrites the result in clear, readable language.

---

## The Big Picture

Here is the journey of a single user message through the system:

```
You type a message in the chat
         ↓
FastAPI receives it at the /chat/ endpoint
         ↓
The ReAct Agent reads it and decides what to do
         ↓
The Agent calls one or more scientific Tools (FBA, FVA, knockout, etc.)
         ↓
COBRApy runs the actual biology math
         ↓
The Agent gets the raw result back
         ↓
A second LLM pass cleans up the answer into plain English
         ↓
You read a clear, formatted response
```

---

## Component 1 — The AI Framework: LlamaIndex

**File:** `agent.py`, `tools.py`  
**Library:** `llama-index` (version 0.12.39)

**What it is:**  
LlamaIndex is an open-source Python framework for building AI agents and RAG (Retrieval-Augmented Generation) systems. Think of it as the "operating system" that the entire AI side of this application runs on.

**What it does here:**  
- Provides the `ReActAgent` class (see Component 2)
- Provides the `FunctionTool` class used to wrap every scientific function
- Provides the `Memory` class for conversation history
- Handles the loop between the LLM and the tools automatically

**Why LlamaIndex and not something else?**  
LlamaIndex makes it straightforward to give an LLM a set of Python functions as "tools" and let it decide which one to call. You don't need to write the reasoning loop yourself — LlamaIndex handles it.

---

## Component 2 — The Agent: ReActAgent

**File:** `agent.py`  
**Class:** `llama_index.core.agent.ReActAgent`

### What is an "Agent"?

An **agent** is an AI that doesn't just answer questions — it *takes actions*. Instead of giving you a one-shot response, it can think step-by-step, decide to run a tool, look at the result, and then think again before giving you a final answer.

### What is "ReAct"?

ReAct stands for **Reasoning + Acting**. It is a specific pattern for how an agent thinks:

```
Thought:  "The user wants FBA results. I should call run_flux_balance_analysis."
Action:   Calls the tool run_flux_balance_analysis()
Result:   Gets back the flux values from COBRApy
Thought:  "I have the FBA result. I can now answer the question."
Answer:   Returns the final response
```

This Thought → Action → Result loop can repeat multiple times in a single conversation turn if the agent needs to call several tools.

### How it is set up in this project

```python
agent = ReActAgent.from_tools(
    tools=all_tools,        # The 12 scientific tools (see Component 3)
    llm=llm,                # Whichever LLM the user picked
    system_prompt=...,      # Instructions telling the agent its role
    context=...,            # Background info about the tools and the system
    verbose=True            # Prints the Thought/Action/Result loop to console
)
```

The agent is re-created every time the user switches the LLM provider or model.

---

## Component 3 — The Tools (12 Scientific Functions)

**File:** `tools.py`  
**Class:** `llama_index.core.tools.FunctionTool`

### What is a "Tool"?

A **tool** is a regular Python function that the agent is allowed to call. The agent reads the tool's name and description to understand what it does, then decides whether to call it based on the user's question.

Each tool is created like this:

```python
my_tool = FunctionTool.from_defaults(
    fn=my_python_function,      # The actual Python function
    name="tool_name",           # Short name the LLM uses to call it
    description="What it does"  # Natural language description the LLM reads
)
```

### The 12 Tools and What They Do

| # | Tool Name | What It Does |
|---|-----------|--------------|
| 1 | `load_model` | Loads a metabolic model from a file or downloads it from BiGG/BioModels by ID |
| 2 | `check_model_loaded` | Confirms whether a model is currently loaded |
| 3 | `get_current_model_id` | Returns the ID of the currently loaded model |
| 4 | `model_data` | Returns key stats: how many reactions, metabolites, and genes the model has |
| 5 | `model_info` | Lists all reactions, metabolites, or genes in the model |
| 6 | `reaction_info` | Returns details about a specific reaction (bounds, formula, gene associations) |
| 7 | `metabolite_info` | Returns details about a specific metabolite (formula, charge, compartment) |
| 8 | `gene_info` | Returns details about a specific gene and which reactions it controls |
| 9 | `run_flux_balance_analysis` | Runs FBA — finds the optimal solution for a given objective (e.g., maximum growth) |
| 10 | `set_model_objective_value` | Changes which reaction is being maximized or minimized |
| 11 | `run_flux_variability_analysis` | Runs FVA — finds the range of possible flux values for every reaction |
| 12 | `gene_knockout_simulation` | Simulates deleting one or two genes and shows the effect on growth |
| 13 | `reaction_knockout_simulation` | Simulates deleting one or two reactions and shows the effect |
| 14 | `sample_metabolic_model` | Runs flux sampling (OptGP or ACHR) to explore the full solution space |

> The actual biology math in all of these tools is done by **COBRApy**, a well-established scientific Python library. The AI does not do the math — it decides *which* math to run and *how* to call it.

---

## Component 4 — The LLM Providers

**File:** `llm_factory.py`

### What is an LLM?

A **Large Language Model** (LLM) is the AI that reads text and writes text. In this project, the LLM serves two roles:
1. The *brain* of the ReActAgent — it reads the user's question and decides which tools to call.
2. A *response refiner* — after the agent produces a raw answer, a second LLM call rewrites it into clean, readable prose.

### Supported Providers

The project supports four different LLM providers, switchable at runtime from the UI:

| Provider | What It Is | Requires |
|----------|-----------|----------|
| **Ollama** | Runs an LLM *locally* on your own machine. No internet needed, no API key. | Ollama installed locally |
| **OpenAI** | Uses GPT-4 or other OpenAI models via the internet. | OpenAI API key |
| **Groq** | Cloud inference service, very fast due to custom LPU hardware. | Groq API key |
| **HuggingFace** | Uses open-source models from the HuggingFace Hub. | HuggingFace token |

### How the Factory Works

`llm_factory.py` has a single function `get_llm()` that takes a provider name, model name, and API key, and returns the correct LlamaIndex LLM object. This is the **factory pattern** — one function, many possible outputs depending on input.

```python
def get_llm(provider, model, api_key=None):
    if provider == "ollama":
        return Ollama(model=model)
    elif provider == "openai":
        return OpenAI(model=model, api_key=api_key)
    elif provider == "groq":
        return Groq(model=model, api_key=api_key)
    elif provider == "Hugging Face":
        return HuggingFaceLLM(model_name=model, tokenizer_name=model)
```

---

## Component 5 — Prompt Engineering

**File:** `prompts.py`

Prompts are the instructions given to the LLM to shape its behavior. This project uses **three separate prompts**, each serving a different role.

### Prompt 1 — System Prompt (for the ReActAgent)

This is the high-level instruction that tells the agent *who it is* and *what its job is*. It is given to the ReActAgent when it is created.

Key things it tells the agent:
- You are a metabolic modeling assistant
- Your users are biologists with no programming background
- You must use your tools to answer questions — never guess or hallucinate
- You are responsible for managing the loaded model and running simulations

### Prompt 2 — Agent Context

This is supplementary background information provided alongside the system prompt. It explains:
- What tools are available and what each one does
- How the COBRApy backend works
- How models are stored and referenced between queries

Think of the System Prompt as the job description and the Agent Context as the employee handbook.

### Prompt 3 — LLM Refinement Prompt

After the ReActAgent produces its raw answer (which may include JSON, numbers, and technical output), a **second LLM call** is made with this prompt to clean it up.

The template looks like:

```
User Query:
<user_input>

Agent Response:
<agentResponse>

Now rewrite the final response in clear, structured language.
Present any tabular data as a formatted table.
Do not add any information that was not in the agent response.
```

This two-stage approach means the agent can focus on *getting the right answer* while the second LLM call focuses on *presenting it well*.

---

## Component 6 — Memory

**File:** `agent.py`  
**Class:** `llama_index.core.memory.Memory`

```python
memory = Memory.from_defaults(session_id="metabolic_agent", token_limit=40000)
```

LlamaIndex's `Memory` class is initialized here to keep track of conversation history. It has a token limit of 40,000 tokens, which prevents the context window from overflowing in a long conversation.

> Note: As of the current codebase, the memory object is initialized but not yet wired into the agent's `from_tools()` call. It is set up as infrastructure for a future improvement where the agent will remember earlier exchanges in the same session.

---

## Component 7 — Session Management

**File:** `main.py`

Each conversation session has its own folder inside `outputs/`. When you start a session with a name like `"E_coli_experiment"`, the server creates:

```
outputs/
  E_coli_experiment/
    fva/              ← FVA result CSVs go here
    knockouts/        ← Knockout result CSVs go here
    flux_sampling/    ← Sampling result CSVs go here
```

This is not an AI feature per se, but it is important to the agentic workflow: tools automatically write their CSV outputs to the session directory, so all simulation results from a single conversation are grouped together.

---

## Component 8 — The Backend: FastAPI

**File:** `main.py`  
**Library:** FastAPI

FastAPI is the web server that sits between the Streamlit UI and the AI agent. It exposes REST API endpoints that the UI calls. The most important one is:

- **`POST /chat/`** — receives the user's message, calls `agent_query()`, and returns the response

Other endpoints handle uploading models, changing the LLM, configuring the sampler, and starting/ending sessions.

---

## Component 9 — The Frontend: Streamlit

**File:** `app.py`  
**Library:** Streamlit

Streamlit is a Python library for building simple web UIs. In this project it provides:
- The chat window where the user types questions and reads answers
- Dialog boxes for configuring the LLM (provider, model, API key)
- Dialog boxes for managing the metabolic model (upload, view stats, set objective)
- Session start/end controls

The Streamlit app talks to the FastAPI server via `requests` HTTP calls.

---

## How Everything Connects — One Complete Example

**User types:** *"Run FBA on the current model and tell me the growth rate."*

1. **Streamlit** sends `POST /chat/` to FastAPI with the message.
2. **FastAPI** calls `agent_query("Run FBA on the current model...")`.
3. **ReActAgent** reads the message. Its LLM (say, Groq) thinks:
   > *"This needs run_flux_balance_analysis. But first I should confirm a model is loaded."*
4. **Agent calls** `check_model_loaded()` → gets back `True, model = e_coli_core`.
5. **Agent thinks:** *"Good, model is loaded. Now I'll call FBA."*
6. **Agent calls** `run_flux_balance_analysis()`.
7. **COBRApy** runs the linear programming optimization and returns flux values including `Biomass_Ecoli_core_w_GAM: 0.8739`.
8. **Agent returns** the raw result to `agent_query()`.
9. **Second LLM call** (refinement prompt) rewrites the raw JSON into:
   > *"The Flux Balance Analysis completed successfully. The predicted growth rate of E. coli under the current conditions is **0.874 mmol/gDW/h**..."*
10. **FastAPI** returns the formatted response to Streamlit.
11. **You read** a clear, plain-English answer.

---

## Summary Table

| Component | Technology | Role |
|-----------|-----------|------|
| Agentic Framework | LlamaIndex 0.12.39 | Orchestrates the entire AI pipeline |
| Agent Type | ReActAgent | Thinks step-by-step, chooses tools, loops until done |
| Tool System | LlamaIndex FunctionTool | Wraps scientific Python functions for the agent to call |
| Biology Engine | COBRApy | Does the actual metabolic math (FBA, FVA, knockouts, sampling) |
| LLM Providers | Ollama / OpenAI / Groq / HuggingFace | The language models that power reasoning and response generation |
| LLM Abstraction | `llm_factory.py` (factory pattern) | Lets you swap LLM providers at runtime without changing any other code |
| Prompt Engineering | `prompts.py` | Three-layer prompt system: agent role, context, and refinement |
| Memory | LlamaIndex Memory (40k token limit) | Tracks conversation history within a session |
| Web Server | FastAPI | REST API connecting UI to agent |
| User Interface | Streamlit | Chat window and configuration dialogs |
| Session Storage | Local filesystem (`outputs/`) | Saves all simulation results per session |

---

## Key Concepts Glossary

| Term | Simple Explanation |
|------|--------------------|
| **LLM** | A large language model — an AI trained on text that can read and write natural language |
| **Agent** | An AI that can take actions (call tools) rather than just generating text responses |
| **ReAct** | A pattern where the agent alternates between Thinking and Acting until it has an answer |
| **Tool** | A Python function the agent is allowed to call when it needs to do something concrete |
| **Prompt** | Text instructions given to an LLM to define its role, constraints, and behavior |
| **FBA** | Flux Balance Analysis — a linear programming method to predict metabolic fluxes |
| **FVA** | Flux Variability Analysis — finds the minimum and maximum possible flux for each reaction |
| **COBRApy** | A Python library for constraint-based metabolic modeling |
| **SBML** | The XML file format used to store metabolic models |
| **Session** | A single user conversation, with its own output folder for saved results |
| **Factory Pattern** | A design pattern where one function creates and returns different objects based on input |
| **Two-Stage LLM** | Using one LLM call to get the right answer, then a second call to format it nicely |
