# 🧬 Metabolic Simulator

A simple AI-powered web application that allows biological researchers to upload, explore, and simulate metabolic models using natural language queries.

## 🚀 Project Goal

This project provides an intuitive interface for interacting with SBML-based metabolic models. It combines AI tools and metabolic simulation to help researchers run analyses like Flux Balance Analysis (FBA), view model statistics, and modify objectives — all through simple text-based interaction.


## ✨ Features

- 🔬 Load SBML models from BiGG, BioModels, or upload your own
- 📊 Get quick statistics from the loaded model
- 🤖 ReAct-style LLM agent for tool-calling and reasoning
- 🧪 Run biological simulations:
  - Set Objective Functions (with Directions)
  - Flux Balance Analysis (FBA)
  - Gene/Reaction Knockouts (single/double)
  - Flux Variability Analysis (FVA)
  - Flux Sampling
- 📊 Export results as CSV for large queries
- 💬 Supports natural language querying for:
  - Reactions, metabolites, gene info
  - Simulation goals and interpretation
- 🔌 Multi-backend LLM support:
  - Local: **Ollama**
  - Remote: **OpenAI**, **Groq**, **HuggingFace**
  - **Note:** Use OpenAI/ Groq for Best Results [Click Here](#-miscellaneous)

## 🚀 Getting Started

1. Installation

```bash
# Clone repository
$ git clone https://github.com/your-org/Agenti_AI_for_COBRA.git
$ cd Agenti_AI_for_COBRA

# Create virtual environment
$ python -m venv venv && ./venv/bin/activate.ps1

# Install dependencies
$ pip install -r requirements.txt
```
2. **Get API KEY or use Ollama**: Supported services include OpenAI, Groq, HuggingFace or use Ollama.
3. **Start the FastAPI Backend**: `uvicorn main:app --reload`
4. **Start the Streamlit Frontened**: `streamlit run app.py`


## 🧪 Example Usage

### Natural Language:
- "What is the metadata of the loaded model?"
- "What are first ten reactions in the model?"
- "Run Flux Balance Analysis on the model."

For more sample queries refer to ```Testing/demo_prompts.txt```

## 📦 MISCELLANEOUS

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Install Ollama (if preferred):**

- Server: [Download Server](https://ollama.com/download)
- Models: [Browse Models](https://ollama.com/search)

**Get API Keys:**

- Groq: [Get Groq Key](https://console.groq.com/keys)
- OpenAI: [Get OpenAI Key](https://platform.openai.com/api-keys)
- HuggingFace: [Get HF Token](https://huggingface.co/settings/tokens) (Required for Gated Repositories)

## 🧹 Future Work

* 🌍 UI for exploring and downloading simulation outputs
* 🔬 Inference over multi-model libraries (VMH, ModelSEED)
* 🧠 Smart model suggestion based on query intent
* 📊 Model visualization (SBML layout rendering)

## 👨‍🔬 Ideal For

- Biological researchers
- Systems biology students
- Anyone exploring SBML-based metabolic models

---

Built with ❤️ for enabling scientific discovery with AI by [Aadhitya Sriram](https://github.com/aadhitya-sriram) and [Pavan Kumar](https://github.com/pavan-kumar-s).
