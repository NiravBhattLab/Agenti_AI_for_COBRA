
# 🧬 Metabolic Simulator

A simple AI-powered web application that allows biological researchers to upload, explore, and simulate metabolic models using natural language queries.

## 🚀 Project Goal

This project provides an intuitive interface for interacting with SBML-based metabolic models. It combines AI tools and metabolic simulation to help researchers run analyses like Flux Balance Analysis (FBA), view model statistics, and modify objectives — all through simple text-based interaction.

## 🏗️ Features

- 📂 Upload SBML models in `.xml` format
- 📊 Get key statistics from the loaded model (e.g., number of reactions, metabolites, genes)
- ⚙️ Run Flux Balance Analysis (FBA) using COBRApy
- 🧠 Ask questions via a built-in AI Agent that responds like a biological researcher
- ✅ Set objective functions dynamically for simulations

## 📁 Folder Structure

```
Metabolic-Simulator/
├── app.py             # FastAPI app with all endpoints
├── agent.py           # AI Agent logic and prompt injection
├── tools.py           # Simulation tools like FBA and stats
├── models.py          # Model loading and management logic
├── main.py            # App entry point
├── prompts.py         # Agent system prompt
└── uploads/           # Example SBML models
```

## ▶️ How to Use

1. **Start the FastAPI server** using `uvicorn main:app --reload`
2. **Start the Streamlit frontened** using `streamlit run app.py`
3. **Upload a model** using the `/upload_model/` endpoint
4. **Use the AI Agent** via the `/agent_query/` endpoint to run simulations or ask about the model

## 📦 Requirements

- Python 3.9+
- FastAPI
- COBRApy
- Uvicorn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 👨‍🔬 Ideal For

- Biological researchers
- Systems biology students
- Anyone exploring SBML-based metabolic models

---

Built with ❤️ for enabling scientific discovery with AI.
