from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import os
from fastapi.responses import JSONResponse, FileResponse
from cobra.io import write_sbml_model
from fastapi.middleware.cors import CORSMiddleware
from tools import set_model_objective, set_model_manager, set_session_dir, set_session_uploads_dir
from llm_factory import get_llm
from pydantic import BaseModel
from agent import agent_query, setup_agent
from models import ModelManager
from pathlib import Path
import shutil
import pandas as pd
import os
import re
import chromadb
from planner.retrieve import GSMRetriever
from planner.planner_core import GSMPlanner
from planner.executor import execute_step as _execute_step
from planner.config import CHROMA_DB_PATH, COLLECTION_NAME

_HF_REPO_ID = "sistasaathvik/gsm_procedural_knowledge_base"

def _ensure_knowledge_base():
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Knowledge base not found at '{CHROMA_DB_PATH}'. Downloading from Hugging Face Hub...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            local_dir=CHROMA_DB_PATH,
        )
        print("Knowledge base downloaded successfully.")

_ensure_knowledge_base()

SESSION_DIR: Path | None = None

def _make_session_dir(name: str) -> Path:
    safe = re.sub(r'[<>:"/\\|?*\s]+', '_', name).strip('_') or "session"
    session = Path("artifacts") / safe
    (session / "outputs" / "fva").mkdir(parents=True, exist_ok=True)
    (session / "outputs" / "knockouts").mkdir(parents=True, exist_ok=True)
    (session / "outputs" / "flux_sampling").mkdir(parents=True, exist_ok=True)
    (session / "uploads").mkdir(parents=True, exist_ok=True)
    return session

model_manager = ModelManager()
set_model_manager(model_manager)

_retriever: GSMRetriever | None = None
_planner: GSMPlanner | None = None


def _make_planner_llm_fn():
    """Return a sync llm_fn(prompt, system_prompt)->str that wraps the current Llama-Index LLM."""
    from llama_index.core.llms import ChatMessage

    def llm_fn(prompt: str, system_prompt: str) -> str:
        import agent as _agent_module
        _llm = _agent_module.llm  # read current value at call time, not at creation time
        if _llm is None:
            raise RuntimeError("LLM not configured. Please set up the LLM via the LLM Configuration dialog first.")
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prompt),
        ]
        response = _llm.chat(messages)
        return response.message.content

    return llm_fn


def _get_planner():
    global _retriever, _planner
    if _retriever is None:
        _retriever = GSMRetriever()
    if _planner is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        _planner = GSMPlanner(llm_fn=_make_planner_llm_fn(), collection=collection)
    return _retriever, _planner

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_llm_config = {
    "provider": "groq",
    "model": "llama-3.1-8b-instant",
    "api_key": None
}

class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: str = None

class ObjectiveInput(BaseModel):
    objective_str: str
    direction: str = "max"

class ChatRequest(BaseModel):
    message: str

class ToolInputsSubmitRequest(BaseModel):
    tool_name: str
    original_message: str
    params: dict

class RetryStepRequest(BaseModel):
    step: dict
    error: str
    history: list  # [{"fix_applied": str, "error": str}, ...]


@app.post("/upload_model/")
async def upload_model(file: UploadFile = File(...)):
    if SESSION_DIR is None:
        raise HTTPException(status_code=400, detail="No active session. Create a session first.")
    if not file.filename.endswith(('.xml', '.sbml')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an SBML file (.xml or .sbml)")
    try:
        file_path = SESSION_DIR / "uploads" / file.filename
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        model_id = model_manager.load_sbml(file_path)
        await file.close()
        return {"status": "success", "model_id": model_id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid SBML file: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    if SESSION_DIR is None:
        raise HTTPException(status_code=400, detail="No active session. Create a session first.")
    try:
        file_path = SESSION_DIR / "uploads" / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        model_manager.bounds_data = pd.read_csv(file_path).values.tolist()
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/upload_data_file/")
async def upload_data_file(file: UploadFile = File(...)):
    """Save an arbitrary tool-input file (expression CSV, FASTA, bounds CSV, ...)
    into the session uploads directory and return its filename. Unlike
    /upload_csv/ this has no side effects on the model or bounds — tools read the
    file themselves by resolving the filename against the session uploads dir."""
    if SESSION_DIR is None:
        raise HTTPException(status_code=400, detail="No active session. Create a session first.")
    try:
        file_path = SESSION_DIR / "uploads" / file.filename
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        await file.close()
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@app.get("/download_model/")
async def download_model():
    if SESSION_DIR is None:
        raise HTTPException(status_code=400, detail="No active session.")
    try:
        model = model_manager.get_current_model()
        export_path = SESSION_DIR / "uploads" / f"{model.id}_manual_step.xml"
        write_sbml_model(model, str(export_path))
        return FileResponse(
            path=str(export_path),
            filename=f"{model.id}.xml",
            media_type="application/xml",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_objective/")
async def set_objective(obj: ObjectiveInput):
    try:
        expr_str, dirr = obj.objective_str, obj.direction.lower()
        if expr_str is None:
            raise HTTPException(status_code=400, detail="Objective expression is empty.")
        clean_str = expr_str.replace(" ", "")
        if not clean_str:
            raise HTTPException(status_code=400, detail="Objective expression is empty after removing spaces.")
        if dirr not in {"max", "min"}:
            raise HTTPException(status_code=400, detail="Direction must be 'max' or 'min'.")
        terms = re.split(r"(?=[+-])", clean_str)
        objective_dict = {}
        for term in terms:
            if not term:
                continue
            m = re.fullmatch(r"([+-]?\d*\.?\d*)\*?([A-Za-z0-9_]+)", term)
            if not m:
                raise HTTPException(status_code=400, detail=f"Could not parse term: {term}")
            coeff_str, rxn_id = m.groups()
            if coeff_str in ("", "+", "-"):
                coeff = 1.0 if coeff_str != "-" else -1.0
            else:
                coeff = float(coeff_str)
            objective_dict[rxn_id] = objective_dict.get(rxn_id, 0.0) + coeff
        if not objective_dict:
            raise HTTPException(status_code=400, detail="No valid terms found in objective expression.")
        result = set_model_objective(objective_dict, dirr)
        if isinstance(result, dict):
            if "objective" in result:
                result["objective"] = str(result["objective"])
            if "direction" in result:
                result["direction"] = str(result["direction"])
            return result
        return {
            "status": "Objective set successfully.",
            "objective": str(getattr(model_manager.get_current_model().objective, "expression", "")),
            "direction": dirr,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_stats/")
async def get_stats():
    try:
        model = model_manager.get_current_model()
        model_id = str(model.id)
        objective_reaction = str(model.objective.expression) if model.objective.expression else "Not Set Yet"
        reactions_count = len(model.reactions)
        metabolites_count = len(model.metabolites)
        genes_count = len(model.genes)

        stats = f"""
        Model ID: {model_id}\n
        Objective Reaction: {objective_reaction}\n
        Reactions Count: {reactions_count}\n
        Metabolites Count: {metabolites_count}\n
        Genes Count: {genes_count}\n
        Groups Count": {len(model.groups)}\n
        Compartments Count": {len(model.compartments)}\n
        Compartments: {str([v for k,v in model.compartments.items()])}\n
        """

        return {"stats": stats, "status_code": 200}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_llm/")
def set_llm(config: LLMConfig):
    global agent, llm, current_llm_config, _planner
    try:
        llm = get_llm(config.provider, config.model, config.api_key)
        current_llm_config.update(config.dict())
        setup_agent(llm)
        _planner = None  # rebuild planner with new LLM on next planning call
        return {"status": "LLM updated", "provider": config.provider, "model": config.model}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Failed to set LLM: {str(e)}"})

@app.post("/set_sampler/")
def set_sampler(req: dict):
    try:
        method = (req or {}).get("method")
        if not method:
            raise HTTPException(status_code=400, detail="'method' is required in JSON body.")
        details = model_manager.set_sampler(method)
        return {"status": "ok", "sampler": details["response"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set Sampler: {e}")


class CreateSessionRequest(BaseModel):
    session_name: str

@app.post("/create_session/")
def create_session(req: CreateSessionRequest):
    global SESSION_DIR
    if not req.session_name.strip():
        raise HTTPException(status_code=400, detail="Session name cannot be empty.")
    SESSION_DIR = _make_session_dir(req.session_name.strip())
    set_session_dir(SESSION_DIR / "outputs")
    set_session_uploads_dir(SESSION_DIR / "uploads")
    return {"status": "created", "session_name": SESSION_DIR.name, "session_dir": str(SESSION_DIR)}

@app.get("/session_info/")
def session_info():
    if SESSION_DIR is None:
        return {"session_dir": None, "session_name": None}
    return {"session_dir": str(SESSION_DIR), "session_name": SESSION_DIR.name}

class EndSessionRequest(BaseModel):
    keep: bool = True

@app.post("/end_session/")
def end_session(req: EndSessionRequest):
    if SESSION_DIR is None:
        return {"status": "no_session"}
    if not req.keep and SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)
        return {"status": "deleted", "session_name": SESSION_DIR.name}
    return {"status": "kept", "session_dir": str(SESSION_DIR)}

@app.post("/shutdown/")
def shutdown(background_tasks: BackgroundTasks):
    import signal, time
    pid = os.getpid()
    def _quit():
        time.sleep(0.3)
        os.kill(pid, signal.SIGKILL)
    background_tasks.add_task(_quit)
    return {"status": "shutting_down"}

class PlanRequest(BaseModel):
    query: str

class ReviseRequest(BaseModel):
    query: str
    current_plan: dict
    feedback: str

class ExecuteStepRequest(BaseModel):
    step: dict

class SummarizeRequest(BaseModel):
    query: str
    plan_summary: str
    steps: list[dict]
    results: list[dict]


@app.post("/plan/generate")
def plan_generate(req: PlanRequest):
    import agent as _agent_module
    if _agent_module.llm is None:
        raise HTTPException(status_code=400, detail="LLM not configured. Open the LLM Configuration dialog and set a provider first.")
    try:
        retriever, planner = _get_planner()
        retrieval_result = retriever.retrieve(req.query)
        plan = planner.plan(req.query, retrieval_result)
        return {"plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan/revise")
def plan_revise(req: ReviseRequest):
    try:
        _, planner = _get_planner()
        revised = planner.revise_plan(req.query, req.current_plan, req.feedback)
        return {"plan": revised}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_STEP_INTERP_SYSTEM = (
    "You are reporting the outcome of a metabolic modeling step. "
    "Start with one line: the step name and whether it succeeded or failed (e.g. 'Load E. coli core model — succeeded.'). "
    "Then report key results concisely: numbers, fluxes, counts, status. "
    "If the output contains a file path or saved artifact, show it explicitly. "
    "Use a markdown table for multiple numeric values. "
    "Do NOT describe what the step does or add biological commentary. "
    "Maximum 6 lines total."
)


def _interpret_step_result(step: dict, result: dict) -> str:
    import json as _json
    try:
        llm_fn = _make_planner_llm_fn()
        prompt = (
            f"Step: {step.get('step_name', '')} — {step.get('description', '')}\n"
            f"Tool: {step.get('tool', '')}\n"
            f"Result: {_json.dumps(result, default=str)[:2000]}"
        )
        return llm_fn(prompt, _STEP_INTERP_SYSTEM)
    except Exception:
        return ""


@app.post("/plan/execute_step")
def plan_execute_step(req: ExecuteStepRequest):
    try:
        result = _execute_step(req.step)
        interpretation = _interpret_step_result(req.step, result)
        return {"result": result, "interpretation": interpretation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_PLAN_SUMMARY_SYSTEM = (
    "You are answering a user's metabolic modeling question based on completed workflow results. "
    "Write a direct answer (3–6 sentences) to the user's original question. "
    "Lead with the key finding. Include critical numbers, flux values, model states, or artifact file paths. "
    "Do not describe what the steps do — only what they found or produced. "
    "Use plain prose; no headers or bullet points."
)


@app.post("/plan/summarize")
def plan_summarize(req: SummarizeRequest):
    try:
        import json as _json
        llm_fn = _make_planner_llm_fn()
        steps_text = "\n".join(
            f"Step {i+1} — {s.get('step_name', '')}: "
            f"tool={s.get('tool', '')} | "
            f"params={_json.dumps({k: v.get('value') for k, v in s.get('user_inputs', {}).items()}, default=str)} | "
            f"result={req.results[i].get('__interpretation__', '') or _json.dumps({k: v for k, v in req.results[i].items() if k != '__interpretation__'}, default=str)[:500]}"
            for i, s in enumerate(req.steps)
            if i < len(req.results)
        )
        prompt = (
            f"User's question: {req.query}\n\n"
            f"Plan goal: {req.plan_summary}\n\n"
            f"Step outcomes:\n{steps_text}"
        )
        return {"summary": llm_fn(prompt, _PLAN_SUMMARY_SYSTEM)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan/retry_step")
def plan_retry_step(req: RetryStepRequest):
    from planner.tool_registry import TOOL_USER_INPUTS
    import json as _json, re as _re

    tool_name = req.step.get("tool", "")
    tool_params = TOOL_USER_INPUTS.get(tool_name, {})
    current_params = {k: v.get("value") for k, v in req.step.get("user_inputs", {}).items()}
    param_descs = {k: v.get("description", "") for k, v in tool_params.items()}
    history_text = "".join(
        f"\nAttempt {i}: fix='{h.get('fix_applied', '')}' → error: {h.get('error', '')}"
        for i, h in enumerate(req.history, 1)
    )

    system = (
        "You are a metabolic modeling assistant fixing a failed tool call. "
        "Analyze the error and output ONLY a JSON object with corrected parameter values. "
        "Output nothing else — just the JSON."
    )
    prompt = (
        f"Tool: {tool_name}\n"
        f"Step: {req.step.get('step_name', '')} — {req.step.get('description', '')}\n"
        f"Parameter descriptions: {_json.dumps(param_descs)}\n"
        f"Current values: {_json.dumps(current_params)}\n"
        f"Error: {req.error}"
        + (f"\nPrevious attempts:{history_text}" if req.history else "")
        + "\n\nOutput corrected parameters as JSON:"
    )

    try:
        llm_fn = _make_planner_llm_fn()
        raw = llm_fn(prompt, system)
        m = _re.search(r'\{.*?\}', raw, _re.DOTALL)
        corrected = _json.loads(m.group() if m else raw.strip())
    except Exception as e:
        return {"result": {"error": f"LLM fix suggestion failed: {e}"}, "fix_applied": "none", "interpretation": ""}

    new_inputs = {
        k: dict(v, value=corrected.get(k, v.get("value")))
        for k, v in req.step.get("user_inputs", {}).items()
    }
    new_step = dict(req.step, user_inputs=new_inputs)

    changes = {k: corrected[k] for k in corrected if corrected[k] != current_params.get(k)}
    fix_applied = f"Changed: {_json.dumps(changes)}" if changes else "No parameter changes"

    result = _execute_step(new_step)
    interpretation = _interpret_step_result(new_step, result) if "error" not in result else ""

    return {
        "result": result,
        "fix_applied": fix_applied,
        "corrected_params": corrected,
        "interpretation": interpretation,
    }


@app.post("/tools/detect_new_metabolites")
async def detect_new_metabolites(body: dict):
    """Return metabolite IDs in an equation string that are not in the current model."""
    import re as _re

    # Explicit no-model check before doing anything else
    if not model_manager.current_model_id:
        return {"error": "no_model", "message": "No model is currently loaded. Load a model first."}
    try:
        model = model_manager.get_current_model()
        if isinstance(model, dict):
            return {"error": "no_model", "message": "No model is currently loaded. Load a model first."}
    except Exception as e:
        return {"error": "no_model", "message": f"Could not access model: {e}"}

    equation = body.get("equation", "")

    # Split on separator (check compound forms first to avoid splitting inside <=>)
    for sep in ["<-->", "<=>", "-->", "=>"]:
        if sep in equation:
            parts = equation.split(sep, 1)
            break
    else:
        parts = equation.split("=", 1) if "=" in equation else [equation, ""]

    # Extract metabolite IDs term-by-term (mirrors _parse_side in add_reaction)
    met_ids = []
    for side in parts:
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            m_space = _re.match(r'^(\d+\.?\d*|\d*\.\d+)\s+(.+)$', term)
            m_paren = _re.match(r'^(\d+\.?\d*|\d*\.\d+)\((.+)\)$', term)
            if m_space:
                mid = m_space.group(2).strip()
            elif m_paren:
                mid = m_paren.group(2).strip()
            else:
                mid = term.strip("() ")
            if mid:
                met_ids.append(mid)

    met_ids = list(dict.fromkeys(met_ids))  # deduplicate, preserve order
    new_mets = [m for m in met_ids if not model.metabolites.has_id(m)]
    return {"new_metabolites": new_mets}


@app.post("/chat/")
async def chat(req: ChatRequest):
    try:
        if model_manager.current_model_id:
            message = f"[Current model: {model_manager.current_model_id}] {req.message}"
        else:
            message = req.message
        result = await agent_query(message)
        if isinstance(result, dict) and result.get("__upload_required__"):
            return {
                "response": result["description"],
                "needs_upload": {
                    "param": result["param"],
                    "file_types": result["file_types"],
                    "description": result["description"],
                },
                "model_id": model_manager.current_model_id,
            }
        if isinstance(result, dict) and result.get("__tool_inputs_required__"):
            import json as _json
            from planner.tool_registry import TOOL_USER_INPUTS
            tool_name = result["tool_name"]
            try:
                prefilled = _json.loads(result.get("prefilled_params", "{}") or "{}")
            except Exception:
                prefilled = {}
            param_specs = {}
            for pname, pspec in TOOL_USER_INPUTS.get(tool_name, {}).items():
                spec = dict(pspec)
                if pname in prefilled:
                    spec["value"] = prefilled[pname]
                param_specs[pname] = spec
            return {
                "response": result["explanation"],
                "needs_tool_inputs": {
                    "tool_name": tool_name,
                    "explanation": result["explanation"],
                    "param_specs": param_specs,
                },
                "model_id": model_manager.current_model_id,
            }
        return {"response": result, "model_id": model_manager.current_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/submit_tool_inputs")
async def chat_submit_tool_inputs(req: ToolInputsSubmitRequest):
    try:
        from planner.tool_registry import TOOL_USER_INPUTS
        from agent import reformat_tool_result

        tool_spec = TOOL_USER_INPUTS.get(req.tool_name, {})
        user_inputs = {}
        for param, value in req.params.items():
            spec = dict(tool_spec.get(param, {}))
            spec["value"] = value
            user_inputs[param] = spec

        step = {"tool": req.tool_name, "user_inputs": user_inputs}
        result = _execute_step(step)
        response = await reformat_tool_result(req.original_message, str(result))
        return {"response": response, "model_id": model_manager.current_model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     model_manager.load_sbml("uploads/e_coli_core.xml")
#     message = "Set the objective of the model to {ATPM: 1.0, EX_o2_e: 2.0} with direction as min"
#     resp = agent_query(message)
#     print(resp)

# @app.post("/set_llm/")
# def set_llm(config: LLMConfig):
#     global agent, llm, current_llm_config
#     llm = get_llm(config.provider, config.model, config.api_key)
#     current_llm_config.update(config.dict())
#     setup_agent(llm)
#     return {"status": "LLM updated", "provider": config.provider, "model": config.model}