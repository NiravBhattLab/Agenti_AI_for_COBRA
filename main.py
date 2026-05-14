from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import os
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tools import set_model_objective, set_model_manager, set_session_dir
from llm_factory import get_llm
from pydantic import BaseModel
from agent import agent_query, setup_agent
from models import ModelManager
from pathlib import Path
import shutil
import pandas as pd
import os
import re

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SESSION_DIR: Path | None = None

def _make_session_dir(name: str) -> Path:
    safe = re.sub(r'[<>:"/\\|?*\s]+', '_', name).strip('_') or "session"
    session = Path("outputs") / safe
    session.mkdir(parents=True, exist_ok=True)
    (session / "fva").mkdir(exist_ok=True)
    (session / "knockouts").mkdir(exist_ok=True)
    (session / "flux_sampling").mkdir(exist_ok=True)
    return session

model_manager = ModelManager()
set_model_manager(model_manager)

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


@app.post("/upload_model/")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xml', '.sbml')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an SBML file (.xml or .sbml)")
    try:
        file_path = UPLOAD_DIR / file.filename
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
    try:
        bounds_dir = UPLOAD_DIR / "bounds_data"
        bounds_dir.mkdir(parents=True, exist_ok=True)
        file_path = bounds_dir / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        model_manager.bounds_data = pd.read_csv(file_path).values.tolist()
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


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
    global agent, llm, current_llm_config
    try:
        llm = get_llm(config.provider, config.model, config.api_key)
        current_llm_config.update(config.dict())
        setup_agent(llm)
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
    set_session_dir(SESSION_DIR)
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
    def _quit():
        import time
        time.sleep(0.3)
        os._exit(0)
    background_tasks.add_task(_quit)
    return {"status": "shutting_down"}

@app.post("/chat/")
async def chat(req: ChatRequest):
    try:
        response = agent_query(req.message)
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