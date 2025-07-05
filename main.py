from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from llm_factory import get_llm
from pydantic import BaseModel
from agent import agent_query, setup_agent
from models import ModelManager
from pathlib import Path
from tools import set_model_manager
import pandas as pd
import os

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
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
    "provider": "ollama",
    "model": "llama3.1:latest",
    "api_key": None
}

class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: str = None


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

    
# @app.post("/set_llm/")
# def set_llm(config: LLMConfig):
#     global agent, llm, current_llm_config
#     llm = get_llm(config.provider, config.model, config.api_key)
#     current_llm_config.update(config.dict())
#     setup_agent(llm)
#     return {"status": "LLM updated", "provider": config.provider, "model": config.model}

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



@app.post("/chat/")
async def chat(req: ChatRequest):
    try:
        response = agent_query(req.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     model_manager.load_sbml("uploads/e_coli_core.xml")
#     message = "Set the objective of the model to {ATPM: 1.0, EX_o2_e: 2.0} with direction as min"
#     resp = agent_query(message)
#     print(resp)