from pydantic import BaseModel

class LoadModelInput(BaseModel):
    model_id: str
