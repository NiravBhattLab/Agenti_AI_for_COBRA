import torch
from transformers import pipeline
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def get_llm(provider: str, model: str, api_key: str = None):
    if provider == "ollama":
        return Ollama(model=model)
    elif provider == "openai":
        return OpenAI(model=model, api_key=api_key)
    elif provider == "hf-local":
        return HuggingFaceLLM(model_name=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")