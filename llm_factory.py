from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.llms.huggingface import HuggingFaceLLM

def get_llm(provider: str, model: str, api_key: str = None):
    if provider == "ollama":
        return Ollama(model=model, request_timeout=300)
    elif provider == "openai":
        return OpenAI(model=model, api_key=api_key)
    elif provider == "hf-local":
        return HuggingFaceLLM(model_name=model)
    elif provider == "groq":
        return Groq(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")