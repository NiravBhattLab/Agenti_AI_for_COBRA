import os
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.llama_cpp import LlamaCPP

def get_llm(provider: str, model: str, api_key: str = None):
    if provider == "ollama":
        return Ollama(model=model, request_timeout=300)
    elif provider == "openai":
        return OpenAI(model=model, api_key=api_key)
    elif provider == "Hugging Face":
        os.environ["HUGGING_FACE_TOKEN"] = api_key
        return HuggingFaceLLM(model_name=model, tokenizer_name=model)
    elif provider == "groq":
        return Groq(model=model, api_key=api_key)
    elif provider == "gemini":
        return GoogleGenAI(model=model, api_key=api_key)
    elif provider == "llama.cpp":
        return LlamaCPP(model_path=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")