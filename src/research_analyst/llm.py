from __future__ import annotations
import os
from langchain_groq import ChatGroq


def get_groq_llm(model_env_var: str, default_model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    model = os.environ.get(model_env_var, default_model)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(model=model, api_key=api_key, temperature=0)
