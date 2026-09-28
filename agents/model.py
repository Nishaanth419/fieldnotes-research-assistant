"""Configure the shared local Ollama chat model used by all research agents."""

import os

from langchain_ollama import ChatOllama


DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def get_chat_model() -> ChatOllama:
    """Create a local Ollama model with enough context for bounded research evidence."""
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        temperature=0,
        num_ctx=16_384,
    )
