"""Configure the shared OpenAI chat model used by all research agents."""

import os

from langchain_openai import ChatOpenAI


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_chat_model() -> ChatOpenAI:
    """Create an OpenAI model for bounded research evidence."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        temperature=0,
    )
