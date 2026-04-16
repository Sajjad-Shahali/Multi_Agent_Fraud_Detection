"""Session ID generation and Langfuse callback configuration.

Session IDs follow the competition-required format: {TEAM_NAME}-{ULID}
"""

import os
from typing import Optional

from dotenv import load_dotenv
from ulid import ULID

load_dotenv()


def generate_session_id() -> str:
    """Return a unique session ID in the format '{TEAM_NAME}-{ULID}'."""
    team = os.getenv("TEAM_NAME", "reply-mirror").replace(" ", "-")
    return f"{team}-{ULID()}"


def langfuse_client():
    """Initialise and return a Langfuse client from environment variables.

    Returns None if credentials are not configured (local runs without Langfuse).
    """
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        )
    except (KeyError, Exception):
        return None


def langfuse_callback_handler(session_id: str):
    """Return a Langfuse LangChain CallbackHandler for the given session ID.

    Returns None if Langfuse is not configured.
    """
    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler(session_id=session_id)
    except Exception:
        return None


def langchain_config(session_id: str, extra_callbacks: Optional[list] = None) -> dict:
    """Return the LangChain config dict that injects the Langfuse session ID.

    Attaches the Langfuse CallbackHandler so token usage, latency, and cost
    are automatically tracked on the competition dashboard.

    Usage:
        graph.invoke(state, config=langchain_config(session_id))
    """
    callbacks = extra_callbacks or []
    handler = langfuse_callback_handler(session_id)
    if handler:
        callbacks = [handler] + callbacks

    cfg: dict = {"metadata": {"langfuse_session_id": session_id}}
    if callbacks:
        cfg["callbacks"] = callbacks
    return cfg
