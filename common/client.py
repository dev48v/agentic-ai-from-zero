"""Shared OpenAI-compatible client factory for the whole series.

Points at NVIDIA NIM by default (https://integrate.api.nvidia.com/v1), but because
NIM speaks the OpenAI wire protocol, you can drop in any other free provider
(Groq, Gemini, OpenRouter) just by changing NIM_BASE_URL + NIM_MODEL in .env.

Every project imports `get_client()` and `DEFAULT_MODEL` from here so there is
exactly one place that knows about credentials and endpoints.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the repo root (and cwd) so any project can `import common.client`.
load_dotenv()

# NVIDIA NIM is OpenAI-compatible.
DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

# A NIM model with solid JSON / tool-calling support that is warm/fast on the free
# tier (the big 70B models cold-start for >100s). Override via NIM_MODEL in .env.
DEFAULT_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")

# Accept a couple of common env var names so this is forgiving.
_API_KEY_ENVS = ("NVIDIA_API_KEY", "NIM_API_KEY", "OPENAI_API_KEY")


def _resolve_api_key() -> str:
    for name in _API_KEY_ENVS:
        val = os.getenv(name)
        if val:
            return val
    raise RuntimeError(
        "No API key found. Set NVIDIA_API_KEY in your .env "
        "(copy .env.example -> .env). Get a free key at https://build.nvidia.com"
    )


def get_client(base_url: str | None = None, timeout: float = 60.0) -> OpenAI:
    """Return an OpenAI SDK client pointed at NVIDIA NIM (or an override base_url).

    `timeout` guards against a cold model hanging forever — it fails fast instead.
    """
    return OpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=_resolve_api_key(),
        timeout=timeout,
        max_retries=1,
    )
