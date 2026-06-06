"""Backend resolution: one switch selects where the agent's LLM calls go.

The agent is just an HTTP client for an OpenAI-compatible endpoint, so the
*only* thing that changes between local development and the final H100 run is
which URL/model/key we point at. This module centralizes that choice so the
rest of the code never reads os.environ directly.

Selection order (first hit wins):
  1. Explicit overrides:  VLLM_BASE_URL / VLLM_MODEL / OPENAI_API_KEY (or
     NEBIUS_API_KEY) - if set, they win regardless of profile.
  2. Profile preset:      LLM_BACKEND in {nebius, h100, cpu, openai}.
  3. Default:             nebius (the same Qwen3-30B served on the H100).

Profiles
--------
  nebius : Nebius Token Factory, hosted Qwen3-30B-A3B (dev default; prompts
           transfer 1:1 to the H100 because it's the identical model).
  h100   : your own vLLM serving the 30B at localhost:8000 (the final run).
  cpu    : your own vLLM serving a tiny Qwen3-0.6B on CPU at localhost:8000
           (Stage A.5 - only used to prove the /metrics plumbing, not quality).
  openai : plain OpenAI (gpt-4o-mini) - a throwaway fallback for dev.

Run `python -m agent.config` to print the resolved backend (key masked).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Idempotent: server.py also calls this, but importing config from a bare
# script (tests, `python -m agent.config`) must still pick up the .env.
load_dotenv()


@dataclass(frozen=True)
class _Preset:
    base_url: str
    model: str
    api_key: str  # default key when none is supplied via env


# vLLM ignores the API key; hosted providers require a real one.
_PROFILES: dict[str, _Preset] = {
    "nebius": _Preset(
        base_url="https://api.tokenfactory.nebius.com/v1",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        api_key="",  # must come from NEBIUS_API_KEY / OPENAI_API_KEY
    ),
    "h100": _Preset(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        api_key="not-needed",
    ),
    "cpu": _Preset(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-0.6B",
        api_key="not-needed",
    ),
    "openai": _Preset(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="",  # must come from OPENAI_API_KEY
    ),
}

DEFAULT_PROFILE = "nebius"


@dataclass(frozen=True)
class LLMBackend:
    """Fully resolved backend the agent should talk to."""

    profile: str
    base_url: str
    model: str
    api_key: str
    temperature: float
    timeout: float

    def describe(self) -> str:
        key = self.api_key or ""
        if len(key) <= 8:
            masked = "<empty>" if not key else "*" * len(key)
        else:
            masked = f"{key[:4]}...{key[-4:]}"
        return (
            f"LLM backend: profile={self.profile} model={self.model} "
            f"base_url={self.base_url} key={masked} "
            f"temperature={self.temperature} timeout={self.timeout}s"
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def resolve_backend() -> LLMBackend:
    """Resolve the active backend from env (see module docstring for order)."""
    profile = os.environ.get("LLM_BACKEND", DEFAULT_PROFILE).strip().lower()
    preset = _PROFILES.get(profile)
    if preset is None:
        valid = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unknown LLM_BACKEND={profile!r}. Choose one of: {valid} "
            f"(or set VLLM_BASE_URL / VLLM_MODEL explicitly)."
        )

    base_url = os.environ.get("VLLM_BASE_URL") or preset.base_url
    model = os.environ.get("VLLM_MODEL") or preset.model
    # Either OPENAI_API_KEY or NEBIUS_API_KEY works; falls back to the preset.
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("NEBIUS_API_KEY")
        or preset.api_key
        or "not-needed"
    )

    backend = LLMBackend(
        profile=profile,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=_env_float("LLM_TEMPERATURE", 0.0),
        timeout=_env_float("LLM_TIMEOUT", 60.0),
    )

    if profile in ("nebius", "openai") and api_key in ("", "not-needed"):
        raise RuntimeError(
            f"Profile {profile!r} needs a real API key. Set NEBIUS_API_KEY "
            f"(or OPENAI_API_KEY) in your .env."
        )
    return backend


if __name__ == "__main__":
    print(resolve_backend().describe())
