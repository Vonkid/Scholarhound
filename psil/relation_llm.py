"""LLM-as-relation-parser for the V3 intake.

Replaces (when enabled) the keyword content reader that structurally cannot detect
contradiction. Each configured model READS (belief + abstract) and returns a relation in
the kernel vocabulary; the kernel's own consensus_evidence_relation then aggregates the
votes (consensus = relation, support<->challenge split = contest).

Enabled only when SCHOLARHOUND_RELATION_READER=llm AND an API key is present; otherwise the
caller falls back to the offline keyword candidates, so existing offline tests are unaffected.

Config (env):
  SCHOLARHOUND_RELATION_READER = "llm"        # turn this reader on
  LLM_API_KEY                  = sk-or-v1-...  # OpenRouter (or any OpenAI-compatible) key
  LLM_BASE_URL                 = https://openrouter.ai/api/v1
  LLM_MODELS                   = deepseek/deepseek-chat,anthropic/claude-opus-4.8,openai/gpt-4o,...
  LLM_TEMP                     = 0.0           # cross-MODEL disagreement is the contest signal

Standard library only (urllib); certifi used for the CA bundle if available.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request
import urllib.error
from typing import Any

try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _CTX = ssl.create_default_context()

_ALLOWED = {"support", "challenge", "neutral", "underdetermined"}

_PROMPT = (
    "Decide this paper's RELATION to the BELIEF. Read the abstract's actual claims.\n"
    "- support: the paper gives evidence FOR the belief.\n"
    "- challenge: the paper gives evidence AGAINST / contradicting the belief.\n"
    "- neutral: the paper is off-topic / not about the belief.\n"
    "- underdetermined: on-topic but not enough to decide a direction.\n"
    'Output ONLY JSON: {"relation":"support|challenge|neutral|underdetermined"}'
)


def read_relation(belief: str, abstract: str, model: str, base_url: str, api_key: str, temp: float) -> str:
    body = json.dumps({
        "model": model,
        "temperature": temp,
        "messages": [
            {"role": "system", "content": "You judge a paper's relation to a research belief. Output only JSON."},
            {"role": "user", "content": f"BELIEF:\n{belief}\n\nPAPER ABSTRACT:\n{abstract}\n\n{_PROMPT}"},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60, context=_CTX) as resp:
        text = json.loads(resp.read())["choices"][0]["message"]["content"]
    obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    rel = str(obj.get("relation", "")).lower().strip()
    return rel if rel in _ALLOWED else "unclear"


def enabled() -> bool:
    return os.environ.get("SCHOLARHOUND_RELATION_READER", "").lower() == "llm" and bool(
        os.environ.get("LLM_API_KEY")
    )


def llm_parse_candidates(belief: str, abstract: str) -> list[dict[str, str]] | None:
    """Return one parse-candidate per configured model, or None to signal fallback."""
    if not enabled():
        return None
    api_key = os.environ["LLM_API_KEY"]
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    models = [m.strip() for m in os.environ.get("LLM_MODELS", "deepseek/deepseek-chat").split(",") if m.strip()]
    temp = float(os.environ.get("LLM_TEMP", "0.0"))
    candidates: list[dict[str, str]] = []
    for model in models:
        try:
            relation = read_relation(belief, abstract, model, base_url, api_key, temp)
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
            continue  # a dead model must not sink the whole read
        candidates.append({"parser": f"llm:{model}", "relation": relation})
    return candidates or None
