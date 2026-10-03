"""scholarhunter.llm_client — thin OpenAI-compatible client for Groq with
model fallback chain and lightweight retry.  stdlib-only (no openai package
needed; we POST straight to https://api.groq.com/openai/v1/chat/completions).

Config via env (see .env.template):
  GROQ_API_KEY        required for live calls
  GROQ_MODEL          optional override for the primary model
  GROQ_MODEL_FALLBACKS optional comma-separated list of fallback models
  GROQ_TIMEOUT_S      default 30
  GROQ_MAX_RETRIES    default 2

All methods that would raise raise LLMError instead — upstream callers
(see crew.py) have deterministic fallbacks.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

DEFAULT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 2
DEFAULT_FALLBACKS = ("llama-3.3-70b-versatile",)


class LLMError(RuntimeError):
    """Raised after the full model-fallback chain exhausts or no API key."""


class LLMClient:
    """Minimal, dependency-free OpenAI-compatible chat client for Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallbacks: Optional[Sequence[str]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        endpoint: str = DEFAULT_ENDPOINT,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile"
        fb_env = os.environ.get("GROQ_MODEL_FALLBACKS", "")
        if fallbacks:
            self.fallbacks = list(fallbacks)
        elif fb_env:
            self.fallbacks = [m.strip() for m in fb_env.split(",") if m.strip()]
        else:
            self.fallbacks = list(DEFAULT_FALLBACKS)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.endpoint = endpoint

    # ------------------------------------------------------------ internals
    def _try_once(self, payload: dict) -> Any:
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:600]
            except Exception:
                pass
            raise LLMError(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"network error: {e.reason}") from e

    @staticmethod
    def _extract_message(data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""


    # --------------------------------------------------------- public API
    def chat(
        self,
        messages: Sequence[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = True,
    ) -> str:
        """Chat completion. Returns the model's text, or raises LLMError."""
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set")

        base = model or self.model
        candidates = [base] + [m for m in self.fallbacks if m != base]
        last_err: Optional[BaseException] = None

        for m in candidates:
            payload: dict[str, Any] = {
                "model": m,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            for attempt in range(self.max_retries + 1):
                try:
                    data = self._try_once(payload)
                    text = self._extract_message(data)
                    if not text:
                        raise LLMError(f"empty response from {m}")
                    return text
                except LLMError as e:
                    last_err = e
                    s = str(e)
                    # Hard 4xx (401/403/422/400) → skip to next model immediately.
                    if ("HTTP 401" in s or "HTTP 403" in s
                            or "HTTP 422" in s or "HTTP 400" in s):
                        break
                    # 429/5xx/network → retry then next model.
                    if attempt < self.max_retries:
                        time.sleep(0.6 * (attempt + 1))
                        continue
            # exhausted retries for this model → next candidate
        raise LLMError(f"all models failed (last: {last_err})") from last_err


# ------------------------------------------------------------ module helper
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Return a process-wide singleton client (lazily built from env vars)."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def demo_ok() -> bool:
    """Return True if we should treat this environment as 'LLM-free demo'."""
    return not bool(os.environ.get("GROQ_API_KEY", ""))
