"""Planner-owned Qwen decision provider (Checkpoint Phase 4 D.0 §7).
Reuses the already-proven Qwen API configuration conventions from the
root RAG implementation (`rag/llm_providers.py::QwenProvider`) --
OpenAI-compatible Chat Completions, `QWEN_API_KEY`/`DASHSCOPE_API_KEY`,
an explicit-HTTPS `QWEN_BASE_URL` (never a guessed region), `enable_thinking:
false`, `response_format: json_object`, `temperature=0` for deterministic
routing -- but does **not** import `rag` at runtime (this submodule never
depends on the root `rag`/`providers` packages, ADR 0009 §6.1); the
request shape is reimplemented fresh, stdlib-only (`urllib.request`,
matching `rag/llm_providers.py`'s own zero-extra-HTTP-dependency choice,
so this checkpoint needs no new third-party HTTP client).

Neither the API key nor the base URL is ever stored as a dataclass
field -- both are read fresh from the environment inside `generate()`,
exactly like `rag.llm_providers.QwenProvider`, so this object can never
leak either value through `repr()`, a log line, a checkpoint, or any
other structured serialization of itself.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Protocol

QWEN_MODEL_DEFAULT = "qwen3.7-flash"


class DecisionProvider(Protocol):
    """The seam both `QwenDecisionProvider` (real) and a test's
    `FakeDecisionProvider` satisfy. Mirrors `rag.llm_providers.LLMProvider`'s
    own proven `generate(system, user) -> str` shape exactly -- used for
    both a Decide-node action-selection call and a Synthesize-node
    narrative-composition call, with different prompts, same interface.
    Returns raw text (expected to be a JSON object for a decision call);
    parsing/validation is the caller's responsibility, never this
    provider's."""

    def generate(self, system: str, user: str) -> str: ...


class QwenConfigurationError(RuntimeError):
    """Raised when `QwenDecisionProvider` is used without a valid,
    explicit HTTPS `QWEN_BASE_URL`. Never guesses a region/workspace
    endpoint. Fails before any network call."""


class QwenTransportError(RuntimeError):
    """Raised for any Qwen HTTP/network/response-shape failure -- the
    message names only a status code or a fixed short phrase, never a
    response body, header, or the request URL (which never contains the
    key: it is sent only via the Authorization header, never as a query
    parameter). The caller (Decide node) maps this to a safe structured
    `status="provider_error"`/`"timeout"` outcome -- it never reaches a
    user-facing result as raw exception text.

    Checkpoint Final Evaluation E.1S: carries `status_code` (the HTTP
    status when one exists, else `None`) and `transient` -- True only for
    a network-level failure with no HTTP response at all (DNS/connect/
    timeout) or an HTTP status in {429, 500, 502, 503, 504}. Every other
    HTTP status (auth/permission/other permanent 4xx) and every
    response-shape failure (invalid JSON, missing content field) is
    `transient=False`. Classified once, here, at the single point that
    already distinguishes these failure kinds -- the caller never
    re-derives this from the message string."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, transient: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.transient = transient


_TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


def _qwen_api_key() -> str:
    """Read fresh at call time, never cached/stored on any object."""
    return os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""


def qwen_api_key_present() -> bool:
    """Presence-only check -- the value itself is never read here."""
    return bool(os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))


def _qwen_base_url() -> str:
    base_url = os.environ.get("QWEN_BASE_URL")
    if not base_url:
        raise QwenConfigurationError(
            "QWEN_BASE_URL is required and must be set explicitly -- no region endpoint is guessed"
        )
    if urllib.parse.urlsplit(base_url).scheme != "https":
        raise QwenConfigurationError("QWEN_BASE_URL must be an explicit https:// URL")
    return base_url


def _qwen_model() -> str:
    return os.environ.get("QWEN_MODEL") or os.environ.get("QWEN_GENERATOR_MODEL") or QWEN_MODEL_DEFAULT


@dataclass(frozen=True)
class QwenDecisionProvider:
    """Real `DecisionProvider`. `model` defaults to `qwen3.7-flash`
    (the same default `rag/llm_providers.py::QWEN_GENERATOR_MODEL`
    already establishes), overridable via `QWEN_MODEL`/
    `QWEN_GENERATOR_MODEL`. Bounded timeout; no retry loop of its own --
    the caller (Decide node) owns the bounded decision-repair budget
    (Checkpoint D.0 §5), since a malformed-JSON repair is a different
    concern from a transport retry and must not share one counter."""

    model: str = field(default_factory=_qwen_model)
    timeout_seconds: float = 25.0

    def generate(self, system: str, user: str) -> str:
        base_url = _qwen_base_url()
        api_key = _qwen_api_key()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,  # deterministic routing, not creative generation
            "response_format": {"type": "json_object"},
            # No raw chain-of-thought is ever requested -- disables
            # extended "thinking" output on Qwen's hybrid-reasoning
            # models where the API supports the parameter.
            "enable_thinking": False,
        }
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "voyager-planner-a/phase4",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise QwenTransportError(
                f"Qwen HTTP error: status={exc.code}",
                status_code=exc.code, transient=exc.code in _TRANSIENT_HTTP_STATUSES,
            ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # No HTTP response was ever received at all (DNS failure --
            # urllib.error.URLError; connection refused/reset -- ConnectionError;
            # socket timeout -- TimeoutError) -- always transient. Checkpoint
            # E.1S.1: deliberately narrower than a blanket `OSError`, which
            # also covers unrelated local failures (e.g. a filesystem error)
            # that are not a transient network condition and should never be
            # silently retried.
            raise QwenTransportError("Qwen transport failure", transient=True) from exc
        except json.JSONDecodeError as exc:
            # A malformed response body is a response-shape problem, not a
            # transient network condition -- retrying would not help.
            raise QwenTransportError("Qwen response was not valid JSON", transient=False) from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenTransportError("Qwen response missing expected content field", transient=False) from exc
