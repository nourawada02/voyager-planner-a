"""The closed structured-output contract for Hybrid Chat C.1's chat-turn
Qwen call (§5). Mirrors `phase4.models.ActionDecision`'s own discipline
exactly: a closed Pydantic model with `extra="forbid"`, a closed intent
enum, and a dedicated `parse_chat_intent_decision` function (never a
model validator) so a malformed response is a typed, repair-worthy error,
never a raw exception. This is a genuinely separate contract from
`ActionDecision` -- a chat turn never selects a tool action, and Qwen may
never use this call to directly construct an MCP/A2A request (CLAUDE.md
Hybrid Chat C.1 constraint); the only two things a chat turn may ever
produce are (a) a user-facing message, and (b) an optional, narrowly
allowlisted `TripPatch` that a SEPARATE, already-existing pipeline
(`RunService.create_run` -> the unmodified bounded ReAct graph) applies
exactly like an ordinary form submission.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from phase1.models import Language, Pace

# Bounded repair budgets, each local to this one step -- never shared
# with MAX_DECISION_REPAIRS/MAX_CAPABILITY_CLASSIFICATION_ATTEMPTS,
# mirroring the project's own established "each loop gets its own small
# named bound" precedent (phase4/models.py's MAX_SPECIALIST_DECISION_REPAIRS
# docstring).
MAX_CHAT_DECISION_REPAIRS = 1  # malformed-JSON-shape repair attempts
MAX_CHAT_LANGUAGE_CORRECTIONS = 1  # "wrong language returned" bounded correction (Hybrid Chat C.1 §6)

_MAX_ASSISTANT_MESSAGE_LENGTH = 2000
_MAX_CLARIFICATION_REASON_LENGTH = 300


class ChatIntent(str, Enum):
    EXPLAIN_PLAN = "explain_plan"
    MODIFY_TRIP = "modify_trip"
    REGENERATE_TRIP = "regenerate_trip"
    CLARIFY = "clarify"
    RESET_TRIP = "reset_trip"


class TripPatch(BaseModel):
    """The only fields a chat-turn Qwen call may ever propose changing --
    a closed allowlist (Hybrid Chat C.1 §5: "only allowlisted patch
    fields may be returned"). Every field is optional and independently
    omittable; an omitted field means "no change requested for this
    field", never "clear this field" -- the caller (chat_service.py)
    merges only the fields actually present onto the existing, already-
    validated trip request, exactly matching "missing fields retain
    previous value"."""

    model_config = ConfigDict(extra="forbid")
    depart_date: Optional[date] = None
    return_date: Optional[date] = None
    traveler_count: Optional[int] = Field(default=None, ge=1, le=12)
    budget_amount_minor_units: Optional[int] = Field(default=None, ge=0)
    budget_currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    pace: Optional[Pace] = None
    add_interests: Optional[list[str]] = Field(default=None, max_length=10)
    remove_interests: Optional[list[str]] = Field(default=None, max_length=10)
    language: Optional[Language] = None

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_none=True)


class ChatIntentDecision(BaseModel):
    """The only shape a chat-turn decision provider (Qwen or a fake) is
    ever allowed to produce."""

    model_config = ConfigDict(extra="forbid")
    intent: ChatIntent
    assistant_message: str = Field(min_length=1, max_length=_MAX_ASSISTANT_MESSAGE_LENGTH)
    response_language: Language
    patch: Optional[TripPatch] = None
    requires_clarification: bool = False
    clarification_reason: Optional[str] = Field(default=None, max_length=_MAX_CLARIFICATION_REASON_LENGTH)


class ChatIntentDecisionValidationError(ValueError):
    """Raised when a raw chat decision-provider response cannot be turned
    into a valid `ChatIntentDecision`. Callers count this against the
    bounded `MAX_CHAT_DECISION_REPAIRS` budget; it never propagates as a
    raw exception to a user-facing result."""


def parse_chat_intent_decision(raw: dict) -> ChatIntentDecision:
    try:
        return ChatIntentDecision.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 -- any parse/validation failure is repair-worthy
        raise ChatIntentDecisionValidationError(f"invalid ChatIntentDecision shape: {exc}") from exc


# --- allowlisted patch-field application (deterministic, never guessed) ------------

_PATCH_TARGET_FIELDS = (
    "depart_date", "return_date", "traveler_count",
    "budget_amount_minor_units", "budget_currency", "pace", "language",
)


def apply_patch_to_trip_request(trip_request: dict[str, Any], patch: TripPatch) -> dict[str, Any]:
    """Deterministically merges `patch` onto a COPY of `trip_request`
    (never the original dict) -- only fields the patch actually set are
    changed; every other field, including nested `preferences`/`budget`
    substructures not touched here, is carried through unchanged. Interest
    add/remove is applied via the caller-resolved normalized lists (see
    `phase4.chat_interests`), not here, since normalization can produce
    warnings the caller must surface -- this function only ever receives
    already-normalized interest names in `patch.add_interests`/
    `remove_interests` when called from `chat_service.py`."""
    merged = dict(trip_request)
    budget = dict(merged.get("budget") or {})
    preferences = dict(merged.get("preferences") or {})

    if patch.depart_date is not None:
        merged["depart_date"] = patch.depart_date.isoformat()
    if patch.return_date is not None:
        merged["return_date"] = patch.return_date.isoformat()
    if patch.traveler_count is not None:
        merged["traveler_count"] = patch.traveler_count
    if patch.budget_amount_minor_units is not None:
        budget["amount_minor_units"] = patch.budget_amount_minor_units
    if patch.budget_currency is not None:
        budget["currency"] = patch.budget_currency
    if patch.pace is not None:
        preferences["pace"] = patch.pace.value
    if patch.language is not None:
        preferences["language"] = patch.language.value

    if patch.add_interests is not None or patch.remove_interests is not None:
        current_interests = list(preferences.get("interests") or [])
        for name in patch.add_interests or []:
            if name not in current_interests:
                current_interests.append(name)
        for name in patch.remove_interests or []:
            if name in current_interests:
                current_interests.remove(name)
        preferences["interests"] = current_interests

    merged["budget"] = budget
    merged["preferences"] = preferences
    return merged
