"""Input and output guards (Checkpoint Phase 4 D.0 §8). Deterministic,
regex/allowlist-based -- never an LLM classification, matching this
project's existing "small explicit host allowlist, never inferred from
free text" precedent (`providers.web_evidence_serpapi.classify_source_type`).
A documented, intentional under-detector for prompt injection: this is a
narrow, explicit pattern list, not an exhaustive classifier -- absence of
a match is "not recognized," never "confirmed safe."
"""

from __future__ import annotations

import json
import re
from datetime import date as date_cls
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phase1.models import TripRequest
from phase4.models import ReasonCode

_MAX_MESSAGE_LENGTH = 2000

# This project's own scope: Istanbul trip planning priced against a
# TRY-first market (see providers/flights_serpapi.py's own
# currency="TRY" default in the root superproject) -- USD/EUR accepted
# as common traveler-budget currencies, nothing else.
_SUPPORTED_CURRENCIES = frozenset({"TRY", "USD", "EUR"})

_INJECTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"disregard (all |any )?(previous|prior|above) (instructions|rules)",
        r"reveal (your |the )?(system prompt|api key|credentials|secret)",
        r"show me (your |the )?(system prompt|api key|credentials|secret)",
        r"what is (your |the )?api[_ ]?key",
        r"execute (this |the )?(code|script)",
        r"run (this |the )?(code|script|command)",
        r"book (the |this |a |my )?(flight|hotel|stay|room)",
        r"purchase (the |this |a )?(ticket|flight|stay)",
        r"pay for",
        r"charge (my|the) card",
        r"credit card number",
    )
)

# Which TripRequest field a pydantic validation failure names -> a
# fixed, closed (safe_error, ReasonCode) pair. Never derived from the
# raw pydantic error message itself (which could echo user input) --
# only the field *name* selects a pre-written, safe string.
_FIELD_REASON_MAP: dict[str, tuple[str, ReasonCode]] = {
    "destination": ("unsupported_destination", ReasonCode.INPUT_REJECTED),
    "origin": ("malformed_iata_code", ReasonCode.INPUT_REJECTED),
    "depart_date": ("missing_or_invalid_date", ReasonCode.MISSING_ESSENTIAL_INPUT),
    "return_date": ("missing_or_invalid_date", ReasonCode.MISSING_ESSENTIAL_INPUT),
    "traveler_count": ("invalid_traveler_count", ReasonCode.INPUT_REJECTED),
    "budget": ("invalid_budget", ReasonCode.INPUT_REJECTED),
}


class InputGuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: bool
    reason_code: Optional[ReasonCode] = None
    safe_error: Optional[str] = Field(default=None, max_length=300)
    normalized_request: dict[str, Any] = Field(default_factory=dict)


def _detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def _first_error_field(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "unknown"
    loc = errors[0].get("loc", ("unknown",))
    return str(loc[0]) if loc else "unknown"


def check_input(user_message: str, trip_request: Optional[dict[str, Any]]) -> InputGuardResult:
    """Format/safety checks only -- never a live network call, never an
    LLM. `trip_request`, when present, is validated against the exact,
    unmodified `phase1.models.TripRequest` (reused, not re-implemented);
    a validation failure there is translated into one of a small, fixed
    set of safe_error strings, never the raw pydantic error text."""
    if len(user_message) > _MAX_MESSAGE_LENGTH:
        return InputGuardResult(accepted=False, reason_code=ReasonCode.INPUT_REJECTED, safe_error="excessive_text")

    if _detect_prompt_injection(user_message):
        return InputGuardResult(
            accepted=False, reason_code=ReasonCode.INPUT_REJECTED, safe_error="unsafe_request_pattern_detected"
        )

    normalized: dict[str, Any] = {"user_message": user_message.strip()}

    if trip_request is not None:
        try:
            trip = TripRequest.model_validate(trip_request)
        except ValidationError as exc:
            field = _first_error_field(exc)
            safe_error, reason = _FIELD_REASON_MAP.get(field, ("invalid_trip_request", ReasonCode.INPUT_REJECTED))
            return InputGuardResult(accepted=False, reason_code=reason, safe_error=safe_error)

        if trip.depart_date < date_cls.today():
            return InputGuardResult(
                accepted=False, reason_code=ReasonCode.MISSING_ESSENTIAL_INPUT, safe_error="depart_date_in_past"
            )
        if trip.return_date < trip.depart_date:
            return InputGuardResult(
                accepted=False, reason_code=ReasonCode.MISSING_ESSENTIAL_INPUT, safe_error="return_date_before_depart_date"
            )
        if trip.budget.currency not in _SUPPORTED_CURRENCIES:
            return InputGuardResult(accepted=False, reason_code=ReasonCode.INPUT_REJECTED, safe_error="unsupported_currency")

        normalized["trip_request"] = trip.model_dump(mode="json")

    return InputGuardResult(accepted=True, normalized_request=normalized)


# --- output guard --------------------------------------------------------------------

_FORBIDDEN_CLAIM_TERMS = (
    "booked", "reservation_confirmed", "seat_available", "payment_processed",
    "purchase_complete", "ticket_issued", "confirmed_booking",
)


def check_output(final_result: dict[str, Any]) -> list[str]:
    """Returns a list of violation codes (empty = passed). Never raises
    -- the caller (Synthesize/graph exit) decides how to react to a
    non-empty list. Deliberately serializes and lowercases the whole
    result rather than checking specific fields, so a violation nested
    anywhere (a stray narrative string, not just a status enum) is still
    caught."""
    serialized = json.dumps(final_result, default=str).lower()
    return [f"forbidden_claim_term:{term}" for term in _FORBIDDEN_CLAIM_TERMS if term in serialized]
