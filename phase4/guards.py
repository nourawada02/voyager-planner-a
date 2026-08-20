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
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from phase1.models import TripRequest
from phase4.models import ReasonCode

_MAX_MESSAGE_LENGTH = 2000

# The one project-wide timezone "today" is resolved in for date validation
# (Manual QA remediation Q.1) -- Istanbul, matching every other IST-timezone
# precedent already in this codebase (providers/flights_serpapi.py's own
# IATA->timezone map for IST/SAW). A request submitted late at night UTC
# that is already tomorrow in Istanbul must see tomorrow as "today", and
# vice versa -- never a bare UTC date, never the server host's local time.
PROJECT_TIMEZONE = ZoneInfo("Europe/Istanbul")


def default_wall_clock() -> datetime:
    return datetime.now(timezone.utc)


def resolve_today(clock: Callable[[], datetime] = default_wall_clock) -> date_cls:
    """Resolves 'today' as a calendar date in PROJECT_TIMEZONE from an
    injectable clock -- never `date.today()` (server local time) and never
    hardcoded. Every caller that needs "today" for date validation goes
    through this one function, so a test can pin an exact instant and every
    layer (API pre-check, graph InputGuard) agrees on the same "today"."""
    now = clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(PROJECT_TIMEZONE).date()

# Manual QA remediation Q.1: TRY and USD only (user correction pass §B --
# supersedes the earlier TRY-only decision now that genuine FX conversion
# exists: providers/fx_frankfurter.py, a real Frankfurter.app/ECB rate
# source, Decimal-exact, cached, with full provenance -- see
# providers/money.py). Travel MCP's accommodation results remain
# contractually TRY-only (`contracts/SearchStaysResult.schema.json`, the
# historical Airbnb-Turkey snapshot has no other currency) -- for a USD
# trip, TRY prices are converted for DISPLAY using the run's one FX quote,
# never re-requested from Travel MCP in a currency it cannot produce.
# EUR remains unsupported: no EUR rate source was verified/wired.
_SUPPORTED_CURRENCIES = frozenset({"TRY", "USD"})

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


def check_input(user_message: str, trip_request: Optional[dict[str, Any]], *, today: date_cls) -> InputGuardResult:
    """Format/safety checks only -- never a live network call, never an
    LLM. `trip_request`, when present, is validated against the exact,
    unmodified `phase1.models.TripRequest` (reused, not re-implemented);
    a validation failure there is translated into one of a small, fixed
    set of safe_error strings, never the raw pydantic error text.

    `today` is required and keyword-only (Manual QA remediation Q.1) --
    every caller must resolve it explicitly via `resolve_today(clock)`
    rather than this function silently reaching for wall-clock time, so
    date validation is always deterministic and testable."""
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

        if trip.depart_date < today:
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
