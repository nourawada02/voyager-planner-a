"""Structured ReAct decision model, allowed-action allowlist, and bounds
(Checkpoint Phase 4 D.0, docs/adr/0013-...md). Qwen (or the injected fake
decision provider) may only ever produce an `ActionDecision` -- a closed
Pydantic model with `extra="forbid"` and an `Action` enum, so an unknown
tool name, an arbitrary URL, code execution, or a booking/payment action
is a validation error, never a value that reaches the graph. Every
argument shape is its own small, closed Pydantic model too -- Qwen can
fill in field values, never invent new fields or a new action.

`ToolObservation` and `PlannerBounds` are additive, orchestration-only
types owned by this package -- they never redefine what a provider
result *is* (that remains exactly `providers.ProviderResponseEnvelope` +
the capability-specific result mirrored in `phase1/models.py`, reused
unchanged). `ToolObservation` only adds the operational status/fingerprint
bookkeeping the bounded ReAct loop itself needs and
`phase1/models.py`'s original 1.0.0-era `ProviderResponseEnvelope` mirror
does not carry (it predates Checkpoint C.0's 1.1.0 status/capability
fields) -- ADR 0013 §"Why ToolObservation is not a competing contract"
records this as a deliberate, narrow addition, not a redesign.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# --- bounds (ADR 0009 §4.3, this checkpoint's concrete implementation) -------------

MAX_GRAPH_TRANSITIONS = 25  # LangGraph recursion_limit -- the whole graph, every node type
MAX_EXTERNAL_TOOL_CALLS = 8  # this ReAct loop's own external-tool-call budget
MAX_DECISION_REPAIRS = 2  # malformed-decision-JSON repair attempts
MAX_CALLS_PER_TOOL = 2  # mirrors providers.policy.RetryPolicy.max_retries's default bound
TOTAL_WORKFLOW_DEADLINE_SECONDS = 60.0

# Checkpoint Phase 4 D.3 (correction pass): the internal Travel Search
# specialist is a genuinely separate, compiled LangGraph `StateGraph`
# (phase4/specialist.py) with its own real transitions -- it needs its
# own real `recursion_limit`, independent of and strictly tighter than
# the supervisor's own MAX_GRAPH_TRANSITIONS=25. A 4-node cycle
# (specialist_decide -> specialist_execute_observe -> specialist_decide
# -> ... -> specialist_end) costs 2 transitions per tool call, so 15
# comfortably bounds up to ~7 real specialist tool calls -- already past
# MAX_SPECIALIST_EXTERNAL_CALLS below, so that external-call cap binds
# first in practice, exactly mirroring the documented relationship
# between MAX_GRAPH_TRANSITIONS and MAX_EXTERNAL_TOOL_CALLS at the
# supervisor level (ADR 0014 §5).
MAX_SPECIALIST_GRAPH_TRANSITIONS = 15
# The specialist's own external-tool-call backstop for a single
# delegation -- independent of, and in addition to, the shared
# MAX_EXTERNAL_TOOL_CALLS/TOTAL_WORKFLOW_DEADLINE_SECONDS bounds it also
# obeys (phase4/specialist.py). Never relies on this alone: it is
# defense in depth against a runaway specialist loop, exactly the same
# "structural backstop beyond the primary bound" pattern
# MAX_CONSECUTIVE_DUPLICATES already establishes for the supervisor
# (phase4/graph.py).
MAX_SPECIALIST_EXTERNAL_CALLS = 5
# The specialist's own decision-format repair budget -- local to the
# specialist loop, never shared with the supervisor's own
# MAX_DECISION_REPAIRS counter (a decision-format repair is not an
# external call, so it is deliberately NOT one of the two bounds this
# checkpoint's instructions name as shared "across both loops").
MAX_SPECIALIST_DECISION_REPAIRS = 2


# --- allowed actions (closed allowlist; unknown values fail enum validation) -------


class Action(str, Enum):
    SEARCH_FLIGHTS = "search_flights"
    SEARCH_STAYS = "search_stays"
    ESTIMATE_FAIR_PRICE = "estimate_fair_price"
    GET_WEATHER = "get_weather"
    WEB_SEARCH = "web_search"
    CALL_ISTANBUL_EXPERT = "call_istanbul_expert"
    ASK_CLARIFICATION = "ask_clarification"
    SYNTHESIZE = "synthesize"
    DEGRADE = "degrade"
    # Checkpoint Phase 4 D.3: the supervisor's own delegation action --
    # hands the 5 travel-search tools off to the internal Travel Search
    # specialist sub-loop (phase4/specialist.py) instead of calling them
    # directly. Carries no arguments (see CallTravelSearchArgs below) --
    # the specialist derives what is still needed from the same shared
    # evidence registry the supervisor itself already uses, exactly like
    # every other action here never carries a reasoning transcript.
    CALL_TRAVEL_SEARCH = "call_travel_search"
    # The specialist's own closed terminal signal -- structurally
    # distinct from Action.SYNTHESIZE so the specialist can never
    # produce a supervisor-only action by construction (Pydantic enum
    # validation rejects it outright), matching this checkpoint's
    # "specialist cannot call System B or synthesize" requirement at
    # the schema level, not just by convention.
    TRAVEL_SEARCH_COMPLETE = "travel_search_complete"


# Actions that call out to an external tool -- everything else (ASK_CLARIFICATION,
# SYNTHESIZE, DEGRADE, CALL_TRAVEL_SEARCH, TRAVEL_SEARCH_COMPLETE) is an
# in-process control decision, never counted against MAX_EXTERNAL_TOOL_CALLS
# and never routed through Execute/Observe itself -- CALL_TRAVEL_SEARCH's
# own delegated sub-calls each count individually when the specialist
# loop actually executes them (phase4/specialist.py), attributed to the
# exact same shared counters the supervisor's own direct tool calls use,
# so the accounting semantics of MAX_EXTERNAL_TOOL_CALLS/MAX_CALLS_PER_TOOL
# are unchanged by this checkpoint -- never redefined, only correctly
# attributed regardless of which loop actually made a given call.
TOOL_CALL_ACTIONS = frozenset({
    Action.SEARCH_FLIGHTS, Action.SEARCH_STAYS, Action.ESTIMATE_FAIR_PRICE,
    Action.GET_WEATHER, Action.WEB_SEARCH, Action.CALL_ISTANBUL_EXPERT,
})

# Checkpoint Phase 4 D.3: the closed set of actions the internal Travel
# Search specialist may ever decide -- exactly the 5 named tools plus its
# own terminal signal, structurally excluding call_istanbul_expert,
# ask_clarification, synthesize, degrade, and call_travel_search itself
# (a specialist can never re-delegate to another specialist -- there is
# exactly one, never nested). The supervisor's own allowed set is
# whatever remains: everything in `Action` except these 5 tools (the
# supervisor no longer decides them directly in production -- it only
# ever proposes CALL_TRAVEL_SEARCH for that evidence) and except the
# specialist-only terminal signal.
SPECIALIST_ACTIONS = frozenset({
    Action.GET_WEATHER, Action.WEB_SEARCH, Action.SEARCH_FLIGHTS,
    Action.SEARCH_STAYS, Action.ESTIMATE_FAIR_PRICE, Action.TRAVEL_SEARCH_COMPLETE,
})
SUPERVISOR_ACTIONS = frozenset(a for a in Action if a not in SPECIALIST_ACTIONS)


class ReasonCode(str, Enum):
    """A fixed, closed vocabulary -- never a free-form reasoning transcript."""

    MISSING_FLIGHT_INFO = "missing_flight_info"
    MISSING_STAY_INFO = "missing_stay_info"
    NEEDS_FAIR_PRICE = "needs_fair_price"
    MISSING_WEATHER_INFO = "missing_weather_info"
    MISSING_CURRENT_INFO = "missing_current_info"
    MISSING_LOCAL_EXPERTISE = "missing_local_expertise"
    MISSING_ESSENTIAL_INPUT = "missing_essential_input"
    IRRELEVANT_TO_REQUEST = "irrelevant_to_request"
    ALL_REQUIRED_EVIDENCE_PRESENT = "all_required_evidence_present"
    DUPLICATE_CALL_AVOIDED = "duplicate_call_avoided"
    BOUND_REACHED = "bound_reached"
    TOOL_UNAVAILABLE = "tool_unavailable"
    DECISION_FORMAT_INVALID = "decision_format_invalid"
    CANCELLED = "cancelled"
    INPUT_REJECTED = "input_rejected"


# --- per-action argument schemas (closed; extra="forbid") --------------------------


class SearchFlightsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    depart_date: date
    passenger_count: int = Field(ge=1, le=12)
    cabin_class: Optional[str] = Field(default=None, pattern=r"^(economy|premium_economy|business|first)$")


class SearchStaysArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    check_in: date
    check_out: date
    guest_count: int = Field(ge=1, le=12)
    district_id: Optional[str] = Field(default=None, pattern=r"^district_[a-z0-9_]+$")


class EstimateFairPriceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stay_id: str = Field(min_length=1, max_length=200)


class GetWeatherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str = Field(default="Istanbul", min_length=1, max_length=100)
    date_from: date
    date_to: date


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=3, ge=1, le=8)


class CallIstanbulExpertArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)


class AskClarificationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    missing_fields: list[str] = Field(min_length=1, max_length=10)
    question: str = Field(min_length=1, max_length=500)


class SynthesizeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DegradeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=300)


class CallTravelSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Deliberately empty -- matches SynthesizeArgs's own "no argument
    # surface for chain-of-thought" pattern. The specialist decides what
    # is still needed from the shared evidence registry itself.


class TravelSearchCompleteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Deliberately empty -- a closed terminal signal only.


ACTION_ARGUMENT_MODELS: dict[Action, type[BaseModel]] = {
    Action.SEARCH_FLIGHTS: SearchFlightsArgs,
    Action.SEARCH_STAYS: SearchStaysArgs,
    Action.ESTIMATE_FAIR_PRICE: EstimateFairPriceArgs,
    Action.GET_WEATHER: GetWeatherArgs,
    Action.WEB_SEARCH: WebSearchArgs,
    Action.CALL_ISTANBUL_EXPERT: CallIstanbulExpertArgs,
    Action.ASK_CLARIFICATION: AskClarificationArgs,
    Action.SYNTHESIZE: SynthesizeArgs,
    Action.DEGRADE: DegradeArgs,
    Action.CALL_TRAVEL_SEARCH: CallTravelSearchArgs,
    Action.TRAVEL_SEARCH_COMPLETE: TravelSearchCompleteArgs,
}


class ActionDecisionValidationError(ValueError):
    """Raised when a raw decision-provider response cannot be turned into
    a valid `ActionDecision` -- an unknown action, arguments that fail
    that action's own schema, or malformed JSON. Callers (the Decide
    node) catch this and count it against the bounded decision-repair
    budget; it never propagates as a raw exception to a user-facing
    result."""


class ActionDecision(BaseModel):
    """The only shape a decision provider (Qwen or a fake) is ever
    allowed to produce. `arguments` is validated against
    `ACTION_ARGUMENT_MODELS[action]` by `parse_action_decision` below --
    Pydantic itself cannot express "the schema of this field depends on
    that field's value", so that cross-field validation happens in a
    dedicated parse function, not a model validator, to keep the error
    reporting precise (which field, which action)."""

    model_config = ConfigDict(extra="forbid")
    action: Action
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason_code: ReasonCode
    # A short, user-safe explanation only -- never a reasoning transcript.
    # The length bound alone makes a multi-paragraph chain-of-thought
    # dump structurally impossible to smuggle through this field.
    explanation: Optional[str] = Field(default=None, max_length=280)


def parse_action_decision(raw: dict) -> ActionDecision:
    """Validates `raw` as an `ActionDecision` and then validates its
    `arguments` against exactly that action's own schema. Raises
    `ActionDecisionValidationError` (never a raw pydantic.ValidationError)
    on any failure -- unknown action, extra field, wrong argument type,
    or an argument schema mismatch."""
    try:
        decision = ActionDecision.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any parse/validation failure is a repair-worthy format error
        raise ActionDecisionValidationError(f"invalid ActionDecision shape: {exc}") from exc

    argument_model = ACTION_ARGUMENT_MODELS[decision.action]
    try:
        validated_arguments = argument_model.model_validate(decision.arguments)
    except Exception as exc:  # noqa: BLE001
        raise ActionDecisionValidationError(
            f"arguments for action={decision.action.value!r} failed schema validation: {exc}"
        ) from exc

    return decision.model_copy(update={"arguments": validated_arguments.model_dump(mode="json")})


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_action(action: Action, arguments: dict) -> str:
    """Deterministic fingerprint of (action, arguments) -- pure, no wall
    clock, no call order -- used for the evidence registry's duplicate-
    call detection (ADR 0009 §4.4). A small, planner-owned duplicate of
    the same canonical-JSON-then-sha256 pattern `providers.fingerprint`
    already established -- deliberately not imported from there (this
    submodule never depends on the root `providers` package, ADR 0009
    §6.1)."""
    return sha256_hex(canonical_json({"action": action.value, "arguments": arguments}))


# --- tool-observation wrapper (additive orchestration bookkeeping) -----------------


class ToolObservation(BaseModel):
    """What Update actually stores per completed (or failed) tool call.
    `envelope` is the reused, unmodified `phase1.models.ProviderResponseEnvelope`
    shape (plus its nested canonical result model, validated separately
    by the caller before this object is constructed) -- this model adds
    only the operational status/fingerprint fields that 1.0.0-era mirror
    does not carry."""

    model_config = ConfigDict(extra="forbid")
    action: Action
    status: str  # one of the shared RESULT_STATUSES-style vocabulary: success/timeout/rate_limited/provider_error/unavailable/cancelled/unsupported/invalid_request
    fingerprint: str
    envelope: Optional[dict[str, Any]] = None  # a validated ProviderResponseEnvelope.model_dump(), or None on failure
    warnings: list[str] = Field(default_factory=list)


class PlannerRequest(BaseModel):
    """The graph's own entry-point parameter bundle -- not a competing
    TripRequest: `trip_request`, when present, is exactly
    `phase1.models.TripRequest`, reused unchanged. This wrapper exists
    only because a real conversational planner request is a free-text
    message optionally accompanied by a fully-structured trip request,
    and the graph needs both without inventing a second TripRequest
    shape for the narrow, single-capability questions (Checkpoint D.0's
    own required test scenarios: weather-only, flight-only, ...) that
    never carry a full TripRequest at all."""

    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    trace_id: UUID
    user_message: str = Field(min_length=1, max_length=4000)
    trip_request: Optional[dict[str, Any]] = None  # TripRequest.model_dump(mode="json") when present
