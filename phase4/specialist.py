"""The internal Travel Search specialist -- a genuine, separately
compiled `langgraph.graph.StateGraph` (Checkpoint Phase 4 D.3 correction
pass, docs/adr/0017-...md §§1-2). Invoked by System A's supervisor graph
(`phase4/graph.py`) from inside its own Execute node, exactly the way any
other in-process Python call is made -- never a second network hop,
never A2A (that boundary is reserved for the System A/System B network
edge, architecture.md §4.2). This module owns only the specialist's own
graph: its typed state, its typed result artifact, its three real nodes,
and the graph builder. It knows nothing about SSE, HTTP, or persistence.

Reuses -- literally, never a second implementation -- the exact same
`ToolExecutor`/`DecisionProvider`/`ActionDecision`/`parse_action_decision`/
`fingerprint_action` machinery (`phase4/models.py`, `phase4/qwen_client.py`,
`phase4/tools.py`) and the shared `validate_tool_result`/
`action_argument_contract` helpers (`phase4/tool_result_validation.py`,
`phase4/prompt_contract.py`) the supervisor's own nodes already use.

A compact, real graph: `specialist_decide -> specialist_execute_observe
-> specialist_decide -> ... -> specialist_end`. Every one of those edges
is a genuine LangGraph transition, counted by this graph's own
`recursion_limit` (`MAX_SPECIALIST_GRAPH_TRANSITIONS`) -- never a plain
Python `for`/`while` loop hiding tool calls from LangGraph's own engine.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from phase4.context import ExecutionContext
from phase4.models import (
    MAX_CALLS_PER_TOOL,
    MAX_EXTERNAL_TOOL_CALLS,
    MAX_SPECIALIST_DECISION_REPAIRS,
    MAX_SPECIALIST_EXTERNAL_CALLS,
    MAX_SPECIALIST_GRAPH_TRANSITIONS,
    SPECIALIST_ACTIONS,
    TOTAL_WORKFLOW_DEADLINE_SECONDS,
    Action,
    ActionDecision,
    ActionDecisionValidationError,
    ReasonCode,
    ToolObservation,
    fingerprint_action,
    parse_action_decision,
)
from phase4.prompt_contract import action_argument_contract
from phase4.qwen_client import DecisionProvider, QwenTransportError
from phase4.tool_result_validation import validate_tool_result
from phase4.tools import ToolExecutor

SCHEMA_VERSION = "1.0.0"

# --- typed state and result artifact ------------------------------------------------


class TravelSearchState(TypedDict, total=False):
    """What the specialist's own graph genuinely owns. `tool_call_count`/
    `tool_call_count_by_action`/`executed_fingerprints`/
    `started_at_monotonic` are SEEDED from the supervisor's own shared
    state at invocation time and updated in place as the specialist's own
    real tool calls happen -- this is what makes the shared budget/
    deadline/duplicate-registry genuinely shared, not a second copy the
    supervisor would have to separately reconcile."""

    session_id: str
    trace_id: str
    normalized_request: dict[str, Any]
    inherited_observations: list[dict[str, Any]]  # read-only: the supervisor's own prior evidence
    specialist_observations: list[dict[str, Any]]  # this delegation's own, growing evidence
    tool_call_count: int  # shared/global total, seeded from the supervisor
    tool_call_count_by_action: dict[str, int]  # shared/global, seeded from the supervisor
    executed_fingerprints: list[str]  # shared/global duplicate registry, seeded from the supervisor
    warnings: list[str]
    started_at_monotonic: float  # the supervisor's own clock origin -- one global 60s deadline
    pending_action: Optional[dict[str, Any]]
    graph_transition_count: int  # this specialist graph's OWN transitions, never the supervisor's
    repair_count: int  # this specialist graph's own decision-repair count
    cancelled: bool
    final_result: Optional[dict[str, Any]]  # a validated TravelSearchResult.model_dump(mode="json")


class TravelSearchResult(BaseModel):
    """The specialist's own typed delegation artifact -- validated before
    it is ever handed back to the supervisor (never an untyped dict).
    `observations` reuses the exact same `ToolObservation` shape the
    supervisor's own Observe node produces for a direct tool call, so the
    supervisor can flatten them into its own public observation list
    unchanged (frontend/result compatibility is preserved structurally,
    by reusing the identical shape, not by convention)."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    status: str  # "success" | "partial" | "degraded" | "cancelled"
    observations: list[ToolObservation] = Field(default_factory=list)
    completed_capabilities: list[str] = Field(default_factory=list)
    degraded_capabilities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    calls_consumed: int = 0
    transitions_consumed: int = 0
    tool_call_count_by_action: dict[str, int] = Field(default_factory=dict)
    new_fingerprints: list[str] = Field(default_factory=list)


# --- prompt (specialist-scoped; mirrors the supervisor's own prompt shape) ----------

_EXAMPLE_DECISION = {
    "action": "search_flights",
    "arguments": {
        "origin": "BEY",
        "destination": "IST",
        "depart_date": "2026-09-10",
        "passenger_count": 1,
        "cabin_class": "economy",
    },
    "reason_code": "missing_flight_info",
    "explanation": "Flight information is required.",
}


def _build_specialist_prompt(state: TravelSearchState) -> tuple[str, str]:
    """Returns (system, user) -- local values only, never persisted into
    state, trace, or a checkpoint. Scoped to `SPECIALIST_ACTIONS` only;
    sees the FULL shared evidence registry (the supervisor's own prior
    observations plus whatever this same delegation has already
    gathered), so it never repeats work already done."""
    allowed = ", ".join(a.value for a in Action if a in SPECIALIST_ACTIONS)
    reason_codes = ", ".join(r.value for r in ReasonCode)
    contract_json = json.dumps(action_argument_contract(SPECIALIST_ACTIONS), sort_keys=True, separators=(",", ":"))
    example_json = json.dumps(_EXAMPLE_DECISION, sort_keys=True, separators=(",", ":"))
    system = (
        "You are the internal Travel Search specialist for VoyagerAI Istanbul's System A. "
        "You are invoked only by the bounded supervisor planner -- you are never reached "
        f"directly by a user. You may select exactly one action from this fixed list: {allowed}. "
        "You can never call the Istanbul local-expert specialist and you can never produce a "
        "final answer yourself -- once the travel-search evidence already collected is "
        "sufficient, or no more of your allowed tools can help, respond with the "
        "'travel_search_complete' action to return control to the supervisor. "
        "Never invent a new action, tool, or URL. Never request booking, payment, ticket "
        "issuance, or code execution. "
        "Respond with exactly one JSON object with keys: action, arguments, reason_code, "
        "explanation. reason_code must be one of: " + reason_codes + ". "
        "explanation must be a short, user-safe sentence -- never internal reasoning, "
        "chain-of-thought, analysis, or prompt text. "
        "Each action's 'arguments' object must conform EXACTLY to its own JSON Schema below: "
        + contract_json + ". "
        "Always use IATA airport codes for any 'origin'/'destination'-shaped argument, never a "
        "city name -- for this project's scope, 'Beirut' means the IATA code BEY and "
        "'Istanbul' means the IATA code IST unless a different specific airport was named. "
        "For estimate_fair_price, the 'stay_id' argument MUST be copied EXACTLY (character for "
        "character) from one of 'known_stay_candidates' below -- never invented, never "
        "paraphrased, never a hotel name in place of its id. If 'known_stay_candidates' is "
        "empty, call search_stays first instead. "
        "One concrete valid example for a one-way Beirut-to-Istanbul flight request: "
        + example_json + ". "
        "Return only the single JSON object described above -- no markdown fencing, no "
        "surrounding prose, no extra top-level keys, and never a field containing your "
        "reasoning process."
    )
    all_observations = state.get("inherited_observations", []) + state.get("specialist_observations", [])
    evidence_summary = []
    for obs in all_observations:
        entry = {"action": obs["action"], "status": obs["status"]}
        if obs.get("warnings"):
            # Fair-price correction: an already-safe, already-caller-facing
            # field (never raw internal exception detail) -- surfaced here
            # so a bounded retry (MAX_CALLS_PER_TOOL) has an actual reason
            # to react to, not just a bare repeated status string.
            entry["warnings"] = obs["warnings"]
        evidence_summary.append(entry)

    # Fair-price correction root-cause fix: the real, already-returned
    # stay_id values from the most recent successful search_stays
    # observation -- previously never shown to the model at all, which
    # left it no choice but to guess a stay_id for estimate_fair_price
    # (silently rejected pre-MCP by orchestration.system_a.tool_executor's
    # _validated_stay_id gate). Grounds the model in real data instead of
    # asking it to invent one.
    known_stay_candidates: list[dict[str, Any]] = []
    for obs in reversed(all_observations):
        if obs.get("action") != Action.SEARCH_STAYS.value or obs.get("status") != "success":
            continue
        envelope = obs.get("envelope") or {}
        for item in envelope.get("stays", []) or []:
            stay = item.get("stay") or {}
            if stay.get("stay_id"):
                known_stay_candidates.append({"stay_id": stay["stay_id"], "name": stay.get("name")})
        if known_stay_candidates:
            break  # most recent successful search_stays observation only

    total_used = state.get("tool_call_count", 0)
    user_payload = {
        "user_message": state.get("normalized_request", {}).get("user_message", ""),
        "trip_request": state.get("normalized_request", {}).get("trip_request"),
        "evidence_collected_so_far": evidence_summary,
        "known_stay_candidates": known_stay_candidates,
        "tool_calls_used": total_used,
        "tool_calls_remaining": MAX_EXTERNAL_TOOL_CALLS - total_used,
    }
    user = "Decide the next travel-search action for this request:\n" + json.dumps(user_payload, default=str)
    return system, user


# --- node factories --------------------------------------------------------------


def _make_specialist_decide_node(
    decision_provider: DecisionProvider, cancellation_check: Callable[[], bool], monotonic_clock: Callable[[], float]
) -> Callable[[TravelSearchState], dict[str, Any]]:
    def _specialist_decide(state: TravelSearchState) -> dict[str, Any]:
        transitions = state.get("graph_transition_count", 0) + 1

        if cancellation_check():
            decision = ActionDecision(action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.CANCELLED)
            return {
                "graph_transition_count": transitions,
                "pending_action": decision.model_dump(mode="json"),
                "cancelled": True,
            }

        started_at = state.get("started_at_monotonic")
        if started_at is None:
            started_at = monotonic_clock()
        if monotonic_clock() - started_at >= TOTAL_WORKFLOW_DEADLINE_SECONDS:
            decision = ActionDecision(
                action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.BOUND_REACHED,
            )
            return {"graph_transition_count": transitions, "pending_action": decision.model_dump(mode="json")}

        if state.get("tool_call_count", 0) >= MAX_EXTERNAL_TOOL_CALLS:
            # The shared, global budget -- reused unmodified, never a
            # second independent ceiling (see TravelSearchState docstring).
            decision = ActionDecision(
                action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.BOUND_REACHED,
            )
            return {"graph_transition_count": transitions, "pending_action": decision.model_dump(mode="json")}

        if len(state.get("specialist_observations", [])) >= MAX_SPECIALIST_EXTERNAL_CALLS:
            # The specialist's OWN, smaller, per-delegation external-call
            # backstop -- distinct from the shared MAX_EXTERNAL_TOOL_CALLS
            # ceiling above: this counts only real tool calls THIS
            # delegation has made (`len(specialist_observations)`), never
            # the supervisor's own running total, so it fires even when
            # the shared 8-call budget still has room left.
            decision = ActionDecision(
                action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.BOUND_REACHED,
            )
            return {"graph_transition_count": transitions, "pending_action": decision.model_dump(mode="json")}

        system, user = _build_specialist_prompt(state)
        decision: Optional[ActionDecision] = None
        repair_count = state.get("repair_count", 0)
        attempts = 0
        max_attempts = MAX_SPECIALIST_DECISION_REPAIRS + 1
        while attempts < max_attempts and decision is None:
            attempts += 1
            try:
                raw_text = decision_provider.generate(system, user)
                raw_obj = json.loads(raw_text)
                candidate = parse_action_decision(raw_obj)
                if candidate.action not in SPECIALIST_ACTIONS:
                    raise ActionDecisionValidationError(
                        f"specialist decision named a non-specialist action {candidate.action.value!r}"
                    )
                decision = candidate
            except (json.JSONDecodeError, ActionDecisionValidationError) as exc:
                last_error = str(exc)[:200]
                if attempts < max_attempts:
                    user = (
                        user + f"\n\nYour previous response was invalid ({last_error}). "
                        "Respond again with a single valid JSON object matching the required schema exactly."
                    )
            except QwenTransportError:
                decision = ActionDecision(
                    action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.TOOL_UNAVAILABLE,
                )
        if decision is None:
            decision = ActionDecision(
                action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.DECISION_FORMAT_INVALID,
            )
        repair_count = repair_count + max(0, attempts - 1)

        # Per-tool cap and exact-fingerprint duplication -- checked here,
        # in Decide, mirroring the supervisor's own decide-node bound-
        # check placement, so specialist_execute_observe stays a simple,
        # unconditional "run whatever Decide already approved" node.
        if decision.action != Action.TRAVEL_SEARCH_COMPLETE:
            fingerprint = fingerprint_action(decision.action, decision.arguments)
            by_action = state.get("tool_call_count_by_action", {})
            if by_action.get(decision.action.value, 0) >= MAX_CALLS_PER_TOOL:
                decision = ActionDecision(action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.BOUND_REACHED)
            elif fingerprint in state.get("executed_fingerprints", []):
                # Deliberately simpler than the supervisor's own skip-
                # then-force-synthesize duplicate handling: the very
                # first exact repeat just ends this delegation (see ADR
                # 0017 §4) -- nothing further within one delegation could
                # usefully change the outcome.
                decision = ActionDecision(action=Action.TRAVEL_SEARCH_COMPLETE, arguments={}, reason_code=ReasonCode.DUPLICATE_CALL_AVOIDED)

        return {
            "graph_transition_count": transitions,
            "pending_action": decision.model_dump(mode="json"),
            "repair_count": repair_count,
        }

    return _specialist_decide


def _make_specialist_execute_observe_node(
    tool_executor: ToolExecutor, cancellation_check: Callable[[], bool]
) -> Callable[[TravelSearchState], dict[str, Any]]:
    def _specialist_execute_observe(state: TravelSearchState) -> dict[str, Any]:
        transitions = state.get("graph_transition_count", 0) + 1
        pending = state["pending_action"] or {}
        action = Action(pending["action"])
        arguments = pending.get("arguments", {})

        if cancellation_check():
            return {"graph_transition_count": transitions, "cancelled": True}

        fingerprint = fingerprint_action(action, arguments)
        started_at = state.get("started_at_monotonic", 0.0)
        context = ExecutionContext(
            session_id=state.get("session_id", ""),
            trace_id=state.get("trace_id", ""),
            normalized_request=state.get("normalized_request", {}),
            observations=tuple(state.get("inherited_observations", []) + state.get("specialist_observations", [])),
            deadline_monotonic=started_at + TOTAL_WORKFLOW_DEADLINE_SECONDS,
            cancellation_check=cancellation_check,
        )
        raw = tool_executor.execute(action, arguments, context)

        status = raw.get("status", "provider_error")
        warnings = list(state.get("warnings", []))
        envelope_dict = None
        if status == "success":
            candidate = raw.get("result")
            validation_error = validate_tool_result(action, candidate)
            if validation_error is not None:
                status = "provider_error"
                warnings.append(f"malformed_tool_result:{action.value}:{validation_error}")
            else:
                envelope_dict = candidate
        elif action == Action.GET_WEATHER:
            # Manual QA remediation Q.1: mirrors the identical fix in
            # phase4/graph.py::_observe_node -- GET_WEATHER is a
            # specialist-owned tool (Checkpoint D.3), so THIS is the node
            # that actually runs for it in production, not the
            # supervisor's own Observe. A degraded-but-well-formed weather
            # envelope (e.g. forecast_not_yet_available, carrying a real
            # earliest_available_forecast_date) was previously discarded
            # entirely just because the call didn't succeed.
            candidate = raw.get("result")
            if isinstance(candidate, dict):
                envelope_dict = candidate

        observation_warnings: list[str] = []
        if envelope_dict is None:
            reason = raw.get("reason")  # additive, internal-only field (orchestration.system_a.tool_executor)
            observation_warnings.append(f"status={status}:{reason}" if reason else f"status={status}")

        observation = {
            "action": action.value,
            "status": status,
            "fingerprint": fingerprint,
            "envelope": envelope_dict,
            "warnings": observation_warnings,
        }

        by_action = dict(state.get("tool_call_count_by_action", {}))
        by_action[action.value] = by_action.get(action.value, 0) + 1

        return {
            "graph_transition_count": transitions,
            "specialist_observations": state.get("specialist_observations", []) + [observation],
            "executed_fingerprints": state.get("executed_fingerprints", []) + [fingerprint],
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "tool_call_count_by_action": by_action,
            "warnings": warnings,
            "pending_action": None,
        }

    return _specialist_execute_observe


def _specialist_end_node(state: TravelSearchState) -> dict[str, Any]:
    transitions = state.get("graph_transition_count", 0) + 1
    observations = state.get("specialist_observations", [])
    completed = [obs["action"] for obs in observations if obs.get("status") == "success"]
    degraded = [obs["action"] for obs in observations if obs.get("status") != "success"]

    if state.get("cancelled"):
        status = "cancelled"
    elif not observations:
        status = "success"  # nothing needed, or already-known -- an honest "no work required" outcome
    elif all(obs.get("status") == "success" for obs in observations):
        status = "success"
    else:
        status = "partial"

    result = TravelSearchResult(
        status=status,
        observations=[ToolObservation.model_validate(obs) for obs in observations],
        completed_capabilities=completed,
        degraded_capabilities=degraded,
        warnings=state.get("warnings", []),
        calls_consumed=len(observations),
        transitions_consumed=transitions,
        tool_call_count_by_action=state.get("tool_call_count_by_action", {}),
        new_fingerprints=[obs["fingerprint"] for obs in observations],
    )
    return {"graph_transition_count": transitions, "final_result": result.model_dump(mode="json")}


# --- routing -----------------------------------------------------------------------


def _route_after_specialist_decide(state: TravelSearchState) -> str:
    pending = state.get("pending_action") or {}
    if pending.get("action") == Action.TRAVEL_SEARCH_COMPLETE.value or state.get("cancelled"):
        return "specialist_end"
    return "specialist_execute_observe"


# --- graph construction --------------------------------------------------------------


def build_specialist_graph(
    tool_executor: ToolExecutor,
    decision_provider: DecisionProvider,
    cancellation_check: Callable[[], bool] = lambda: False,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> CompiledStateGraph:
    """Compiles the internal Travel Search specialist's own real
    `StateGraph` -- `decision_provider` here is the SPECIALIST's own,
    separate `DecisionProvider` instance (never the supervisor's own,
    never inferred from prompt content -- see ADR 0017 §3)."""
    graph = StateGraph(TravelSearchState)
    graph.add_node("specialist_decide", _make_specialist_decide_node(decision_provider, cancellation_check, monotonic_clock))
    graph.add_node("specialist_execute_observe", _make_specialist_execute_observe_node(tool_executor, cancellation_check))
    graph.add_node("specialist_end", _specialist_end_node)

    graph.add_edge("__start__", "specialist_decide")
    graph.add_conditional_edges(
        "specialist_decide", _route_after_specialist_decide,
        {"specialist_execute_observe": "specialist_execute_observe", "specialist_end": "specialist_end"},
    )
    graph.add_edge("specialist_execute_observe", "specialist_decide")
    graph.add_edge("specialist_end", "__end__")

    # A fresh, ephemeral, function-call-scoped MemorySaver -- never
    # shared across delegations or persisted to disk. It exists so a
    # genuine `GraphRecursionError` (the 15-transition ceiling) can still
    # recover whatever real progress the last successfully-completed
    # superstep already made (LangGraph checkpoints after every
    # superstep), rather than discarding it -- see
    # `invoke_travel_search_specialist`'s own recovery path below.
    return graph.compile(checkpointer=MemorySaver())


# --- supervisor-facing invocation boundary --------------------------------------------


def _degraded_result_from_partial_state(values: dict[str, Any]) -> dict[str, Any]:
    """Used only when the specialist's own `recursion_limit` fires before
    `specialist_end` was ever reached -- `values` is whatever the last
    successfully-completed superstep actually persisted (real tool calls
    already made are NOT discarded), matching the honest, already-
    established `_safe_recursion_limit_result` pattern the supervisor's
    own graph uses for its own recursion-limit backstop (ADR 0014 §5)."""
    observations = values.get("specialist_observations", [])
    completed = [obs["action"] for obs in observations if obs.get("status") == "success"]
    degraded = [obs["action"] for obs in observations if obs.get("status") != "success"]
    result = TravelSearchResult(
        status="partial" if observations else "degraded",
        observations=[ToolObservation.model_validate(obs) for obs in observations],
        completed_capabilities=completed,
        degraded_capabilities=degraded,
        warnings=list(values.get("warnings", [])) + ["specialist_transition_limit_reached"],
        calls_consumed=len(observations),
        transitions_consumed=values.get("graph_transition_count", 0),
        tool_call_count_by_action=values.get("tool_call_count_by_action", {}),
        new_fingerprints=[obs["fingerprint"] for obs in observations],
    )
    return result.model_dump(mode="json")


def invoke_travel_search_specialist(
    specialist_graph: CompiledStateGraph,
    *,
    session_id: str,
    trace_id: str,
    normalized_request: dict[str, Any],
    inherited_observations: list[dict[str, Any]],
    tool_call_count: int,
    tool_call_count_by_action: dict[str, int],
    executed_fingerprints: list[str],
    started_at_monotonic: float,
    on_event: Optional[Callable[[dict[str, Any]], None]] = None,
) -> TravelSearchResult:
    """The one boundary function `phase4/graph.py`'s own Execute node
    calls for a `call_travel_search` delegation. Seeds a fresh
    `TravelSearchState` from the supervisor's own current shared counters
    (never a second independent budget) and drives the compiled
    specialist graph via `.stream(..., stream_mode="updates")` -- never
    `.invoke()` -- specifically so `on_event`, when provided, can be
    called with a genuinely real-time event for each specialist
    transition AS IT HAPPENS (ADR 0017 §6: real live streaming, not a
    batch emitted after the whole delegation already finished). Returns
    a validated `TravelSearchResult`, never an untyped dict."""
    thread_id = f"specialist-{uuid4()}"
    initial_state: TravelSearchState = {
        "session_id": session_id,
        "trace_id": trace_id,
        "normalized_request": normalized_request,
        "inherited_observations": list(inherited_observations),
        "specialist_observations": [],
        "tool_call_count": tool_call_count,
        "tool_call_count_by_action": dict(tool_call_count_by_action),
        "executed_fingerprints": list(executed_fingerprints),
        "warnings": [],
        "started_at_monotonic": started_at_monotonic,
        "graph_transition_count": 0,
        "repair_count": 0,
    }
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_SPECIALIST_GRAPH_TRANSITIONS}

    try:
        for update in specialist_graph.stream(initial_state, config=config, stream_mode="updates"):
            if on_event is not None:
                _emit_specialist_events(update, on_event)
        final_values = specialist_graph.get_state(config).values
        result_dict = final_values.get("final_result")
        if result_dict is None:
            # The graph ended (or was interrupted) without ever reaching
            # specialist_end -- build an honest result from whatever real
            # progress the last completed superstep actually made.
            result_dict = _degraded_result_from_partial_state(final_values or initial_state)
    except GraphRecursionError:
        final_values = specialist_graph.get_state(config).values or initial_state
        result_dict = _degraded_result_from_partial_state(final_values)

    return TravelSearchResult.model_validate(result_dict)


def _emit_specialist_events(update: dict[str, dict[str, Any]], on_event: Callable[[dict[str, Any]], None]) -> None:
    """Translates one real LangGraph `stream_mode="updates"` update from
    the specialist's own graph into sanitized, real-time progress events
    -- action names and statuses only, never a prompt, an argument, or an
    explanation. Called DURING `specialist_graph.stream(...)`'s own
    iteration, i.e. genuinely as each specialist transition happens, not
    after the whole delegation has already completed."""
    for node_name, node_output in update.items():
        if node_name == "specialist_decide":
            pending = node_output.get("pending_action") or {}
            action = pending.get("action")
            if action and action != Action.TRAVEL_SEARCH_COMPLETE.value:
                on_event({"stage": "action_started", "action": action})
        elif node_name == "specialist_execute_observe":
            observations = node_output.get("specialist_observations") or []
            if observations:
                obs = observations[-1]
                stage = "action_completed" if obs.get("status") == "success" else "action_failed"
                on_event({"stage": stage, "action": obs.get("action"), "status": obs.get("status")})
