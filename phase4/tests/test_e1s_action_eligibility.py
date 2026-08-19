"""Final Evaluation Checkpoint E.1S.1 -- deterministic tests for the
typed per-turn capability-plan classification (`phase4.graph.
_classify_capability`), the scope-driven action-eligibility layer
(`phase4.graph.compute_eligible_supervisor_actions`), the post-decision
eligibility gate + one-shot correction in `_decide_node`, and the
at-most-one transient-transport retry (`phase4.graph.
_generate_with_transport_retry` / `phase4.qwen_client.
QwenTransportError.transient`).

These are hermetic (no network call) and prove the MECHANISM behaves as
designed under scripted/fake conditions. They do not, and cannot,
measure live-Qwen routing accuracy -- that requires an actual live-model
holdout run, explicitly out of scope for this remediation-only
checkpoint. See the final report for the honest, tested scope and limits
of what this mechanism does and does not structurally fix."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from phase4.graph import (
    PlannerState,
    _build_decision_prompt,
    _compute_request_signature,
    _generate_with_transport_retry,
    _make_decide_node,
    build_graph,
    compute_eligible_supervisor_actions,
    start_session,
)
from phase4.models import Action, PlannerRequest
from phase4.qwen_client import QwenTransportError
from phase4.tools import FakeToolExecutor


def _decision(action: str, arguments: dict, reason_code: str = "all_required_evidence_present") -> str:
    return json.dumps({"action": action, "arguments": arguments, "reason_code": reason_code, "explanation": "ok"})


def _classification(scope: str, reason_code: str = "requires_travel_evidence") -> str:
    return json.dumps({"scope": scope, "reason_code": reason_code})


def _obs(action: str, status: str = "success") -> dict:
    return {"action": action, "status": status}


def _plan(scope: str, signature: str = "sig-1", succeeded: bool = True, reason_code: str = "requires_travel_evidence") -> dict:
    return {"scope": scope, "request_signature": signature, "classification_succeeded": succeeded, "reason_code": reason_code}


def _state(**overrides) -> PlannerState:
    base: PlannerState = {
        "observations": [], "tool_call_count": 0,
        "travel_search_attempted_signature": None, "istanbul_expert_attempted_signature": None,
        "capability_plan": _plan("combined"),
    }
    base.update(overrides)
    return base


# --- 1-8: compute_eligible_supervisor_actions, direct state-level tests, all 5 scopes -


def test_no_capability_plan_yields_only_degrade_eligibility():
    """Checkpoint Final Evaluation E.1Y: with no classification at all
    (and no safe prior-scope fallback -- E.1W's own fallback-inheritance
    already turns a recoverable failure into `classification_succeeded=
    True` before this state is ever reached in production), `degrade` is
    now part of THIS function's own policy, never a caller-side
    unconditional union -- an honest, structured degradation is the only
    legal action."""
    assert compute_eligible_supervisor_actions(_state(capability_plan=None)) == frozenset({Action.DEGRADE})


def test_failed_classification_yields_only_degrade_eligibility():
    assert compute_eligible_supervisor_actions(
        _state(capability_plan=_plan("clarification_required", succeeded=False))
    ) == frozenset({Action.DEGRADE})


def test_travel_only_before_terminal_with_trip_request_offers_only_call_travel_search():
    """Checkpoint Final Evaluation E.1Y: essential input IS present (a
    validated, structured `trip_request`) -- `ask_clarification` must not
    be offered alongside a fully-actionable specialist call."""
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("travel_only"),
        normalized_request={"user_message": "x", "trip_request": {"origin": "BEY"}},
    ))
    assert eligible == frozenset({Action.CALL_TRAVEL_SEARCH})


def test_travel_only_before_terminal_without_trip_request_also_offers_clarification():
    """Checkpoint Final Evaluation E.1Y: no validated `trip_request` is
    present -- essential input MAY be missing, so `ask_clarification`
    becomes legitimately co-eligible alongside `call_travel_search`.
    Deliberately NOT an exclusive `ask_clarification`-only result: a
    narrow single-capability message (e.g. a plain weather-date question)
    can still carry everything one specialist tool needs directly in its
    own text even with no structured `trip_request` object at all (this
    project's own established `PlannerRequest` design, ADR 0013) -- this
    deterministic, text-blind policy cannot itself tell those two cases
    apart, so both remain legally available and the one judgment call it
    cannot make safely on its own is left, gated and still correctable,
    to the model's next decision."""
    eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("travel_only")))
    assert eligible == frozenset({Action.CALL_TRAVEL_SEARCH, Action.ASK_CLARIFICATION})


def test_travel_only_after_terminal_offers_only_synthesize():
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("travel_only"), travel_search_attempted_signature="sig-1",
    ))
    assert eligible == frozenset({Action.SYNTHESIZE})


def test_travel_only_never_offers_call_istanbul_expert():
    for travel_terminal in (None, "sig-1"):
        eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("travel_only"), travel_search_attempted_signature=travel_terminal))
        assert Action.CALL_ISTANBUL_EXPERT not in eligible


def test_istanbul_local_only_before_terminal_offers_only_call_istanbul_expert():
    eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("istanbul_local_only", reason_code="requires_istanbul_local_grounding")))
    assert eligible == frozenset({Action.CALL_ISTANBUL_EXPERT})


def test_istanbul_local_only_after_terminal_offers_only_synthesize():
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("istanbul_local_only", reason_code="requires_istanbul_local_grounding"),
        istanbul_expert_attempted_signature="sig-1",
    ))
    assert eligible == frozenset({Action.SYNTHESIZE})


def test_istanbul_local_only_never_offers_call_travel_search():
    for istanbul_terminal in (None, "sig-1"):
        eligible = compute_eligible_supervisor_actions(_state(
            capability_plan=_plan("istanbul_local_only", reason_code="requires_istanbul_local_grounding"),
            istanbul_expert_attempted_signature=istanbul_terminal,
        ))
        assert Action.CALL_TRAVEL_SEARCH not in eligible


def test_combined_requires_travel_search_first():
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("combined", reason_code="requires_both"),
        normalized_request={"user_message": "x", "trip_request": {"origin": "BEY"}},
    ))
    assert eligible == frozenset({Action.CALL_TRAVEL_SEARCH})


def test_combined_requires_istanbul_expert_after_travel_search_terminal():
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("combined", reason_code="requires_both"), travel_search_attempted_signature="sig-1",
    ))
    assert eligible == frozenset({Action.CALL_ISTANBUL_EXPERT})


def test_combined_offers_synthesize_only_after_both_terminal():
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("combined", reason_code="requires_both"),
        travel_search_attempted_signature="sig-1", istanbul_expert_attempted_signature="sig-1",
    ))
    assert eligible == frozenset({Action.SYNTHESIZE})


def test_clarification_required_offers_only_ask_clarification():
    eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("clarification_required", reason_code="insufficient_information")))
    assert eligible == frozenset({Action.ASK_CLARIFICATION})


def test_out_of_scope_offers_only_degrade():
    eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("out_of_scope", reason_code="outside_project_scope")))
    assert eligible == frozenset({Action.DEGRADE})


def test_exhausted_tool_budget_forces_synthesize_only_regardless_of_scope():
    for scope in ("travel_only", "istanbul_local_only", "combined"):
        eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan(scope), tool_call_count=8))
        assert eligible == frozenset({Action.SYNTHESIZE})


def test_eligible_action_mask_is_always_a_subset_of_the_full_action_set_and_never_empty():
    """Checkpoint Final Evaluation E.1Y: the registered set now covers all
    5 actions the policy may ever return (specialist/synthesize plus
    ask_clarification/degrade, now part of the same deterministic
    policy) -- and invariant #9 ("at least one safe action always
    remains") is checked directly: the result is never empty."""
    registered = {
        Action.CALL_TRAVEL_SEARCH, Action.CALL_ISTANBUL_EXPERT, Action.SYNTHESIZE,
        Action.ASK_CLARIFICATION, Action.DEGRADE,
    }
    for scope in ("travel_only", "istanbul_local_only", "combined", "clarification_required", "out_of_scope"):
        for travel_terminal in (None, "sig-1"):
            for istanbul_terminal in (None, "sig-1"):
                for tool_calls in (0, 3, 8):
                    eligible = compute_eligible_supervisor_actions(_state(
                        capability_plan=_plan(scope), tool_call_count=tool_calls,
                        travel_search_attempted_signature=travel_terminal, istanbul_expert_attempted_signature=istanbul_terminal,
                    ))
                    assert eligible <= registered
                    assert eligible  # never empty


def test_a_prior_turns_terminal_flag_never_leaks_into_a_new_turns_signature():
    """The core E.1S.1 fix over E.1S: "terminal" is scoped to the CURRENT
    turn's signature, not the whole session -- a prior turn's completed
    delegation must never make the specialist ineligible for a genuinely
    new turn (different signature)."""
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("travel_only", signature="sig-NEW"), travel_search_attempted_signature="sig-OLD",
        normalized_request={"user_message": "x", "trip_request": {"origin": "BEY"}},
    ))
    assert eligible == frozenset({Action.CALL_TRAVEL_SEARCH})


# --- 9-11: full-graph scripted tests -- rejection, correction, safe termination -------


class _QueueDecisionProvider:
    """A scripted fake that can return raw JSON strings OR raise
    `QwenTransportError` on a given turn -- `items` is a list where each
    entry is either a str (raw JSON to return) or an exception instance
    to raise."""

    def __init__(self, items: list):
        self.items = list(items)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _run_graph(decider, user_message: str, trip_request):
    graph = build_graph(FakeToolExecutor(), decider, decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message=user_message, trip_request=trip_request)
    return start_session(graph, request, f"t-e1s1-{uuid4()}")


def test_repeated_call_to_terminal_istanbul_expert_is_prevented():
    """Section 6: repeated-call prevention within the SAME turn. A
    second, non-identical (different question, so the existing
    fingerprint-based duplicate skip does not fire) call_istanbul_expert
    decision, offered right after the first one already completed, is
    rejected as ineligible and corrected to synthesize rather than
    silently re-executed."""
    decider = _QueueDecisionProvider([
        _classification("istanbul_local_only", "requires_istanbul_local_grounding"),
        _decision("call_istanbul_expert", {"question": "What should I see first?"}),
        _decision("call_istanbul_expert", {"question": "What about day two?"}),  # ineligible: already terminal this turn
        _decision("synthesize", {}),  # the one bounded correction attempt
    ])
    result = _run_graph(decider, "Plan my day.", None)
    actions = [o["action"] for o in result["observations"]]
    assert actions.count("call_istanbul_expert") == 1
    assert result["final_result"]["status"] in ("success", "partial")
    rejected = [t for t in result["trace"] if t.get("status") == "action_ineligible_rejected"]
    corrected = [t for t in result["trace"] if t.get("status") == "action_ineligible_corrected"]
    assert rejected and rejected[0]["action"] == "call_istanbul_expert"
    assert corrected and corrected[0]["action"] == "synthesize"


def test_invalid_model_action_followed_by_successful_correction():
    decide = _make_decide_node(
        decision_provider=_QueueDecisionProvider([
            _decision("call_istanbul_expert", {"question": "x"}),  # ineligible: already attempted this turn
            _decision("synthesize", {}),  # valid correction
        ]),
        cancellation_check=lambda: False, monotonic_clock=lambda: 0.0,
    )
    signature = _compute_request_signature({"normalized_request": {"user_message": "x", "trip_request": None}})
    state: PlannerState = {
        "started_at_monotonic": 0.0, "tool_call_count": 1, "tool_call_count_by_action": {"call_istanbul_expert": 1},
        "executed_fingerprints": [], "observations": [_obs("call_istanbul_expert")], "trace": [],
        "graph_transition_count": 3, "repair_count": 0, "consecutive_duplicate_count": 0,
        "normalized_request": {"user_message": "x", "trip_request": None},
        "capability_plan": _plan("istanbul_local_only", signature=signature, reason_code="requires_istanbul_local_grounding"),
        "istanbul_expert_attempted_signature": signature,
    }
    updates = decide(state)
    assert updates["pending_action"]["action"] == "synthesize"
    assert updates["repair_count"] == 1
    statuses = [t.get("status") for t in updates["trace"]]
    assert "action_ineligible_rejected" in statuses
    assert "action_ineligible_corrected" in statuses


def test_two_invalid_decisions_in_a_row_terminate_safely_via_degrade():
    """Section 6: two invalid decisions -> safe termination. The
    correction attempt itself also names an ineligible action -> DEGRADE
    with a structured reason, never a silently executed ineligible
    action and never a second correction attempt (bounded to exactly
    one)."""
    decide = _make_decide_node(
        decision_provider=_QueueDecisionProvider([
            _decision("call_istanbul_expert", {"question": "x"}),  # ineligible
            _decision("call_istanbul_expert", {"question": "y"}),  # still ineligible on the one correction try
        ]),
        cancellation_check=lambda: False, monotonic_clock=lambda: 0.0,
    )
    signature = _compute_request_signature({"normalized_request": {"user_message": "x", "trip_request": None}})
    state: PlannerState = {
        "started_at_monotonic": 0.0, "tool_call_count": 1, "tool_call_count_by_action": {"call_istanbul_expert": 1},
        "executed_fingerprints": [], "observations": [_obs("call_istanbul_expert")], "trace": [],
        "graph_transition_count": 3, "repair_count": 0, "consecutive_duplicate_count": 0,
        "normalized_request": {"user_message": "x", "trip_request": None},
        "capability_plan": _plan("istanbul_local_only", signature=signature, reason_code="requires_istanbul_local_grounding"),
        "istanbul_expert_attempted_signature": signature,
    }
    updates = decide(state)
    assert updates["pending_action"]["action"] == "degrade"
    assert updates["pending_action"]["arguments"]["reason"] == "ineligible_action_after_correction"
    statuses = [t.get("status") for t in updates["trace"]]
    assert statuses.count("action_ineligible_rejected") == 1  # exactly one rejection ever recorded
    assert "action_ineligible_correction_failed" in statuses
    # no hidden reasoning/chain-of-thought ever lands in the trace -- only structured fields
    for entry in updates["trace"]:
        assert "raw_text" not in entry and "prompt" not in entry


# --- 12-13: transient-transport retry -------------------------------------------------


def test_one_retry_after_a_transient_transport_error():
    trace: list = []
    provider = _QueueDecisionProvider([
        QwenTransportError("Qwen HTTP error: status=503", status_code=503, transient=True),
        _decision("synthesize", {}),
    ])
    result = _generate_with_transport_retry(provider, "sys", "usr", trace, backoff_seconds=0.0)
    assert json.loads(result)["action"] == "synthesize"
    assert len(provider.calls) == 2  # exactly one retry -- two total attempts
    retry_entries = [t for t in trace if t.get("status") == "transport_retry"]
    assert len(retry_entries) == 1
    assert retry_entries[0]["status_code"] == 503


def test_no_retry_for_a_permanent_authentication_failure():
    trace: list = []
    provider = _QueueDecisionProvider([
        QwenTransportError("Qwen HTTP error: status=401", status_code=401, transient=False),
        _decision("synthesize", {}),  # must never be reached
    ])
    with pytest.raises(QwenTransportError):
        _generate_with_transport_retry(provider, "sys", "usr", trace, backoff_seconds=0.0)
    assert len(provider.calls) == 1  # no retry attempted
    assert not [t for t in trace if t.get("status") == "transport_retry"]
    assert [t for t in trace if t.get("status") == "transport_failed"]


def test_no_retry_for_a_response_shape_validation_failure():
    """Malformed JSON / missing content field is `transient=False` --
    never retried, matching "no retry for... validation errors, or
    unsupported model responses"."""
    trace: list = []
    provider = _QueueDecisionProvider([
        QwenTransportError("Qwen response was not valid JSON", transient=False),
    ])
    with pytest.raises(QwenTransportError):
        _generate_with_transport_retry(provider, "sys", "usr", trace, backoff_seconds=0.0)
    assert len(provider.calls) == 1


def test_transport_retry_at_most_two_total_attempts_even_if_both_are_transient():
    trace: list = []
    provider = _QueueDecisionProvider([
        QwenTransportError("Qwen transport failure", transient=True),
        QwenTransportError("Qwen transport failure", transient=True),
    ])
    with pytest.raises(QwenTransportError):
        _generate_with_transport_retry(provider, "sys", "usr", trace, backoff_seconds=0.0)
    assert len(provider.calls) == 2  # never a third attempt


def test_full_decide_node_falls_back_to_degrade_after_transport_retry_exhausted():
    """A transport failure at the CLASSIFICATION step (the first thing
    Decide does for a new turn) degrades safely without ever attempting
    an action decision."""
    decide = _make_decide_node(
        decision_provider=_QueueDecisionProvider([
            QwenTransportError("Qwen transport failure", transient=True),
            QwenTransportError("Qwen transport failure", transient=True),
        ]),
        cancellation_check=lambda: False, monotonic_clock=lambda: 0.0,
    )
    state: PlannerState = {
        "started_at_monotonic": 0.0, "tool_call_count": 0, "tool_call_count_by_action": {},
        "executed_fingerprints": [], "observations": [], "trace": [], "graph_transition_count": 0,
        "repair_count": 0, "consecutive_duplicate_count": 0,
        "normalized_request": {"user_message": "x", "trip_request": None},
    }
    updates = decide(state)
    assert updates["pending_action"]["action"] == "degrade"
    assert updates["pending_action"]["reason_code"] == "decision_format_invalid"
    assert updates["capability_plan"]["classification_succeeded"] is False


def test_classification_transient_failure_retries_then_recovers():
    """A transient failure during classification itself gets the same
    one-retry treatment as an action-decision call."""
    decide = _make_decide_node(
        decision_provider=_QueueDecisionProvider([
            QwenTransportError("Qwen transport failure", transient=True),
            _classification("travel_only"),
            _decision("call_travel_search", {}),
        ]),
        cancellation_check=lambda: False, monotonic_clock=lambda: 0.0,
    )
    state: PlannerState = {
        "started_at_monotonic": 0.0, "tool_call_count": 0, "tool_call_count_by_action": {},
        "executed_fingerprints": [], "observations": [], "trace": [], "graph_transition_count": 0,
        "repair_count": 0, "consecutive_duplicate_count": 0,
        "normalized_request": {"user_message": "weather please", "trip_request": None},
    }
    updates = decide(state)
    assert updates["capability_plan"]["classification_succeeded"] is True
    assert updates["capability_plan"]["scope"] == "travel_only"
    assert updates["pending_action"]["action"] == "call_travel_search"


def test_no_retry_around_specialist_tool_execution():
    """Retry only ever wraps the supervisor's own Decide-node generate()
    calls -- a real tool/specialist execution failure (e.g. `unavailable`)
    is never retried by this mechanism at all, it is simply recorded as
    an honest observation status."""
    decider = _QueueDecisionProvider([
        _classification("travel_only"),
        _decision("call_travel_search", {}),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
        _decision("travel_search_complete", {}),
        _decision("synthesize", {}),
    ])
    tools = FakeToolExecutor(scenario_by_action={Action.GET_WEATHER: "unavailable"})
    graph = build_graph(tools, decider, decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="weather please", trip_request=None)
    result = start_session(graph, request, f"t-e1s1-{uuid4()}")
    assert len(tools.call_log) == 1  # never retried
    weather_obs = [o for o in result["observations"] if o["action"] == "get_weather"]
    assert weather_obs and weather_obs[0]["status"] == "unavailable"


# --- 14: multilingual invariance -------------------------------------------------------


@pytest.mark.parametrize("user_message,question", [
    ("Plan my day.", "What should I see today?"),
    ("Gunumu planla.", "Bugun ne gormeliyim?"),
    ("خطط ليومي.", "ماذا يجب أن أرى اليوم؟"),
])
def test_eligibility_and_correction_behave_identically_across_languages(user_message, question):
    """compute_eligible_supervisor_actions never reads user_message, so
    the same ineligible-action-rejection behavior must fire identically
    regardless of request language."""
    decider = _QueueDecisionProvider([
        _classification("istanbul_local_only", "requires_istanbul_local_grounding"),
        _decision("call_istanbul_expert", {"question": question}),
        _decision("call_istanbul_expert", {"question": question + "2"}),  # ineligible: already terminal
        _decision("synthesize", {}),
    ])
    result = _run_graph(decider, user_message, None)
    actions = [o["action"] for o in result["observations"]]
    assert actions.count("call_istanbul_expert") == 1
    rejected = [t for t in result["trace"] if t.get("status") == "action_ineligible_rejected"]
    assert rejected


@pytest.mark.parametrize("user_message", ["Plan my day.", "Gunumu planla.", "خطط ليومي."])
def test_classification_prompt_is_identical_regardless_of_request_language(user_message):
    from phase4.graph import _build_capability_classification_prompt

    system_a, _ = _build_capability_classification_prompt({"normalized_request": {"user_message": user_message, "trip_request": None}})
    system_b, _ = _build_capability_classification_prompt({"normalized_request": {"user_message": "different text entirely", "trip_request": None}})
    assert system_a == system_b  # the classifier's SYSTEM prompt is architectural, never example/language-specific


# --- 15-16: ReAct loop and specialist-delegation regression proof ---------------------


def test_react_decide_act_observe_cycle_remains_intact_through_delegation_and_synthesis():
    decider = _QueueDecisionProvider([
        _classification("combined", "requires_both"),
        _decision("call_travel_search", {}),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
        _decision("travel_search_complete", {}),
        _decision("call_istanbul_expert", {"question": "What should I see?"}),
        _decision("synthesize", {}),
    ])
    result = _run_graph(decider, "Plan my full Istanbul trip, including things to see.", None)
    nodes_in_order = [t["node"] for t in result["trace"]]
    assert "InputGuard" in nodes_in_order and "LoadSession" in nodes_in_order
    assert nodes_in_order.count("Decide") >= 2  # supervisor decided at least twice (before and after delegation)
    assert "Execute" in nodes_in_order
    assert "Observe" in nodes_in_order
    assert "Synthesize" in nodes_in_order and "End" in nodes_in_order
    assert result["final_result"]["status"] == "success"


def test_supervisor_delegates_through_the_real_specialist_subgraph_not_a_direct_tool_call():
    """Regression proof: `call_travel_search` is executed by invoking the
    genuinely separate, already-compiled Travel Search specialist
    `StateGraph` -- never a direct tool call from the supervisor's own
    Execute node."""
    decider = _QueueDecisionProvider([
        _classification("travel_only"),
        _decision("call_travel_search", {}),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
        _decision("travel_search_complete", {}),
        _decision("synthesize", {}),
    ])
    result = _run_graph(decider, "What's the weather?", None)
    execute_entries = [t for t in result["trace"] if t["node"] == "Execute" and t.get("action") == "call_travel_search"]
    assert execute_entries and execute_entries[0]["status"] == "delegated"
    assert execute_entries[0]["specialist_actions"] == ["get_weather"]
    assert result["observations"][0]["action"] == "get_weather"  # merged from the specialist, not a supervisor-level tool call


# --- 17-19: travel-only skips System B; istanbul-only skips Travel Search; combined ----


def test_travel_only_request_never_calls_istanbul_expert_end_to_end():
    decider = _QueueDecisionProvider([
        _classification("travel_only"),
        _decision("call_travel_search", {}),
        _decision("search_flights", {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1}),
        _decision("travel_search_complete", {}),
        _decision("synthesize", {}),
    ])
    result = _run_graph(decider, "Just find me a flight.", None)
    actions = [o["action"] for o in result["observations"]]
    assert "call_istanbul_expert" not in actions
    assert result["final_result"]["status"] == "success"


def test_istanbul_only_request_never_delegates_to_travel_search_end_to_end():
    decider = _QueueDecisionProvider([
        _classification("istanbul_local_only", "requires_istanbul_local_grounding"),
        _decision("call_istanbul_expert", {"question": "What should I see?"}),
        _decision("synthesize", {}),
    ])
    result = _run_graph(decider, "What should I see in Sultanahmet?", None)
    execute_entries = [t for t in result["trace"] if t["node"] == "Execute" and t.get("action") == "call_travel_search"]
    assert execute_entries == []
    assert result["final_result"]["status"] == "success"


def test_combined_request_full_sequence_travel_then_istanbul_then_synthesize():
    decider = _QueueDecisionProvider([
        _classification("combined", "requires_both"),
        _decision("call_travel_search", {}),
        _decision("search_flights", {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1}),
        _decision("travel_search_complete", {}),
        _decision("call_istanbul_expert", {"question": "What should I see?"}),
        _decision("synthesize", {}),
    ])
    result = _run_graph(decider, "Plan my whole trip including things to see.", None)
    actions = [o["action"] for o in result["observations"]]
    assert actions == ["search_flights", "call_istanbul_expert"]  # travel evidence strictly before Istanbul Expert
    assert result["final_result"]["status"] == "success"


def test_clarification_required_scope_calls_no_specialist():
    decider = _QueueDecisionProvider([
        _classification("clarification_required", "insufficient_information"),
        _decision("ask_clarification", {"missing_fields": ["destination"], "question": "Which city?"}, "missing_essential_input"),
    ])
    result = _run_graph(decider, "I want to travel.", None)
    execute_entries = [t for t in result["trace"] if t["node"] == "Execute"]
    assert execute_entries == []
    assert result["final_result"]["status"] == "needs_clarification"


def test_out_of_scope_calls_no_specialist():
    decider = _QueueDecisionProvider([
        _classification("out_of_scope", "outside_project_scope"),
        _decision("degrade", {"reason": "outside_project_scope"}, "irrelevant_to_request"),
    ])
    result = _run_graph(decider, "Can you book me a hotel room right now?", None)
    execute_entries = [t for t in result["trace"] if t["node"] == "Execute"]
    assert execute_entries == []
    assert result["final_result"]["status"] == "degraded"


def test_no_early_synthesis_when_required_travel_work_remains():
    """A model that tries to synthesize before Travel Search has even
    been attempted (travel_only scope) is rejected and corrected --
    proves synthesize is NOT always eligible."""
    decider = _QueueDecisionProvider([
        _classification("travel_only"),
        _decision("synthesize", {}, "all_required_evidence_present"),  # ineligible: no evidence gathered yet
        _decision("call_travel_search", {}),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
        _decision("travel_search_complete", {}),
        _decision("synthesize", {}),
    ])
    result = _run_graph(decider, "weather please", None)
    rejected = [t for t in result["trace"] if t.get("status") == "action_ineligible_rejected"]
    assert rejected and rejected[0]["action"] == "synthesize"
    assert result["final_result"]["status"] == "success"


# --- 20-21: terminal partial/degraded/failed states still permit progress -------------


def test_partial_travel_search_result_still_terminal_for_eligibility():
    decider = _QueueDecisionProvider([
        _classification("combined", "requires_both"),
        _decision("call_travel_search", {}),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
        _decision("travel_search_complete", {}),
        _decision("call_istanbul_expert", {"question": "What should I see?"}),
        _decision("synthesize", {}),
    ])
    tools = FakeToolExecutor(scenario_by_action={Action.GET_WEATHER: "unavailable"})
    graph = build_graph(tools, decider, decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="plan my trip", trip_request=None)
    result = start_session(graph, request, f"t-e1s1-{uuid4()}")
    assert "call_istanbul_expert" in [o["action"] for o in result["observations"]]  # proceeded past the degraded travel result
    assert result["final_result"]["status"] == "partial"


# --- fixture mode determinism -----------------------------------------------------------


def test_fixture_mode_classification_and_action_decision_are_both_deterministic():
    """Checkpoint Final Evaluation E.1S.1 §2: fixture mode (no live/paid
    provider call) must classify deterministically too. Reuses the real
    `orchestration.system_a.fixture_decision_provider.
    SupervisorFixtureDecisionProvider` -- two identical runs must produce
    byte-identical results, and the provider must never open a socket."""
    from orchestration.system_a.fixture_decision_provider import (
        SpecialistFixtureDecisionProvider,
        SupervisorFixtureDecisionProvider,
    )

    def _run(thread_id):
        tools = FakeToolExecutor()
        graph = build_graph(tools, SupervisorFixtureDecisionProvider(), SpecialistFixtureDecisionProvider())
        request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="weather please", trip_request=None)
        return start_session(graph, request, thread_id)

    result_a = _run("t-fixture-a")
    result_b = _run("t-fixture-b")
    assert result_a["final_result"] == result_b["final_result"]
    assert result_a["capability_plan"]["classification_succeeded"] is True


# --- prompt-building sanity: eligible set actually drives the prompt ------------------


def test_prompt_only_lists_currently_eligible_actions():
    state = _state(capability_plan=_plan("combined", reason_code="requires_both"))  # travel search not yet terminal
    system, _ = _build_decision_prompt(state)
    allowed_line = system.split("currently-eligible list: ")[1].split(".")[0]
    assert "call_travel_search" in allowed_line
    assert "call_istanbul_expert" not in allowed_line  # combined, but travel search not yet terminal
    assert "synthesize" not in allowed_line


def test_prompt_excludes_istanbul_expert_once_it_is_already_terminal():
    state = _state(
        capability_plan=_plan("istanbul_local_only", reason_code="requires_istanbul_local_grounding"),
        istanbul_expert_attempted_signature="sig-1",
    )
    system, _ = _build_decision_prompt(state)
    allowed_line = system.split("currently-eligible list: ")[1].split(".")[0]
    assert "call_istanbul_expert" not in allowed_line
    assert "synthesize" in allowed_line
