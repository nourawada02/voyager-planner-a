"""Final Evaluation Checkpoint E.1Y -- deterministic ground-truth tests
for the action-eligibility repair (`phase4.graph.
compute_eligible_supervisor_actions` now returning the COMPLETE legal
action set, including `ask_clarification`/`degrade`, instead of those two
being unconditionally unioned in by every caller).

Every expected eligible set below is declared explicitly from the
architecture contract itself (the scope-eligibility table + the input-
guard-validated `trip_request` signal + the existing `_synthesize_node`/
`_degrade_node` output contract) -- never inferred from a scripted future
action-decision queue. Hermetic throughout: no network call.

Generalized re-creations of the two genuine E.1X production defects this
checkpoint fixes (V2-S09: istanbul_local_only wrongly offered
ask_clarification; V2-S11: a terminal failed-but-evidenced istanbul_local_
only turn wrongly offered degrade) use new wording, never E.1X's own case
text."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from phase4.graph import (
    PlannerState,
    _make_decide_node,
    build_graph,
    compute_eligible_supervisor_actions,
    start_session,
)
from phase4.models import Action, PlannerRequest
from phase4.tools import FakeToolExecutor


def _plan(scope: str, signature: str = "sig-1", succeeded: bool = True, reason_code: str = "requires_travel_evidence") -> dict:
    return {"scope": scope, "request_signature": signature, "classification_succeeded": succeeded, "reason_code": reason_code}


def _state(**overrides) -> PlannerState:
    base: PlannerState = {
        "observations": [], "tool_call_count": 0,
        "travel_search_attempted_signature": None, "istanbul_expert_attempted_signature": None,
        "capability_plan": _plan("combined"), "normalized_request": {"user_message": "x", "trip_request": None},
    }
    base.update(overrides)
    return base


TRIP = {
    "session_id": "11111111-1111-1111-1111-111111111111", "trace_id": "22222222-2222-2222-2222-222222222222",
    "origin": "BEY", "destination": "IST", "depart_date": "2027-01-10", "return_date": "2027-01-17",
    "traveler_count": 1, "budget": {"amount_minor_units": 300000, "currency": "TRY"},
    "preferences": {"interests": [], "pace": "relaxed", "language": "en", "mobility_constraints": []},
}


# --- 1-2: travel scope, complete vs. missing essential input -----------------------


def test_travel_only_complete_inputs_allows_specialist_and_prohibits_clarification():
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("travel_only"), normalized_request={"user_message": "x", "trip_request": TRIP},
    ))
    assert eligible == frozenset({Action.CALL_TRAVEL_SEARCH})
    assert Action.ASK_CLARIFICATION not in eligible


def test_travel_only_missing_trip_request_allows_clarification_alongside_specialist():
    """A validated, structured `trip_request` is this project's own
    existing input-guard-checked signal that concrete travel parameters
    exist; its absence does not itself prove essential input is missing
    (a narrow single-capability message, e.g. a plain weather-date
    question, can carry everything one specialist tool needs directly in
    its own text -- ADR 0013's own established `PlannerRequest` design).
    This deterministic, text-blind policy cannot tell those two cases
    apart, so BOTH `call_travel_search` and `ask_clarification` remain
    legally available -- never a hard prohibition of the specialist call
    that would regress every already-passing narrow single-capability
    case (e.g. E.1V/E.1X's own weather-only cases)."""
    eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("travel_only")))
    assert eligible == frozenset({Action.CALL_TRAVEL_SEARCH, Action.ASK_CLARIFICATION})


# --- 3-4: local scope -----------------------------------------------------------


def test_istanbul_local_only_sufficient_context_allows_only_istanbul_expert():
    """Generalized re-creation of E.1X V2-S09: `call_istanbul_expert`'s
    own argument schema never requires structured trip data (only a
    free-text `question`), so once scope is classified istanbul_local_
    only, `ask_clarification` must never be additionally offered --
    exactly the genuine E.1X production defect this checkpoint fixes."""
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("istanbul_local_only", reason_code="requires_istanbul_local_grounding"),
    ))
    assert eligible == frozenset({Action.CALL_ISTANBUL_EXPERT})
    assert Action.ASK_CLARIFICATION not in eligible


def test_material_missing_context_is_resolved_at_classification_not_eligibility():
    """"Local scope with material missing context" is resolved by the
    CLASSIFICATION step producing `clarification_required` in the first
    place -- not by istanbul_local_only eligibility offering
    ask_clarification as a side option. Once classification lands on
    clarification_required, eligibility offers exactly ask_clarification."""
    eligible = compute_eligible_supervisor_actions(_state(
        capability_plan=_plan("clarification_required", reason_code="insufficient_information"),
    ))
    assert eligible == frozenset({Action.ASK_CLARIFICATION})


# --- 5: combined scope transition order ---------------------------------------


def test_combined_scope_transition_order_travel_then_istanbul_then_synthesize():
    plan = _plan("combined", reason_code="requires_both")
    not_started = compute_eligible_supervisor_actions(_state(capability_plan=plan, normalized_request={"user_message": "x", "trip_request": TRIP}))
    assert not_started == frozenset({Action.CALL_TRAVEL_SEARCH})

    travel_done = compute_eligible_supervisor_actions(_state(
        capability_plan=plan, normalized_request={"user_message": "x", "trip_request": TRIP},
        travel_search_attempted_signature="sig-1",
    ))
    assert travel_done == frozenset({Action.CALL_ISTANBUL_EXPERT})

    both_done = compute_eligible_supervisor_actions(_state(
        capability_plan=plan, normalized_request={"user_message": "x", "trip_request": TRIP},
        travel_search_attempted_signature="sig-1", istanbul_expert_attempted_signature="sig-1",
    ))
    assert both_done == frozenset({Action.SYNTHESIZE})


# --- 6-8: synthesis vs. degradation contract -----------------------------------


def _obs(action: str, status: str) -> dict:
    return {"action": action, "status": status}


def _run_graph(decider, user_message: str, trip_request=None):
    graph = build_graph(FakeToolExecutor(), decider, decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message=user_message, trip_request=trip_request)
    return start_session(graph, request, f"t-e1y-{uuid4()}")


class _Scripted:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def generate(self, system: str, user: str) -> str:
        return self.responses.pop(0)


def _cls(scope: str, reason_code: str = "requires_istanbul_local_grounding") -> str:
    return json.dumps({"scope": scope, "reason_code": reason_code, "is_continuation": False})


def _dec(action: str, arguments: dict, reason_code: str = "all_required_evidence_present") -> str:
    return json.dumps({"action": action, "arguments": arguments, "reason_code": reason_code, "explanation": "ok"})


def test_successful_specialist_result_produces_synthesis():
    decider = _Scripted([
        _cls("istanbul_local_only"),
        _dec("call_istanbul_expert", {"question": "What should I see?"}),
        _dec("synthesize", {}),
    ])
    result = _run_graph(decider, "Tell me what to see.")
    assert result["final_result"]["status"] == "success"


def test_failed_specialist_with_usable_evidence_produces_partial_synthesis_not_degrade():
    """Generalized re-creation of E.1X V2-S11: once a terminal (even
    failed) call_istanbul_expert observation exists, `degrade` is no
    longer eligible for istanbul_local_only -- only `synthesize`, which
    `_synthesize_node`'s own existing contract turns into an honest
    "partial" status (never "success", never a routed degrade) whenever
    at least one observation exists but not all succeeded."""
    tools = FakeToolExecutor(scenario_by_action={Action.CALL_ISTANBUL_EXPERT: "unavailable"})
    decider = _Scripted([
        _cls("istanbul_local_only"),
        _dec("call_istanbul_expert", {"question": "What should I see even if the guide is down?"}),
        _dec("synthesize", {}),
    ])
    graph = build_graph(tools, decider, decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="Continue please.", trip_request=None)
    result = start_session(graph, request, f"t-e1y-{uuid4()}")
    assert result["final_result"]["status"] == "partial"
    eligible_at_terminal = compute_eligible_supervisor_actions({
        **_state(), "capability_plan": {**_plan("istanbul_local_only", signature="sig-x"), },
        "istanbul_expert_attempted_signature": "sig-x",
    })
    assert eligible_at_terminal == frozenset({Action.SYNTHESIZE})
    assert Action.DEGRADE not in eligible_at_terminal


def test_unrecoverable_out_of_scope_still_degrades():
    """No observations, no specialist eligible at all -- degrade remains
    the one legal action for a genuinely out-of-scope request."""
    decider = _Scripted([_cls("out_of_scope", "outside_project_scope"), _dec("degrade", {"reason": "outside_project_scope"})])
    result = _run_graph(decider, "Can you book my flight and charge my card?")
    assert result["final_result"]["status"] == "degraded"


# --- 9: clarification_required / out_of_scope scopes ---------------------------


def test_clarification_required_scope_offers_only_ask_clarification():
    eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("clarification_required", reason_code="insufficient_information")))
    assert eligible == frozenset({Action.ASK_CLARIFICATION})


def test_out_of_scope_scope_offers_only_degrade():
    eligible = compute_eligible_supervisor_actions(_state(capability_plan=_plan("out_of_scope", reason_code="outside_project_scope")))
    assert eligible == frozenset({Action.DEGRADE})


# --- 10: exhausted budget -------------------------------------------------------


def test_exhausted_budget_forces_synthesize_regardless_of_scope_or_trip_request():
    for scope in ("travel_only", "istanbul_local_only", "combined"):
        for trip_request in (None, TRIP):
            eligible = compute_eligible_supervisor_actions(_state(
                capability_plan=_plan(scope), tool_call_count=8,
                normalized_request={"user_message": "x", "trip_request": trip_request},
            ))
            assert eligible == frozenset({Action.SYNTHESIZE})


# --- 11: invalid model selection outside the complete eligible set -------------


def test_action_outside_the_complete_eligible_set_is_rejected_and_corrected():
    decider = _Scripted([
        _cls("istanbul_local_only"),
        _dec("degrade", {"reason": "giving up"}),  # ineligible: degrade is not legal for istanbul_local_only pre-terminal
        _dec("call_istanbul_expert", {"question": "What should I see?"}),  # the one bounded correction
        _dec("synthesize", {}),
    ])
    result = _run_graph(decider, "Tell me what to see.")
    rejected = [t for t in result["trace"] if t.get("status") == "action_ineligible_rejected"]
    corrected = [t for t in result["trace"] if t.get("status") == "action_ineligible_corrected"]
    assert rejected and rejected[0]["action"] == "degrade"
    assert corrected and corrected[0]["action"] == "call_istanbul_expert"
    assert result["final_result"]["status"] == "success"


# --- 12: English/Turkish/Arabic equivalence -------------------------------------


@pytest.mark.parametrize("user_message", [
    "Tell me what to see today.", "Bugun ne gormeliyim soyle.", "أخبرني بما يجب أن أراه اليوم.",
])
def test_eligibility_policy_behaves_identically_across_languages(user_message):
    """compute_eligible_supervisor_actions never reads user_message, so
    the same eligibility/correction behavior must fire identically
    regardless of request language."""
    decider = _Scripted([
        _cls("istanbul_local_only"),
        _dec("degrade", {"reason": "giving up"}),
        _dec("call_istanbul_expert", {"question": user_message}),
        _dec("synthesize", {}),
    ])
    result = _run_graph(decider, user_message)
    rejected = [t for t in result["trace"] if t.get("status") == "action_ineligible_rejected"]
    assert rejected and rejected[0]["action"] == "degrade"
    assert result["final_result"]["status"] == "success"


# --- 13: invariant #9, always non-empty -----------------------------------------


def test_eligible_action_set_is_never_empty():
    for scope in ("travel_only", "istanbul_local_only", "combined", "clarification_required", "out_of_scope"):
        for succeeded in (True, False):
            for trip_request in (None, TRIP):
                eligible = compute_eligible_supervisor_actions(_state(
                    capability_plan=_plan(scope, succeeded=succeeded),
                    normalized_request={"user_message": "x", "trip_request": trip_request},
                ))
                assert eligible
    assert compute_eligible_supervisor_actions(_state(capability_plan=None))
