"""Final Evaluation Checkpoint E.1W -- deterministic ground-truth tests
for the capability-classification CONTEXT and continuity repair
(`phase4.graph._build_capability_classification_prompt`,
`phase4.graph._classify_capability`, and the new
`last_successful_capability_scope` cross-turn state field).

These tests are independent of, and never derive their expected scope
from, any scripted future action-decision queue -- every test below
either calls `_classify_capability`/`_build_capability_classification_prompt`
directly and asserts on the returned `CapabilityPlan` (`scope`,
`scope_source`, `classification_succeeded`, `reason_code`) with an
explicitly hard-coded expected value, or (for the two multi-turn/full-
graph tests) asserts on `capability_plan` read back from the real graph's
own result state -- never on which action happened to be selected
afterward as a proxy for scope correctness.

Every case here is a generalized/paraphrased re-creation of one of the
ten E.1V scope-misclassification failure patterns (V-S02/03/04/06/08/10/
12/14/16/18) with new wording, never copied or lightly reworded from
`evaluation/e1v_cases.json` itself -- E.1V's own frozen case text and
labels are never touched by this file.

Hermetic throughout: no network call, `_ScriptedClassificationProvider`
below never calls a real model."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from phase4.graph import (
    PlannerState,
    _build_capability_classification_prompt,
    _classify_capability,
    _compute_request_signature,
    _make_decide_node,
    build_graph,
    resume_session,
    start_session,
)
from phase4.models import Action, CapabilityReasonCode, CapabilityScope, PlannerRequest, ScopeSource
from phase4.qwen_client import QwenTransportError
from phase4.tools import FakeToolExecutor


class _ScriptedClassificationProvider:
    """A scripted fake `DecisionProvider` -- `items` is a list where each
    entry is either a raw JSON string to return or an exception instance
    to raise, consumed in order. Records every `(system, user)` call pair
    so tests can assert on exactly what context reached the model."""

    def __init__(self, items: list):
        self.items = list(items)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _classification_response(scope: str, reason_code: str, is_continuation: bool = False) -> str:
    return json.dumps({"scope": scope, "reason_code": reason_code, "is_continuation": is_continuation})


def _obs(action: str, status: str) -> dict:
    return {"action": action, "status": status}


def _state(user_message: str, trip_request=None, observations=None, last_scope=None) -> PlannerState:
    return {
        "normalized_request": {"user_message": user_message, "trip_request": trip_request},
        "observations": observations or [],
        "last_successful_capability_scope": last_scope,
    }


def _classify(user_message, provider_items, trip_request=None, observations=None, last_scope=None):
    state = _state(user_message, trip_request, observations, last_scope)
    signature = _compute_request_signature(state)
    provider = _ScriptedClassificationProvider(provider_items)
    trace: list = []
    plan = _classify_capability(provider, state, signature, trace)
    return plan, trace, provider


TRIP_EN = {
    "origin": "BEY", "destination": "IST", "depart_date": "2026-11-03", "return_date": "2026-11-09",
    "traveler_count": 2, "budget": {"amount_minor_units": 400000, "currency": "TRY"},
    "preferences": {"interests": ["food"], "pace": "relaxed", "language": "en", "mobility_constraints": []},
}


# --- 1-3: initial (no prior scope) requests classify fresh, per scope --------------


def test_initial_travel_only_request_classifies_fresh():
    plan, _, _ = _classify(
        "Find me a one-way flight to Istanbul next month.",
        [_classification_response("travel_only", "requires_travel_evidence")],
    )
    assert plan["scope"] == "travel_only"
    assert plan["scope_source"] == "classified"
    assert plan["classification_succeeded"] is True


def test_initial_istanbul_local_only_request_classifies_fresh():
    plan, _, _ = _classify(
        "What neighborhoods should I explore on foot?",
        [_classification_response("istanbul_local_only", "requires_istanbul_local_grounding")],
    )
    assert plan["scope"] == "istanbul_local_only"
    assert plan["scope_source"] == "classified"


def test_initial_combined_request_classifies_fresh():
    plan, _, _ = _classify(
        "Sort out my whole Istanbul trip, travel and sightseeing both.",
        [_classification_response("combined", "requires_both")],
        trip_request=TRIP_EN,
    )
    assert plan["scope"] == "combined"
    assert plan["scope_source"] == "classified"


# --- 4-6: short multilingual elliptical follow-ups inherit the PREVIOUS scope ------
# (generalized re-creations of the V-S02/V-S04/V-S08/V-S10 failure pattern: a
# short "wrap it up"/"continue" message lost its scope because the old
# classifier had no previous-scope context at all)


@pytest.mark.parametrize("user_message,previous_scope", [
    ("Good, that's everything -- move on.", "travel_only"),
    ("Tamam, devam edebiliriz.", "istanbul_local_only"),
    ("جيد، يمكننا المتابعة الآن.", "combined"),
])
def test_elliptical_followup_inherits_previous_scope_ignoring_models_own_scope_guess(user_message, previous_scope):
    """The scripted provider deliberately returns a NONSENSE scope/reason
    alongside is_continuation=true, proving the FINAL scope comes from
    code-side inheritance of `previous_scope`, never from re-trusting the
    model's own (here, deliberately wrong) scope field on a continuation
    turn."""
    plan, _, provider = _classify(
        user_message,
        [_classification_response("out_of_scope", "outside_project_scope", is_continuation=True)],
        last_scope=previous_scope,
    )
    assert plan["scope"] == previous_scope
    assert plan["scope_source"] == "inherited"
    assert plan["classification_succeeded"] is True
    assert len(provider.calls) == 1  # first attempt succeeded, no repair needed


# --- 7: an explicit follow-up scope change overrides inheritance -------------------
# (generalized re-creation of V-S06: a stale prior scope must never block an
# explicit, genuinely different new request)


def test_explicit_task_change_overrides_previous_scope():
    plan, _, _ = _classify(
        "Actually, forget that -- I need a completely different flight on a new date.",
        [_classification_response("travel_only", "requires_travel_evidence", is_continuation=False)],
        last_scope="istanbul_local_only",
    )
    assert plan["scope"] == "travel_only"
    assert plan["scope_source"] == "explicit"


# --- 8: structured trip_request facts reach the classifier's own input ------------


def test_structured_trip_request_fields_reach_the_classification_prompt():
    system, user = _build_capability_classification_prompt(_state(
        "Let's get everything sorted for the trip.", trip_request=TRIP_EN,
    ))
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["trip_request"] == TRIP_EN
    assert "flight/stay" in system or "flight/stay/fair-price" in system  # policy text present


# --- 9: existing terminal observations reach the classifier's own input -----------
# (generalized re-creation of V-S14/V-S16: travel evidence already gathered was
# invisible to the old classifier, which never saw `observations` at all)


def test_prior_observations_reach_the_classification_prompt_and_drive_combined_scope():
    observations = [_obs("search_flights", "success"), _obs("search_stays", "success")]
    system, user = _build_capability_classification_prompt(_state(
        "Flights and the hotel are handled -- now let's plan the daily itinerary.",
        trip_request=TRIP_EN, observations=observations,
    ))
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["evidence_collected_so_far"] == [
        {"action": "search_flights", "status": "success"}, {"action": "search_stays", "status": "success"},
    ]

    plan, _, _ = _classify(
        "Flights and the hotel are handled -- now let's plan the daily itinerary.",
        [_classification_response("combined", "requires_both")],
        trip_request=TRIP_EN, observations=observations,
    )
    assert plan["scope"] == "combined"


# --- 10: Istanbul-as-destination alone never implies local-expert work ------------


def test_system_prompt_states_istanbul_destination_does_not_imply_local_work():
    system, _ = _build_capability_classification_prompt(_state("Book my flight to Istanbul.", trip_request=TRIP_EN))
    assert "merely because Istanbul is named as the destination" in system


# --- 11: a structured trip object alone never implies Travel Search ---------------


def test_system_prompt_states_trip_request_alone_does_not_imply_travel_search():
    system, _ = _build_capability_classification_prompt(_state("Just tell me what to see.", trip_request=TRIP_EN))
    assert "merely because a structured trip_request is" in system


def test_structured_trip_object_present_but_user_wants_local_guidance_only_classifies_istanbul_local_only():
    plan, _, _ = _classify(
        "I already booked everything -- just tell me what to see and do.",
        [_classification_response("istanbul_local_only", "requires_istanbul_local_grounding")],
        trip_request=TRIP_EN,
    )
    assert plan["scope"] == "istanbul_local_only"


# --- 12: genuinely ambiguous request requires clarification -----------------------


def test_genuinely_ambiguous_request_requires_clarification():
    plan, _, _ = _classify(
        "Can you help me figure something out?",
        [_classification_response("clarification_required", "insufficient_information")],
    )
    assert plan["scope"] == "clarification_required"
    assert plan["scope_source"] == "classified"


# --- 13: a clearly unrelated request degrades --------------------------------------


def test_unrelated_request_is_out_of_scope():
    plan, _, _ = _classify(
        "What's the exchange rate between the dollar and the euro today?",
        [_classification_response("out_of_scope", "outside_project_scope")],
    )
    assert plan["scope"] == "out_of_scope"


# --- 14: no previous scope is explicitly signaled as null/false, not omitted ------


def test_no_prior_scope_is_explicitly_signaled_as_null_and_not_resumed():
    _, user = _build_capability_classification_prompt(_state("I've gathered everything, summarize it now."))
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["previous_successful_scope"] is None
    assert payload["is_resumed_session"] is False


# --- 15-16: classification failure, with and without a prior plan -----------------


def test_classification_failure_with_prior_plan_safely_inherits_it():
    plan, trace, provider = _classify(
        "???", [
            QwenTransportError("Qwen response was not valid JSON", transient=False),
            QwenTransportError("Qwen response was not valid JSON", transient=False),
        ],
        last_scope="travel_only",
    )
    assert plan["classification_succeeded"] is True
    assert plan["scope"] == "travel_only"
    assert plan["scope_source"] == "fallback"
    assert plan["reason_code"] == "fallback_inherited_previous_scope"
    assert any(t.get("status") == "capability_classification_fallback_inherited" for t in trace)


def test_classification_failure_with_no_prior_plan_degrades_honestly():
    plan, trace, _ = _classify(
        "???", [
            QwenTransportError("Qwen response was not valid JSON", transient=False),
            QwenTransportError("Qwen response was not valid JSON", transient=False),
        ],
    )
    assert plan["classification_succeeded"] is False
    assert plan["scope"] == "clarification_required"
    assert plan["scope_source"] == "fallback"
    assert plan["reason_code"] == "classification_failed"
    assert any(t.get("status") == "capability_classification_failed" for t in trace)


# --- 17: no production prompt or raw model output is ever persisted ---------------


def test_no_raw_prompt_or_model_output_persisted_in_plan_or_trace():
    plan, trace, _ = _classify(
        "Continue please.",
        [_classification_response("out_of_scope", "outside_project_scope", is_continuation=True)],
        last_scope="travel_only",
    )
    assert set(plan.keys()) == {"scope", "request_signature", "classification_succeeded", "reason_code", "scope_source"}
    for entry in trace:
        assert "raw_text" not in entry and "prompt" not in entry and "system" not in entry and "user" not in entry


# --- 18-19: full-graph tests -- new-turn inheritance + same-turn prohibition ------
# (independent of the scripted-action-queue "peeking" pattern: expected scope is
# asserted directly from the returned capability_plan, never inferred from which
# action was picked)


def _decision(action: str, arguments: dict, reason_code: str = "all_required_evidence_present") -> str:
    return json.dumps({"action": action, "arguments": arguments, "reason_code": reason_code, "explanation": "ok"})


def test_new_turn_inherits_scope_and_reopens_specialist_eligibility_with_a_new_signature():
    decider = _ScriptedClassificationProvider([
        _classification_response("travel_only", "requires_travel_evidence"),
        _decision("call_travel_search", {}),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-11-03", "date_to": "2026-11-03"}),
        _decision("travel_search_complete", {}),
        _decision("synthesize", {}),
        # turn 2 (a genuinely new, different follow-up signature):
        _classification_response("out_of_scope", "outside_project_scope", is_continuation=True),
        _decision("call_travel_search", {}),
        _decision("search_flights", {"origin": "BEY", "destination": "IST", "depart_date": "2026-11-10", "passenger_count": 1}),
        _decision("travel_search_complete", {}),
        _decision("synthesize", {}),
    ])
    graph = build_graph(FakeToolExecutor(), decider, decider)
    thread_id = f"t-e1w-{uuid4()}"
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="Find me a flight to Istanbul.", trip_request=None)
    turn1 = start_session(graph, request, thread_id)
    assert turn1["capability_plan"]["scope"] == "travel_only"
    assert turn1["capability_plan"]["scope_source"] == "classified"
    assert turn1["last_successful_capability_scope"] == "travel_only"

    turn2 = resume_session(graph, thread_id, user_message="Now find a different one, new dates entirely.")
    assert turn2["capability_plan"]["scope"] == "travel_only"
    assert turn2["capability_plan"]["scope_source"] == "inherited"  # ignores the bogus out_of_scope the model returned
    actions = [o["action"] for o in turn2["observations"]]
    assert "search_flights" in actions  # specialist eligibility genuinely reopened for the new turn/signature


def test_same_turn_repeated_specialist_call_remains_prohibited_after_inherited_scope():
    """The inheritance mechanism must never bypass the pre-existing
    same-turn duplicate-specialist prohibition. Turn 1 makes one genuine
    call_istanbul_expert call (istanbul_local_only, classified fresh).
    Turn 2 is a genuine new signature whose scope is INHERITED (not
    freshly classified); its own first call_istanbul_expert call is
    legitimately eligible (istanbul_expert_attempted_signature is turn
    1's signature, not turn 2's) and brings the whole-session count to
    the shared MAX_CALLS_PER_TOOL=2 cap. A second, same-turn attempt right
    after is still never executed -- forced to synthesize by the
    pre-existing per-tool budget check, exactly as the already-existing
    mechanism guarantees regardless of scope_source. (Checkpoint E.1Y:
    `ask_clarification` is no longer eligible for istanbul_local_only at
    all once call_istanbul_expert itself is eligible -- unlike this
    test's own earlier draft, turn 1 must make a real specialist call to
    reach a valid state.)"""
    decider = _ScriptedClassificationProvider([
        _classification_response("istanbul_local_only", "requires_istanbul_local_grounding"),
        _decision("call_istanbul_expert", {"question": "What should I see first?"}),
        _decision("synthesize", {}),
        # turn 2: a new signature, scope inherited (model's own bogus scope/reason ignored)
        _classification_response("out_of_scope", "outside_project_scope", is_continuation=True),
        _decision("call_istanbul_expert", {"question": "What about tomorrow?"}),
        _decision("call_istanbul_expert", {"question": "And the day after?"}),  # forced to synthesize: per-tool cap reached
    ])
    graph = build_graph(FakeToolExecutor(), decider, decider)
    thread_id = f"t-e1w-{uuid4()}"
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="Show me around.", trip_request=None)
    turn1 = start_session(graph, request, thread_id)
    assert turn1["capability_plan"]["scope_source"] == "classified"
    assert turn1["tool_call_count_by_action"].get("call_istanbul_expert", 0) == 1

    turn2 = resume_session(graph, thread_id, user_message="Keep going with more ideas.")
    assert turn2["capability_plan"]["scope"] == "istanbul_local_only"
    assert turn2["capability_plan"]["scope_source"] == "inherited"
    actions = [o["action"] for o in turn2["observations"]]
    assert actions.count("call_istanbul_expert") == 2  # turn 1's + turn 2's ONE genuine call -- the duplicate never executed
    assert turn2["tool_call_count_by_action"]["call_istanbul_expert"] == 2
    bound_rejected = [
        t for t in turn2["trace"]
        if t.get("node") == "Decide" and t.get("action") == "synthesize" and t.get("reason_code") == "bound_reached"
    ]
    assert bound_rejected  # the duplicate attempt was forced to synthesize by the shared per-tool cap


# --- 20-21: generalized re-creations of the remaining E.1V failure patterns -------
# (V-S12: fresh classification, no evidence, no prior scope, tempting wording;
#  V-S18: fresh combined classification with a full structured trip_request)


def test_v_s12_pattern_no_prior_scope_no_evidence_still_classifies_fresh_not_inherited():
    plan, _, _ = _classify(
        "I think that's everything -- can you wrap it up for me?",
        [_classification_response("istanbul_local_only", "requires_istanbul_local_grounding")],
    )
    assert plan["scope"] == "istanbul_local_only"
    assert plan["scope_source"] == "classified"  # no previous scope existed -- never "inherited"


def test_v_s18_pattern_fresh_combined_request_with_full_trip_request():
    plan, _, _ = _classify(
        "Put together my whole Istanbul trip please.",
        [_classification_response("combined", "requires_both")],
        trip_request=TRIP_EN,
    )
    assert plan["scope"] == "combined"
    assert plan["scope_source"] == "classified"
