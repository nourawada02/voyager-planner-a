"""Final Evaluation Checkpoint E.1R §5 / E.1S.1 -- deterministic
prompt-contract tests for the supervisor routing mechanism in
`phase4/graph.py::_build_decision_prompt`. These are hermetic (no
network call) and validate two things together: (1) the generated
prompt TEXT actually reflects the currently-eligible action set, and
(2) the real compiled graph's own deterministic (scripted) behavior for
each scenario is architecture-consistent -- never a substitute for, and
never confused with, the live-Qwen holdout evaluation these tests
precede.

Checkpoint Final Evaluation E.1S.1 replaced the E.1S heuristic
(`trip_request is None`/a persistent "already attempted" bool) with a
typed, once-per-turn `capability_plan` classification
(`phase4.models.CapabilityScope`) -- every scripted scenario below now
supplies its own classification response as the FIRST queued decision,
exactly as the real supervisor's Decide node requires once per turn.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from phase4.graph import _build_decision_prompt, build_graph, start_session
from phase4.models import Action, PlannerRequest


class _ScriptedDecisionProvider:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self._i = 0

    def generate(self, system: str, user: str) -> str:
        response = self.responses[self._i]
        self._i += 1
        return response


def _decision(action: str, arguments: dict, reason_code: str = "all_required_evidence_present") -> str:
    return json.dumps({"action": action, "arguments": arguments, "reason_code": reason_code, "explanation": "ok"})


def _classification(scope: str, reason_code: str = "requires_travel_evidence") -> str:
    return json.dumps({"scope": scope, "reason_code": reason_code})


def _obs(action: str, status: str) -> dict:
    return {"action": action, "status": status}


TRIP_FULL = {
    "session_id": "11111111-1111-1111-1111-111111111111",
    "trace_id": "22222222-2222-2222-2222-222222222222",
    "origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "return_date": "2026-09-15",
    "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
}


# --- prompt-content assertions --------------------------------------------------------


def test_prompt_states_the_corrected_routing_policy():
    """The prompt lists only the actions currently structurally legal
    (per the seeded `capability_plan`), plus the always-available
    ask_clarification/degrade, and states the one thing eligibility
    cannot determine structurally -- whether Istanbul-local grounding is
    actually needed -- in one short sentence."""
    plan = {"scope": "combined", "request_signature": "sig-1", "classification_succeeded": True, "reason_code": "requires_both"}
    system, _ = _build_decision_prompt({
        "observations": [], "normalized_request": {"user_message": "x", "trip_request": None},
        "tool_call_count": 0, "capability_plan": plan,
    })
    assert "currently-eligible list" in system
    assert "call_istanbul_expert' is listed as eligible, select it only if" in system
    assert "call_travel_search" in system  # combined scope, travel search not yet terminal: eligible
    assert "Follow this routing policy" not in system  # the old E.1R 7-point prose is gone, not merely supplemented


def test_prompt_content_is_identical_regardless_of_request_language():
    """The eligible-action computation is architectural, not
    example-specific -- its presence/content must not depend on the
    user_message's own language (compute_eligible_supervisor_actions
    never reads user_message, only the seeded capability_plan)."""
    plan = {"scope": "combined", "request_signature": "sig-1", "classification_succeeded": True, "reason_code": "requires_both"}
    prompts = []
    for user_message in ("Plan my trip.", "Gezimi planla.", "خطط لرحلتي."):
        system, _ = _build_decision_prompt({
            "observations": [], "normalized_request": {"user_message": user_message, "trip_request": None},
            "tool_call_count": 0, "capability_plan": plan,
        })
        assert "currently-eligible list" in system
        prompts.append(system)
    assert prompts[0] == prompts[1] == prompts[2]  # identical system prompt text across all 3 languages


# --- deterministic (scripted) graph behavior, one case per required scenario ----------


def _run(decisions: list[str], user_message: str, trip_request):
    from phase4.tools import FakeToolExecutor

    decider = _ScriptedDecisionProvider(decisions)
    graph = build_graph(FakeToolExecutor(), decider, decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message=user_message, trip_request=trip_request)
    return start_session(graph, request, f"t-e1r-{uuid4()}")


def test_travel_evidence_missing_routes_to_travel_search():
    result = _run(
        [
            _classification("travel_only"),
            _decision("call_travel_search", {}), _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
            _decision("travel_search_complete", {}), _decision("synthesize", {}),
        ],
        "What's the weather in Istanbul?", None,
    )
    assert result["final_result"]["status"] == "success"
    assert [o["action"] for o in result["observations"]] == ["get_weather"]


def test_sufficient_travel_only_evidence_routes_to_synthesis():
    """Once the travel-only evidence the request actually needs (flight +
    weather) is gathered, synthesize directly -- no Istanbul Expert call,
    since the request was classified travel_only."""
    result = _run(
        [
            _classification("travel_only"),
            _decision("call_travel_search", {}),
            _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
            _decision("search_flights", {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 2}),
            _decision("travel_search_complete", {}),
            _decision("synthesize", {}),
        ],
        "I just need the flight and weather for my trip.", TRIP_FULL,
    )
    assert result["final_result"]["status"] == "success"
    assert [o["action"] for o in result["observations"]] == ["get_weather", "search_flights"]
    assert "call_istanbul_expert" not in [o["action"] for o in result["observations"]]


def test_sufficient_combined_request_evidence_routes_to_istanbul_expert():
    """Once travel evidence is terminal AND the request was classified
    combined (needs Istanbul-local grounding too), call the Istanbul
    Expert before synthesis -- and synthesis is not even eligible until
    it does."""
    result = _run(
        [
            _classification("combined", "requires_both"),
            _decision("call_travel_search", {}),
            _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
            _decision("search_flights", {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 2}),
            _decision("search_stays", {"check_in": "2026-09-10", "check_out": "2026-09-15", "guest_count": 2}),
            _decision("travel_search_complete", {}),
            _decision("call_istanbul_expert", {"question": "What should I see?"}),
            _decision("synthesize", {}),
        ],
        "Plan my full Istanbul trip, including things to see.", TRIP_FULL,
    )
    assert result["final_result"]["status"] == "success"
    assert "call_istanbul_expert" in [o["action"] for o in result["observations"]]


def test_istanbul_only_request_routes_directly_to_istanbul_expert():
    result = _run(
        [
            _classification("istanbul_local_only", "requires_istanbul_local_grounding"),
            _decision("call_istanbul_expert", {"question": "What should I see near Sultanahmet?"}), _decision("synthesize", {}),
        ],
        "What should I see near Sultanahmet?", None,
    )
    assert result["final_result"]["status"] == "success"
    assert [o["action"] for o in result["observations"]] == ["call_istanbul_expert"]


def test_completed_specialist_state_is_never_called_again():
    """A specialist that already returned a terminal result for THIS
    turn is never redundantly re-invoked. Proven directly at the
    decide-node level (matching the other bound-enforcement tests in
    this file): a seeded state where the Istanbul Expert is already
    terminal for the current turn's `capability_plan` signature never
    proposes `call_istanbul_expert` again -- only `synthesize` is
    eligible, matching the single scripted response."""
    from phase4.graph import PlannerState, _compute_request_signature, _make_decide_node

    normalized_request = {"user_message": "Is everything ready?", "trip_request": TRIP_FULL}
    signature = _compute_request_signature({"normalized_request": normalized_request})
    decide = _make_decide_node(
        decision_provider=_ScriptedDecisionProvider([_decision("synthesize", {})]),
        cancellation_check=lambda: False, monotonic_clock=lambda: 0.0,
    )
    state: PlannerState = {
        "started_at_monotonic": 0.0, "tool_call_count": 1, "tool_call_count_by_action": {"call_istanbul_expert": 1},
        "executed_fingerprints": [], "observations": [{"action": "call_istanbul_expert", "status": "success"}],
        "trace": [], "graph_transition_count": 3, "repair_count": 0, "consecutive_duplicate_count": 0,
        "normalized_request": normalized_request,
        "capability_plan": {
            "scope": "istanbul_local_only", "request_signature": signature,
            "classification_succeeded": True, "reason_code": "requires_istanbul_local_grounding",
        },
        "istanbul_expert_attempted_signature": signature,
    }
    updates = decide(state)
    assert updates["pending_action"]["action"] == "synthesize"


def test_exhausted_budget_state_forces_synthesize_never_invalid_redelegation():
    """A time-bounded/exhausted state must not be met with a fresh
    delegation attempt -- proven directly at the decide-node level,
    matching the existing D.3 bound-enforcement tests."""
    from phase4.graph import PlannerState, _compute_request_signature, _make_decide_node

    decide = _make_decide_node(
        decision_provider=_ScriptedDecisionProvider([_decision("call_travel_search", {})]),
        cancellation_check=lambda: False, monotonic_clock=lambda: 0.0,
    )
    normalized_request = {"user_message": "x", "trip_request": None}
    signature = _compute_request_signature({"normalized_request": normalized_request})
    state: PlannerState = {
        "started_at_monotonic": 0.0, "tool_call_count": 8, "tool_call_count_by_action": {},
        "executed_fingerprints": [], "observations": [], "trace": [], "graph_transition_count": 5,
        "repair_count": 0, "consecutive_duplicate_count": 0, "normalized_request": normalized_request,
        "capability_plan": {
            "scope": "travel_only", "request_signature": signature,
            "classification_succeeded": True, "reason_code": "requires_travel_evidence",
        },
    }
    updates = decide(state)
    assert updates["pending_action"]["action"] == "synthesize"
    assert updates["pending_action"]["reason_code"] == "bound_reached"


def test_failed_partial_evidence_state_recovers_architecture_consistently():
    """A genuinely degraded/failed Travel Search result (get_weather
    really returns "unavailable" through the specialist, not merely
    claimed in free text) still counts as terminal for this turn -- the
    supervisor proceeds to the Istanbul Expert rather than
    re-delegating."""
    from phase4.tools import FakeToolExecutor

    decider = _ScriptedDecisionProvider([
        _classification("combined", "requires_both"),
        _decision("call_travel_search", {}),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}),
        _decision("travel_search_complete", {}),
        _decision("call_istanbul_expert", {"question": "what to see?"}),
        _decision("synthesize", {}),
    ])
    graph = build_graph(FakeToolExecutor(scenario_by_action={Action.GET_WEATHER: "unavailable"}), decider, decider)
    request = PlannerRequest(
        session_id=uuid4(), trace_id=uuid4(), user_message="Continue even though the weather check failed.", trip_request=TRIP_FULL,
    )
    result = start_session(graph, request, f"t-e1s-{uuid4()}")

    weather_obs = [o for o in result["observations"] if o["action"] == "get_weather"]
    assert weather_obs and weather_obs[0]["status"] == "unavailable"  # genuinely failed, not merely claimed
    assert "call_istanbul_expert" in [o["action"] for o in result["observations"]]  # proceeded anyway
    assert result["final_result"]["status"] == "partial"  # honest partial result, not a fabricated "success"


@pytest.mark.parametrize("user_message", [
    "What should I see near Sultanahmet?",
    "Sultanahmet cevresinde ne gormeliyim?",
    "ماذا يجب أن أرى بالقرب من السلطان أحمد؟",
])
def test_istanbul_only_request_multilingual_equivalents(user_message):
    result = _run(
        [
            _classification("istanbul_local_only", "requires_istanbul_local_grounding"),
            _decision("call_istanbul_expert", {"question": user_message}), _decision("synthesize", {}),
        ],
        user_message, None,
    )
    assert result["final_result"]["status"] == "success"
    assert [o["action"] for o in result["observations"]] == ["call_istanbul_expert"]
