"""Hermetic regression tests for the Decide-node prompt repair
(Checkpoint Phase 4 D.0): the first live Qwen gate failed because the
prompt told Qwen the outer `ActionDecision` shape but never the exact
per-action `arguments` fields/constraints, so Qwen returned
understandable-but-invalid aliases (city names instead of IATA codes,
`date`/`passengers`/`trip_type` instead of the real field names). The
fix generates the prompt's action-contract section directly from
`ACTION_ARGUMENT_MODELS` -- the same canonical registry
`parse_action_decision` validates against -- rather than a second,
hand-maintained schema. No network call anywhere in this file.
"""

from __future__ import annotations

import json

import pytest

from phase4.graph import _EXAMPLE_DECISION, _action_argument_contract, _build_decision_prompt
from phase4.models import (
    ACTION_ARGUMENT_MODELS,
    Action,
    ActionDecisionValidationError,
    parse_action_decision,
)

# The literal, previously-observed malformed Qwen response for the
# BEY/IST live-gate scenario, reconstructed from the sanitized validation
# error the prior live-gate attempt reported -- contains no secret, no
# prompt text, nothing but the (wrong) argument shape itself.
KNOWN_MALFORMED_LIVE_GATE_RESPONSE = {
    "action": "search_flights",
    "arguments": {
        "origin": "Beirut",
        "destination": "Istanbul",
        "date": "2026-09-10",
        "passengers": 1,
        "trip_type": "one-way",
    },
    "reason_code": "missing_flight_info",
    "explanation": "The user wants a one-way flight from Beirut to Istanbul.",
}


def _minimal_state() -> dict:
    return {"observations": [], "normalized_request": {"user_message": "Find me a flight."}}


def test_generated_prompt_contains_every_canonical_action():
    system, _ = _build_decision_prompt(_minimal_state())
    for action in Action:
        assert action.value in system


def test_search_flights_contract_exposes_exact_canonical_fields():
    contract = _action_argument_contract()
    properties = set(contract["search_flights"]["properties"].keys())
    assert properties == {"origin", "destination", "depart_date", "passenger_count", "cabin_class"}
    assert set(contract["search_flights"]["required"]) == {"origin", "destination", "depart_date", "passenger_count"}
    assert "cabin_class" not in contract["search_flights"]["required"]  # optional


def test_iata_regex_is_represented_in_the_contract_and_prompt():
    contract = _action_argument_contract()
    assert contract["search_flights"]["properties"]["origin"]["pattern"] == r"^[A-Z]{3}$"
    assert contract["search_flights"]["properties"]["destination"]["pattern"] == r"^[A-Z]{3}$"
    system, _ = _build_decision_prompt(_minimal_state())
    assert r"^[A-Z]{3}$" in system
    assert "IATA" in system


def test_extra_arguments_remain_forbidden_in_every_action_contract():
    contract = _action_argument_contract()
    for action_name, schema in contract.items():
        assert schema.get("additionalProperties") is False, action_name


def test_prompt_uses_deterministic_ordering():
    state = _minimal_state()
    system_a, _ = _build_decision_prompt(state)
    system_b, _ = _build_decision_prompt(state)
    assert system_a == system_b


def test_contract_is_generated_from_canonical_models_not_duplicated():
    """Proves the prompt's schema section is byte-for-byte what
    ACTION_ARGUMENT_MODELS itself produces -- not a second,
    hand-maintained dictionary that could silently drift from the real
    validation contract."""
    contract = _action_argument_contract()
    for action, model in ACTION_ARGUMENT_MODELS.items():
        expected = model.model_json_schema()
        expected.pop("title", None)
        for value in expected.get("properties", {}).values():
            value.pop("title", None)
        assert contract[action.value] == expected


def test_known_malformed_live_gate_response_is_still_rejected():
    """The contract must not be loosened to accommodate the earlier
    malformed response -- Qwen must conform to it, not the reverse."""
    with pytest.raises(ActionDecisionValidationError):
        parse_action_decision(KNOWN_MALFORMED_LIVE_GATE_RESPONSE)


def test_city_name_aliases_are_still_rejected_individually():
    for bad_field, bad_args in (
        ("origin_city_name", {"origin": "Beirut", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1}),
        ("destination_city_name", {"origin": "BEY", "destination": "Istanbul", "depart_date": "2026-09-10", "passenger_count": 1}),
        ("alias_date_field", {"origin": "BEY", "destination": "IST", "date": "2026-09-10", "passenger_count": 1}),
        ("alias_passengers_field", {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passengers": 1}),
    ):
        with pytest.raises(ActionDecisionValidationError):
            parse_action_decision({"action": "search_flights", "arguments": bad_args, "reason_code": "missing_flight_info"})


def test_example_decision_is_itself_schema_valid():
    decision = parse_action_decision(_EXAMPLE_DECISION)
    assert decision.action == Action.SEARCH_FLIGHTS
    assert decision.arguments["origin"] == "BEY"
    assert decision.arguments["destination"] == "IST"
    assert decision.arguments["depart_date"] == "2026-09-10"
    assert decision.arguments["passenger_count"] == 1
    assert decision.arguments["cabin_class"] == "economy"


def test_valid_bey_ist_response_is_accepted_and_selects_search_flights():
    raw = {
        "action": "search_flights",
        "arguments": {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1},
        "reason_code": "missing_flight_info",
        "explanation": "Flight information is required.",
    }
    decision = parse_action_decision(raw)
    assert decision.action == Action.SEARCH_FLIGHTS
    assert decision.arguments["origin"] == "BEY"
    assert decision.arguments["destination"] == "IST"


def test_no_prompt_or_secret_leaks_into_graph_state_trace_or_checkpoint():
    from uuid import uuid4

    from phase4.graph import build_graph, start_session
    from phase4.models import PlannerRequest
    from phase4.tools import FakeToolExecutor

    class _Scripted:
        def __init__(self, responses):
            self.responses = responses
            self.call_log = []
            self._i = 0

        def generate(self, system, user):
            self.call_log.append((system, user))
            response = self.responses[self._i]
            self._i += 1
            return response

    decider = _Scripted([
        json.dumps({
            "action": "search_flights",
            "arguments": {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1},
            "reason_code": "missing_flight_info",
            "explanation": "Flight information is required.",
        }),
        json.dumps({"action": "synthesize", "arguments": {}, "reason_code": "all_required_evidence_present", "explanation": "Done."}),
    ])
    graph = build_graph(FakeToolExecutor(), decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="Find me a flight to Istanbul.")
    result = start_session(graph, request, "t-prompt-privacy")

    serialized_state = json.dumps(result, default=str)
    # The full generated system prompt (which legitimately contains the
    # word "search_flights" and IATA patterns) must never appear verbatim
    # inside persisted state/trace/output -- only the decision provider's
    # own local call_log (never persisted) saw it.
    system_prompt_used, _ = decider.call_log[0]
    assert system_prompt_used not in serialized_state
    assert "chain_of_thought" not in serialized_state.lower()
    assert "chain-of-thought" not in serialized_state.lower()
    assert "authorization" not in serialized_state.lower()
    assert "bearer" not in serialized_state.lower()
