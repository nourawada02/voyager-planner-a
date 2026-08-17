"""Hermetic tests for the ActionDecision allowlist/argument-schema model
(Checkpoint Phase 4 D.0 §4). No network, no LangGraph execution."""

import pytest
from pydantic import ValidationError

from phase4.models import (
    Action,
    ActionDecision,
    ActionDecisionValidationError,
    fingerprint_action,
    parse_action_decision,
)


def test_valid_decision_parses():
    decision = parse_action_decision({
        "action": "get_weather",
        "arguments": {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"},
        "reason_code": "missing_weather_info",
        "explanation": "Weather is needed for the trip dates.",
    })
    assert decision.action == Action.GET_WEATHER


def test_unknown_action_is_rejected():
    with pytest.raises(ActionDecisionValidationError):
        parse_action_decision({
            "action": "delete_database",
            "arguments": {},
            "reason_code": "missing_weather_info",
        })


def test_booking_is_not_an_allowed_action():
    with pytest.raises(ActionDecisionValidationError):
        parse_action_decision({"action": "book_flight", "arguments": {}, "reason_code": "needs_fair_price"})


def test_arbitrary_url_argument_is_rejected_by_web_search_schema():
    with pytest.raises(ActionDecisionValidationError):
        parse_action_decision({
            "action": "web_search",
            "arguments": {"query": "test", "fetch_url": "https://evil.example.com"},
            "reason_code": "missing_current_info",
        })


def test_code_execution_argument_is_rejected():
    with pytest.raises(ActionDecisionValidationError):
        parse_action_decision({
            "action": "web_search",
            "arguments": {"query": "test", "exec": "import os; os.system('rm -rf /')"},
            "reason_code": "missing_current_info",
        })


def test_extra_top_level_field_is_rejected():
    with pytest.raises(ActionDecisionValidationError):
        parse_action_decision({
            "action": "get_weather",
            "arguments": {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"},
            "reason_code": "missing_weather_info",
            "chain_of_thought": "step 1: ...",
        })


def test_explanation_is_length_bounded():
    with pytest.raises(ValidationError):
        ActionDecision(action=Action.SYNTHESIZE, arguments={}, reason_code="all_required_evidence_present", explanation="x" * 500)


def test_mismatched_arguments_for_action_are_rejected():
    with pytest.raises(ActionDecisionValidationError):
        parse_action_decision({
            "action": "search_flights",
            "arguments": {"query": "flights please"},  # wrong shape entirely
            "reason_code": "missing_flight_info",
        })


def test_fingerprint_is_deterministic():
    a = fingerprint_action(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"})
    b = fingerprint_action(Action.GET_WEATHER, {"date_from": "2026-09-10", "location": "Istanbul", "date_to": "2026-09-10"})
    assert a == b  # key order does not matter


def test_fingerprint_differs_for_different_arguments():
    a = fingerprint_action(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"})
    b = fingerprint_action(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-11", "date_to": "2026-09-11"})
    assert a != b
