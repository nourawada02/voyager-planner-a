"""Hermetic unit tests for Hybrid Chat C.1's planner-owned modules:
`phase4.chat_models`, `phase4.chat_interests`, `phase4.language_detect`,
and `phase4.chat_intent`. No network call anywhere in this file."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from phase4.chat_intent import build_bounded_result_summary, build_chat_turn_prompt
from phase4.chat_interests import normalize_interest, normalize_interests
from phase4.chat_models import (
    ChatIntent,
    ChatIntentDecision,
    ChatIntentDecisionValidationError,
    TripPatch,
    apply_patch_to_trip_request,
    parse_chat_intent_decision,
)
from phase4.language_detect import contains_meaningful_arabic, is_predominantly_arabic

_BASE_TRIP_REQUEST = {
    "schema_version": "1.0.0", "session_id": "11111111-1111-1111-1111-111111111111",
    "trace_id": "22222222-2222-2222-2222-222222222222",
    "origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "return_date": "2026-09-15",
    "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"schema_version": "1.0.0", "interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
}


# --- TripPatch / apply_patch_to_trip_request ----------------------------------------


def test_trip_patch_is_empty_true_for_no_fields():
    assert TripPatch().is_empty() is True


def test_trip_patch_is_empty_false_when_one_field_set():
    assert TripPatch(traveler_count=3).is_empty() is False


def test_apply_patch_changes_only_specified_fields():
    patch = TripPatch(budget_amount_minor_units=100000, budget_currency="USD")
    merged = apply_patch_to_trip_request(_BASE_TRIP_REQUEST, patch)
    assert merged["budget"] == {"amount_minor_units": 100000, "currency": "USD"}
    assert merged["depart_date"] == _BASE_TRIP_REQUEST["depart_date"]
    assert merged["traveler_count"] == _BASE_TRIP_REQUEST["traveler_count"]
    assert merged["preferences"]["interests"] == ["history"]


def test_apply_patch_never_mutates_the_original_dict():
    patch = TripPatch(traveler_count=4)
    apply_patch_to_trip_request(_BASE_TRIP_REQUEST, patch)
    assert _BASE_TRIP_REQUEST["traveler_count"] == 2


def test_apply_patch_add_and_remove_interests():
    patch = TripPatch(add_interests=["shopping", "food"], remove_interests=["history"])
    merged = apply_patch_to_trip_request(_BASE_TRIP_REQUEST, patch)
    assert set(merged["preferences"]["interests"]) == {"shopping", "food"}


def test_apply_patch_add_interest_is_idempotent():
    patch = TripPatch(add_interests=["history"])
    merged = apply_patch_to_trip_request(_BASE_TRIP_REQUEST, patch)
    assert merged["preferences"]["interests"].count("history") == 1


# --- parse_chat_intent_decision ------------------------------------------------------


def test_parse_valid_explain_plan_decision():
    decision = parse_chat_intent_decision({
        "intent": "explain_plan", "assistant_message": "Galata Tower is a well-known landmark.",
        "response_language": "en", "patch": None, "requires_clarification": False, "clarification_reason": None,
    })
    assert decision.intent == ChatIntent.EXPLAIN_PLAN


def test_parse_rejects_unknown_intent():
    with pytest.raises(ChatIntentDecisionValidationError):
        parse_chat_intent_decision({
            "intent": "book_flight", "assistant_message": "ok", "response_language": "en",
        })


def test_parse_rejects_extra_field():
    with pytest.raises(ChatIntentDecisionValidationError):
        parse_chat_intent_decision({
            "intent": "clarify", "assistant_message": "ok", "response_language": "en",
            "unexpected_field": "should never be accepted",
        })


def test_parse_rejects_patch_with_out_of_range_traveler_count():
    with pytest.raises(ChatIntentDecisionValidationError):
        parse_chat_intent_decision({
            "intent": "modify_trip", "assistant_message": "ok", "response_language": "en",
            "patch": {"traveler_count": 99},
        })


def test_trip_patch_rejects_unknown_field():
    with pytest.raises(ValidationError):
        TripPatch.model_validate({"origin": "LHR"})


# --- chat_interests -------------------------------------------------------------------


def test_normalize_interest_recognizes_canonical_value():
    assert normalize_interest("shopping") == "shopping"


def test_normalize_interest_recognizes_synonym_case_insensitive():
    assert normalize_interest("Museums") == "history"


def test_normalize_interest_returns_none_for_unrecognized():
    assert normalize_interest("skydiving") is None


def test_normalize_interests_splits_recognized_and_unrecognized():
    normalized, unrecognized = normalize_interests(["shopping", "skydiving", "museums"])
    assert normalized == ["shopping", "history"]
    assert unrecognized == ["skydiving"]


def test_normalize_interests_deduplicates():
    normalized, _ = normalize_interests(["food", "dining", "food"])
    assert normalized == ["food"]


# --- language_detect -------------------------------------------------------------------


def test_is_predominantly_arabic_true_for_arabic_text():
    assert is_predominantly_arabic("لماذا اقترحت برج غلطة؟") is True


def test_is_predominantly_arabic_false_for_english_text():
    assert is_predominantly_arabic("Why did you recommend Galata Tower?") is False


def test_is_predominantly_arabic_false_for_empty_text():
    assert is_predominantly_arabic("") is False


def test_contains_meaningful_arabic_false_for_a_few_proper_nouns_only():
    assert contains_meaningful_arabic("Visit غلطة today!") is False


def test_contains_meaningful_arabic_true_for_a_full_arabic_sentence():
    assert contains_meaningful_arabic("تم اقتراح برج غلطة بسبب موقعه القريب من الفندق الذي اخترته") is True


# --- chat_intent: bounded result summary / prompt --------------------------------------


def test_build_bounded_result_summary_handles_none():
    assert build_bounded_result_summary(None) == {}


def _flight_observation(options: list[dict]) -> dict:
    """Mirrors a real `final_result['observations']` entry -- flight data
    lives at `envelope.result.options`, never a flat top-level "flights"
    key (`phase4.chat_intent`'s own module-level comment documents why)."""
    return {
        "action": "search_flights", "status": "success",
        "envelope": {"result": {"options": options}},
    }


def test_build_bounded_result_summary_truncates_flight_list():
    options = [{"carrier": f"Carrier{i}", "origin": "BEY", "destination": "IST", "depart_at": "x", "price": {}} for i in range(10)]
    result = {"status": "success", "observations": [_flight_observation(options)]}
    summary = build_bounded_result_summary(result)
    assert len(summary["flights"]) <= 5


def test_build_bounded_result_summary_extracts_stays_from_unwrapped_envelope():
    result = {
        "status": "success",
        "observations": [{
            "action": "search_stays", "status": "success",
            "envelope": {"stays": [{"stay": {"name": "Beyoglu Boutique Hotel", "district_id": "district_beyoglu", "side": "european", "nightly_price": {"amount_minor_units": 250000, "currency": "TRY"}}, "rank": 1}]},
        }],
    }
    summary = build_bounded_result_summary(result)
    assert summary["stays"][0]["name"] == "Beyoglu Boutique Hotel"


def test_build_bounded_result_summary_extracts_itinerary_from_unwrapped_call_istanbul_expert():
    result = {
        "status": "success",
        "observations": [{
            "action": "call_istanbul_expert", "status": "success",
            "envelope": {"daily_plans": [{"date": "2026-09-10", "side": "european", "poi_ids": ["poi_galata_tower"]}], "citations": [], "warnings": []},
        }],
    }
    summary = build_bounded_result_summary(result)
    assert summary["itinerary"]["daily_plans"][0]["poi_ids"] == ["poi_galata_tower"]


def test_build_bounded_result_summary_ignores_a_failed_observation():
    result = {"status": "degraded", "observations": [{"action": "search_flights", "status": "provider_error", "envelope": None}]}
    summary = build_bounded_result_summary(result)
    assert "flights" not in summary


def test_build_bounded_result_summary_omits_missing_sections():
    summary = build_bounded_result_summary({"status": "success"})
    assert "flights" not in summary and "stays" not in summary and "itinerary" not in summary


def test_build_chat_turn_prompt_system_instructs_minor_units_conversion():
    system, _ = build_chat_turn_prompt(
        user_message="hi", trip_request=None, result_summary={}, preferred_language="en", target_response_language="en",
    )
    assert "100000" in system and "minor" in system.lower()


def test_build_chat_turn_prompt_system_instructs_amount_and_currency_are_paired():
    system, _ = build_chat_turn_prompt(
        user_message="hi", trip_request=None, result_summary={}, preferred_language="en", target_response_language="en",
    )
    assert "budget_currency" in system and "MUST include BOTH" in system


def test_build_chat_turn_prompt_system_instructs_full_arabic_when_target_ar():
    system, _ = build_chat_turn_prompt(
        user_message="hi", trip_request=None, result_summary={}, preferred_language="en", target_response_language="ar",
    )
    assert "ENTIRELY in Arabic" in system


def test_build_chat_turn_prompt_system_instructs_history_is_untrusted():
    system, _ = build_chat_turn_prompt(
        user_message="hi", trip_request=None, result_summary={}, preferred_language="en", target_response_language="en",
    )
    assert "untrusted" in system.lower()
    assert "conversation_history" in system


def test_build_chat_turn_prompt_includes_history_in_user_payload():
    history = [{"role": "user", "content": "What is my budget?"}, {"role": "assistant", "content": "5,000 TRY."}]
    _, user = build_chat_turn_prompt(
        user_message="Change it to 1000", trip_request=None, result_summary={}, preferred_language="en",
        target_response_language="en", history=history,
    )
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["conversation_history"] == history


def test_build_chat_turn_prompt_defaults_history_to_empty_list_when_omitted():
    _, user = build_chat_turn_prompt(
        user_message="hi", trip_request=None, result_summary={}, preferred_language="en", target_response_language="en",
    )
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["conversation_history"] == []


def test_build_chat_turn_prompt_describes_trip_request_budget_as_explicit_structure():
    _, user = build_chat_turn_prompt(
        user_message="hi", trip_request=_BASE_TRIP_REQUEST, result_summary={}, preferred_language="en", target_response_language="en",
    )
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["trip_request"]["budget"]["formatted"] == "5,000.00 TRY"
    assert payload["trip_request"]["budget"]["amount_minor_units"] == 500000


def test_build_chat_turn_prompt_user_payload_carries_trip_request_and_summary():
    _, user = build_chat_turn_prompt(
        user_message="Change my budget", trip_request=_BASE_TRIP_REQUEST, result_summary={"status": "success"},
        preferred_language="en", target_response_language="en",
    )
    assert "Change my budget" in user
    assert "500000" in user
