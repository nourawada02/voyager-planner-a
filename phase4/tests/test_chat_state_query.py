"""Hermetic unit tests for `phase4.chat_state_query` -- the deterministic
authoritative-state-answer fast path (never an LLM call)."""

from __future__ import annotations

from phase4.chat_state_query import StateQueryField, answer_state_query, detect_state_query

_TRIP_REQUEST = {
    "depart_date": "2026-09-10", "return_date": "2026-09-15", "traveler_count": 2,
    "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"interests": ["history", "food"], "pace": "moderate", "language": "en"},
}


# --- detection: English --------------------------------------------------------------


def test_detects_budget_question_english():
    assert detect_state_query("What is my budget?") == StateQueryField.BUDGET


def test_detects_budget_question_variant_english():
    assert detect_state_query("How much is my budget?") == StateQueryField.BUDGET


def test_detects_dates_question_english():
    assert detect_state_query("What are my travel dates?") == StateQueryField.DATES


def test_detects_travelers_question_english():
    assert detect_state_query("How many travelers are on this trip?") == StateQueryField.TRAVELERS


def test_detects_pace_question_english():
    assert detect_state_query("What is my pace?") == StateQueryField.PACE


def test_detects_interests_question_english():
    assert detect_state_query("What are my interests?") == StateQueryField.INTERESTS


def test_detects_language_question_english():
    assert detect_state_query("What is my preferred language?") == StateQueryField.LANGUAGE


# --- detection: Arabic -----------------------------------------------------------------


def test_detects_budget_question_arabic():
    assert detect_state_query("ما هي ميزانيتي؟") == StateQueryField.BUDGET


def test_detects_dates_question_arabic():
    assert detect_state_query("متى سأسافر؟") == StateQueryField.DATES


# --- detection: must never fire on a modification request ------------------------------


def test_change_budget_request_is_not_detected_as_a_state_query():
    assert detect_state_query("Change my budget to 1000 USD") is None


def test_isolated_pronoun_request_is_not_detected():
    assert detect_state_query("Change it to 1000") is None


def test_unrelated_message_is_not_detected():
    assert detect_state_query("Why did you recommend Galata Tower?") is None


# --- answers -----------------------------------------------------------------------------


def test_answer_budget_english_uses_money_formatting():
    answer = answer_state_query(StateQueryField.BUDGET, _TRIP_REQUEST, "en")
    assert "5,000.00 TRY" in answer


def test_answer_budget_arabic_uses_money_formatting():
    answer = answer_state_query(StateQueryField.BUDGET, _TRIP_REQUEST, "ar")
    assert "5,000.00 TRY" in answer
    assert "ميزانيتك" in answer


def test_answer_dates_english():
    answer = answer_state_query(StateQueryField.DATES, _TRIP_REQUEST, "en")
    assert "2026-09-10" in answer and "2026-09-15" in answer


def test_answer_travelers_english():
    answer = answer_state_query(StateQueryField.TRAVELERS, _TRIP_REQUEST, "en")
    assert "2" in answer


def test_answer_pace_english():
    answer = answer_state_query(StateQueryField.PACE, _TRIP_REQUEST, "en")
    assert "moderate" in answer


def test_answer_interests_english():
    answer = answer_state_query(StateQueryField.INTERESTS, _TRIP_REQUEST, "en")
    assert "history" in answer and "food" in answer


def test_answer_language_english():
    answer = answer_state_query(StateQueryField.LANGUAGE, _TRIP_REQUEST, "en")
    assert "English" in answer


def test_answer_returns_none_without_a_trip_request():
    assert answer_state_query(StateQueryField.BUDGET, None, "en") is None
