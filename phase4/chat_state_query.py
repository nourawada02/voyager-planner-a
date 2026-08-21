"""Deterministic authoritative-state answers (Hybrid Chat C.1 persistent-
history/grounding correction, §7). A small, conservative, explicit
regex-pattern classifier -- never an LLM call -- matching this project's
existing `phase4.guards` "deterministic, regex/allowlist-based" precedent
exactly. Fires ONLY on a confident match; anything not confidently
recognized here falls through to the bounded Qwen chat-decision path
(never a forced/guessed classification).

These questions never need a model call at all: the answer is already
sitting, unambiguous, in the authoritative stored `TripRequest` -- asking
Qwen would spend a real provider call (and risk a hallucinated or
misformatted answer) for a fact this code can just read.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from phase4.chat_money import describe_money

_PACE_LABELS_EN = {"relaxed": "relaxed", "moderate": "moderate", "packed": "packed"}
_PACE_LABELS_AR = {"relaxed": "مريحة", "moderate": "متوسطة", "packed": "مكثفة"}
_LANGUAGE_LABELS_EN = {"en": "English", "tr": "Turkish", "ar": "Arabic"}
_LANGUAGE_LABELS_AR = {"en": "الإنجليزية", "tr": "التركية", "ar": "العربية"}


class StateQueryField(str, Enum):
    BUDGET = "budget"
    DATES = "dates"
    TRAVELERS = "travelers"
    PACE = "pace"
    INTERESTS = "interests"
    LANGUAGE = "language"


# Deliberately narrow, high-precision patterns -- a false NEGATIVE (falling
# through to Qwen for a state question) is always safe; a false POSITIVE
# (answering the wrong field) is not, so every pattern here is specific.
_EN_PATTERNS: dict[StateQueryField, tuple[str, ...]] = {
    StateQueryField.BUDGET: (
        r"\bwhat'?s?\s+(is\s+)?my\s+budget\b", r"\bhow much\s+(is|was)\s+my\s+budget\b",
        r"\bmy\s+current\s+budget\b", r"\bwhat\s+budget\s+did\s+i\s+set\b",
    ),
    StateQueryField.DATES: (
        r"\bwhat\s+(are\s+)?my\s+(travel\s+)?dates\b", r"\bwhen\s+am\s+i\s+travel(l)?ing\b",
        r"\bmy\s+departure\s+date\b", r"\bmy\s+return\s+date\b", r"\bwhen\s+do\s+i\s+(depart|leave|return)\b",
    ),
    StateQueryField.TRAVELERS: (
        r"\bhow many\s+travel(l)?ers\b", r"\bhow many\s+people\s+are\s+(going|traveling|travelling)\b",
        r"\bmy\s+traveler\s+count\b",
    ),
    StateQueryField.PACE: (
        r"\bwhat'?s?\s+(is\s+)?my\s+pace\b", r"\bwhat\s+pace\s+did\s+i\s+(choose|pick|select)\b",
    ),
    StateQueryField.INTERESTS: (
        r"\bwhat\s+(are\s+)?my\s+interests\b", r"\bwhich\s+interests\s+did\s+i\s+(choose|pick|select)\b",
    ),
    StateQueryField.LANGUAGE: (
        r"\bwhat'?s?\s+(is\s+)?my\s+(preferred\s+)?language\b",
    ),
}

_AR_PATTERNS: dict[StateQueryField, tuple[str, ...]] = {
    StateQueryField.BUDGET: (r"ما\s+هي\s+ميزانيتي", r"كم\s+(هي\s+)?ميزانيتي", r"ميزانيتي\s+الحالية"),
    StateQueryField.DATES: (r"ما\s+هي\s+تواريخ\s+رحلتي", r"متى\s+(سأسافر|أسافر)", r"تاريخ\s+(المغادرة|العودة)"),
    StateQueryField.TRAVELERS: (r"كم\s+عدد\s+المسافرين", r"كم\s+شخص(اً)?\s+سيسافر"),
    StateQueryField.PACE: (r"ما\s+هي\s+وتيرة\s+رحلتي", r"ما\s+هو\s+إيقاع\s+رحلتي"),
    StateQueryField.INTERESTS: (r"ما\s+هي\s+اهتماماتي", r"ما\s+هي\s+اهتمامات(ي)?\s+المختارة"),
    StateQueryField.LANGUAGE: (r"ما\s+هي\s+لغتي\s+المفضلة", r"ما\s+هي\s+اللغة\s+المفضلة"),
}

_EN_COMPILED = {field: tuple(re.compile(p, re.IGNORECASE) for p in patterns) for field, patterns in _EN_PATTERNS.items()}
_AR_COMPILED = {field: tuple(re.compile(p) for p in patterns) for field, patterns in _AR_PATTERNS.items()}


def detect_state_query(message: str) -> Optional[StateQueryField]:
    """Returns the confidently-matched field, or None -- never a guess.
    Only ever called on a READ question; a request to CHANGE a field
    (e.g. "change my budget") must never be routed here, so callers check
    `phase4.chat_models`-level intent-shaped verbs are absent first, or
    simply accept that "change ..." never matches these read-only
    patterns in the first place (verified by test)."""
    for field, patterns in _EN_COMPILED.items():
        if any(p.search(message) for p in patterns):
            return field
    for field, patterns in _AR_COMPILED.items():
        if any(p.search(message) for p in patterns):
            return field
    return None


def _budget_sentence(trip_request: dict[str, Any], language: str) -> str:
    budget = describe_money((trip_request.get("budget") or {}).get("amount_minor_units"), (trip_request.get("budget") or {}).get("currency"))
    if budget is None:
        return "غير متوفرة حالياً." if language == "ar" else "not currently available."
    if language == "ar":
        return f"ميزانيتك الحالية هي {budget['formatted']}."
    return f"Your current budget is {budget['formatted']}."


def _dates_sentence(trip_request: dict[str, Any], language: str) -> str:
    depart, ret = trip_request.get("depart_date"), trip_request.get("return_date")
    if not depart or not ret:
        return "غير متوفرة حالياً." if language == "ar" else "not currently available."
    if language == "ar":
        return f"تبدأ رحلتك في {depart} وتنتهي في {ret}."
    return f"Your trip runs from {depart} to {ret}."


def _travelers_sentence(trip_request: dict[str, Any], language: str) -> str:
    count = trip_request.get("traveler_count")
    if count is None:
        return "غير متوفر حالياً." if language == "ar" else "not currently available."
    if language == "ar":
        return f"عدد المسافرين في رحلتك هو {count}."
    return f"You have {count} traveler(s) on this trip."


def _pace_sentence(trip_request: dict[str, Any], language: str) -> str:
    pace = (trip_request.get("preferences") or {}).get("pace")
    if not pace:
        return "غير متوفرة حالياً." if language == "ar" else "not currently available."
    label = (_PACE_LABELS_AR if language == "ar" else _PACE_LABELS_EN).get(pace, pace)
    if language == "ar":
        return f"وتيرة رحلتك هي {label}."
    return f"Your trip pace is {label}."


def _interests_sentence(trip_request: dict[str, Any], language: str) -> str:
    interests = (trip_request.get("preferences") or {}).get("interests") or []
    if not interests:
        return "لم يتم اختيار اهتمامات بعد." if language == "ar" else "no interests are currently selected."
    joined = "، ".join(interests) if language == "ar" else ", ".join(interests)
    if language == "ar":
        return f"اهتماماتك المختارة هي: {joined}."
    return f"Your selected interests are: {joined}."


def _language_sentence(trip_request: dict[str, Any], language: str) -> str:
    pref = (trip_request.get("preferences") or {}).get("language")
    if not pref:
        return "غير متوفرة حالياً." if language == "ar" else "not currently available."
    label = (_LANGUAGE_LABELS_AR if language == "ar" else _LANGUAGE_LABELS_EN).get(pref, pref)
    if language == "ar":
        return f"لغتك المفضلة هي {label}."
    return f"Your preferred language is {label}."


_SENTENCE_BUILDERS = {
    StateQueryField.BUDGET: _budget_sentence,
    StateQueryField.DATES: _dates_sentence,
    StateQueryField.TRAVELERS: _travelers_sentence,
    StateQueryField.PACE: _pace_sentence,
    StateQueryField.INTERESTS: _interests_sentence,
    StateQueryField.LANGUAGE: _language_sentence,
}


def answer_state_query(field: StateQueryField, trip_request: Optional[dict[str, Any]], language: str) -> Optional[str]:
    """Returns a deterministic, authoritative-state answer sentence, or
    None if there is no trip_request at all to answer from (the caller
    then falls through to the ordinary clarify path -- never fabricates
    an answer with no real data)."""
    if not isinstance(trip_request, dict):
        return None
    return _SENTENCE_BUILDERS[field](trip_request, language)
