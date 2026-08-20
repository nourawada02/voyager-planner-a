"""Closed interest vocabulary for chat-driven trip modifications (Hybrid
Chat C.1 §5: "interests normalized using existing project vocabulary").
`phase1.models.TripPreferences.interests` is intentionally an
unconstrained `list[str]` (no closed vocabulary exists at the contract
level), but the one vocabulary this project's own UI actually offers a
user is the fixed 7-item `st.multiselect` list in
`services/frontend/phase6/app.py::_render_trip_form`
(history/food/shopping/nightlife/art/nature/family) -- this module reuses
that exact list rather than inventing a new taxonomy or importing
`services/istanbul-expert-b`'s own separate, broader RAG interest
taxonomy (`phase4/interests.py` in that submodule), which is a different
system's concern and would cross the submodule boundary
(CLAUDE.md's "Repository / submodule boundaries" rule)."""

from __future__ import annotations

SUPPORTED_INTERESTS = ("history", "food", "shopping", "nightlife", "art", "nature", "family")
_SUPPORTED_INTERESTS_SET = frozenset(SUPPORTED_INTERESTS)

# A small, explicit synonym map for the handful of natural phrasings a
# user is likely to type in chat (English and transliterated Arabic
# equivalents) -- never a fuzzy/similarity match, so an unrecognized word
# is always honestly reported as unrecognized rather than silently mapped
# to the nearest guess.
_SYNONYMS: dict[str, str] = {
    "museums": "history", "museum": "history", "historical": "history", "culture": "history",
    "cultural": "history", "heritage": "history",
    "dining": "food", "restaurants": "food", "cuisine": "food", "eating": "food",
    "markets": "shopping", "bazaar": "shopping", "bazaars": "shopping", "malls": "shopping",
    "clubs": "nightlife", "bars": "nightlife", "nightclubs": "nightlife",
    "galleries": "art", "gallery": "art", "museums_of_art": "art",
    "outdoors": "nature", "parks": "nature", "scenery": "nature",
    "kids": "family", "children": "family", "family_friendly": "family",
}


def normalize_interest(raw: str) -> str | None:
    """Returns one of `SUPPORTED_INTERESTS`, or None if `raw` cannot be
    confidently mapped -- never a guess. Case/whitespace-insensitive."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _SUPPORTED_INTERESTS_SET:
        return key
    return _SYNONYMS.get(key)


def normalize_interests(raw: list[str]) -> tuple[list[str], list[str]]:
    """Splits `raw` into (normalized, unrecognized) -- `normalized`
    preserves first-seen order with no duplicates; `unrecognized` holds
    each original string that could not be mapped, for an honest
    clarification/warning message, never silently dropped without
    disclosure."""
    normalized: list[str] = []
    unrecognized: list[str] = []
    for item in raw:
        mapped = normalize_interest(item)
        if mapped is None:
            unrecognized.append(item)
        elif mapped not in normalized:
            normalized.append(mapped)
    return normalized, unrecognized
