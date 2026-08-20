"""Fair-price correction checkpoint: hermetic tests for the specialist-side
half of the fix. Root cause (see orchestration/tests/test_tool_executor.py's
own header comment for the full trace): `_build_specialist_prompt` reduced
every prior observation down to `{action, status}`, so Qwen was never shown
the real `stay_id` values a prior `search_stays` call had already returned
and had to guess one for `estimate_fair_price` -- silently rejected before
ever reaching Travel MCP. These tests prove the prompt is now grounded in
the real data, that a rejection's closed reason code reaches the prompt
as an already-safe `warnings` entry, and that this stays fully backward
compatible with an observation shape that never carries `reason` at all
(a payload from before this fix, or an MCP-originated rejection).
"""

from __future__ import annotations

import json

from phase4.specialist import _build_specialist_prompt, _make_specialist_execute_observe_node

_SEARCH_STAYS_SUCCESS_ENVELOPE = {
    "stays": [
        {"stay": {"stay_id": "stay_real_001", "name": "Beyoglu Boutique Hotel"}, "fair_price": {}, "rank": 1},
        {"stay": {"stay_id": "stay_real_002", "name": "Sultanahmet Palace Suites"}, "fair_price": {}, "rank": 2},
    ]
}


def _state(observations: list[dict], **overrides) -> dict:
    base = {
        "normalized_request": {"user_message": "Plan my trip.", "trip_request": None},
        "inherited_observations": [],
        "specialist_observations": observations,
        "tool_call_count": 0,
    }
    base.update(overrides)
    return base


def test_prompt_exposes_known_stay_candidates_from_a_successful_search_stays_observation():
    obs = {"action": "search_stays", "status": "success", "envelope": _SEARCH_STAYS_SUCCESS_ENVELOPE, "warnings": []}
    _, user = _build_specialist_prompt(_state([obs]))
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["known_stay_candidates"] == [
        {"stay_id": "stay_real_001", "name": "Beyoglu Boutique Hotel"},
        {"stay_id": "stay_real_002", "name": "Sultanahmet Palace Suites"},
    ]


def test_prompt_known_stay_candidates_empty_when_no_successful_search_stays_yet():
    _, user = _build_specialist_prompt(_state([]))
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["known_stay_candidates"] == []


def test_prompt_known_stay_candidates_ignores_a_failed_search_stays_observation():
    obs = {"action": "search_stays", "status": "provider_error", "envelope": None, "warnings": ["status=provider_error"]}
    _, user = _build_specialist_prompt(_state([obs]))
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["known_stay_candidates"] == []


def test_prompt_known_stay_candidates_uses_most_recent_successful_search_stays_only():
    older = {"action": "search_stays", "status": "success", "envelope": _SEARCH_STAYS_SUCCESS_ENVELOPE, "warnings": []}
    newer_envelope = {"stays": [{"stay": {"stay_id": "stay_newest", "name": "Newest Hotel"}, "fair_price": {}, "rank": 1}]}
    newer = {"action": "search_stays", "status": "success", "envelope": newer_envelope, "warnings": []}
    _, user = _build_specialist_prompt(_state([older, newer]))
    payload = json.loads(user.split("\n", 1)[1])
    assert payload["known_stay_candidates"] == [{"stay_id": "stay_newest", "name": "Newest Hotel"}]


def test_prompt_instructs_stay_id_must_be_copied_exactly_from_known_candidates():
    system, _ = _build_specialist_prompt(_state([]))
    assert "known_stay_candidates" in system
    assert "never invented" in system


def test_prompt_evidence_summary_surfaces_a_prior_observations_warnings():
    """A prior invalid_request observation's `warnings` (already a safe,
    caller-facing field) reaches the prompt as a concrete correction
    signal -- not just a bare repeated status string."""
    obs = {"action": "estimate_fair_price", "status": "invalid_request", "envelope": None, "warnings": ["status=invalid_request:stay_id_not_recognized"]}
    _, user = _build_specialist_prompt(_state([obs]))
    payload = json.loads(user.split("\n", 1)[1])
    entry = payload["evidence_collected_so_far"][0]
    assert entry["action"] == "estimate_fair_price"
    assert entry["status"] == "invalid_request"
    assert entry["warnings"] == ["status=invalid_request:stay_id_not_recognized"]


def test_prompt_evidence_summary_omits_warnings_key_when_none_present():
    """Backward compatible with the pre-fix, minimal {action, status}
    shape -- no `warnings` key is added when the observation carries none."""
    obs = {"action": "search_flights", "status": "success", "envelope": {"options": []}, "warnings": []}
    _, user = _build_specialist_prompt(_state([obs]))
    payload = json.loads(user.split("\n", 1)[1])
    entry = payload["evidence_collected_so_far"][0]
    assert "warnings" not in entry


# --- observe-node: the `reason` field is folded into `warnings` ------------


def test_observe_node_folds_a_reason_into_the_observation_warnings():
    class _StubExecutor:
        def execute(self, action, arguments, context=None):
            return {"status": "invalid_request", "result": None, "reason": "stay_id_not_recognized"}

    node = _make_specialist_execute_observe_node(_StubExecutor(), cancellation_check=lambda: False)
    state = {
        "pending_action": {"action": "estimate_fair_price", "arguments": {"stay_id": "invented"}},
        "graph_transition_count": 0, "session_id": "s1", "trace_id": "t1", "normalized_request": {},
        "inherited_observations": [], "specialist_observations": [], "started_at_monotonic": 0.0,
        "tool_call_count_by_action": {}, "warnings": [],
    }
    updates = node(state)
    observation = updates["specialist_observations"][-1]
    assert observation["status"] == "invalid_request"
    assert observation["warnings"] == ["status=invalid_request:stay_id_not_recognized"]


def test_observe_node_backward_compatible_with_a_raw_result_carrying_no_reason_at_all():
    """A tool result with no `reason` key at all (an MCP-originated
    invalid_request, or any pre-fix code path) must still produce the
    exact old warnings shape, never raise a KeyError."""

    class _StubExecutor:
        def execute(self, action, arguments, context=None):
            return {"status": "invalid_request", "result": None}  # no "reason" key

    node = _make_specialist_execute_observe_node(_StubExecutor(), cancellation_check=lambda: False)
    state = {
        "pending_action": {"action": "estimate_fair_price", "arguments": {"stay_id": "x"}},
        "graph_transition_count": 0, "session_id": "s1", "trace_id": "t1", "normalized_request": {},
        "inherited_observations": [], "specialist_observations": [], "started_at_monotonic": 0.0,
        "tool_call_count_by_action": {}, "warnings": [],
    }
    updates = node(state)
    observation = updates["specialist_observations"][-1]
    assert observation["warnings"] == ["status=invalid_request"]


def test_observe_node_still_bounds_estimate_fair_price_to_max_calls_per_tool():
    """Retry count is bounded: two rejected attempts at the SAME tool
    already hit MAX_CALLS_PER_TOOL via tool_call_count_by_action -- this
    fix adds no new counter and loosens no existing one."""
    from phase4.models import MAX_CALLS_PER_TOOL

    class _StubExecutor:
        def execute(self, action, arguments, context=None):
            return {"status": "invalid_request", "result": None, "reason": "stay_id_not_recognized"}

    node = _make_specialist_execute_observe_node(_StubExecutor(), cancellation_check=lambda: False)
    state = {
        "pending_action": {"action": "estimate_fair_price", "arguments": {"stay_id": "x"}},
        "graph_transition_count": 0, "session_id": "s1", "trace_id": "t1", "normalized_request": {},
        "inherited_observations": [], "specialist_observations": [], "started_at_monotonic": 0.0,
        "tool_call_count_by_action": {}, "warnings": [],
    }
    for _ in range(MAX_CALLS_PER_TOOL):
        updates = node(state)
        state = {**state, **updates, "pending_action": {"action": "estimate_fair_price", "arguments": {"stay_id": "x"}}}
    assert state["tool_call_count_by_action"]["estimate_fair_price"] == MAX_CALLS_PER_TOOL
