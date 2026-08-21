"""Hermetic tests for the Checkpoint Phase 4 D.1 additive `ExecutionContext`
and the corrected real-shaped weather validation it exists alongside. No
network call anywhere in this file."""

from __future__ import annotations

from phase4.context import ExecutionContext
from phase4.models import Action
from phase4.tool_result_validation import (
    _validate_estimate_fair_price_result,
    _validate_search_stays_result,
    _validate_weather_result,
)
from phase4.tools import FakeToolExecutor


def test_execution_context_observations_for_action_filters_correctly():
    context = ExecutionContext(
        session_id="s1",
        trace_id="t1",
        normalized_request={},
        observations=(
            {"action": "search_stays", "status": "success", "envelope": {"marker": 1}},
            {"action": "get_weather", "status": "success", "envelope": {"marker": 2}},
            {"action": "search_stays", "status": "success", "envelope": {"marker": 3}},
        ),
        deadline_monotonic=100.0,
        cancellation_check=lambda: False,
    )
    stays = context.observations_for_action("search_stays")
    assert len(stays) == 2
    assert stays[-1]["envelope"]["marker"] == 3  # most recent last


def test_fake_tool_executor_accepts_and_ignores_context():
    executor = FakeToolExecutor()
    context = ExecutionContext(
        session_id="s1", trace_id="t1", normalized_request={}, observations=(),
        deadline_monotonic=60.0, cancellation_check=lambda: False,
    )
    result_with_context = executor.execute(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, context)
    result_without_context = FakeToolExecutor(clock=executor.clock).execute(
        Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}
    )
    assert result_with_context == result_without_context


def test_weather_validation_accepts_real_current_observation_shape():
    result = {
        "location": "Istanbul",
        "kind": "current_observation",
        "observation": {"condition": "clear_sky", "temperature": 24},
    }
    assert _validate_weather_result(result) is None


def test_weather_validation_accepts_real_forecast_shape():
    result = {
        "location": "Istanbul",
        "kind": "forecast",
        "forecast_days": [{"date": "2026-09-10", "condition": "partly_cloudy", "high": 27, "low": 19}],
    }
    assert _validate_weather_result(result) is None


def test_weather_validation_accepts_real_historical_climate_shape():
    """Manual QA remediation Q.1 (user correction pass §C): a real
    successful get_weather result for a future date beyond the forecast
    horizon has kind='historical' -- this was previously rejected as
    'missing_or_unknown_kind' and silently demoted to 'provider_error',
    caught only by this checkpoint's own extended live verification."""
    result = {
        "location": "Istanbul",
        "kind": "historical",
        "forecast_days": [],
        "historical_climate_days": [
            {"date": "2026-10-15", "condition": "overcast", "avg_high": 20.5, "avg_low": 13.2, "years_sampled": [2023, 2024, 2025]},
        ],
    }
    assert _validate_weather_result(result) is None


def test_weather_validation_accepts_real_mixed_coverage_shape():
    result = {
        "location": "Istanbul",
        "kind": "mixed",
        "forecast_days": [{"date": "2026-09-01", "condition": "clear_sky", "high": 29, "low": 21}],
        "historical_climate_days": [
            {"date": "2026-09-10", "condition": "overcast", "avg_high": 26.0, "avg_low": 19.0, "years_sampled": [2023, 2024, 2025]},
        ],
    }
    assert _validate_weather_result(result) is None


def test_weather_validation_rejects_empty_historical_climate_days():
    result = {"location": "Istanbul", "kind": "historical", "historical_climate_days": []}
    assert _validate_weather_result(result) == "missing_historical_climate_condition"


def test_weather_validation_rejects_mixed_missing_forecast_days():
    result = {"location": "Istanbul", "kind": "mixed", "forecast_days": [], "historical_climate_days": [{"condition": "clear"}]}
    assert _validate_weather_result(result) == "missing_forecast_condition"


def test_weather_validation_rejects_mixed_missing_historical_days():
    result = {"location": "Istanbul", "kind": "mixed", "forecast_days": [{"condition": "clear"}], "historical_climate_days": []}
    assert _validate_weather_result(result) == "missing_historical_climate_condition"


def test_weather_validation_rejects_the_old_flat_shape():
    """The original D.0 fixture shape ({"location", "condition"} flat)
    never matched the real provider contract -- this proves the fixed
    validation now correctly rejects it as malformed, not accepts it."""
    result = {"location": "Istanbul", "condition": "clear"}
    assert _validate_weather_result(result) == "missing_or_unknown_kind"


def test_weather_validation_rejects_missing_location():
    assert _validate_weather_result({"kind": "forecast", "forecast_days": [{"condition": "clear"}]}) == "missing_location"


def test_weather_validation_rejects_empty_forecast_days():
    result = {"location": "Istanbul", "kind": "forecast", "forecast_days": []}
    assert _validate_weather_result(result) == "missing_forecast_condition"


def test_fake_weather_fixture_now_matches_real_shape_and_passes_validation():
    executor = FakeToolExecutor()
    result = executor.execute(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"})
    weather_result = result["result"]["result"]
    assert _validate_weather_result(weather_result) is None


def test_search_stays_validation_rejects_the_old_flat_options_shape():
    """The original D.0 fixture used {"options": [StayOption...]} -- no
    real Travel MCP response is shaped that way. Proves it is now
    correctly rejected as malformed."""
    assert _validate_search_stays_result({"options": [{"stay_id": "x"}]}) == "missing_stays"


def test_search_stays_validation_accepts_real_mcp_shape():
    result = {
        "stays": [
            {
                "stay": {"stay_id": "stay_001", "nightly_price": {"amount_minor_units": 100, "currency": "TRY"}},
                "fair_price": {"stay_id": "stay_001", "estimated_fair_price": {"amount_minor_units": 90, "currency": "TRY"}},
                "rank": 1,
            }
        ]
    }
    assert _validate_search_stays_result(result) is None


def test_fake_search_stays_fixture_matches_real_shape_and_passes_validation():
    executor = FakeToolExecutor()
    result = executor.execute(Action.SEARCH_STAYS, {"check_in": "2026-09-10", "check_out": "2026-09-15", "guest_count": 2})
    # search_stays results are NOT ProviderResponseEnvelope-wrapped (ADR
    # 0009 §5 applies only to the root providers/ package) -- result["result"]
    # IS the SearchStaysResult directly, one level only.
    assert _validate_search_stays_result(result["result"]) is None


def test_estimate_fair_price_validation_rejects_the_old_flat_deal_score_shape():
    assert _validate_estimate_fair_price_result({"deal_score": 0.8, "components": {}}) == "missing_fair_price_fields"


def test_fake_estimate_fair_price_fixture_matches_real_shape_and_passes_validation():
    executor = FakeToolExecutor()
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_fake_d0_001"})
    assert _validate_estimate_fair_price_result(result["result"]) is None
