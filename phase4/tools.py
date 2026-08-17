"""Planner-owned tool-executor boundary (Checkpoint Phase 4 D.0 §6).
`ToolExecutor` is the seam D.1 will fill with real, network-calling
adapters injected at the composition boundary -- this checkpoint
implements only `FakeToolExecutor`: deterministic, no socket, no
external quota consumed. Never imports the root `providers` package
(ADR 0009 §6.1) -- fixtures are built directly from the existing,
unmodified `phase1.models` mirrors.

Production composition rule for D.1 (recorded here, not implemented):
System A owns orchestration; root live-provider adapters are injected at
this exact boundary (a real class satisfying `ToolExecutor`, swapped in
for `FakeToolExecutor`); Travel MCP remains accommodation owner; System B
is invoked only through real A2A, never receives provider credentials;
A2A is never used between LangGraph nodes (only at the System A/System B
network boundary, architecture.md §4.2).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date as date_cls
from typing import Any, Callable, Optional, Protocol

from phase1.models import (
    AccessibilityScore,
    DailyPlan,
    DataMode,
    DataProvenance,
    DataQuality,
    DealScoreComponent,
    FairPriceEstimate,
    FlightOption,
    LocalItinerary,
    Money,
    ProviderResponseEnvelope,
    StayOption,
)
from phase4.models import Action

FIXED_CLOCK = "2026-08-17T12:00:00Z"

# Deterministic fake scenarios a test can request per action -- never a
# real network condition, always a fixed, reproducible fixture path.
Scenario = str  # "success" | "unavailable" | "timeout" | "rate_limited" | "malformed"


class ToolExecutor(Protocol):
    """The seam Checkpoint D.1 fills with real provider adapters. Always
    returns a plain dict: `{"status": <RESULT_STATUSES-style str>, "result": <dict | None>}`
    -- `result` is the raw, not-yet-validated payload; the graph's Observe
    node is solely responsible for schema validation before any state
    insertion (this executor never silently pre-validates, since a real
    adapter's own malformed response must be observable the same way a
    fake's is)."""

    def execute(self, action: Action, arguments: dict[str, Any]) -> dict[str, Any]: ...


# A fixed, pinned namespace (a random constant, generated once) so
# uuid5-derived ids are stable across processes -- never uuid4/random,
# matching this project's established `providers.fingerprint.deterministic_request_id`
# precedent: identical fixture inputs must produce byte-identical output
# (Checkpoint D.0 §10, "deterministic identical-input behavior").
_ID_NAMESPACE = uuid.UUID("7e6d9b3a-4c1a-4e3a-9b7a-6f2d8c1a9b3e")


def _deterministic_id(*parts: Any) -> uuid.UUID:
    return uuid.uuid5(_ID_NAMESPACE, ":".join(str(p) for p in parts))


def _envelope(provider: str, retrieved_at: str, currency: Optional[str], result: dict) -> dict:
    request_id = _deterministic_id(provider, retrieved_at, json.dumps(result, sort_keys=True, default=str))
    return ProviderResponseEnvelope(
        request_id=request_id,
        provider=provider,
        data_mode=DataMode.FIXTURE,
        retrieved_at=retrieved_at,
        currency=currency,
        source_urls=[],
        quality=DataQuality(completeness=1.0, freshness=DataMode.FIXTURE, assumptions=["Deterministic Checkpoint D.0 fixture -- not a live call."]),
        result=result,
    ).model_dump(mode="json")


def _build_search_flights(arguments: dict, now: str) -> dict:
    depart_date = arguments["depart_date"]
    option = FlightOption(
        flight_id="flight_fake_d0_001",
        origin=arguments["origin"],
        destination=arguments["destination"],
        depart_at=f"{depart_date}T06:30:00Z",
        arrive_at=f"{depart_date}T08:15:00Z",
        carrier="FakeAir",
        stops=0,
        price=Money(amount_minor_units=450000, currency="TRY"),
        provenance=DataProvenance(provider="fake-flight-search-provider", data_mode=DataMode.FIXTURE, retrieved_at=now, source_urls=[]),
    )
    return _envelope("fake-flight-search-provider", now, "TRY", {"options": [option.model_dump(mode="json")]})


def _build_search_stays(arguments: dict, now: str) -> dict:
    option = StayOption(
        stay_id="stay_fake_d0_001",
        name="Fake Boutique Hotel",
        district_id=arguments.get("district_id") or "district_sultanahmet",
        side="european",
        coordinates={"lat": 41.0086, "lon": 28.9802},
        nightly_price=Money(amount_minor_units=250000, currency="TRY"),
        rating=4.5,
        review_count=120,
        amenities=["wifi", "breakfast"],
        provenance=DataProvenance(provider="fake-accommodation-provider", data_mode=DataMode.FIXTURE, retrieved_at=now, source_urls=[]),
    )
    return _envelope("fake-accommodation-provider", now, "TRY", {"options": [option.model_dump(mode="json")]})


def _build_estimate_fair_price(arguments: dict, now: str) -> dict:
    estimate = FairPriceEstimate(
        stay_id=arguments["stay_id"],
        estimated_fair_price=Money(amount_minor_units=230000, currency="TRY"),
        deal_score=0.72,
        components={"location": DealScoreComponent(weight=0.5, raw_value=0.8, normalized_value=0.8)},
        model_version="fake-d0-v1",
        baseline_beaten=True,
        provenance=DataProvenance(provider="fake-fair-price-model", data_mode=DataMode.FIXTURE, retrieved_at=now, source_urls=[]),
    )
    return _envelope("fake-fair-price-model", now, "TRY", estimate.model_dump(mode="json"))


def _build_get_weather(arguments: dict, now: str) -> dict:
    result = {
        "location": arguments.get("location", "Istanbul"),
        "date_from": str(arguments["date_from"]),
        "date_to": str(arguments["date_to"]),
        "condition": "partly_cloudy",
        "temperature_high_c": 27,
        "temperature_low_c": 19,
    }
    return _envelope("fake-weather-provider", now, None, result)


def _build_web_search(arguments: dict, now: str) -> dict:
    result = {
        "query": arguments["query"],
        "items": [
            {
                "title": "Hagia Sophia visiting hours",
                "canonical_url": "https://example-fixture.voyagerai.dev/hagia-sophia",
                "snippet": "Deterministic fake evidence snippet for Checkpoint D.0 -- never a real fetched fact.",
                "source_type": "secondary",
            }
        ],
    }
    return _envelope("fake-web-search-provider", now, None, result)


def _build_call_istanbul_expert(arguments: dict, now: str) -> dict:
    fixture_date = date_cls.fromisoformat(now[:10])
    daily_plan = DailyPlan(
        date=fixture_date,
        side="european",
        poi_ids=["poi_hagia_sophia"],
        legs=[],
        walking_minutes=15.0,
        transfer_minutes=0.0,
        activity_minutes=60.0,
        meal_minutes=0.0,
        slack_minutes=5.0,
        warnings=[],
    )
    itinerary = LocalItinerary(
        session_id=_deterministic_id("call_istanbul_expert", "session", now, json.dumps(arguments, sort_keys=True, default=str)),
        trace_id=_deterministic_id("call_istanbul_expert", "trace", now, json.dumps(arguments, sort_keys=True, default=str)),
        contract_version="1.0.0",
        recommended_base_candidate_id="candidate_fake_d0_001",
        accessibility_scores=[AccessibilityScore(candidate_id="candidate_fake_d0_001", score=0.8)],
        selected_poi_ids=["poi_hagia_sophia"],
        daily_plans=[daily_plan],
        estimated_travel_minutes=15.0,
        expected_walking_minutes=15.0,
        side_crossings=0,
        citations=[],
        assumptions=["Deterministic Checkpoint D.0 fixture -- not a real A2A call to System B."],
        warnings=[],
        data_quality=DataQuality(completeness=1.0, freshness=DataMode.FIXTURE, assumptions=[]),
        hard_constraint_validation_passed=True,
    )
    return _envelope("fake-istanbul-expert", now, None, itinerary.model_dump(mode="json"))


_BUILDERS: dict[Action, Callable[[dict, str], dict]] = {
    Action.SEARCH_FLIGHTS: _build_search_flights,
    Action.SEARCH_STAYS: _build_search_stays,
    Action.ESTIMATE_FAIR_PRICE: _build_estimate_fair_price,
    Action.GET_WEATHER: _build_get_weather,
    Action.WEB_SEARCH: _build_web_search,
    Action.CALL_ISTANBUL_EXPERT: _build_call_istanbul_expert,
}


@dataclass
class FakeToolExecutor:
    """Deterministic fake -- never opens a socket, never consumes real
    SerpApi/Open-Meteo/Groq/Qwen/MCP/A2A quota. `scenario_by_action`
    lets a test force a specific action into "unavailable"/"timeout"/
    "rate_limited"/"malformed" instead of the default "success" fixture."""

    scenario_by_action: dict[Action, Scenario] = field(default_factory=dict)
    clock: Callable[[], str] = field(default=lambda: FIXED_CLOCK)
    call_log: list[tuple[Action, dict]] = field(default_factory=list)

    def execute(self, action: Action, arguments: dict[str, Any]) -> dict[str, Any]:
        self.call_log.append((action, dict(arguments)))
        scenario = self.scenario_by_action.get(action, "success")
        now = self.clock()

        if scenario == "timeout":
            return {"status": "timeout", "result": None}
        if scenario == "unavailable":
            return {"status": "unavailable", "result": None}
        if scenario == "rate_limited":
            return {"status": "rate_limited", "result": None}
        if scenario == "malformed":
            # Deliberately missing every field a valid envelope/result
            # needs -- proves Observe rejects it rather than inserting
            # it as if it were valid.
            return {"status": "success", "result": {"unexpected_shape": True}}

        builder = _BUILDERS.get(action)
        if builder is None:
            return {"status": "unavailable", "result": None}
        return {"status": "success", "result": builder(arguments, now)}
