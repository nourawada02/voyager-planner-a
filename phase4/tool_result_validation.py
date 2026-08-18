"""Raw tool-result validation against the reused, unmodified `phase1.models`
mirrors (Checkpoint Phase 4 D.0, extracted to its own module in the D.3
correction pass so both the supervisor's own Observe/Execute nodes
(`phase4/graph.py`) and the internal Travel Search specialist's own
Execute/Observe node (`phase4/specialist.py`) share exactly one
validation path -- never two independent copies that could silently
drift apart.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError

from phase1.models import FlightOption, LocalItinerary, ProviderResponseEnvelope
from phase4.models import Action


def _first_error_field(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "unknown"
    loc = errors[0].get("loc", ("unknown",))
    return str(loc[0]) if loc else "unknown"


def _validate_weather_result(result: dict) -> Optional[str]:
    """Matches the REAL `providers.weather`/`providers.weather_openmeteo`
    WeatherResult contract: a "kind"-discriminated shape ("current_observation"
    with a nested `observation.condition`, or "forecast" with
    `forecast_days[i].condition") -- not a flat {"location","condition"}
    shape. Checkpoint D.1 audit finding: the original D.0 check only
    matched this checkpoint's own fake fixture, not the real provider's
    actual shape; both the fixture (`phase4/tools.py`) and this check
    were corrected together."""
    if "location" not in result:
        return "missing_location"
    kind = result.get("kind")
    if kind == "current_observation":
        observation = result.get("observation")
        if not isinstance(observation, dict) or "condition" not in observation:
            return "missing_observation_condition"
    elif kind == "forecast":
        days = result.get("forecast_days")
        if not isinstance(days, list) or not days or "condition" not in days[0]:
            return "missing_forecast_condition"
    else:
        return "missing_or_unknown_kind"
    return None


def _validate_search_stays_result(result: dict) -> Optional[str]:
    """Matches the REAL Travel MCP `search_stays` result contract
    (services/travel-mcp/phase2/serving/models.py::SearchStaysResult) --
    a nested `{"stays": [{"stay": {...}, "fair_price": {...}, "rank": ...}]}`
    shape, structurally distinct from `phase1.models.StayOption`."""
    stays = result.get("stays")
    if not isinstance(stays, list) or not stays:
        return "missing_stays"
    for item in stays:
        if not isinstance(item, dict):
            return "malformed_stay_item"
        stay = item.get("stay")
        if not isinstance(stay, dict) or "stay_id" not in stay or "nightly_price" not in stay:
            return "malformed_stay_item"
        if "fair_price" not in item:
            return "missing_fair_price"
    return None


def _validate_estimate_fair_price_result(result: dict) -> Optional[str]:
    """Matches the REAL Travel MCP `estimate_fair_price` result contract
    (`EstimateFairPriceResult`) -- nested `fair_price.estimated_fair_price`,
    not the flat `deal_score`/`components` shape D.0's original fixture
    invented."""
    if "stay_id" not in result or "fair_price" not in result:
        return "missing_fair_price_fields"
    fair_price = result.get("fair_price")
    if not isinstance(fair_price, dict) or "estimated_fair_price" not in fair_price:
        return "malformed_fair_price"
    return None


# Checkpoint D.1 audit finding: `ProviderResponseEnvelope` (ADR 0009 §5)
# is specifically the root `providers/` package's own convention --
# search_flights/get_weather/web_search results really are wrapped in it.
# Travel MCP (search_stays/estimate_fair_price) and System B's A2A
# artifact (call_istanbul_expert) each have their own, different, real
# result contract and were never wrapped in this envelope at all.
ENVELOPE_WRAPPED_ACTIONS = frozenset({Action.SEARCH_FLIGHTS, Action.GET_WEATHER, Action.WEB_SEARCH})


def validate_tool_result(action: Action, candidate: Any) -> Optional[str]:
    """Validates a raw tool result against the reused, unmodified
    `phase1.models` mirrors before it is ever allowed into state.
    Returns None on success, or a short, safe validation-failure code
    otherwise -- never the raw pydantic error text. Shared, unmodified,
    by both the supervisor's own direct tool calls and every real tool
    call the internal Travel Search specialist makes."""
    if not isinstance(candidate, dict):
        return "not_a_dict"

    if action in ENVELOPE_WRAPPED_ACTIONS:
        try:
            envelope = ProviderResponseEnvelope.model_validate(candidate)
        except ValidationError as exc:
            return f"envelope_invalid:{_first_error_field(exc)}"
        result = envelope.result
        if not isinstance(result, dict):
            return "result_not_a_dict"
    else:
        result = candidate

    try:
        if action == Action.SEARCH_FLIGHTS:
            options = result.get("options")
            if not isinstance(options, list) or not options:
                return "missing_options"
            for opt in options:
                FlightOption.model_validate(opt)
        elif action == Action.SEARCH_STAYS:
            stays_error = _validate_search_stays_result(result)
            if stays_error is not None:
                return stays_error
        elif action == Action.ESTIMATE_FAIR_PRICE:
            price_error = _validate_estimate_fair_price_result(result)
            if price_error is not None:
                return price_error
        elif action == Action.GET_WEATHER:
            weather_error = _validate_weather_result(result)
            if weather_error is not None:
                return weather_error
        elif action == Action.WEB_SEARCH:
            if not isinstance(result.get("items"), list):
                return "missing_web_search_items"
        elif action == Action.CALL_ISTANBUL_EXPERT:
            LocalItinerary.model_validate(result)
    except ValidationError as exc:
        return f"result_invalid:{_first_error_field(exc)}"

    return None
