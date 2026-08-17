"""Hermetic tests for the fake tool-executor boundary (Checkpoint Phase 4
D.0 §6). No real network call anywhere in this file."""

import socket

from phase4.models import Action
from phase4.tools import FakeToolExecutor


def test_success_scenario_returns_schema_shaped_fixture():
    executor = FakeToolExecutor()
    result = executor.execute(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"})
    assert result["status"] == "success"
    assert result["result"]["provider"] == "fake-weather-provider"


def test_unavailable_scenario():
    executor = FakeToolExecutor(scenario_by_action={Action.SEARCH_FLIGHTS: "unavailable"})
    result = executor.execute(Action.SEARCH_FLIGHTS, {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1})
    assert result["status"] == "unavailable"
    assert result["result"] is None


def test_timeout_scenario():
    executor = FakeToolExecutor(scenario_by_action={Action.SEARCH_FLIGHTS: "timeout"})
    result = executor.execute(Action.SEARCH_FLIGHTS, {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1})
    assert result["status"] == "timeout"


def test_rate_limited_scenario():
    executor = FakeToolExecutor(scenario_by_action={Action.WEB_SEARCH: "rate_limited"})
    result = executor.execute(Action.WEB_SEARCH, {"query": "test"})
    assert result["status"] == "rate_limited"


def test_malformed_scenario_returns_unusable_shape():
    executor = FakeToolExecutor(scenario_by_action={Action.SEARCH_FLIGHTS: "malformed"})
    result = executor.execute(Action.SEARCH_FLIGHTS, {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1})
    assert result["status"] == "success"
    assert "options" not in result["result"]


def test_records_call_log():
    executor = FakeToolExecutor()
    executor.execute(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"})
    assert len(executor.call_log) == 1
    assert executor.call_log[0][0] == Action.GET_WEATHER


def test_identical_arguments_produce_byte_identical_fixture():
    executor_a = FakeToolExecutor()
    executor_b = FakeToolExecutor()
    args = {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1}
    result_a = executor_a.execute(Action.SEARCH_FLIGHTS, args)
    result_b = executor_b.execute(Action.SEARCH_FLIGHTS, args)
    assert result_a == result_b


def test_every_capability_has_a_success_fixture():
    executor = FakeToolExecutor()
    fixtures = {
        Action.SEARCH_FLIGHTS: {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1},
        Action.SEARCH_STAYS: {"check_in": "2026-09-10", "check_out": "2026-09-15", "guest_count": 2},
        Action.ESTIMATE_FAIR_PRICE: {"stay_id": "stay_fake_d0_001"},
        Action.GET_WEATHER: {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"},
        Action.WEB_SEARCH: {"query": "Hagia Sophia hours"},
        Action.CALL_ISTANBUL_EXPERT: {"question": "What should I see near Sultanahmet?"},
    }
    # search_stays/estimate_fair_price/call_istanbul_expert are NOT
    # ProviderResponseEnvelope-wrapped (Travel MCP/System B A2A each have
    # their own real, unwrapped result contract, ADR 0009 §5) -- only
    # search_flights/get_weather/web_search carry a top-level "provider" key.
    unwrapped_actions = {Action.SEARCH_STAYS, Action.ESTIMATE_FAIR_PRICE, Action.CALL_ISTANBUL_EXPERT}
    for action, args in fixtures.items():
        result = executor.execute(action, args)
        assert result["status"] == "success", action
        if action in unwrapped_actions:
            assert isinstance(result["result"], dict) and result["result"], action
        else:
            assert result["result"]["provider"], action


def test_no_real_socket_is_ever_opened(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("no network call is permitted from a hermetic fake-tool test")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    executor = FakeToolExecutor()
    result = executor.execute(Action.GET_WEATHER, {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"})
    assert result["status"] == "success"
