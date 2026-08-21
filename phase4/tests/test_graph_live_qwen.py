"""Narrowly scoped OPTIONAL live smoke test for the real Qwen decision
provider (Checkpoint Phase 4 D.0 §11). Disabled during normal `pytest`
runs -- enabled only by setting the explicit environment flag below.

Tests **only structured action selection** -- exactly one Qwen Chat
Completions request, never a tool call, never a `FakeToolExecutor`
execution, never MCP, never A2A. Requires both `QWEN_API_KEY` (or
`DASHSCOPE_API_KEY`) and an explicit `QWEN_BASE_URL` to be set in the
environment. Prints only sanitized, non-secret fields -- never the
prompt, the raw response, headers, the key, or a base URL that could
contain a workspace-specific credential-bearing path.

Enable with:
    VOYAGER_LIVE_QWEN_DECISION_GATE=1 python -m pytest phase4/tests/test_graph_live_qwen.py -v -s
"""

from __future__ import annotations

import json
import os
import time

import pytest

from phase4.graph import _build_decision_prompt
from phase4.models import Action, parse_action_decision
from phase4.qwen_client import QwenConfigurationError, QwenDecisionProvider, QwenTransportError, qwen_api_key_present

LIVE_GATE_ENV_VAR = "VOYAGER_LIVE_QWEN_DECISION_GATE"

pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_GATE_ENV_VAR) != "1",
    reason=f"live Qwen network gate is disabled by default; set {LIVE_GATE_ENV_VAR}=1 to enable",
)


def test_live_qwen_selects_search_flights_action():
    key_present = qwen_api_key_present()
    print(f"QWEN_API_KEY present: {str(key_present).lower()}")
    if not key_present:
        pytest.skip("QWEN_API_KEY/DASHSCOPE_API_KEY is not set in this environment -- cannot attempt the keyed live gate")

    provider = QwenDecisionProvider()

    # A minimal state slice -- exactly what _build_decision_prompt needs,
    # never a full PlannerState/full graph run. No tool executor, no
    # FakeToolExecutor, no LangGraph invocation happens in this test.
    state = {
        "normalized_request": {
            "user_message": "Find a one-way flight from Beirut to Istanbul on 2026-09-10 for one adult."
        },
        "observations": [],
        "tool_call_count": 0,
    }
    system, user = _build_decision_prompt(state)

    start = time.monotonic()
    try:
        raw_text = provider.generate(system, user)
    except QwenConfigurationError as exc:
        pytest.skip(f"Qwen is not fully configured in this environment ({exc})")
    except QwenTransportError as exc:
        pytest.fail(f"Qwen transport failure: {exc}")
    runtime_seconds = time.monotonic() - start

    raw_obj = json.loads(raw_text)
    decision = parse_action_decision(raw_obj)  # raises ActionDecisionValidationError if not schema-valid -- never swallowed

    assert decision.action == Action.SEARCH_FLIGHTS
    assert decision.arguments["origin"] == "BEY"
    assert decision.arguments["destination"] == "IST"
    assert runtime_seconds < 30.0

    serialized = json.dumps(decision.model_dump(mode="json"))
    for forbidden in ("chain_of_thought", "chain-of-thought", "api_key", "authorization", "bearer"):
        assert forbidden not in serialized.lower()

    print(f"model={provider.model}")
    print(f"selected_action={decision.action.value}")
    print("schema_valid=true")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print("exit_result=pass")
