"""Parametrized tests proving planner-a's Pydantic mirrors accept the
canonical valid examples, reject what the schemas reject, and produce
output that still validates against the canonical JSON Schema.

Run with: python -m pytest phase1/tests -v
"""

from __future__ import annotations

import glob
import json
import os

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from phase1 import models as m

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRACTS_DIR = os.path.abspath(
    os.path.join(THIS_DIR, "..", "..", "..", "..", "contracts")
)
VALID_DIR = os.path.join(CONTRACTS_DIR, "examples", "valid")
INVALID_DIR = os.path.join(CONTRACTS_DIR, "examples", "invalid")

# Models planner-a owns/consumes/produces, mapped to their canonical
# contract name (see docs/adr/0001-phase1-contracts.md).
MODEL_MAP = {
    "SourceReference": m.SourceReference,
    "DataProvenance": m.DataProvenance,
    "DataQuality": m.DataQuality,
    "ErrorEnvelope": m.ErrorEnvelope,
    "ProviderResponseEnvelope": m.ProviderResponseEnvelope,
    "TripPreferences": m.TripPreferences,
    "TripRequest": m.TripRequest,
    "FlightOption": m.FlightOption,
    "StayOption": m.StayOption,
    "FairPriceEstimate": m.FairPriceEstimate,
    "CandidateCombination": m.CandidateCombination,
    "TravelLeg": m.TravelLeg,
    "DailyPlan": m.DailyPlan,
    "LocalPlanRequest": m.LocalPlanRequest,
    "LocalItinerary": m.LocalItinerary,
    "BudgetBreakdown": m.BudgetBreakdown,
    "ConstraintConflict": m.ConstraintConflict,
    "PartialFailure": m.PartialFailure,
    "TripPlan": m.TripPlan,
    "StreamEvent": m.StreamEvent,
}

# Contracts closed with additionalProperties:false whose invalid examples
# specifically test "forbidden extra property" -- Pydantic (extra="forbid")
# must reject those too.
CLOSED_CONTRACT_INVALID_EXAMPLES = [
    "ErrorEnvelope_forbidden_extra_property.json",
    "TripPlan_forbidden_extra_property.json",
]


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def schema_registry() -> Registry:
    resources = []
    for path in glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")):
        schema = _load(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _schema_validator(name: str, registry: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, f"{name}.schema.json"))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


@pytest.mark.parametrize("name", sorted(MODEL_MAP))
def test_model_accepts_canonical_valid_example(name: str):
    model_cls = MODEL_MAP[name]
    instance = _load(os.path.join(VALID_DIR, f"{name}.json"))
    model_cls.model_validate(instance)


@pytest.mark.parametrize("name", sorted(MODEL_MAP))
def test_model_dump_still_validates_against_canonical_schema(name: str, schema_registry: Registry):
    model_cls = MODEL_MAP[name]
    instance = _load(os.path.join(VALID_DIR, f"{name}.json"))
    obj = model_cls.model_validate(instance)
    dumped = obj.model_dump(mode="json", exclude_none=True)
    validator = _schema_validator(name, schema_registry)
    errors = list(validator.iter_errors(dumped))
    assert not errors, f"{name} round-trip mismatch: {[e.message for e in errors]}"


@pytest.mark.parametrize("filename", CLOSED_CONTRACT_INVALID_EXAMPLES)
def test_model_rejects_unknown_fields_on_closed_contracts(filename: str):
    contract_name = filename.split("_")[0]
    model_cls = MODEL_MAP[contract_name]
    instance = _load(os.path.join(INVALID_DIR, filename))
    with pytest.raises(Exception):
        model_cls.model_validate(instance)


@pytest.mark.parametrize(
    "filename,contract_name",
    [
        ("TripRequest_missing_required_field.json", "TripRequest"),
        ("DataProvenance_invalid_enum.json", "DataProvenance"),
        ("LocalPlanRequest_malformed_session_id.json", "LocalPlanRequest"),
        ("StayOption_missing_provenance.json", "StayOption"),
        ("SourceReference_invalid_timestamp.json", "SourceReference"),
        ("StreamEvent_invalid_enum.json", "StreamEvent"),
    ],
)
def test_model_rejects_other_invalid_examples(filename: str, contract_name: str):
    model_cls = MODEL_MAP[contract_name]
    instance = _load(os.path.join(INVALID_DIR, filename))
    with pytest.raises(Exception):
        model_cls.model_validate(instance)


# ---------------------------------------------------------------------------
# Cross-service shared-contract agreement: the same canonical valid example
# for a contract shared across services must be accepted here, exactly as
# it is accepted by istanbul-expert-b's and travel-mcp's independent
# mirrors (see their own phase1/tests/test_models.py).
# ---------------------------------------------------------------------------

SHARED_CONTRACTS = ["SourceReference", "DataProvenance", "DataQuality", "ErrorEnvelope"]


@pytest.mark.parametrize("name", SHARED_CONTRACTS)
def test_shared_contract_example_accepted_identically(name: str):
    model_cls = MODEL_MAP[name]
    instance = _load(os.path.join(VALID_DIR, f"{name}.json"))
    obj = model_cls.model_validate(instance)
    assert obj.model_dump(mode="json", exclude_none=True)["schema_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Checkpoint Phase 4 D.2A: StreamStage's additive ReAct action-loop values
# (docs/adr/0015-phase4-checkpoint-d2a-system-a-api.md). Proves the
# Pydantic mirror itself -- not just the JSON Schema (see
# contracts/tests/test_stream_event_react_stages.py) -- accepts every new
# value and round-trips through the canonical schema.
# ---------------------------------------------------------------------------

NEW_D2A_STAGES = [
    "run_started", "action_started", "action_completed", "action_failed",
    "run_completed", "run_degraded", "run_failed", "run_cancelled",
]


@pytest.mark.parametrize("stage", NEW_D2A_STAGES)
def test_stream_event_accepts_new_react_action_loop_stage(stage: str, schema_registry: Registry):
    instance = {
        "schema_version": "1.1.0",
        "event_id": "55555555-5555-4555-8555-555555555555",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "trace_id": "22222222-2222-4222-8222-222222222222",
        "sequence": 0,
        "timestamp": "2026-08-01T12:00:00Z",
        "stage": stage,
        "payload": {},
    }
    obj = m.StreamEvent.model_validate(instance)
    assert obj.stage.value == stage
    dumped = obj.model_dump(mode="json", exclude_none=True)
    validator = _schema_validator("StreamEvent", schema_registry)
    errors = list(validator.iter_errors(dumped))
    assert not errors, [e.message for e in errors]


def test_stream_event_still_rejects_an_unknown_stage_value():
    instance = {
        "schema_version": "1.1.0",
        "event_id": "55555555-5555-4555-8555-555555555555",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "trace_id": "22222222-2222-4222-8222-222222222222",
        "sequence": 0,
        "timestamp": "2026-08-01T12:00:00Z",
        "stage": "not_a_real_stage",
        "payload": {},
    }
    with pytest.raises(Exception):
        m.StreamEvent.model_validate(instance)
