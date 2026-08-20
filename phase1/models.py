"""Pydantic v2 mirrors of the canonical contracts System A (planner) owns,
consumes, or produces: orchestration, trip, stream, aggregation, and error
contracts, per docs/adr/0001-phase1-contracts.md.

These are hand-mirrored from ../../../contracts/*.schema.json, per
architecture.md §15.1 ("mirrored in Pydantic per service — never shared as
imported Python"). The canonical JSON Schema files remain the source of
truth for shape; contracts/*/tests validate the two stay in agreement.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class DataMode(str, Enum):
    LIVE = "live"
    CACHED = "cached"
    HISTORICAL = "historical"
    FIXTURE = "fixture"
    ESTIMATED = "estimated"
    # Checkpoint Phase 4 D.1 additive value: matches the root
    # ProviderResponseEnvelope.schema.json 1.1.0 correction pass
    # (docs/adr/0009-...md) exactly -- "no data and no computed estimate
    # exist at all", distinct from ESTIMATED. Every pre-existing document
    # using only the original five values remains valid unchanged.
    UNAVAILABLE = "unavailable"


class Pace(str, Enum):
    RELAXED = "relaxed"
    MODERATE = "moderate"
    PACKED = "packed"


class Language(str, Enum):
    EN = "en"
    TR = "tr"
    AR = "ar"


class Side(str, Enum):
    EUROPEAN = "european"
    ASIAN = "asian"


class Money(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_minor_units: int = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    source_id: str = Field(min_length=1)
    title: Optional[str] = None
    url: Optional[str] = None
    retrieved_at: datetime
    chunk_ids: Optional[list[str]] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class DataProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    provider: str = Field(min_length=1)
    data_mode: DataMode
    retrieved_at: datetime
    valid_for: Optional[str] = None
    source_urls: Optional[list[str]] = None


class DataQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    completeness: float = Field(ge=0, le=1)
    freshness: DataMode
    assumptions: list[str]


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    error_code: str = Field(pattern=r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")
    message: str = Field(min_length=1)
    trace_id: UUID
    retriable: bool


class ProviderResponseEnvelope(BaseModel):
    """Checkpoint Phase 4 D.1 additive change: `capability`, `status`,
    `query_fingerprint`, `cache_status`, and `cache_age_seconds` mirror
    the root ProviderResponseEnvelope.schema.json's own 1.1.0 additive
    fields exactly (docs/adr/0009-phase4-checkpointc0-live-data-and-react-design.md)
    -- required because every real root provider adapter (Open-Meteo,
    SerpApi web evidence, SerpApi Google Flights) has produced 1.1.0
    envelopes since Checkpoint C.0/C.1, and this mirror previously only
    had the original 1.0.0 fields, so it would have rejected every real
    envelope outright. All five are optional, so every pre-existing
    1.0.0-only instance/fixture remains valid unchanged -- the same
    additive-minor-version discipline the root schema itself follows."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    request_id: UUID
    provider: str = Field(min_length=1)
    capability: Optional[str] = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    data_mode: DataMode
    status: Optional[str] = None
    query_fingerprint: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cache_status: Optional[str] = None
    cache_age_seconds: Optional[int] = Field(default=None, ge=0)
    retrieved_at: datetime
    valid_for: Optional[str] = None
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    source_urls: list[str]
    quality: DataQuality
    result: dict


class TripPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    interests: list[str]
    pace: Pace
    language: Language
    mobility_constraints: list[str]


class TripRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    session_id: UUID
    trace_id: UUID
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^IST$")
    depart_date: date
    return_date: date
    traveler_count: int = Field(ge=1, le=12)
    budget: Money
    preferences: TripPreferences


class FlightLeg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    depart_at: datetime
    arrive_at: datetime
    carrier: str = Field(min_length=1)
    flight_number: Optional[str] = Field(default=None, pattern=r"^[A-Z0-9]{2,3}[0-9]{1,4}[A-Z]?$")


class FlightOption(BaseModel):
    """Checkpoint Phase 4 D.1 additive change: `flight_number`,
    `duration_minutes`, `legs`, `provider_reference`, `offer_expires_at`,
    `baggage_limitations`, and `fare_limitations` mirror the root
    FlightOption.schema.json's own 1.1.0 additive fields exactly
    (docs/adr/0009-...md) -- required because the real
    `providers.flights_serpapi` adapter (Checkpoint C.3) has produced
    1.1.0 options since that checkpoint, and this mirror previously only
    had the original 1.0.0 fields, so it would have rejected every real
    flight option outright. All seven are optional, so every
    pre-existing 1.0.0-only instance/fixture remains valid unchanged."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    flight_id: str = Field(min_length=1)
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    depart_at: datetime
    arrive_at: datetime
    carrier: str = Field(min_length=1)
    flight_number: Optional[str] = Field(default=None, pattern=r"^[A-Z0-9]{2,3}[0-9]{1,4}[A-Z]?$")
    stops: int = Field(ge=0)
    duration_minutes: Optional[int] = Field(default=None, ge=0)
    legs: Optional[list[FlightLeg]] = Field(default=None, min_length=2)
    price: Money
    provider_reference: Optional[str] = Field(default=None, min_length=1)
    offer_expires_at: Optional[datetime] = None
    baggage_limitations: Optional[str] = Field(default=None, max_length=512)
    fare_limitations: Optional[str] = Field(default=None, max_length=512)
    provenance: DataProvenance


class StayOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    stay_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    district_id: str = Field(pattern=r"^district_[a-z0-9_]+$")
    side: Side
    coordinates: dict
    nightly_price: Money
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    review_count: Optional[int] = Field(default=None, ge=0)
    amenities: Optional[list[str]] = None
    provenance: DataProvenance


class DealScoreComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weight: float = Field(ge=0, le=1)
    raw_value: float
    normalized_value: float = Field(ge=0, le=1)


class FairPriceEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    stay_id: str = Field(min_length=1)
    estimated_fair_price: Money
    deal_score: float = Field(ge=0, le=1)
    components: dict[str, DealScoreComponent]
    model_version: str = Field(min_length=1)
    baseline_beaten: bool
    provenance: DataProvenance


class CandidateCombination(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    combination_id: str = Field(min_length=1)
    flight_id: str = Field(min_length=1)
    stay_id: str = Field(min_length=1)
    total_price: Money
    feasible: bool
    score: Optional[float] = Field(default=None, ge=0, le=1)


class TravelLeg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    from_poi_id: str = Field(pattern=r"^poi_[a-z0-9_]+$")
    to_poi_id: str = Field(pattern=r"^poi_[a-z0-9_]+$")
    mode: str = Field(pattern=r"^(walk|ferry|transit|drive)$")
    duration_minutes: float = Field(ge=0)
    distance_km: Optional[float] = Field(default=None, ge=0)
    is_estimated: bool
    side_crossing: bool


class DailyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    date: date
    side: str = Field(pattern=r"^(european|asian|mixed)$")
    poi_ids: list[str]
    legs: list[TravelLeg]
    walking_minutes: float = Field(ge=0)
    transfer_minutes: float = Field(ge=0)
    activity_minutes: float = Field(ge=0)
    meal_minutes: float = Field(ge=0)
    slack_minutes: float = Field(ge=0)
    warnings: list[str]


class StayCandidateRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class LocalPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    session_id: UUID
    trace_id: UUID
    contract_version: str = SCHEMA_VERSION
    trip_start_date: date
    trip_end_date: date
    available_local_time_minutes: Optional[int] = Field(default=None, ge=0)
    interests: list[str] = Field(min_length=1)
    pace: Pace
    language: Language
    mobility_constraints: list[str]
    daily_activity_budget_minutes: int = Field(ge=0)
    stay_candidates: list[StayCandidateRef] = Field(min_length=1, max_length=3)
    weather_context: Optional[dict] = None
    hard_constraints: list[str]
    soft_constraints: list[str]


class AccessibilityScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class CandidateProvenanceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    poi_id: str = Field(pattern=r"^poi_[a-z0-9_]+$")
    candidate_origin: str = Field(pattern=r"^(rag|catalog_fallback)$")
    display_name: Optional[str] = Field(default=None, min_length=1)
    matched_interests: Optional[list[str]] = None
    source_id: Optional[str] = Field(default=None, min_length=1)
    chunk_id: Optional[str] = Field(default=None, min_length=1)
    retrieval_query: Optional[str] = Field(default=None, min_length=1)
    retrieval_score: Optional[float] = None


class AccessibilityEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requested: bool
    status: str = Field(pattern=r"^(satisfied|evidence_available_not_candidate_verified|unsupported)$")
    evidence_count: int = Field(ge=0)
    note: str = Field(min_length=1)


class LocalItinerary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    session_id: UUID
    trace_id: UUID
    contract_version: str = SCHEMA_VERSION
    recommended_base_candidate_id: str = Field(min_length=1)
    accessibility_scores: list[AccessibilityScore]
    selected_poi_ids: list[str]
    daily_plans: list[DailyPlan] = Field(min_length=1)
    estimated_travel_minutes: float = Field(ge=0)
    expected_walking_minutes: float = Field(ge=0)
    side_crossings: int = Field(ge=0)
    weather_substitutions: Optional[list[str]] = None
    citations: list[SourceReference]
    assumptions: list[str]
    warnings: list[str]
    data_quality: DataQuality
    hard_constraint_validation_passed: bool
    candidate_provenance: Optional[list[CandidateProvenanceEntry]] = None
    uncovered_interests: Optional[list[str]] = None
    accessibility_evaluation: Optional[AccessibilityEvaluation] = None


class FxSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(min_length=1)
    rate: float = Field(gt=0)
    timestamp: datetime
    inverse_rate_method: str = Field(min_length=1)


class BudgetBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    known_costs_minor_units: int = Field(ge=0)
    estimated_costs_minor_units: int = Field(ge=0)
    contingency_minor_units: int = Field(ge=0)
    total_minor_units: int = Field(ge=0)
    fx_snapshot: Optional[FxSnapshot] = None


class RankedRelaxation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraint: str = Field(min_length=1)
    description: str = Field(min_length=1)
    impact: str = Field(min_length=1)


class ConstraintConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    session_id: UUID
    trace_id: UUID
    minimum_budget_gap: Money
    ranked_relaxations: list[RankedRelaxation] = Field(min_length=1)


class FailureComponent(str, Enum):
    FLIGHT_PROVIDER = "flight_provider"
    STAY_SOURCE = "stay_source"
    SYSTEM_B_A2A = "system_b_a2a"
    QDRANT = "qdrant"
    ROUTE_PROVIDER = "route_provider"
    WEATHER = "weather"
    OPERATIONAL_STATUS = "operational_status"
    ML_ARTIFACT = "ml_artifact"


class PartialFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    trace_id: UUID
    component: FailureComponent
    degraded_behavior: str = Field(min_length=1)


class TripPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    session_id: UUID
    trace_id: UUID
    created_at: datetime
    flights: list[FlightOption]
    stays: list[StayOption]
    itinerary: Optional[LocalItinerary] = None
    budget: BudgetBreakdown
    warnings: list[str]
    partial_failures: list[PartialFailure]
    data_quality: DataQuality


class StreamStage(str, Enum):
    REQUEST_ACCEPTED = "request.accepted"
    SEARCH_STARTED = "search.started"
    SEARCH_COMPLETED = "search.completed"
    LOCAL_PLAN_STARTED = "local_plan.started"
    LOCAL_PLAN_COMPLETED = "local_plan.completed"
    BUDGET_VALIDATED = "budget.validated"
    WARNING = "warning"
    PARTIAL_FAILURE = "partial_failure"
    PLAN_COMPLETED = "plan.completed"
    ERROR = "error"
    # Checkpoint Phase 4 D.2A additive values (contracts/StreamEvent.schema.json
    # 1.1.0): the original ten values above describe architecture.md
    # §5.3's linear pipeline; System A's real shape since Checkpoint D.0
    # is the bounded ReAct action loop (ADR 0009 §4), which this
    # checkpoint's public SSE endpoint streams sanitized progress for
    # instead. Purely additive -- every pre-existing 1.0.0 StreamEvent
    # instance using only the original ten values remains valid
    # unchanged (docs/adr/0015-phase4-checkpoint-d2a-system-a-api.md).
    RUN_STARTED = "run_started"
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"
    RUN_COMPLETED = "run_completed"
    RUN_DEGRADED = "run_degraded"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class StreamEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    event_id: UUID
    session_id: UUID
    trace_id: UUID
    sequence: int = Field(ge=0)
    timestamp: datetime
    stage: StreamStage
    payload: dict
