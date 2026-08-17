"""Typed execution context passed to a `ToolExecutor` (Checkpoint Phase 4
D.1 additive protocol change). Exists so a real tool binding (D.1) can
map an action like `estimate_fair_price`/`call_istanbul_expert` using
already-validated prior observations and the normalized request, without
asking Qwen to reproduce/invent a stay candidate id, coordinates, or
other model output it never actually saw. Optional and backward
compatible: `FakeToolExecutor.execute()` (Checkpoint D.0) accepts and
ignores it, so every existing hermetic test keeps working unchanged.

Contains only operational data -- never a credential, a raw prompt/
response, chain-of-thought, an arbitrary URL, or a stack trace. Frozen
and built fresh per Execute-node call from already-validated state, never
mutated afterward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ExecutionContext:
    session_id: str
    trace_id: str
    normalized_request: dict[str, Any]
    observations: tuple[dict[str, Any], ...]
    deadline_monotonic: float
    cancellation_check: Callable[[], bool]

    def observations_for_action(self, action_value: str) -> tuple[dict[str, Any], ...]:
        """Validated prior observations for exactly one action, most
        recent last -- the shape a real binding needs to, e.g., pick a
        `stay_id` from the last successful `search_stays` result without
        asking Qwen to restate it."""
        return tuple(obs for obs in self.observations if obs.get("action") == action_value)
