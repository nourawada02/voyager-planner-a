"""Shared decision-prompt argument-contract generation (Checkpoint Phase
4 D.0, extracted to its own module in the D.3 correction pass so the
supervisor's own prompt builder (`phase4/graph.py`) and the internal
Travel Search specialist's own prompt builder (`phase4/specialist.py`)
generate their per-action JSON-Schema contract from exactly the same
canonical `ACTION_ARGUMENT_MODELS` registry -- never two independent
copies that could silently drift apart.
"""

from __future__ import annotations

from typing import Any, Optional

from phase4.models import ACTION_ARGUMENT_MODELS


def strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: strip_titles(value) for key, value in node.items() if key != "title"}
    if isinstance(node, list):
        return [strip_titles(item) for item in node]
    return node


def action_argument_contract(allowed_actions: Optional[frozenset] = None) -> dict[str, Any]:
    """The prompt's per-action argument contract, generated directly from
    `ACTION_ARGUMENT_MODELS` -- the exact same canonical registry
    `parse_action_decision` validates a decision's arguments against.
    Each entry is that action's own `BaseModel.model_json_schema()`
    (Pydantic's own JSON Schema, carrying `required`, enum/pattern/
    format/min/max constraints, and `additionalProperties: false`
    automatically from `ConfigDict(extra="forbid")`) -- with only the
    purely-cosmetic `title` keys stripped to keep the prompt compact.
    Stable action ordering (`Action`'s own declared enum order) makes
    this deterministic across calls.

    `allowed_actions` restricts the contract to exactly that closed
    subset -- the supervisor's own prompt never even shows the
    specialist-only tool schemas as an option once `SUPERVISOR_ACTIONS`
    is passed, and vice versa; `None` returns every action's contract."""
    models = ACTION_ARGUMENT_MODELS if allowed_actions is None else {
        action: model for action, model in ACTION_ARGUMENT_MODELS.items() if action in allowed_actions
    }
    return {action.value: strip_titles(model.model_json_schema()) for action, model in models.items()}
