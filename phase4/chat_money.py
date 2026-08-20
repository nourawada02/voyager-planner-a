"""Decimal-exact money normalization for Hybrid Chat C.1's grounding
context (persistent-history/grounding correction). A live test found the
chat assistant describing a 1,685,800-minor-units TRY amount as
"1,685,800 TRY" -- a 100x error, because the bounded summary handed Qwen
only the bare `amount_minor_units` integer with no explicit statement of
what unit it was in. This module is the one place that turns a raw
Money-shaped dict into an explicit, self-describing structure before it
ever reaches a prompt, mirroring `providers/money.py`'s own
Decimal/ROUND_HALF_UP discipline -- reimplemented fresh, not imported
(this submodule never depends on the root `providers` package, ADR 0009
§6.1)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Optional


def describe_money(amount_minor_units: Any, currency: Any) -> Optional[dict[str, Any]]:
    """Returns the explicit, safe structure Hybrid Chat C.1 §8 requires:
    `amount_minor_units`, ISO `currency`, a normalized major-unit decimal
    string, and a human-formatted display string. Returns None -- never a
    fabricated value -- when either input is missing or not a genuine
    integer amount."""
    if amount_minor_units is None or not currency:
        return None
    try:
        minor = int(amount_minor_units)
    except (TypeError, ValueError):
        return None
    currency_code = str(currency).upper()
    major = (Decimal(minor) / Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "amount_minor_units": minor,
        "currency": currency_code,
        "major_units": str(major),
        "formatted": f"{major:,} {currency_code}",
    }


def describe_money_dict(money: Any) -> Optional[dict[str, Any]]:
    """Convenience wrapper over a raw `{"amount_minor_units": ..., "currency": ...}`
    dict (the shape every Money field in this project already uses)."""
    if not isinstance(money, dict):
        return None
    return describe_money(money.get("amount_minor_units"), money.get("currency"))


def whole_units_to_minor_units(amount: Any) -> Optional[int]:
    """Converts a whole-currency-unit amount (e.g. a user-stated '1000
    USD', already parsed to a plain number by the caller) into integer
    minor units exactly once, Decimal-exact, ROUND_HALF_UP. Returns None
    for a non-numeric input rather than raising -- the caller treats that
    as an unresolvable patch value, never a guess."""
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return int((value * 100).to_integral_value(rounding=ROUND_HALF_UP))
