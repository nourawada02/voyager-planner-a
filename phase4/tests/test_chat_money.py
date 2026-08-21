"""Hermetic unit tests for `phase4.chat_money` -- the Decimal-exact
money-normalization module added to close the 100x display-regression
found during Hybrid Chat C.1 live testing."""

from __future__ import annotations

from phase4.chat_money import describe_money, describe_money_dict, whole_units_to_minor_units


def test_describe_money_500000_try_formats_as_5000():
    result = describe_money(500000, "TRY")
    assert result["major_units"] == "5000.00"
    assert result["formatted"] == "5,000.00 TRY"
    assert result["amount_minor_units"] == 500000
    assert result["currency"] == "TRY"


def test_describe_money_1685800_try_formats_as_16858_not_1685800():
    """The exact regression this module exists to close: a live test saw
    the assistant describe 1,685,800 minor units as '1,685,800 TRY'
    instead of the correct 16,858.00 TRY."""
    result = describe_money(1685800, "TRY")
    assert result["major_units"] == "16858.00"
    assert result["formatted"] == "16,858.00 TRY"


def test_describe_money_100000_usd_formats_as_1000():
    result = describe_money(100000, "USD")
    assert result["major_units"] == "1000.00"
    assert result["formatted"] == "1,000.00 USD"


def test_describe_money_lowercases_currency_is_uppercased():
    result = describe_money(100, "try")
    assert result["currency"] == "TRY"


def test_describe_money_none_amount_returns_none():
    assert describe_money(None, "TRY") is None


def test_describe_money_none_currency_returns_none():
    assert describe_money(100, None) is None


def test_describe_money_non_numeric_amount_returns_none():
    assert describe_money("not-a-number", "TRY") is None


def test_describe_money_dict_wraps_a_money_shaped_dict():
    result = describe_money_dict({"amount_minor_units": 250000, "currency": "TRY"})
    assert result["formatted"] == "2,500.00 TRY"


def test_describe_money_dict_none_for_non_dict():
    assert describe_money_dict(None) is None
    assert describe_money_dict("not a dict") is None


def test_describe_money_dict_none_for_missing_fields():
    assert describe_money_dict({"currency": "TRY"}) is None


def test_whole_units_to_minor_units_exact_conversion():
    assert whole_units_to_minor_units("1000") == 100000
    assert whole_units_to_minor_units(1000) == 100000
    assert whole_units_to_minor_units(1000.5) == 100050


def test_whole_units_to_minor_units_non_numeric_returns_none():
    assert whole_units_to_minor_units("not a number") is None
    assert whole_units_to_minor_units(None) is None
