"""Modellauswahl und Kostenanzeige (core/models.py)."""

from core.models import (
    MODELS,
    format_cost,
    format_price,
    model_name,
    model_options,
    model_price,
)


def test_format_cost_shows_fractions_of_a_cent():
    assert format_cost(0.0000123) == "0.000012 USD"
    assert format_cost(0) == "0.000000 USD"
    assert format_cost(0.0000123, unit=False) == "0.000012"


def test_prices_are_shown_like_on_openrouter():
    assert format_price(0.10) == "$0.10"
    assert format_price(0.075) == "$0.075"
    assert format_price(5) == "$5.00"
    assert model_name("anthropic/claude-haiku-4.5") == "claude-haiku-4.5"
    assert model_price("anthropic/claude-haiku-4.5") == "$1.00 in · $5.00 out"
    assert model_price("anderes/modell") is None  # ohne bekannten Preis


def test_model_options_are_sorted_by_price():
    options = model_options(None)
    prices = [MODELS[m] for m in options]
    assert prices == sorted(prices)
    assert (
        options[0] == "mistralai/mistral-small-3.2-24b-instruct"
    )  # günstigstes zuerst
    # Das Standardmodell ändert die Reihenfolge nicht, es wird nur vorausgewählt
    assert model_options("google/gemini-2.5-flash-lite") == options
    # Ein unbekanntes Standardmodell (ohne Preis) kommt ans Ende
    assert model_options("anderes/modell") == [*options, "anderes/modell"]
