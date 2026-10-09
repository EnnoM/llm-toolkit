"""Modellauswahl und Kostenanzeige, von allen Modulen gemeinsam genutzt. Keine Oberfläche."""

from core.llm import Usage

# Auswahl in den Modulen: drei günstige Modelle und eins zum Vergleich, das rund zehnmal mehr
# kostet. Preise in USD pro Million Tokens (Eingabe, Ausgabe), Stand 18.09.2026 laut
# openrouter.ai/models. Bei Mistral hängt der Preis vom Anbieter ab; angegeben ist der
# günstigste (DeepInfra). Die tatsächlichen Kosten jeder Antwort kommen aus dem usage-Feld.
MODELS = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "mistralai/mistral-small-3.2-24b-instruct": (0.075, 0.20),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
}


def model_options(default: str | None) -> list[str]:
    """Modelle für die Auswahl, nach Preis sortiert (günstigstes zuerst).

    Ein anderes Standardmodell aus OPENROUTER_MODEL ohne bekannten Preis kommt ans Ende.
    """
    # Sortiert nach (Eingabepreis, Ausgabepreis); bei Gleichstand bleibt die Reihenfolge oben
    options = sorted(MODELS, key=lambda model_id: MODELS[model_id])
    if default and default not in MODELS:
        options.append(default)
    return options


def format_price(usd_per_million: float) -> str:
    """0.1 -> '$0.10', 0.075 -> '$0.075', 5 -> '$5.00'."""
    if round(usd_per_million, 2) == usd_per_million:
        return f"${usd_per_million:.2f}"
    return f"${usd_per_million:.3f}"


def model_name(model_id: str) -> str:
    """Anzeigename ohne Anbieter-Präfix: 'google/gemini-2.5-flash-lite' -> 'gemini-2.5-flash-lite'."""
    return model_id.split("/", 1)[-1]


def model_price(model_id: str) -> str | None:
    """Preiszeile für die Auswahl, z. B. '$0.10 in · $0.40 out' (USD pro Million Tokens)."""
    if model_id not in MODELS:
        return None  # z. B. ein anderes Standardmodell aus OPENROUTER_MODEL
    price_in, price_out = MODELS[model_id]
    return f"{format_price(price_in)} in · {format_price(price_out)} out"


def format_cost(cost_usd: float, unit: bool = True) -> str:
    """Kosten mit sechs Nachkommastellen, weil einzelne Antworten Bruchteile eines Cents kosten.

    Englisches Zahlenformat (Dezimalpunkt), weil die Module englisch sind.
    """
    amount = f"{cost_usd:.6f}"
    return f"{amount} USD" if unit else amount


def format_usage(usage: Usage) -> str:
    """Eine Zeile mit Tokens und Kosten, z. B. für unter eine Antwort."""
    return (
        f"Input {usage.prompt_tokens} tokens · Output {usage.completion_tokens} tokens"
        f" · {format_cost(usage.cost_usd)}"
    )
