"""Arbeits-Checkliste: lädt und prüft docs/modules.yaml. Keine Oberfläche."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

MODULES_FILE = Path(__file__).resolve().parent.parent / "docs" / "modules.yaml"

Status = Literal["geplant", "in_arbeit", "fertig"]
STATUS_LABELS = {"geplant": "geplant", "in_arbeit": "in Arbeit", "fertig": "fertig"}
NOT_STARTED = "noch nicht begonnen"
DONE = "abgeschlossen"
NO_NEXT_STEP = "-"

# Text, der ohne umgebende Leerzeichen nicht leer ist
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ModulesError(Exception):
    """docs/modules.yaml fehlt, ist kein gültiges YAML oder passt nicht zum Schema."""


class Module(BaseModel):
    """Ein Eintrag der Checkliste."""

    # Unbekannte Felder verbieten: Tippfehler in Feldnamen fallen sofort auf
    model_config = ConfigDict(extra="forbid")

    id: Text
    name: Text
    phase: int = Field(ge=1, le=6)
    status: Status
    kurz: Text
    beschreibung: Text
    oberflaeche: Text
    lernziele: list[Text] = Field(min_length=3, max_length=5)
    akzeptanz: list[Text] = Field(min_length=3, max_length=5)
    arbeitsstand: Text
    naechster_schritt: Text
    notizen: str  # Pflichtfeld, darf aber leer sein


def load_modules(path: Path = MODULES_FILE) -> list[Module]:
    """Liest die YAML, prüft jedes Modul und liefert die Module nach Phase sortiert."""
    try:
        # safe_load erzeugt nur einfache Datentypen, nie beliebige Python-Objekte
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ModulesError(f"Datei nicht gefunden: {path}") from None
    except yaml.YAMLError as err:
        mark = getattr(err, "problem_mark", None)
        where = f" in Zeile {mark.line + 1}" if mark else ""
        problem = getattr(err, "problem", None) or err
        raise ModulesError(f"YAML-Syntaxfehler{where}: {problem}") from None

    if not isinstance(raw, list) or not raw:
        raise ModulesError(
            "Die Datei muss eine Liste von Modulen sein (je Modul '- id: ...')."
        )

    modules, problems = [], []
    for number, entry in enumerate(raw, start=1):
        label = entry.get("id") if isinstance(entry, dict) else None
        try:
            modules.append(Module.model_validate(entry))
        except ValidationError as err:
            for error in err.errors():
                field = (
                    ".".join(str(part) for part in error["loc"]) or "(ganzer Eintrag)"
                )
                problems.append(
                    f"Modul '{label or number}', Feld '{field}': {error['msg']}"
                )
    if problems:
        raise ModulesError("\n".join(problems))

    ids = [module.id for module in modules]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ModulesError(f"Doppelte id: {', '.join(duplicates)}")

    # sorted ist stabil: innerhalb einer Phase bleibt die Reihenfolge der YAML erhalten
    return sorted(modules, key=lambda module: module.phase)


def status_problem(module: Module) -> str | None:
    """Prüft die Regeln für arbeitsstand und naechster_schritt (siehe Kopf der YAML)."""
    stand, step = module.arbeitsstand, module.naechster_schritt
    if module.status == "geplant":
        ok = stand == NOT_STARTED and step == NOT_STARTED
        rule = f"arbeitsstand und naechster_schritt müssen '{NOT_STARTED}' sein"
    elif module.status == "fertig":
        ok = stand == DONE and step == NO_NEXT_STEP
        rule = f"arbeitsstand muss '{DONE}' und naechster_schritt '{NO_NEXT_STEP}' sein"
    else:  # in_arbeit
        placeholders = {NOT_STARTED, DONE, NO_NEXT_STEP}
        ok = stand not in placeholders and step not in placeholders and "\n" not in step
        rule = (
            "arbeitsstand und naechster_schritt brauchen echten Inhalt, "
            "naechster_schritt ist genau eine Aufgabe in einer Zeile"
        )
    return None if ok else f"Status '{module.status}': {rule}."


def filter_modules(
    modules: list[Module], phases: list[int], statuses: list[str]
) -> list[Module]:
    """Behält nur Module, deren Phase und Status ausgewählt sind."""
    return [m for m in modules if m.phase in phases and m.status in statuses]


def count_done(modules: list[Module]) -> int:
    """Anzahl der fertigen Module."""
    return sum(module.status == "fertig" for module in modules)
