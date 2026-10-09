"""Tests für die Arbeits-Checkliste docs/modules.yaml. Keine API-Aufrufe."""

import pytest
import yaml

from core.modules import Module, ModulesError, load_modules, status_problem

EXPECTED_IDS = {
    "chat",
    "prompt_labor",
    "datenextraktion",
    "rag_chatbot",
    "tool_use",
    "ki_datenanalyst",
    "workflow_freigabe",
    "mcp_server",
    "agent",
    "evaluation",
    "security",
}


def module_data(**changes) -> dict:
    """Minimal gültiges Modul als dict, einzelne Felder überschreibbar."""
    data = {
        "id": "beispiel",
        "name": "Beispiel",
        "phase": 2,
        "status": "geplant",
        "kurz": "Kurz.",
        "beschreibung": "Beschreibung.",
        "oberflaeche": "Oberfläche.",
        "lernziele": ["a", "b", "c"],
        "akzeptanz": ["a", "b", "c"],
        "arbeitsstand": "noch nicht begonnen",
        "naechster_schritt": "noch nicht begonnen",
        "notizen": "",
    }
    return data | changes


def test_yaml_is_valid_and_has_all_modules():
    # load_modules prüft jedes Pflichtfeld per Pydantic und wirft sonst ModulesError
    modules = load_modules()
    assert {m.id for m in modules} == EXPECTED_IDS


def test_status_rules_are_kept():
    problems = {m.id: status_problem(m) for m in load_modules() if status_problem(m)}
    assert problems == {}


@pytest.mark.parametrize(
    ("changes", "valid"),
    [
        ({}, True),
        ({"arbeitsstand": "halb fertig"}, False),
        ({"status": "in_arbeit"}, False),
        (
            {
                "status": "in_arbeit",
                "arbeitsstand": "Seite steht",
                "naechster_schritt": "Tests",
            },
            True,
        ),
        (
            {
                "status": "in_arbeit",
                "arbeitsstand": "Seite steht",
                "naechster_schritt": "a\nb",
            },
            False,
        ),
        (
            {
                "status": "fertig",
                "arbeitsstand": "abgeschlossen",
                "naechster_schritt": "-",
            },
            True,
        ),
        (
            {
                "status": "fertig",
                "arbeitsstand": "abgeschlossen",
                "naechster_schritt": "Doku",
            },
            False,
        ),
    ],
)
def test_status_rule_check_detects_violations(changes, valid):
    module = Module.model_validate(module_data(**changes))
    assert (status_problem(module) is None) is valid


def test_broken_yaml_names_the_line(tmp_path):
    path = tmp_path / "modules.yaml"
    path.write_text("- id: chat\n  name: [kaputt\n", encoding="utf-8")
    with pytest.raises(ModulesError, match="Zeile"):
        load_modules(path)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (module_data(id="chat", status="erledigt"), "Modul 'chat', Feld 'status'"),
        (
            {k: v for k, v in module_data(id="chat").items() if k != "akzeptanz"},
            "Modul 'chat', Feld 'akzeptanz'",
        ),
    ],
)
def test_invalid_module_names_module_and_field(tmp_path, entry, expected):
    path = tmp_path / "modules.yaml"
    path.write_text(yaml.safe_dump([entry], allow_unicode=True), encoding="utf-8")
    with pytest.raises(ModulesError, match=expected):
        load_modules(path)
