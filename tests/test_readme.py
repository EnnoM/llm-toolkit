"""README-Modultabelle (core/readme.py)."""

import pytest

from core.modules import Module, load_modules
from core.readme import END, README_FILE, START, render_table, update_readme


def test_readme_table_matches_modules_yaml():
    text = README_FILE.read_text(encoding="utf-8")
    assert update_readme(text, load_modules()) == text, (
        "README veraltet: python -m core.readme ausführen"
    )


def test_pipe_in_short_text_does_not_break_the_table():
    module = load_modules()[0].model_copy(update={"kurz": "A | B"})
    assert "A \\| B" in render_table([module])


def test_missing_markers_raise_clear_error():
    with pytest.raises(ValueError, match="Markierungen"):
        update_readme("# README ohne Tabelle", [])


def test_only_the_marked_block_changes():
    text = f"Vorher\n{START}\nalt\n{END}\nNachher"
    modules: list[Module] = load_modules()[:1]
    result = update_readme(text, modules)
    assert result.startswith(f"Vorher\n{START}\n| Modul |")
    assert result.endswith(f"{END}\nNachher")
    assert "alt" not in result
