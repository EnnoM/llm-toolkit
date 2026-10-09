"""Erzeugt die Modultabelle im README aus docs/modules.yaml.

Aufruf: python -m core.readme (läuft auch automatisch als pre-commit-Hook).
Rückgabewert 1 heißt: README wurde geändert und muss erneut mit git add hinzugefügt werden.
"""

import sys
from pathlib import Path

from core.modules import STATUS_LABELS, Module, ModulesError, load_modules

README_FILE = Path(__file__).resolve().parent.parent / "README.md"
START = (
    "<!-- modules:start (automatisch aus docs/modules.yaml, nicht von Hand ändern) -->"
)
END = "<!-- modules:end -->"


def render_table(modules: list[Module]) -> str:
    """Markdown-Tabelle mit Modul, Phase, Status und Kurzbeschreibung."""
    lines = ["| Modul | Phase | Status | Worum es geht |", "|---|---|---|---|"]
    for m in modules:
        # Ein senkrechter Strich im Text würde die Tabelle zerteilen
        kurz = m.kurz.replace("|", "\\|")
        lines.append(f"| {m.name} | {m.phase} | {STATUS_LABELS[m.status]} | {kurz} |")
    return "\n".join(lines)


def update_readme(text: str, modules: list[Module]) -> str:
    """Ersetzt den Bereich zwischen den beiden Markierungen durch die aktuelle Tabelle."""
    before, found_start, rest = text.partition(START)
    _, found_end, after = rest.partition(END)
    if not (found_start and found_end):
        raise ValueError(f"README.md braucht die Markierungen {START} und {END}.")
    return f"{before}{START}\n{render_table(modules)}\n{END}{after}"


def main() -> int:
    try:
        text = README_FILE.read_text(encoding="utf-8")
        new_text = update_readme(text, load_modules())
    except (ModulesError, ValueError) as err:
        print(f"README nicht aktualisiert: {err}")
        return 1
    if new_text == text:
        return 0
    README_FILE.write_text(new_text, encoding="utf-8")
    print(
        "README.md: Modultabelle aktualisiert. Bitte erneut hinzufügen: git add README.md"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
