"""core/catalog.py: passt zu docs/modules.yaml und zu den Seiten in app_pages/."""

from pathlib import Path

from core.catalog import CATALOG, LEVELS, levels_newest_first
from core.modules import load_modules

ROOT = Path(__file__).resolve().parent.parent


def test_catalog_matches_modules_yaml():
    # Gleiche ids in gleicher Reihenfolge: neue Module dürfen auf Home nicht fehlen
    assert [entry.id for entry in CATALOG] == [m.id for m in load_modules()]


def test_only_planned_modules_are_wip():
    status = {m.id: m.status for m in load_modules()}
    for entry in CATALOG:
        assert (entry.page is None) == (status[entry.id] == "geplant"), entry.id


def test_pages_exist_and_icons_are_material_symbols():
    for entry in CATALOG:
        assert entry.icon.startswith(":material/"), entry.id
        if entry.page:
            assert (ROOT / entry.page).is_file(), entry.page


def test_level_follows_the_phase():
    phase = {m.id: m.phase for m in load_modules()}
    for entry in CATALOG:
        assert entry.level == phase[entry.id] - 1, entry.id
        assert entry.level in LEVELS


def test_home_shows_the_newest_built_module_first():
    groups = levels_newest_first()
    assert sorted(e.id for _, entries in groups for e in entries) == sorted(
        e.id for e in CATALOG
    )
    newest = [e for e in CATALOG if e.page][-1]
    assert groups[0][1][0] == newest
    # Stufen mit gebauten Modulen vor rein geplanten Stufen
    started = [bool(entries[0].page) for _, entries in groups]
    assert started == sorted(started, reverse=True)
