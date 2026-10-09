"""core/catalog.py: passt zu docs/modules.yaml und zu den Seiten in app_pages/."""

from pathlib import Path

from core.catalog import CATALOG
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
