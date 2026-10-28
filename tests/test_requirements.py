"""requirements.txt enthält jedes Fremdpaket, das die App direkt importiert.

Anlass (28.10.2026): core/tools.py importierte httpx, das nur über chromadb mitkam. Nach dem
Entfernen von Chroma lief lokal alles weiter (httpx war noch installiert), online fehlte es.
"""

import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_FILES = [
    ROOT / "app.py",
    *(ROOT / "core").glob("*.py"),
    *(ROOT / "app_pages").glob("*.py"),
]
LOCAL = {"core", "app_pages", "tests"}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def imported_modules() -> set[str]:
    names = set()
    for path in APP_FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names) - LOCAL


def required_distributions() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    names = [re.split(r"[<>=!~#\[ ]", line.strip())[0] for line in lines]
    return {normalize(n) for n in names if n}


def test_every_third_party_import_is_in_requirements():
    distributions = packages_distributions()
    required = required_distributions()
    missing = {
        module
        for module in imported_modules()
        if not {normalize(d) for d in distributions.get(module, [module])} & required
    }
    assert not missing, f"In requirements.txt fehlt: {sorted(missing)}"
