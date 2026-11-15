"""Fetch scripts: no machine-specific paths (inventory #6) and correct query planning."""

import importlib.util
import re

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_hard_coded_user_or_scratchpad_paths():
    """Inventory #6: scripts resolve paths from the repo root, never from one machine's home or temp dirs."""
    bad = re.compile(r"/Users/|/home/|/private/tmp|scratchpad/")
    offenders = [p.name for p in SCRIPTS.glob("*.py") if bad.search(p.read_text())]
    assert offenders == []


def test_fetch_script_docstrings_name_the_script():
    """Inventory #6: the usage line names the file it is in, and the unimplemented "0 tried first" claim is gone."""
    for name in ("nvd_count_only", "fetch_sources"):
        doc = load(name).__doc__
        assert doc.lstrip().startswith(f"{name}.py"), doc[:40]
        assert "tried first" not in doc
