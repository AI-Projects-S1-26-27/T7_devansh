"""D28 run folders and frozen effect labels are never committable (OQ-M2-1, OQ-M2-2)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", [
    "results/d28_data/cve_master/2026-11-15T000000Z-abc/manifest.json",
    "results/d28_data/cve_master/2026-11-15T000000Z-abc/report.md",  # the *.md re-include must not reach it
    "results/d28_effect/2026-11-15T000000Z-abc/work/pool.json",
    "data/d28/effect/p3-labels-v2.1/p3-labels-v2.1.jsonl.gz",
])
def test_d28_outputs_are_ignored(path):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    # check-ignore exits 0 when the path is ignored; --no-index so a tracked file would not mask the rule
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not ignored"
