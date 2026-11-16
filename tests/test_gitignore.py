"""Commit policy in .gitignore (OQ-M2-1, OQ-M2-2, OQ-M2-4): nothing new under results/ or data/ is committable,
except the test-access log; README, LICENSE, NOTICE, the .gitkeep files and the existing data/derived tables stay."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")


def _ignored(path: str) -> bool:
    # check-ignore exits 0 when the path is ignored; --no-index so a tracked file does not mask the rule
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT).returncode == 0


@pytest.mark.parametrize("path", [
    "results/d28_data/cve_master/2026-11-15T000000Z-abc/manifest.json",
    "results/d28_data/cve_master/2026-11-15T000000Z-abc/report.md",
    "results/d28_effect/2026-11-15T000000Z-abc/work/pool.json",
    "data/d28/effect/p3-labels-v2.1/p3-labels-v2.1.jsonl.gz",
    # old-study folders: new files are ignored (OQ-M2-4); D28 code writes runs into the first three
    "results/training/onpolicy/run/manifest.json",
    "results/training/grid/run/report.md",
    "results/evaluation/run/raw.json",
    "results/landscape/run/report.md",
    "results/feasibility/run/raw.json",
    "results/analysis/run/raw.json",
    "results/cve_semantics/run/raw.json",
    "results/lit_search/run/raw.json",
    "results/raw/new.json",
    "results/new.json",
    "data/derived/new_table.csv",  # only the existing data/derived tables are committed
])
def test_ignored(path):
    assert _ignored(path), f"{path} is not ignored"


@pytest.mark.parametrize("path", ["results/test_access_log.jsonl", "README.md", "LICENSE", "NOTICE"])
def test_committable(path):
    assert not _ignored(path), f"{path} is ignored"


def test_existing_gitkeeps_and_derived_tables_stay_tracked():
    tracked = subprocess.run(["git", "ls-files", "results", "data"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    assert {"results/raw/.gitkeep", "results/landscape/.gitkeep", "results/test_access_log.jsonl"} <= set(tracked)
    assert any(t.startswith("data/derived/") for t in tracked)
