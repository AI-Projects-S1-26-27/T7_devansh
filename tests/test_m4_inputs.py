"""M4: after the d28-fixtures, d28-gen and d28-pipe merges every configured input resolves inside this checkout.

A config path that starts with "../" reads a sibling track worktree (the OQ-M3-1 defect): the run then depends on
a folder outside the repository and its manifest. Descriptive provenance text (e.g. "clean run in ../ahg-fixtures")
is not a path and is not checked."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def strings(o, key=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from strings(v, f"{key}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from strings(v, f"{key}[{i}]")
    elif isinstance(o, str):
        yield key, o


def test_no_config_reads_a_path_outside_the_checkout():
    bad = [(str(p.relative_to(ROOT)), k, v) for p in sorted((ROOT / "configs").rglob("*.json"))
           for k, v in strings(json.loads(p.read_text())) if v.startswith("../") or v.startswith("/")]
    assert not bad, bad


def test_effect_config_reads_the_master_dataset_and_ghsa_of_this_checkout():
    cfg = json.loads((ROOT / "configs" / "effect" / "p3.json").read_text())
    assert cfg["master_dataset"] == "data/d28/master/2026-11-15T124455Z-cf2dc62/cves.jsonl.gz"
    assert cfg["ghsa_folder"] == "data/raw/ghsa/2026-11-15"
    gen = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())
    assert gen["master"]["path"] == cfg["master_dataset"]  # the generator pins the same master dataset
