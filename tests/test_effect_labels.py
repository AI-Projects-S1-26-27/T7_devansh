"""Frozen effect-label sets: tracked config and hash-checked loader (OQ-M2-1)."""

import gzip
import hashlib
import json
import re

import pytest

from ahg import effect as ef

ROWS = [{"cve_id": "CVE-1", "label": "RCE_CONFIRMED"}, {"cve_id": "CVE-2", "label": "NON_ADMIN"}]


def _setup(tmp_path, rows=ROWS, sha=None):
    blob = gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode())
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "v.jsonl.gz").write_bytes(blob)
    cfg = tmp_path / "labels.json"
    cfg.write_text(json.dumps({"versions": {"v": {"path": "data/v.jsonl.gz",
                                                  "sha256": sha or hashlib.sha256(blob).hexdigest()}}}))
    return cfg


def test_matching_hash_loads_rows(tmp_path):
    assert ef.load_frozen_labels("v", config=_setup(tmp_path), root=tmp_path) == ROWS


def test_changed_file_is_refused(tmp_path):
    cfg = _setup(tmp_path)
    other = gzip.compress(json.dumps({"cve_id": "CVE-1", "label": "UNKNOWN"}).encode() + b"\n")
    (tmp_path / "data" / "v.jsonl.gz").write_bytes(other)
    with pytest.raises(ValueError, match="does not match"):
        ef.load_frozen_labels("v", config=cfg, root=tmp_path)


def test_wrong_recorded_hash_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        ef.load_frozen_labels("v", config=_setup(tmp_path, sha="0" * 64), root=tmp_path)


def test_unknown_version_raises(tmp_path):
    with pytest.raises(KeyError):
        ef.load_frozen_labels("v9", config=_setup(tmp_path), root=tmp_path)


def test_tracked_config_lists_the_three_frozen_versions():
    versions = json.loads(ef.LABELS_CONFIG.read_text())["versions"]
    assert set(versions) == {"p3-labels-v1", "p3-labels-v2", "p3-labels-v2.1"}
    for v, e in versions.items():
        assert e["path"] == f"data/d28/effect/{v}/{v}.jsonl.gz"  # untracked under data/, never results/
        assert re.fullmatch(r"[0-9a-f]{64}", e["sha256"])
    assert len({e["sha256"] for e in versions.values()}) == 3


@pytest.mark.parametrize("version", ["p3-labels-v1", "p3-labels-v2", "p3-labels-v2.1"])
def test_local_copy_matches_config(version):
    entry = json.loads(ef.LABELS_CONFIG.read_text())["versions"][version]
    if not (ef.ROOT / entry["path"]).exists():
        pytest.skip("frozen label files are untracked (data/, Zenodo); not present in this checkout")
    rows = ef.load_frozen_labels(version)
    assert rows and all("cve_id" in r and "label" in r for r in rows)
