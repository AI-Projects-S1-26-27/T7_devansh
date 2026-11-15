"""scripts/build_cve_master.py helpers: hash-verified raw inputs and the MSRC entry choice."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_cve_master.py"
spec = importlib.util.spec_from_file_location("build_cve_master", SCRIPT)
bcm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcm)


def test_verified_files_reads_parent_fetch_log_and_detects_edits(tmp_path):
    (tmp_path / "msrc").mkdir()
    (tmp_path / "msrc" / "cvrf_2020-Jan.json").write_bytes(b"{}")
    (tmp_path / "other.html").write_bytes(b"x")
    log = [{"name": "msrc/cvrf_2020-Jan.json", "sha256": hashlib.sha256(b"{}").hexdigest()},
           {"name": "other.html", "sha256": hashlib.sha256(b"x").hexdigest()},
           {"name": "msrc/failed.json", "sha256": None}]  # fmt: skip
    (tmp_path / "fetch_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    assert list(bcm.verified_files(tmp_path / "msrc")) == ["cvrf_2020-Jan.json"]
    (tmp_path / "msrc" / "cvrf_2020-Jan.json").write_bytes(b"{ }")
    with pytest.raises(SystemExit, match="modified"):
        bcm.verified_files(tmp_path / "msrc")


def test_latest_msrc_entry_by_revision_date():
    a = {"doc": "2020-Jan", "revision_dates": ["2020-01-14T08:00:00"]}
    b = {"doc": "2019-Dec", "revision_dates": ["2019-12-10T08:00:00", "2020-02-11T08:00:00"]}
    assert bcm.latest([a, b]) is b
