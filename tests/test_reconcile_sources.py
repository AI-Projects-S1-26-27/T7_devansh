"""scripts/reconcile_sources.py: every file logged and verified, or explained; missing files re-fetched."""

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconcile_sources.py"
spec = importlib.util.spec_from_file_location("reconcile_sources", SCRIPT)
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def write_log(folder, recs):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "fetch_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def test_reconcile(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    old, new = tmp_path / "old", tmp_path / "new"
    write_log(old, [{"name": "a.html", "sha256": sha(b"a")}, {"name": "gone.html", "sha256": None},
                    {"name": "b.html", "sha256": sha(b"b")}])  # fmt: skip
    (old / "a.html").write_bytes(b"a")
    (old / "b.html").write_bytes(b"b")
    (old / "lit").mkdir()
    (old / "lit" / "q.xml").write_bytes(b"q")
    write_log(new, [{"name": "gone.html", "http": 200, "sha256": sha(b"g")}])
    (new / "gone.html").write_bytes(b"g")
    r = rs.reconcile(old, {"lit/*": "lit_search output"}, {"gone.html": "new"})
    assert r["errors"] == [] and r["verified"] == 2
    assert [u["name"] for u in r["unlogged"]] == ["lit/q.xml"] and r["unlogged"][0]["origin"] == "lit_search output"
    assert r["missing"][0]["refetch_record"]["sha256"] == sha(b"g")
    (old / "b.html").write_bytes(b"edited")
    (old / "stray.pdf").write_bytes(b"s")
    errs = rs.reconcile(old, {"lit/*": "x"}, {})["errors"]
    assert any("b.html: SHA-256 differs" in e for e in errs)
    assert any("stray.pdf: unlogged and unexplained" in e for e in errs)
    assert any("gone.html: logged but missing" in e for e in errs)
