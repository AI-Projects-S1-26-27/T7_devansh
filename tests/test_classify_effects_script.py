"""P3 driver script: per-CVE ingest, repair batches, per-ruling ingest, Wilson interval."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("classify_effects", ROOT / "scripts" / "classify_effects.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def item(cid, ls="remote"):
    return {"cve_id": cid, "label_set": ls, "texts": {"nvd_description": "allows remote code execution"},
            "units": [{"product": "p", "role_question": False}], "supporting": {}}


def ann(cid, label="RCE_CONFIRMED"):
    return {"cve_id": cid, "label": label, "target_privilege": None, "text_product_mismatch": False, "effects": ["code_execution"],
            "roles": {}, "evidence": [{"source": "nvd_description", "span": "remote code execution"}]}


def make_run(tmp_path, anns):
    run = tmp_path / "run"
    w = run / "work"
    items = [item("CVE-1"), item("CVE-2")]
    (w / "stage2" / "A").mkdir(parents=True)
    (w / "pool.json").write_text(json.dumps(items))
    (w / "stage2" / "A" / "batch_00.json").write_text(json.dumps({"rubric": "r", "items": items}))
    (w / "stage2" / "A" / "out_00.json").write_text(json.dumps({"annotations": anns}))
    return run


def test_ingest_keeps_valid_cves_and_repair_batches_only_the_rest(tmp_path):
    bad = ann("CVE-2", "ADMIN_CONFIRMED")  # wrong label set for a remote item
    run = make_run(tmp_path, [ann("CVE-1"), bad])
    ce.cmd_ingest(SimpleNamespace(run=str(run), stage="A", scope="stage2"))
    assert [x["cve_id"] for x in json.loads((run / "work/stage2/A/ann_00.json").read_text())] == ["CVE-1"]
    ce.cmd_repair(SimpleNamespace(run=str(run), stage="A", scope="stage2", size=10))
    b = json.loads((run / "work/stage2/A/batch_r00.json").read_text())
    assert [i["cve_id"] for i in b["items"]] == ["CVE-2"]
    # the repair output is then ingested like any batch and annotations() merges both files
    (run / "work/stage2/A/out_r00.json").write_text(json.dumps({"annotations": [ann("CVE-2")]}))
    ce.cmd_ingest(SimpleNamespace(run=str(run), stage="A", scope="stage2"))
    assert sorted(ce.annotations(run, "stage2", "A")) == ["CVE-1", "CVE-2"]


def test_missing_annotation_counts_as_invalid(tmp_path):
    run = make_run(tmp_path, [ann("CVE-1")])
    ce.cmd_ingest(SimpleNamespace(run=str(run), stage="A", scope="stage2"))
    assert json.loads((run / "work/stage2/A/errors_00.json").read_text()) == {"CVE-2": ["no annotation"]}


def test_adjudicate_ingest_keeps_valid_rulings_only(tmp_path):
    run = tmp_path / "run"
    d = run / "work/stage2/adj-s"
    d.mkdir(parents=True)
    (run / "work/pool.json").write_text(json.dumps([item("CVE-1")]))
    (d / "batch_00.json").write_text(json.dumps({"items": [{"cve_id": "CVE-1", "disputes": ["label", "target_privilege"]}]}))
    good = {"cve_id": "CVE-1", "field": "label", "pick": "A", "source": "nvd_description", "span": "remote code execution"}
    bad = {"cve_id": "CVE-1", "field": "target_privilege", "pick": "A", "source": "nvd_description", "span": None}
    (d / "out_00.json").write_text(json.dumps({"rulings": [good, bad]}))
    ce.cmd_adjudicate_ingest(SimpleNamespace(run=str(run), name="s", scope="stage2"))
    assert list(ce.load_rulings(run, "stage2", "s")["CVE-1"]) == ["label"]


def test_wilson_interval():
    assert ce.wilson(0, 0) is None
    lo, hi = ce.wilson(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.04
    lo, hi = ce.wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61


def test_freeze_rows_sources_roles_and_unresolved(tmp_path):
    p3 = tmp_path / "p3"
    w = p3 / "work"
    units = [{"product": "p", "family": "http", "os": "linux", "cohort": "core", "stage": "role_not_server", "role": "ambiguous", "role_question": True}]
    items = [{**item("C1"), "units": units}, {**item("C2", "local"), "units": [{**units[0], "family": "os", "role": "os_component", "role_question": False, "stage": "local_candidate"}]},
             {**item("C3", "local"), "units": [{**units[0], "family": "os", "role": "os_component", "role_question": False, "stage": "local_candidate"}]}]
    (w / "stage2" / "A").mkdir(parents=True)
    (w / "stage2" / "B").mkdir(parents=True)
    (w / "pool.json").write_text(json.dumps(items))
    rules = {"C1": {"label": "unresolved", "rule": "none", "version": "r1", "source": None, "span": None},
             "C2": {"label": "NON_ADMIN", "rule": "local.non_wording", "version": "r1", "source": "nvd_description", "span": "x"},
             "C3": {"label": "unresolved", "rule": "none", "version": "r1", "source": None, "span": None}}
    (w / "rules.json").write_text(json.dumps(rules))
    a = {**ann("C1"), "roles": {"p": {"role": "server", "source": "nvd_description", "span": "remote code execution"}}}
    (w / "stage2" / "A" / "ann_00.json").write_text(json.dumps([a]))
    (w / "stage2" / "B" / "ann_00.json").write_text(json.dumps([a]))
    master = [{"cve_id": c, "units": [{"product": "p", "family": f, "os": o, "ac": "L", "ac_status": "explicit", "role": r}]}
              for c, f, o, r in (("C1", "http", "linux", "ambiguous"), ("C2", "os", "linux", "os_component"), ("C3", "os", "linux", "os_component"))]
    rows = {r["cve_id"]: r for r in ce.freeze_rows(p3, master, {})}
    assert (rows["C1"]["label"], rows["C1"]["labelled_by"], rows["C1"]["role"], rows["C1"]["role_by"]) == ("RCE_CONFIRMED", "annotator", "server", "annotator")
    assert rows["C1"]["evidence"][0]["span"] == "remote code execution" and rows["C1"]["ac_status"] == "explicit"
    assert (rows["C2"]["label"], rows["C2"]["labelled_by"], rows["C2"]["role"]) == ("NON_ADMIN", "rule", "os_component")
    assert (rows["C3"]["label"], rows["C3"]["labelled_by"]) == ("UNRESOLVED", "rule")


def test_v2_candidates_r2_hedge_and_review_required():
    pool = {"H": {**item("H"), "texts": {"nvd_description": "possible remote code execution"}},
            "K": item("K"), "R": item("R"), "N": item("N")}
    ev = [{"source": "nvd_description", "span": "possible remote code execution"}]
    rows = [{"cve_id": "H", "label": "RCE_CONFIRMED", "evidence": ev},
            {"cve_id": "K", "label": "RCE_CONFIRMED", "evidence": [{"source": "nvd_description", "span": "remote code execution"}]},
            {"cve_id": "R", "label": "REVIEW_REQUIRED", "evidence": []},
            {"cve_id": "N", "label": "NON_RCE", "evidence": []}]
    assert ce.v2_candidates(rows, pool) == {"H": "r2_hedge", "R": "readjudicate"}


def test_v2_freeze_maps_non_label_of_hedge_candidate_to_unknown(tmp_path, monkeypatch):
    import gzip
    import hashlib

    p4, p3, run = tmp_path / "p4", tmp_path / "p3", tmp_path / "v2run"
    for d in (p4 / "frozen", p4 / "work", run / "work" / "v2" / "A", run / "work" / "v2" / "B"):
        d.mkdir(parents=True)
    base = {"family": "http", "os": "linux", "cohort": "core", "label_set": "remote", "ac": "L", "ac_status": "explicit", "role": "server",
            "target_privilege": None, "evidence": [], "product": "p"}
    v1 = [{**base, "cve_id": c, "label": "RCE_CONFIRMED"} for c in ("H", "K")]
    data = gzip.compress("".join(json.dumps(r) + "\n" for r in v1).encode(), mtime=0)
    (p4 / "frozen" / "v1.jsonl.gz").write_bytes(data)
    ev = [{"source": "nvd_description", "span": "remote code execution"}]
    a = {"cve_id": "H", "label": "NON_RCE", "target_privilege": None, "text_product_mismatch": False, "effects": [], "roles": {}, "evidence": ev}
    for slot in "AB":
        (run / "work" / "v2" / slot / "ann_00.json").write_text(json.dumps([a, {**a, "cve_id": "K", "label": "RCE_CONFIRMED"}]))
    (run / "work" / "pool.json").write_text(json.dumps([item("H"), item("K")]))
    (run / "work" / "v2_select.json").write_text(json.dumps({"H": {"reason": "r2_hedge", "v1": "RCE_CONFIRMED"}}))
    (p4 / "work" / "pool.json").write_text(json.dumps([item("H")]))
    (p4 / "work" / "val_select.json").write_text(json.dumps({"H": {"cell": ["svc", "remote", "RCE_CONFIRMED"], "p3_label": "RCE_CONFIRMED", "labelled_by": "annotator"}}))
    (p4 / "reference.json").write_text(json.dumps({"labels": {"H": "RCE_CONFIRMED"}, "metrics": {k: {"accuracy": 1.0, "kappa": 1.0, "classes": {c: {"precision": 1.0, "recall": 1.0} for c in ce.ef.REMOTE_LABELS}} for k in ("remote|all", "remote|svc")}}))
    (run / "manifest.json").write_text(json.dumps({"p4_run": str(p4), "p3_run": str(p3), "v1": "v1.jsonl.gz", "v1_sha256": hashlib.sha256(data).hexdigest()}))
    monkeypatch.setattr(ce, "ROOT", ROOT)
    ce.cmd_v2_freeze(SimpleNamespace(run=str(run)))
    rows = {r["cve_id"]: r for r in ce.load_frozen(run / "frozen" / "p3-labels-v2.jsonl.gz")}
    assert (rows["H"]["label"], rows["H"]["v2_annotator_label"], rows["H"]["v2_from"]["label"]) == ("UNKNOWN", "NON_RCE", "RCE_CONFIRMED")
    assert rows["K"]["label"] == "RCE_CONFIRMED" and "v2_from" not in rows["K"]
    assert json.loads((run / "eligible.json").read_text())["RCE_CONFIRMED|http|linux|core"]["eligible_units"] == 1
    assert json.loads((run / "eligible_remote_by_product.json").read_text()) == {"http|linux|core|p": {"eligible_units": 1}}


def test_v21_candidates_are_v2_hedge_cves_no_longer_demoted(tmp_path):
    ev = [{"source": "nvd_description", "span": "might allow remote attackers to execute arbitrary code"}]
    pool_item = {**item("S"), "texts": {"nvd_description": "pkg might allow remote attackers to execute arbitrary code"}}
    assert not ce.er.r2_demotes(pool_item, "RCE_CONFIRMED", ev)
    hedged = {**item("H"), "texts": {"nvd_description": "possible remote code execution"}}
    assert ce.er.r2_demotes(hedged, "RCE_CONFIRMED", [{"source": "nvd_description", "span": "possible remote code execution"}])


def test_v21_freeze_applies_verified_label_and_keeps_v2_untouched(tmp_path):
    import gzip
    import hashlib

    p4, v2, run = tmp_path / "p4", tmp_path / "v2run", tmp_path / "v21run"
    for d in (p4 / "work", v2 / "frozen", run / "work" / "v21" / "A", run / "work" / "v21" / "B"):
        d.mkdir(parents=True)
    base = {"family": "http", "os": "linux", "cohort": "core", "label_set": "remote", "ac": "L", "ac_status": "explicit", "role": "server",
            "target_privilege": None, "evidence": [], "product": "p"}
    rows = [{**base, "cve_id": "S", "label": "UNKNOWN", "role_by": "master"}, {**base, "cve_id": "K", "label": "UNKNOWN"}]
    data = gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode(), mtime=0)
    (v2 / "frozen" / "v2.jsonl.gz").write_bytes(data)
    a = {"cve_id": "S", "label": "RCE_CONFIRMED", "target_privilege": None, "text_product_mismatch": False, "effects": [],
         "roles": {}, "evidence": [{"source": "nvd_description", "span": "remote code execution"}]}
    for slot in "AB":
        (run / "work" / "v21" / slot / "ann_00.json").write_text(json.dumps([a]))
    (run / "work" / "pool.json").write_text(json.dumps([item("S")]))
    (run / "work" / "v21_select.json").write_text(json.dumps({"S": {"reason": "stock_allow_exempt", "v2": "UNKNOWN"}}))
    (p4 / "work" / "pool.json").write_text(json.dumps([item("S")]))
    (p4 / "work" / "val_select.json").write_text(json.dumps({"S": {"cell": ["svc", "remote", "UNKNOWN"], "p3_label": "UNKNOWN", "labelled_by": "annotator"}}))
    (p4 / "reference.json").write_text(json.dumps({"labels": {"S": "RCE_CONFIRMED"}, "metrics": {k: {"accuracy": 0.0, "kappa": 0.0, "classes": {c: {"precision": 0.0, "recall": 0.0} for c in ce.ef.REMOTE_LABELS}} for k in ("remote|all", "remote|svc")}}))
    (run / "manifest.json").write_text(json.dumps({"p4_run": str(p4), "p3_run": "x", "v1": "v1", "v1_sha256": "y", "v2_run": str(v2), "v2": "v2.jsonl.gz",
                                                   "v2_sha256": hashlib.sha256(data).hexdigest()}))
    ce.cmd_v21_freeze(SimpleNamespace(run=str(run)))
    out = {r["cve_id"]: r for r in ce.load_frozen(run / "frozen" / "p3-labels-v2.1.jsonl.gz")}
    assert out["S"]["label"] == "RCE_CONFIRMED" and out["S"]["v21_from"]["label"] == "UNKNOWN" and out["K"]["label"] == "UNKNOWN"
    assert json.loads((run / "eligible_remote_by_product.json").read_text()) == {"http|linux|core|p": {"eligible_units": 1}}
    assert (v2 / "frozen" / "v2.jsonl.gz").read_bytes() == data
