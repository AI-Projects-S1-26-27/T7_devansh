"""D28 evaluation runner (P7b, OQ-P7-11): per-task records for every baseline and learned model, both regimes,
both tie rules, Fast Downward; the checks and the test gate. Tiny hand-written tasks; FD is faked here (the
real FD runs in test_p7b_e2e.py and test_fd_timing.py)."""

import json

import pytest

from ahg import cohorts as co, d28eval as ev, d28labels as dl, fd_timing
from ahg.measure import HardFailure
from ahg.splits import EVALUATION, LABEL_GENERATION
from p7_helpers import checkpoint, tiny_doc

REG = co.load_registry()
LCFG = dl.with_budget(dl.load_config(), 1000)
CFG = ev.with_budget(ev.load_config(), 1000) | {"repeats": 2, "warmup": 0}


def fake_fd(cost_of):
    """fd_timing.run stand-in: solved, cost from cost_of(domain text), plausible wall and memory records."""
    def run(domain, problem, heuristic, repeats, fd_dir=None, *, warmup):
        wall = {"ns": [5_000_000] * repeats, "resolution_ns": 1, "censored": False}
        return {"fd_version": fd_timing.FD_VERSION, "heuristic": heuristic, "status": "solved", "exit_code": 0,
                "cost": cost_of(domain.read_text()), "expanded": 3, "repeats": repeats, "warmup": warmup,
                "wall": {"translate": wall, "search": wall, "end_to_end": wall},
                "peak_rss_bytes": {"translate": [1 << 20] * repeats, "search": [1 << 20] * repeats,
                                   "end_to_end": [3 << 20, 2 << 20][:repeats]},
                "fd_reported": []}
    return run


@pytest.fixture
def fd_ok(monkeypatch):
    # tiny_doc's optimum is 12 under ac and 3 under uniform; the PDDL domain carries the regime's costs.
    monkeypatch.setattr(fd_timing, "run", fake_fd(lambda d: 12 if "(total-cost) 4)" in d else 3))


def member(doc):
    from ahg.gen.generate import doc_digest
    return {"digest": doc_digest(doc), "seed": doc["params"]["seed"], "n": doc["params"]["n"],
            "solvable": doc["solvable"], "reason": doc["unsolvable_reason"]}


def flags(doc):
    return doc | {"flags": {"unverified_release": False}}


def label(doc, regime):
    return dl.label_doc(doc, regime, LCFG)


def test_config_is_complete_and_the_budget_must_be_given_until_frozen():
    cfg = ev.load_config()
    assert set(cfg) == ev.CONFIG_KEYS and cfg["search_budget"] is None
    assert sorted(cfg["regimes"]) == ["ac", "uniform"] and sorted(cfg["tie_rules"]) == ["fifo", "lifo"]
    assert sorted(cfg["fast_downward"]) == ["hmax", "lmcut"] and (cfg["repeats"], cfg["warmup"]) == (5, 1)
    with pytest.raises(ValueError, match="not frozen"):
        ev.with_budget(cfg, None)
    with pytest.raises(ValueError):
        ev.check_config(cfg | {"tie_rules": ["random"]})
    with pytest.raises(ValueError):
        ev.check_config(cfg | {"extra": 1})


def run_task(doc, regime, ckpts=(), cfg=CFG):
    return ev.evaluate_task(doc, member(doc), regime, label(doc, regime), cfg, list(ckpts))


def test_every_method_runs_under_both_tie_rules_with_the_report_fields(fd_ok, tmp_path):
    doc = flags(tiny_doc())
    for regime, opt in (("ac", 12), ("uniform", 3)):
        ck = ev.load_checkpoints([checkpoint(tmp_path / f"{k}{regime}.pt", k, 0, regime) for k in ev.MODELS])[regime]
        out = run_task(doc, regime, ck)
        assert set(out) == {"fifo", "lifo"}
        for tie, rec in out.items():
            assert rec["solvable"] and rec["certified"] and rec["hstar"] == opt and rec["label_outcome"] == dl.LABELLED
            assert set(rec["runs"]) == {"h0", "hmax", "hadd", "hff", "lmcut", "hsp", "gnn@0", "mlp@0", "wl@0"}
            for m, r in rec["runs"].items():
                assert r["expanded"] > 0 and r["peak_open"] >= 1 and r["peak_closed"] >= 1
                assert len(r["python_time"]["ns"]) == 2 and r["overestimation"]["states"] == len(label(doc, regime)["rows"])
                assert (r["fd_time"] is not None) == (m in ("lmcut", "hmax"))
                if m in ("h0", "hmax", "lmcut", "hsp"):
                    assert r["cost"] == opt and r["overestimation"]["violations"] == 0
            assert rec["runs"]["lmcut"]["peak_memory_bytes"] == 3 << 20  # the max over the timed repeats
            assert rec["runs"]["lmcut"]["fd"]["cost"] == opt and rec["runs"]["h0"]["fd"] is None


def test_budget_stops_a_search_without_a_cost(fd_ok):
    rec = run_task(flags(tiny_doc()), "ac", cfg=CFG | {"search_budget": 1})["fifo"]
    assert rec["runs"]["h0"]["budget_hit"] and rec["runs"]["h0"]["cost"] is None


def test_unsolvable_and_failed_members_are_kept_without_search(fd_ok):
    doc = flags(tiny_doc("t/1", solvable=False))
    rec = run_task(doc, "ac")["lifo"]
    assert not rec["solvable"] and rec["runs"] == {} and rec["label_outcome"] == dl.DEAD_END
    assert rec["unsolvable_reason"] == "no_foothold_on_goal_host_reachable"
    failed = {"digest": None, "seed": "t/2", "n": 3, "solvable": False, "reason": "infeasible_cell: x"}
    rec = ev.evaluate_task(None, failed, "ac", None, CFG, [])["fifo"]
    assert rec["task"] is None and rec["unsolvable_reason"] == "infeasible_cell: x" and rec["runs"] == {}


def test_a_wrong_certified_optimum_or_fd_cost_aborts(monkeypatch):
    doc = flags(tiny_doc())
    bad = label(doc, "ac") | {"hstar_init": 11}
    monkeypatch.setattr(fd_timing, "run", fake_fd(lambda d: 11))
    with pytest.raises(HardFailure, match="A\\*\\+h0"):
        ev.evaluate_task(doc, member(doc), "ac", bad, CFG, [])
    monkeypatch.setattr(fd_timing, "run", fake_fd(lambda d: 13))
    with pytest.raises(HardFailure, match="FD"):
        ev.evaluate_task(doc, member(doc), "ac", label(doc, "ac"), CFG, [])


def test_missing_label_or_contradicting_generator_flag_aborts(fd_ok):
    doc = flags(tiny_doc())
    with pytest.raises(FileNotFoundError):
        ev.evaluate_task(doc, member(doc), "ac", None, CFG, [])
    with pytest.raises(HardFailure, match="generator solvable"):
        ev.evaluate_task(doc | {"solvable": False}, member(doc), "ac", label(doc, "ac"), CFG, [])


def test_checkpoints_are_named_from_their_config_and_regime_is_required(tmp_path):
    paths = [checkpoint(tmp_path / f"{k}{s}{r}.pt", k, s, r) for k in ev.MODELS for s in (0, 1) for r in ("ac", "uniform")]
    ck = ev.load_checkpoints(paths)
    assert sorted(c.name for c in ck["ac"]) == ["gnn@0", "gnn@1", "mlp@0", "mlp@1", "wl@0", "wl@1"]
    assert len(ev.checkpoint_hashes(ck)) == 12 and all(len(h) == 64 for h in ev.checkpoint_hashes(ck).values())
    with pytest.raises(ValueError, match="two checkpoints"):
        ev.load_checkpoints(paths[:1] + paths[:1])
    with pytest.raises(ValueError, match="no cost regime"):
        ev.load_checkpoints([checkpoint(tmp_path / "old.pt", "mlp", 0, None)])
    for k in ev.MODELS:  # P7c: D28 tasks are encoded only by the D28 encoder
        with pytest.raises(ValueError, match="D28 encoder"):
            ev.load_checkpoints([checkpoint(tmp_path / f"d19{k}.pt", k, 0, "ac", features=None)])


def full_checkpoints(tmp_path):
    return ev.load_checkpoints([checkpoint(tmp_path / f"{k}{s}{r}.pt", k, s, r)
                                for k in ev.MODELS for s in ev.D23_SEEDS for r in ("ac", "uniform")])


def test_test_cohort_needs_the_complete_batch_and_its_own_evaluation_read(tmp_path, fd_ok):
    c, log = REG["p7_smoke_test"], tmp_path / "log.jsonl"
    record = {"cohort": c.name, "members": [], "dataset_sha256": "d"}
    ck = full_checkpoints(tmp_path)
    assert ev.completeness_problems(c, CFG, ck) == []
    assert ev.completeness_problems(c, CFG | {"tie_rules": ["fifo"]}, ck)
    assert ev.completeness_problems(c, CFG, {"ac": ck["ac"]})
    kw = dict(labels_root=tmp_path / "labels", store=tmp_path / "store", log=log)
    with pytest.raises(ValueError, match="timing protocol"):  # only a smoke cohort may use fewer repeats
        ev.evaluate_cohort({**record, "cohort": "d28_val"}, REG, CFG, ck, LCFG, **kw)
    with pytest.raises(co.TestAccessError, match="complete"):
        ev.evaluate_cohort(record, REG, CFG | {"regimes": ["ac"]}, ck, LCFG, **kw)
    with pytest.raises(co.TestAccessError, match="evaluation read"):
        ev.evaluate_cohort(record, REG, CFG, ck, LCFG, **kw)
    gen_tok = co.record_test_read(REG, c.name, LABEL_GENERATION, "abc", dataset_sha256=None, log=log)
    with pytest.raises(co.TestAccessError, match="evaluation read"):
        ev.evaluate_cohort(record, REG, CFG, ck, LCFG, access=gen_tok, **kw)
    other = co.record_test_read(REG, c.name, EVALUATION, "abc", dataset_sha256="d", eval_config_sha256="x",
                                checkpoint_sha256=ev.checkpoint_hashes(ck), log=log)
    with pytest.raises(co.TestAccessError, match="different config"):
        ev.evaluate_cohort(record, REG, CFG, ck, LCFG, access=other, **kw)
    ok = co.record_test_read(REG, c.name, EVALUATION, "abc", dataset_sha256="d", eval_config_sha256=ev.config_sha(CFG),
                             checkpoint_sha256=ev.checkpoint_hashes(ck), permitted_by="D29 test", log=log)
    raws = ev.evaluate_cohort(record, REG, CFG, ck, LCFG, access=ok, **kw)
    assert set(raws) == {(r, t) for r in ("ac", "uniform") for t in ("fifo", "lifo")}
    assert all(r["test_access"] == ok and r["records"] == [] for r in raws.values())


def test_cohort_runs_write_one_report_readable_run_per_regime_and_tie_rule(tmp_path, fd_ok):
    from ahg import report
    store, labels = tmp_path / "store", tmp_path / "labels"
    docs = [flags(tiny_doc(f"t/{i}")) for i in range(3)] + [flags(tiny_doc("t/9", solvable=False))]
    record = {"cohort": "p7_smoke_val", "dataset_sha256": "d",
              "members": [member(d) | {"digest": co.store_doc(d, store)} for d in docs]}
    assert ev.missing_labels(record, LCFG, labels) == 2 * len(docs)
    dl.label_docs(docs, LCFG, root=labels)
    assert ev.missing_labels(record, LCFG, labels) == 0
    ck = full_checkpoints(tmp_path)
    raws = ev.evaluate_cohort(record, REG, CFG, ck, LCFG, labels_root=labels, store=store)
    dirs = ev.write_runs(raws, tmp_path / "eval")
    assert len(dirs) == 4 and {d.parent.name for d in dirs} == {f"p7_smoke_val.{r}.{t}" for r in ("ac", "uniform")
                                                                  for t in ("fifo", "lifo")}
    runs = report.load_runs(tmp_path / "eval")
    assert len(runs) == 4 and all(len(r["records"]) == 4 for r in runs)
    raw = next(r for r in runs if (r["regime"], r["tie_break"]) == ("uniform", "lifo"))
    assert {r["hstar"] for r in raw["records"] if r["solvable"]} == {3}
    assert json.loads((dirs[0] / "provenance.json").read_text())["seeds"] == [0, 1, 2]
    text = report.build_report(tmp_path / "eval", tmp_path / "rep", primary=set(), val_split="p7_smoke_val",
                               n_boot=50).read_text()
    assert "peak open" in text and "### p7_smoke_val / tie rule lifo" in text
    assert all(r["leakage_audit"]["ok"] for r in runs)  # LRN6 audit of the encoded split, kept in every raw (P7c)


def test_a_leaking_encoding_stops_the_cohort_before_any_model_runs(tmp_path, fd_ok, monkeypatch):
    # P7c (LRN6): an account name where the principal class belongs fails the audit; nothing is searched.
    store, labels = tmp_path / "store", tmp_path / "labels"
    doc = flags(tiny_doc("t/0"))
    doc["network"]["principal_classes"][1] = "nobody"
    record = {"cohort": "p7_smoke_val", "dataset_sha256": "d", "members": [member(doc) | {"digest": co.store_doc(doc, store)}]}
    dl.label_docs([doc], LCFG, root=labels)
    searched = []
    monkeypatch.setattr(ev, "evaluate_task", lambda *a, **k: searched.append(1))
    from ahg.learning.leakage import LeakageError
    with pytest.raises(LeakageError):
        ev.evaluate_cohort(record, REG, CFG, full_checkpoints(tmp_path), LCFG, labels_root=labels, store=store)
    assert searched == []
