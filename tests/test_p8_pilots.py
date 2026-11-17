"""P8 pilots on the merged tree: the training-size pilot config (P8a) and N4 before the freeze (P8b).

The N4 test generates networks and needs the pinned generator inputs (skipped without them)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ahg import cohorts as co, n4

ROOT = Path(__file__).resolve().parents[1]
GCFG = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())
HAVE = all((ROOT / p).exists() for p in (GCFG["master"]["path"], GCFG["catalogue"]["path"]))
CFG8 = json.loads((ROOT / "configs" / "gen" / "p8_pilot.json").read_text())
REG = co.load_registry()


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pilot = load("gen_d28_pilot")


def test_p8_config_mirrors_the_registry_training_and_validation_cohorts():
    """§19 P7d: the pilot's cell is the training cohort (and validation, which draws the same cell under its own
    seeds): tree, core, Linux 0.5, p 1, n = 8..20, exim and the CVE holdout v2 excluded; quotas 231 and 31."""
    from ahg.gen.pool import cve_key
    t, v = REG.training, REG["d28_val"]
    (cell,) = CFG8["cells"]
    assert (CFG8["namespace"], CFG8["run_name"], CFG8["tasks_per_cell"]) == ("p8-pilot", "p8_pilot", 2401)
    assert (cell["archetype"], cell["cohort"], cell["os_linux"], cell["p"]) == \
        (t.archetype, t.gen_cohort, t.os_linux, t.p) == (v.archetype, v.gen_cohort, v.os_linux, v.p)
    assert tuple(cell["sizes"]) == t.sizes == v.sizes == tuple(range(8, 21))
    assert (CFG8["training"]["solvable_per_size"], CFG8["validation"]["solvable_per_size"]) == (t.per_size, v.per_size)
    assert (CFG8["training"]["cell"], CFG8["validation"]["cell"]) == ("train", "train")
    assert (t.exclude, v.exclude) == ("heldout_products", "heldout_products") and REG.excludes_heldout_cves(t)
    assert tuple(CFG8["heldout_products"]) == REG.heldout_products and (cell["products"], cell["cves"]) == ("heldout", "holdout")
    held = json.loads((ROOT / CFG8["cve_holdout"]).read_text())["cves"]
    assert tuple(sorted(held, key=cve_key)) == REG.heldout_cves
    # development seeds: a namespace no cohort, pilot or earlier pilot uses
    used = {c.namespace for c in REG.cohorts.values()} | {"p6-pilot", "p7-smoke", "d28.train.n4", "p6-smoke"}
    assert not any(u.startswith("p8-pilot") or "p8-pilot".startswith(u) for u in used)
    seeds = [j[4] for j in pilot.jobs(CFG8, 2, ["CVE-2020-1"], [])]
    assert len(seeds) == len(set(seeds)) == 26 and all(s.startswith("p8-pilot/train/n") for s in seeds)


def test_training_and_validation_caps_and_report_on_synthetic_rows():
    cfg = {**CFG8, "cells": [{**CFG8["cells"][0], "sizes": [8, 9]}]}
    rows = [{"cell": "train", "n": n, "k": k, "seed": f"p8-pilot/train/n{n}/{k}", "solvable": k % 3 == 0,
             "reason": None if k % 3 == 0 else "goal_unreachable", "seconds": 0.01, "actions": 40}
            for n in (8, 9) for k in range(300)]
    s = pilot.summarise(rows, cfg, 2.0, 2)
    low = s["cells"]["train|n=8"]["wilson_low"]
    assert s["training_budget"]["train|n=8"]["budget"] == pilot.budget(231, low, 0.001)
    assert s["validation_budget"]["train|n=8"]["budget"] == pilot.budget(31, low, 0.001)
    # all sizes together: eps split over the 2 sizes (union bound), so never a smaller cap
    assert s["training_budget"]["train|n=9"]["budget_all_sizes"] == pilot.budget(231, low, 0.0005)
    assert s["training_budget"]["train|n=9"]["budget_all_sizes"] >= s["training_budget"]["train|n=9"]["budget"]
    assert s["test_sizes"] == {}
    text = pilot.report({"summary": s, "config": cfg, "holdout": {"count": 89, "sha256": "x"}})
    assert text.startswith("# P8a training-size pilot (seed namespace p8-pilot)")
    assert "validation candidate budget" in text and "| total | 462 |" in text and "fixed test cohort" not in text


def test_n4_reports_the_out_of_fold_confusion_matrix():
    rows = [{"archetype": a, "n": 12, **{f: (i if a == "tree" else -i) + 0.0 for f in n4.FEATURES}}
            for a in ("chain", "tree") for i in range(1, 11)]
    cm = n4.classify(rows)["12"]["confusion"]  # separable by sign: every network classified right
    assert cm == {"labels": ["chain", "tree"], "rows": "true", "columns": "predicted", "matrix": [[10, 0], [0, 10]]}
    rows[0] = {**rows[0], **{f: 5.0 for f in n4.FEATURES}}  # one chain network looks like a tree
    res = n4.classify(rows)["12"]
    m = res["confusion"]["matrix"]
    assert [sum(r) for r in m] == [10, 10]  # every network once, by the fold that held it out
    assert m != [[10, 0], [0, 10]]
    assert res["pooled_oof_balanced_accuracy"] == pytest.approx((m[0][0] / 10 + m[1][1] / 10) / 2)


@pytest.mark.skipif(not HAVE, reason="pinned generator inputs not in this checkout")
def test_n4_runs_before_the_freeze_on_training_seeds_with_trainings_exclusions(tmp_path, monkeypatch):
    """P8 (DECIDED-BY-AGENT, OQ-P8-2): N4 generates no labels, so neither the registry freeze nor the hypotheses
    freeze (§13.1: before any label generation) gates it; it runs on the training-style N4 seeds with training's
    exclusions (exim and, since M4, the CVE holdout)."""
    from ahg.gen import generate as gen
    pipe = load("d28_pipe")
    pipe.RUNS = tmp_path / "runs"
    reg_cfg = json.loads(co.REGISTRY.read_text())
    reg_cfg["n4"] = {**reg_cfg["n4"], "sizes": [12], "per_family_size": 5}
    small = tmp_path / "d28_cohorts.json"
    small.write_text(json.dumps(reg_cfg))
    monkeypatch.setattr(co, "REGISTRY", small)
    assert not REG.frozen and REG.problems(REG.training)  # unfrozen: nothing else may build
    seen = []

    def spy(*a, _real=gen.generate, **k):
        seen.append((a[2], k["excluded"], k["excluded_cves"]))
        return _real(*a, **k)

    monkeypatch.setattr(gen, "generate", spy)
    pipe.cmd_n4(type("A", (), {"smoke": False})(), REG, gen.Context.load())
    assert len(seen) == 10 and all(s.startswith("d28.train.n4/n12/") for s, _, _ in seen)
    assert all(e == {"exim:exim"} and c == frozenset(REG.heldout_cves) for _, e, c in seen)
    from ahg import runlog
    raw = runlog.load_raw(next((pipe.RUNS / "n4").iterdir()))
    assert raw["results"]["12"]["verdict"] in ("PASS", "FAIL") and raw["results"]["12"]["confusion"]["labels"] == ["chain", "tree"]
    assert set(raw["feature_summary"]) == {"chain|12", "tree|12"}
    assert set(raw["feature_summary"]["tree|12"]) == set(n4.FEATURES)


# ------------------------------------------------------------------ P8c/d: the pilots after the freeze (17 Nov 2026)

def test_landscape_uniform_column_uses_uniform_labels_and_reports_plan_length():
    from ahg import d28labels as dl, pilots
    from p7_helpers import tiny_doc
    cfg = {"walk_cap": 20, "walks_per_sample": 3, "successor_cap": 50, "oracle_all_max_n": 60, "oracle_states": 20,
           "oracle_max_pops": 10000, "enum_cap": 1000, "regimes": ["ac", "uniform"], "state_budget": 1000}
    d = tiny_doc()
    ac = pilots.landscape_task(d, dl.label_doc(d, "ac", cfg), 10000)
    un = pilots.landscape_task(d, dl.label_doc(d, "uniform", cfg), 10000, "uniform")
    assert (ac["regime"], un["regime"]) == ("ac", "uniform") and un["hstar"] == 3 and ac["hstar"] == 12
    with pytest.raises(ValueError, match="certified uniform"):
        pilots.landscape_task(d, dl.label_doc(d, "ac", cfg), 10000, "uniform")
    with pytest.raises(ValueError, match="neither ac"):
        pilots.landscape_task(d, dl.label_doc(d, "ac", cfg), 10000, "other")
    s = pilots.landscape_summary([ac, ac, {**ac, "plan_actions": 0, "R": {h: None for h in ac["R"]}}])["tree|3"]
    assert s["plan_actions_histogram"] == {0: 1, 3: 2} and s["plan_actions"]["median"] == 3


def test_budget_sample_covers_every_training_size():
    pipe = load("d28_pipe")
    docs = [{"params": {"n": n}, "i": i} for n in (8, 9, 10) for i in range(5)]
    got = pipe.stratified(docs, 7)  # ceil(7 / 3) = 3 per size
    assert [(d["params"]["n"], d["i"]) for d in got] == [(n, i) for n in (8, 9, 10) for i in range(3)]
    assert pipe.stratified(docs, 200) == docs and pipe.stratified([], 5) == []


def test_registry_takes_per_size_test_sizes_and_generation_caps():
    """P8 freeze: test cohort sizes and training/validation caps differ by size, so the registry takes either one
    number or a {size: number} mapping that names exactly the cohort's sizes."""
    from dataclasses import replace
    raw = {"name": "d28_x", "role": "test", "analysis": "primary", "hypothesis": "H-ID", "namespace": "d28.x",
           "archetype": "tree", "sizes": [12, 20], "gen_cohort": "core", "os_linux": 0.5, "p": 1.0,
           "exclude": "heldout_products", "selection": "fixed", "per_size": {"12": 650, "20": 674},
           "max_candidates_per_size": None, "requires": [], "blocked_by": []}
    c = co._cohort(raw, smoke=False)
    assert (c.size_at(12), c.size_at(20), c.cap_at(12)) == (650, 674, None)
    with pytest.raises(ValueError, match="names sizes"):
        co._cohort({**raw, "per_size": {"12": 650}}, smoke=False)
    t = replace(REG.training, max_candidates_per_size={n: 300 + n for n in REG.training.sizes})
    others = [x for x in REG.cohorts.values() if x.name != t.name]
    co.validate([t, c] + others, REG.totals, [1.0, .75, .5, .25])
    assert t.cap_at(8) == 308 and t.size_at(8) == 231
    with pytest.raises(ValueError, match="generation limit"):
        co.validate([replace(t, max_candidates_per_size={**t.max_candidates_per_size, 8: 230})] + others, REG.totals,
                    [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="one equal quota"):
        co.validate([replace(t, per_size={n: 231 for n in t.sizes})] + others, REG.totals, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="positive per_size"):
        co.validate([replace(c, per_size={12: 650, 20: 0})] + list(REG.cohorts.values()), REG.totals,
                    [1.0, .75, .5, .25])


def test_build_uses_the_size_specific_cap(tmp_path, monkeypatch):
    from dataclasses import replace
    from ahg.gen import generate as gen
    seen = []

    def fake(ctx, arch, seed, n, **kw):
        seen.append(seed)
        return {"seed": seed, "solvable": False, "unsolvable_reason": "x", "flags": {"unverified_release": False}}

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setattr(co, "store_doc", lambda doc, store: doc["seed"])
    monkeypatch.setattr(co.Registry, "problems", lambda self, c, prereg=None: [])
    t = replace(REG.training, sizes=(8, 9), per_size=2, max_candidates_per_size={8: 3, 9: 5})
    reg = replace(REG, cohorts={**REG.cohorts, t.name: t})

    class Ctx:
        input_hashes: dict = {}

    rec = co.build(Ctx(), reg, t.name, store=tmp_path)
    assert rec["by_size"]["8"]["candidates"] == 3 and rec["by_size"]["9"]["candidates"] == 5
    assert rec["shortfall"] == {8: 2, 9: 2} and len(seen) == 8
