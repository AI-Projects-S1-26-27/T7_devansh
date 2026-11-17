"""End to end on a tiny generated cohort under the seed namespace "p7-smoke": build, label, pilots, test gate, report.

Needs the pinned generator inputs (master, labels, catalogue); skipped without them."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ahg import cohorts as co, d28labels as dl, report
from ahg.gen import generate as gen
from ahg.splits import EVALUATION, LABEL_GENERATION

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())
HAVE = all((ROOT / p).exists() for p in (CFG["master"]["path"], CFG["catalogue"]["path"]))
pytestmark = pytest.mark.skipif(not HAVE, reason="pinned generator inputs not in this checkout")

BUDGET = 5000


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p7")
    spec = importlib.util.spec_from_file_location("d28_pipe", ROOT / "scripts" / "d28_pipe.py")
    pipe = importlib.util.module_from_spec(spec)
    sys.modules["d28_pipe"] = pipe
    spec.loader.exec_module(pipe)
    pipe.RECORDS, pipe.RUNS = tmp / "records", tmp / "runs"
    return {"tmp": tmp, "pipe": pipe, "reg": co.load_registry(), "ctx": gen.Context.load(), "store": tmp / "tasks"}


def test_smoke_cohorts_build_deterministically_and_keep_unsolvable_tasks(env):
    reg, ctx, store = env["reg"], env["ctx"], env["store"]
    train = co.build(ctx, reg, "p7_smoke_train", store=store)
    assert train["by_size"]["12"]["solvable_members"] == train["by_size"]["12"]["members"] == 2 - train["shortfall"][12]
    assert len(train["candidates"]) >= len(train["members"]) and all(
        {"seed", "archetype", "n", "os_linux", "p", "solvable", "reason"} <= set(c) for c in train["candidates"])
    again = co.build(ctx, reg, "p7_smoke_train", store=store)
    assert again["dataset_sha256"] == train["dataset_sha256"] and again["candidates"] == train["candidates"]
    val = co.build(ctx, reg, "p7_smoke_val", store=store)
    # validation: a solvable quota per size, like training (decided 17 Nov 2026, OQ-P7-1)
    assert all(m["solvable"] for m in val["members"]) and len(val["members"]) == 2 - val["shortfall"][12]
    assert all(m["seed"].startswith("p7-smoke.val/n12/") for m in val["members"])
    assert {m["seed"] for m in val["members"]}.isdisjoint({m["seed"] for m in train["members"]})
    env["train"], env["val"] = train, val


def test_labels_both_regimes_resume_and_match_reference_solver(env):
    ctx, reg, store = env["ctx"], env["reg"], env["store"]
    docs = [d for _, d in co.open_members(env["train"], reg, store=store)]
    cfg = dl.with_budget(dl.load_config(), BUDGET)
    root = env["tmp"] / "labels"
    first = dl.label_docs(docs, cfg, root=root)
    assert first["counts"]["ac|LABELLED"] == len(docs) == first["counts"]["uniform|LABELLED"]
    assert dl.label_docs(docs, cfg, root=root)["counts"]["ac|resumed"] == len(docs)
    for d in docs:
        for regime in ("ac", "uniform"):
            rec = dl.load_labels([d], cfg, root=root)[(gen.doc_digest(d), regime)]
            assert rec["hstar_init"] == gen.reference_cost(d, regime)  # independent reference solver agrees
    env["labels_root"], env["cfg"] = root, cfg


def test_cli_build_label_and_pilots_honour_a_redirected_task_store(env, monkeypatch):
    # OQ-P7b-4 (P7c): build, label and the pilots look the task store and label root up when they run, so a
    # redirect reaches every command and nothing goes to data/d28/tasks or data/labels/d28.
    pipe, reg, ctx = env["pipe"], env["reg"], env["ctx"]
    store, labels = env["tmp"] / "cli_tasks", env["tmp"] / "cli_labels"
    monkeypatch.setattr(co, "TASK_STORE", store)
    monkeypatch.setattr(dl, "LABEL_ROOT", labels)
    stores = []
    for fn in ("build", "open_members"):
        def spy(*a, _real=getattr(co, fn), **k):
            stores.append(k.get("store"))
            return _real(*a, **k)
        monkeypatch.setattr(co, fn, spy)
    ns = lambda **k: type("A", (), k)()
    pipe.cmd_build(ns(cohort="p7_smoke_train", n4_pass=None), reg, ctx)
    pipe.cmd_label(ns(cohort="p7_smoke_train", state_budget=BUDGET, shard=None), reg, ctx)
    for name, args in (("cmd_n4", ns(smoke=True)), ("cmd_budget", ns(smoke=True)),
                       ("cmd_landscape", ns(smoke=True, state_budget=BUDGET, search_budget=50_000))):
        getattr(pipe, name)(args, reg, ctx)
    assert len(stores) >= 4 and all(s == store for s in stores)  # build, label, budget, landscape
    rec = json.loads((pipe.RECORDS / "p7_smoke_train.json").read_text())
    assert all(co.load_doc(m["digest"], store)["params"]["seed"] == m["seed"] for m in rec["members"])
    n = len(rec["members"])  # label: both regimes; the landscape pilot adds its own ac-only records
    assert len(list((labels / "uniform").rglob("*.json.gz"))) == n and len(list((labels / "ac").rglob("*.json.gz"))) >= n
    runs = sorted(p.name for p in pipe.RUNS.iterdir())
    assert runs == ["budget", "label_p7_smoke_train", "landscape", "n4"]
    for r in runs:
        d = next((pipe.RUNS / r).iterdir())
        assert (d / "raw.json.gz").exists() and (d / "report.md").exists() and (d / "provenance.json").exists()
    from ahg import runlog
    n4_raw = runlog.load_raw(next((pipe.RUNS / "n4").iterdir()))
    assert set(n4_raw["results"]) == {"12", "20"}
    assert len(n4_raw["rows"]) == 2 * 2 * 6  # sizes x archetypes x smoke networks, none left out (OQ-P7-6)
    assert sum(n4_raw["with_unreachable_hosts"].values()) == sum(r["unreachable_frac"] > 0 for r in n4_raw["rows"])


def test_test_cohort_needs_the_access_log_and_second_read_is_refused(env):
    reg, ctx, store = env["reg"], env["ctx"], env["store"]
    log = env["tmp"] / "test_log.jsonl"
    c = reg["p7_smoke_test"]
    with pytest.raises(co.TestAccessError):
        co.build(ctx, reg, c.name, store=store, log=log)  # no token
    tok = co.record_test_read(reg, c.name, LABEL_GENERATION, "abc", dataset_sha256=None, log=log)
    rec = co.build(ctx, reg, c.name, store=store, access=tok, log=log)
    assert len(rec["members"]) == 4
    ev = dict(dataset_sha256=rec["dataset_sha256"], eval_config_sha256="e", checkpoint_sha256={"gnn@0": "c"}, log=log)
    co.record_test_read(reg, c.name, EVALUATION, "abc", **ev)
    with pytest.raises(co.TestAccessError):
        co.record_test_read(reg, c.name, EVALUATION, "abc", **ev)


def test_report_regenerates_from_a_results_directory(env):
    from p7_helpers import record
    root = env["tmp"] / "eval"
    import gzip
    for split in ("p7_smoke_val", "p7_smoke_test"):
        d = root / split / "ac-fifo"
        d.mkdir(parents=True)
        (d / "raw.json.gz").write_bytes(gzip.compress(json.dumps(
            {"split": split, "regime": "ac", "tie_break": "fifo", "records": [record(i) for i in range(20)]}).encode()))
    text = report.build_report(root, env["tmp"] / "report", primary={"p7_smoke_test"}, val_split="p7_smoke_val",
                               n_boot=100).read_text()
    assert "p7_smoke_test|GNN vs BASE" in text


def test_real_cohorts_refuse_to_build_while_the_registry_is_unfrozen(env, monkeypatch):
    # The real registry is frozen since P8 (17 Nov 2026); the CLI sees it as it stood before, so this gate test can
    # never build a real cohort or log a real test read.
    from dataclasses import replace
    real = co.load_registry
    monkeypatch.setattr(co, "load_registry", lambda *a, **k: replace(real(*a, **k), frozen=False))
    pipe = env["pipe"]
    before = co.TEST_ACCESS_LOG.read_bytes() if co.TEST_ACCESS_LOG.exists() else b""
    # The label-generating pilots (budget, landscape) stay refused until the freeze (§13.1, OQ-P8-1). N4 generates no
    # labels and runs unfrozen since P8 (OQ-P8-2; tests/test_p8_pilots.py checks it uses training's exclusions).
    for argv in (["build", "d28_val"], ["build", "d28_test_id"], ["label", "d28_test_id", "--state-budget", "9"],
                 ["budget"], ["landscape"]):
        with pytest.raises(co.BlockedCohort):
            pipe.main(argv)
    after = co.TEST_ACCESS_LOG.read_bytes() if co.TEST_ACCESS_LOG.exists() else b""
    assert after == before  # a refused build logs no read


def test_ac_counts_command_lists_the_pool(env, capsys):
    env["pipe"].main(["ac-counts"])
    counts = json.loads(capsys.readouterr().out)
    assert counts["core|windows|local|L"] > 0 and sum(counts.values()) == len(env["ctx"].units)
