"""End to end on p7-smoke through the report builder (P7b, OQ-P7-11): build and label the smoke validation and test
cohorts, evaluate both through the CLI (every baseline, GNN/MLP/WL x seeds 0-2 per regime, both regimes, both tie
rules, real Fast Downward), refuse a second test read, regenerate the report.

Needs the pinned generator inputs and Fast Downward release-24.06.1; skipped without them. Checkpoints are
untrained (seeded initialisations): the run tests the pipeline, never a model."""

import json
import sys
from pathlib import Path

import pytest

from ahg import cohorts as co, d28eval as ev, d28labels as dl, fd_timing, runlog
from ahg.gen import generate as gen
from ahg.splits import LABEL_GENERATION
from d28_context import generating_with_context
from p7_helpers import checkpoint

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())
HAVE = all((ROOT / p).exists() for p in (CFG["master"]["path"], CFG["catalogue"]["path"]))
FD_OK = (fd_timing.FD_DIR / "fast-downward.py").exists() and fd_timing.fd_version() == fd_timing.FD_VERSION
pytestmark = pytest.mark.skipif(not (HAVE and FD_OK), reason="pinned generator inputs or Fast Downward missing")

STATE_BUDGET, SEARCH_BUDGET = 5000, 20_000


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    import importlib.util
    tmp = tmp_path_factory.mktemp("p7b")
    spec = importlib.util.spec_from_file_location("d28_pipe_p7b", ROOT / "scripts" / "d28_pipe.py")
    pipe = importlib.util.module_from_spec(spec)
    sys.modules["d28_pipe_p7b"] = pipe
    spec.loader.exec_module(pipe)
    mp = pytest.MonkeyPatch()
    # Everything the CLI reads or writes goes to the temporary directory; the real test-access log is untouched.
    mp.setattr(pipe, "RECORDS", tmp / "records")
    mp.setattr(co, "TASK_STORE", tmp / "tasks")
    mp.setattr(dl, "LABEL_ROOT", tmp / "labels")
    mp.setattr(co, "TEST_ACCESS_LOG", tmp / "test_log.jsonl")
    # The D28 encoder reads the planner-visible host OS and principal classes, which the generator writes once the
    # d28-gen change is merged (P7c, OQ-P7c-1); until then the documents get them from the test stand-in.
    mp.setattr(gen, "generate", generating_with_context(gen.generate))
    reg, ctx = co.load_registry(), gen.Context.load()
    lcfg = dl.with_budget(dl.load_config(), STATE_BUDGET)
    (tmp / "records").mkdir()
    docs = {}
    for name in ("p7_smoke_val", "p7_smoke_test"):
        c = reg[name]
        tok = None if not c.is_test else co.record_test_read(reg, name, LABEL_GENERATION, "abc", dataset_sha256=None,
                                                             log=co.TEST_ACCESS_LOG)
        rec = co.build(ctx, reg, name, store=co.TASK_STORE, access=tok, log=co.TEST_ACCESS_LOG)
        (tmp / "records" / f"{name}.json").write_text(json.dumps(rec))
        docs[name] = [d for _, d in co.open_members(rec, reg, store=co.TASK_STORE, access=tok, log=co.TEST_ACCESS_LOG)
                      if d is not None]
        dl.label_docs(docs[name], lcfg, root=dl.LABEL_ROOT)
    ckpts = [str(checkpoint(tmp / f"{k}{s}{r}.pt", k, s, r)) for k in ev.MODELS for s in ev.D23_SEEDS
             for r in ("ac", "uniform")]
    yield {"tmp": tmp, "pipe": pipe, "reg": reg, "docs": docs, "ckpts": ckpts, "lcfg": lcfg}
    mp.undo()


def evaluate(env, cohort):
    args = ["evaluate", cohort, "--state-budget", str(STATE_BUDGET), "--search-budget", str(SEARCH_BUDGET),
            "--repeats", "1", "--warmup", "0", "--eval-root", str(env["tmp"] / "eval")]
    for p in env["ckpts"]:
        args += ["--checkpoint", p]
    env["pipe"].main(args)


def test_validation_and_test_cohorts_evaluate_once_then_the_report_regenerates(env):
    log = co.TEST_ACCESS_LOG
    evaluate(env, "p7_smoke_val")
    assert sum(json.loads(x)["purpose"] == "evaluation" for x in log.read_text().splitlines()) == 0
    evaluate(env, "p7_smoke_test")
    entries = [json.loads(x) for x in log.read_text().splitlines()]
    ev_reads = [e for e in entries if e["purpose"] == "evaluation"]
    assert len(ev_reads) == 1 and len(ev_reads[0]["checkpoint_sha256"]) == 18 and ev_reads[0]["eval_config_sha256"]
    with pytest.raises(co.TestAccessError, match="already read"):
        evaluate(env, "p7_smoke_test")
    assert log.read_text().count("\n") == len(entries)  # the refused read wrote nothing

    runs = sorted((env["tmp"] / "eval").glob("*/*/raw.json.gz"))
    assert len(runs) == 8  # 2 cohorts x 2 regimes x 2 tie rules
    for path in runs:
        raw = runlog.load_raw(path.parent)
        labels = dl.load_labels(env["docs"][raw["split"]], env["lcfg"], root=dl.LABEL_ROOT)
        assert len(raw["records"]) == len(json.loads((env["tmp"] / "records" / f"{raw['split']}.json").read_text())["members"])
        assert raw["leakage_audit"]["ok"] and raw["leakage_audit"]["split"] == raw["split"]  # LRN6, P7c
        for r in raw["records"]:
            if not r["solvable"]:
                assert r["runs"] == {} and r["unsolvable_reason"]
                continue
            assert set(r["runs"]) == {"h0", "hmax", "hadd", "hff", "lmcut", "hsp"} | {
                f"{m}@{s}" for m in ev.MODELS for s in ev.D23_SEEDS}
            lab = labels[(r["task"], raw["regime"])]  # oracle-checked when labelled; test_p7_e2e compares the
            assert (r["certified"], r["hstar"]) == (lab["certified"], lab["hstar_init"])  # reference solver too
            if r["certified"]:
                for h in ("lmcut", "hmax"):
                    x = r["runs"][h]
                    assert x["fd"]["cost"] == r["hstar"] and x["peak_memory_bytes"] > 0 and x["fd_time"]["ns"][0] > 0
                    assert x["cost"] == r["hstar"] or x["budget_hit"]
            assert all(x["peak_open"] >= 1 and x["peak_closed"] >= 0 for x in r["runs"].values())

    out = env["tmp"] / "report"
    env["pipe"].main(["report", "--eval-root", str(env["tmp"] / "eval"), "--out", str(out), "--primary",
                      "p7_smoke_test", "--val-split", "p7_smoke_val", "--boot", "200"])
    text = (out / "report.md").read_text()
    assert "Strongest classical baseline on `p7_smoke_val`" in text and "Family size 3" in text
    for split in ("p7_smoke_val", "p7_smoke_test"):
        for tie in ("fifo", "lifo"):
            assert f"### {split} / tie rule {tie}" in text
    assert "## Regime `uniform` (secondary comparison, no verdict)" in text
    tables = json.loads((out / "tables.json").read_text())
    assert len(tables) == 8
    env["pipe"].main(["report", "--eval-root", str(env["tmp"] / "eval"), "--out", str(env["tmp"] / "report2"),
                      "--primary", "p7_smoke_test", "--val-split", "p7_smoke_val", "--boot", "200"])
    assert (env["tmp"] / "report2" / "report.md").read_text() == text  # regenerated byte for byte
