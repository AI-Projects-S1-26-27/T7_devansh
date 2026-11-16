"""P9a script ports (OQ-P7-2): explicit timing repeats, Timing records, commonly solved tasks, loud failures;
and the on-policy run configs the CLI builds."""

import importlib.util
from pathlib import Path

import pytest

from ahg.onpolicy import plan, run_config
from ahg.timing import REPEATS, WARMUP

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(f"p9a_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(expanded, cost, ns=(5_000_000,) * 5):
    return {"expanded": expanded, "cost": cost, "budget_hit": cost is None,
            "time": {"ns": list(ns), "resolution_ns": 1, "censored": False}}


def raw_eval(n=8, budget_hit_task=None, censored_task=None):
    recs = []
    for i in range(n):
        runs = {"hmax": run(100 + i, 10), "lmcut": run(50 + i, 10)}
        for s in (0, 1):
            runs[f"M-s{s}"] = run(30 + i + s, 10)
        if i == budget_hit_task:
            runs["M-s1"] = run(1_000, None)  # a budget stop: no cost, no ratio
        if i == censored_task:
            runs["lmcut"]["time"] = {"ns": [1] * 5, "resolution_ns": 41, "censored": True}
        recs.append({"seed": i, "hstar": 10, "repeats": 5, "warmup": 1, "runs": runs})
    over = {"walk": {"rate": 0.0}}
    return {"split": "val", "records": recs, "budget": 1_000,
            "models": [{"name": f"M-s{s}", "run": f"results/training/x/M-s{s}", "combine": None} for s in (0, 1)],
            "summary": {f"M-s{s}": {"overestimation": over} for s in (0, 1)}}


FROZEN = {"runs": [{"id": "M", "seed": s, "run": f"results/training/x/M-s{s}", "role": "primary"} for s in (0, 1)]}


def test_analyse_refuses_pre_m1_records():
    analyse = script("analyse")
    raw = raw_eval()
    del raw["records"][3]["runs"]["hmax"]["time"]
    raw["records"][3]["runs"]["hmax"]["seconds"] = 0.01
    with pytest.raises(SystemExit, match="pre-M1"):
        analyse.analyse(raw, FROZEN)
    raw = raw_eval()
    del raw["records"][0]["repeats"]
    with pytest.raises(SystemExit, match="pre-M1"):
        analyse.analyse(raw, FROZEN)


def test_analyse_uses_commonly_solved_tasks_and_never_the_budget():
    analyse = script("analyse")
    a = analyse.analyse(raw_eval(budget_hit_task=2, censored_task=5), FROZEN)
    vs = a["models"]["M"]["vs"]
    assert vs["hmax"]["common"] == 7 and vs["lmcut"]["common"] == 7  # task 2 dropped, never priced at the budget
    assert vs["lmcut"]["time_censored"] == 1 and vs["hmax"]["time_censored"] == 0
    assert a["h1"][0]["n_nonzero"] <= 7


def test_evaluate_and_diagnose_record_explicit_repeats():
    for name in ("evaluate", "diagnose"):
        mod = script(name)
        args = mod.parser().parse_args(["--batches", "b", "--models", "m"])
        assert (args.repeats, args.warmup) == (REPEATS, WARMUP) == (5, 1)
        args = mod.parser().parse_args(["--batches", "b", "--models", "m", "--repeats", "3", "--warmup", "0"])
        assert (args.repeats, args.warmup) == (3, 0)
    args = script("evaluate").parser().parse_args(["--batches", "b", "--tie-break", "lifo"])
    assert args.tie_break == "lifo"
    with pytest.raises(SystemExit):
        script("evaluate").parser().parse_args(["--batches", "b", "--tie-break", "random"])


def test_run_config_follows_the_plan_and_the_frozen_grid():
    agg = {"pre_agg": "sum", "readout": "mean", "sha256": "x"}
    base = {"hidden": 8, "train_batches": ["t"], "val_batches": ["v"]}
    for i, (regime, kind, seed) in enumerate(plan()):
        cfg = run_config(i, base, agg if kind == "gnn" else None)
        assert (cfg["cost_regime"], cfg["kind"], cfg["seed"]) == (regime, kind, seed)
        assert (cfg.get("pre_agg"), cfg.get("readout")) == (("sum", "mean") if kind == "gnn" else (None, None))
        with pytest.raises(ValueError):  # the GNN needs the frozen choice; the others must not get one
            run_config(i, base, None if kind == "gnn" else agg)
    with pytest.raises(ValueError):
        run_config(0, {**base, "cost_regime": "uniform"}, None)
    with pytest.raises(IndexError):
        run_config(len(plan()), base, None)
