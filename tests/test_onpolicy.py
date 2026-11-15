"""D28 on-policy orchestration (Round 9) and cost regimes (Round 5) on a tiny synthetic task set (P9a)."""

import gzip
import hashlib
import json

import pytest

from ahg import onpolicy
from ahg.cost import AC_COSTS, check_priced, check_regime, regime_cost
from ahg.learning import encode
from ahg.learning.runs import make_config
from ahg.learning.train import load, train_model
from ahg.measure import HardFailure, enumerate_exact
from ahg.onpolicy import (ROUNDS, check_protocol, label_states, plan, refuse_non_training, run_onpolicy,
                          source_name, write_onpolicy_run)
from ahg.runlog import load_raw
from ahg.task import Task

from synthetic import PROTOCOL, offline_batch, synthetic_seeds, synthetic_task

SMALL = {"max_epochs": 2, "irls_rounds": 2, "hidden": 8, "layers": 2, "patience": 2}
SEEDS = synthetic_seeds(8)  # 6 training networks, 2 validation networks


@pytest.fixture(autouse=True)
def no_shared_cache(tmp_path, monkeypatch):
    """Encodings go to a per-test cache, never to data/cache/."""
    monkeypatch.setattr(encode, "CACHE", tmp_path / "cache")


@pytest.fixture(scope="module")
def corpus():
    return {r: (offline_batch(f"syn-train-{r}", "train", r, SEEDS[:6]), offline_batch(f"syn-val-{r}", "val", r, SEEDS[6:]))
            for r in ("ac", "uniform")}


def cfg(kind, regime="ac", seed=0, **over):
    extra = {"pre_agg": "mean", "readout": "sum"} if kind == "gnn" else {}
    return make_config(kind=kind, seed=seed, cost_regime=regime, **SMALL, **extra, **over)


# ---- cost regimes (Round 5) ----

def test_regime_costs():
    assert AC_COSTS == {"L": 4, "H": 7}
    assert [regime_cost("ac", a) for a in "LH"] == [4, 7]
    assert [regime_cost("uniform", a) for a in "LH"] == [1, 1]


@pytest.mark.parametrize("regime", ["ac", "uniform"])
@pytest.mark.parametrize("ac", [None, "", "N", "l", 4])
def test_missing_or_invalid_ac_is_never_defaulted(regime, ac):
    with pytest.raises(ValueError):
        regime_cost(regime, ac)


@pytest.mark.parametrize("name", ["AC", "primary", "", None, "sensitivity"])
def test_unknown_regime_names_are_refused(name):
    with pytest.raises(ValueError):
        check_regime(name)
    with pytest.raises(ValueError):
        regime_cost(name, "L")


def test_a_task_priced_under_one_regime_is_refused_under_the_other():
    ac, uni = synthetic_task(SEEDS[0], "ac"), synthetic_task(SEEDS[0], "uniform")
    check_priced(ac, "ac")
    check_priced(uni, "uniform")
    with pytest.raises(ValueError):
        check_priced(ac, "uniform")
    with pytest.raises(ValueError):
        check_priced(uni, "ac")
    d9 = Task.from_names(["a", "b"], ["a"], ["b"], [("x", ["a"], ["b"], 38)])  # the pre-D28 D9 units
    for regime in ("ac", "uniform"):
        with pytest.raises(ValueError):
            check_priced(d9, regime)


def test_both_regimes_cost_the_same_instance():
    ac, uni = synthetic_task(SEEDS[0], "ac"), synthetic_task(SEEDS[0], "uniform")
    assert [(a.name, a.pre, a.add) for a in ac.actions] == [(a.name, a.pre, a.add) for a in uni.actions]
    assert {a.cost for a in uni.actions} == {1} and {a.cost for a in ac.actions} <= {4, 7}


def test_training_refuses_labels_of_the_other_regime(corpus):
    (tr_ac, va_ac), (tr_uni, va_uni) = corpus["ac"], corpus["uniform"]
    with pytest.raises(ValueError, match="regime"):
        train_model(cfg("mlp", "uniform"), [tr_ac], [va_ac])
    with pytest.raises(ValueError, match="regime"):
        train_model(cfg("mlp", "ac"), [tr_ac], [va_uni])
    with pytest.raises(ValueError, match="regime"):  # a pre-D28 config (no regime) on D28 labels
        train_model(make_config(kind="mlp", **SMALL), [tr_ac], [va_ac])
    with pytest.raises(ValueError):
        train_model(cfg("mlp", "AC"), [tr_ac], [va_ac])


# ---- labelling ----

def test_label_states_are_exact_and_checked():
    task = synthetic_task(SEEDS[0], "ac")
    states = [task.init] + [s for _, s in task.successors(task.init)]
    out = label_states(task, states, PROTOCOL)
    assert out["labels"] and out["budget"] == [] and out["stats"]["oracle_skipped"] == 0
    exact = enumerate_exact(task, 10**6)  # the independent reference solver (Round 8 C2)
    for s, h in out["labels"].items():
        assert h == exact[s]
    assert out["stats"]["oracle_checked"] == len(out["labels"]) + len(out["dead_ends"])


def test_label_budget_gives_no_label_and_dead_ends_are_kept_apart():
    chain = Task.from_names(["a", "b", "c", "d"], ["a"], ["d"],
                            [("x", ["a"], ["b"], 4), ("y", ["b"], ["c"], 7), ("z", ["c"], ["d"], 4)])
    out = label_states(chain, [chain.init], {**PROTOCOL, "state_budget": 1})
    assert out["labels"] == {} and out["budget"] == [chain.init]  # BUDGET_EXHAUSTED: no value of any kind
    dead_task = Task.from_names(["a", "b", "c"], ["a"], ["c"], [("x", ["a"], ["c"], 4), ("y", ["b"], ["c"], 7)])
    out = label_states(dead_task, [0b000, 0b001], PROTOCOL)  # {} has no applicable action: a dead end
    assert out["dead_ends"] == [0] and out["labels"] == {1: 4}


def test_an_oracle_mismatch_aborts(monkeypatch):
    task = synthetic_task(SEEDS[0], "ac")
    monkeypatch.setattr(onpolicy, "optimal_cost", lambda t, s, max_pops=None: 10**6)
    with pytest.raises(HardFailure, match="oracle"):
        label_states(task, [task.init], PROTOCOL)


def test_protocol_has_no_defaults():
    check_protocol(dict(PROTOCOL))
    for key in PROTOCOL:
        with pytest.raises(ValueError):
            check_protocol({k: v for k, v in PROTOCOL.items() if k != key})
    with pytest.raises(ValueError):
        check_protocol({**PROTOCOL, "rounds": 3})


# ---- orchestration ----

def test_plan_is_every_model_seed_and_regime():
    p = plan()
    assert len(p) == len(set(p)) == 2 * 3 * 3
    assert {r for r, _, _ in p} == {"ac", "uniform"} and {k for _, k, _ in p} == {"gnn", "mlp", "wl"}
    assert {s for _, _, s in p} == {0, 1, 2} and ROUNDS == 2


def test_collection_refuses_validation_and_test_networks(corpus):
    tr, va = corpus["ac"]
    with pytest.raises(PermissionError):
        refuse_non_training([va])
    with pytest.raises(PermissionError):
        run_onpolicy(cfg("mlp"), [va], [va], PROTOCOL)
    test = onpolicy.make_batch("syn-test", "test_id", "ac", tr.instances, tr.examples, [])
    with pytest.raises(PermissionError):
        run_onpolicy(cfg("mlp"), [tr, test], [va], PROTOCOL)


def test_run_refuses_bad_inputs(corpus):
    tr, va = corpus["ac"]
    with pytest.raises(ValueError):  # regime mismatch between the run and its batches
        run_onpolicy(cfg("mlp", "uniform"), [tr], [va], PROTOCOL)
    with pytest.raises(ValueError):  # a seed that is not a D23 seed
        run_onpolicy(cfg("mlp", seed=3), [tr], [va], PROTOCOL)
    with pytest.raises(ValueError):  # a GNN without the frozen aggregator choice
        run_onpolicy(make_config(kind="gnn", seed=0, cost_regime="ac", **SMALL), [tr], [va], PROTOCOL)
    with pytest.raises(ValueError):  # a missing budget
        run_onpolicy(cfg("mlp"), [tr], [va], {k: v for k, v in PROTOCOL.items() if k != "search_budget"})


@pytest.mark.parametrize("kind", ["mlp", "wl", "gnn"])
@pytest.mark.parametrize("regime", ["ac", "uniform"])
def test_two_rounds_end_to_end(corpus, kind, regime):
    tr, va = corpus[regime]
    before = len(tr.examples)
    res = run_onpolicy(cfg(kind, regime), [tr], [va], PROTOCOL)
    assert len(res["models"]) == ROUNDS + 1 and [r["round"] for r in res["rounds"]] == [0, 1, 2]
    assert len(tr.examples) == before  # the offline corpus is never modified
    seen = {(tr.manifest["instances"][ex.instance]["task_digest"], ex.state) for ex in tr.examples}
    for k, (b, r) in enumerate(zip(res["batches"], res["rounds"][1:]), start=1):
        s = r["collect"]
        assert b.manifest["split"] == "train" and b.manifest["cost_regime"] == regime
        assert s["networks"] == len(tr.instances) and s["search_expansions"] >= s["states_expanded"] > 0
        assert s["labelled"] + s["label_budget_hits"] + s["dead_ends"] == s["states_new"] == len(b.examples) + len(b.failures)
        assert s["oracle_checked"] == s["labelled"] + s["dead_ends"] and s["oracle_skipped"] == 0
        for ex in b.examples:
            key = (b.manifest["instances"][ex.instance]["task_digest"], ex.state)
            assert key not in seen  # deduplicated per task against the corpus so far
            seen.add(key)
            assert ex.sources == [source_name(k)]
            assert ex.hstar == enumerate_exact(b.instances[ex.instance].task, 10**6)[ex.state]
        assert r["train"]["train_examples"] == before + sum(len(x.examples) for x in res["batches"][:k])
    assert res["rounds"][1]["collect"]["labelled"] > 0  # the synthetic set is big enough to collect something


def test_runs_are_isolated_and_reproducible(corpus):
    tr, va = corpus["ac"]
    a = run_onpolicy(cfg("mlp", seed=0), [tr], [va], PROTOCOL)
    b = run_onpolicy(cfg("mlp", seed=0), [tr], [va], PROTOCOL)
    c = run_onpolicy(cfg("mlp", seed=1), [tr], [va], PROTOCOL)
    strip = lambda res: [{k: v for k, v in r.get("collect", {}).items() if not k.endswith("seconds")} for r in res["rounds"]]
    assert strip(a) == strip(b)
    assert [x.manifest["content_sha256"] for x in a["batches"]] == [x.manifest["content_sha256"] for x in b["batches"]]
    # Each run's corpus is its own: seed 1 collects from its own model, and starts from the same offline data.
    assert all(x.manifest["batch_id"].endswith(f"mlp-s1-ac-{source_name(k)}") for k, x in enumerate(c["batches"], 1))


def test_run_manifest_records_regime_rounds_and_checkpoints(corpus, tmp_path):
    tr, va = corpus["uniform"]
    c = cfg("wl", "uniform")
    res = run_onpolicy(c, [tr], [va], PROTOCOL)
    out = write_onpolicy_run(c, [tr], [va], PROTOCOL, res, root=tmp_path / "results", labels_root=tmp_path / "labels")
    raw = load_raw(out)
    assert raw["cost_regime"] == "uniform" and raw["config"]["cost_regime"] == "uniform"
    assert raw["protocol"] == PROTOCOL and raw["rounds_planned"] == 2 and len(raw["rounds"]) == 3
    assert raw["offline_content_sha256"] == [tr.manifest["content_sha256"]]
    for k in range(3):
        _, saved = load(out / f"round{k}.pt")
        assert saved["cost_regime"] == "uniform"
    assert not list(out.glob("*.jsonl.gz"))  # labels stay out of the tracked run folder (OQ-P9a-7)
    for rec, b in zip(raw["onpolicy_batches"], res["batches"]):
        f = tmp_path / "labels" / f"{rec['labels_sha256']}.jsonl.gz"
        assert rec["labels_file"] == str(f) and hashlib.sha256(f.read_bytes()).hexdigest() == rec["labels_sha256"]
        lines = gzip.decompress(f.read_bytes()).decode().splitlines()
        assert [json.loads(x)["h"] for x in lines] == [ex.hstar for ex in b.examples]
    assert "uniform" in (out / "report.md").read_text()
    assert json.loads((out / "provenance.json").read_text())["config"]["cost_regime"] == "uniform"
