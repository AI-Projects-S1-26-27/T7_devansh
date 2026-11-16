"""D28 GNN aggregator grid (Round 11, CA2): run on `ac` offline data, select, freeze once, reuse under `uniform`.
Tested on the tiny synthetic task set only (P9a); no real data."""

import json

import pytest

from ahg import aggregator_grid as grid
from ahg.learning import encode
from ahg.learning.models import AGGREGATOR_GRID
from ahg.learning.runs import make_config
from ahg.onpolicy import make_batch, run_onpolicy, source_name
from ahg.runlog import load_raw

from synthetic import PROTOCOL, offline_batch, synthetic_seeds

SMALL = {"max_epochs": 2, "hidden": 8, "layers": 2, "patience": 2}
SEEDS = synthetic_seeds(5)
# The decided rule (P9a2, OQ-P9a-1) with an exact tie, so the order tests see only exact ties.
RULE = {"simplicity_order": [list(c) for c in AGGREGATOR_GRID], "seed_combination": "mean", "tie_relative": 0.0}


@pytest.fixture(autouse=True)
def no_shared_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(encode, "CACHE", tmp_path / "cache")


@pytest.fixture(scope="module")
def ac_data():
    return offline_batch("syn-train-ac", "train", "ac", SEEDS[:3]), offline_batch("syn-val-ac", "val", "ac", SEEDS[3:])


def base(regime="ac", **over):
    return make_config(kind="gnn", cost_regime=regime, **SMALL, **over)


def rows(scores: dict) -> list[dict]:
    """Grid rows from {cell: [val MAE per seed 0, 1, 2]}."""
    return [{"pre_agg": c[0], "readout": c[1], "seed": s, "val_mae": v}
            for c, maes in scores.items() for s, v in zip((0, 1, 2), maes)]


def test_grid_is_six_cells_times_three_seeds():
    cfgs = grid.grid_configs(base())
    assert len(cfgs) == 18
    assert sorted({(c["pre_agg"], c["readout"]) for c in cfgs}) == sorted(AGGREGATOR_GRID)
    assert {c["seed"] for c in cfgs} == {0, 1, 2} and all(c["cost_regime"] == "ac" for c in cfgs)


@pytest.mark.parametrize("bad", [dict(regime="uniform"), dict(regime=None), dict(pre_agg="sum"), dict(readout="sum")])
def test_grid_refuses_uniform_and_preset_cells(bad):
    regime = bad.pop("regime", "ac")
    cfg = base(regime, **bad) if regime else make_config(kind="gnn", **SMALL, **bad)
    with pytest.raises(ValueError):
        grid.grid_configs(cfg)


def test_grid_is_gnn_only():
    with pytest.raises(ValueError):
        grid.grid_configs(make_config(kind="mlp", cost_regime="ac", **SMALL))


def test_grid_refuses_uniform_labels_and_on_policy_data(ac_data):
    tr, va = ac_data
    uni_tr = offline_batch("syn-train-uni", "train", "uniform", SEEDS[:3])
    with pytest.raises(ValueError):
        grid.run_grid(base(), [uni_tr], [va])
    on = make_batch("syn-on", "train", "ac", tr.instances,
                    [type(ex)(ex.instance, ex.state, ex.hstar, [source_name(1)]) for ex in tr.examples[:3]], [])
    with pytest.raises(ValueError, match="non-offline"):
        grid.run_grid(base(), [tr, on], [va])


def test_select_lowest_mae_then_simplicity_order():
    s = {c: [0.5, 0.5, 0.5] for c in AGGREGATOR_GRID}
    s[("max", "sum")] = [0.125, 0.25, 0.375]  # binary-exact: mean 0.25
    assert (grid.select(rows(s), RULE)["pre_agg"], grid.select(rows(s), RULE)["readout"]) == ("max", "sum")
    s[("sum", "mean")] = [0.25, 0.25, 0.25]  # an exact tie on the mean: the first in simplicity order wins
    order = [["sum", "mean"], *[list(c) for c in AGGREGATOR_GRID if c != ("sum", "mean")]]
    sel = grid.select(rows(s), {**RULE, "simplicity_order": order})
    assert (sel["pre_agg"], sel["readout"]) == ("sum", "mean") and len(sel["tied"]) == 2
    rev = grid.select(rows(s), {**RULE, "simplicity_order": order[::-1]})
    assert (rev["pre_agg"], rev["readout"]) == ("max", "sum")
    # 2% relative (the decided tolerance): 1% worse ties, 3% worse does not.
    near = grid.select(rows({**s, ("sum", "mean"): [0.2525] * 3}), {**RULE, "simplicity_order": order, "tie_relative": 0.02})
    assert (near["pre_agg"], near["readout"]) == ("sum", "mean") and len(near["tied"]) == 2
    far = grid.select(rows({**s, ("sum", "mean"): [0.2575] * 3}), {**RULE, "simplicity_order": order, "tie_relative": 0.02})
    assert (far["pre_agg"], far["readout"]) == ("max", "sum") and len(far["tied"]) == 1 and len(far["scores"]) == 6


def test_decided_rule_file():
    rule = grid.load_rule()
    assert rule == {"simplicity_order": [["mean", "mean"], ["mean", "sum"], ["sum", "mean"], ["sum", "sum"],
                                         ["max", "mean"], ["max", "sum"]],
                    "seed_combination": "mean", "tie_relative": 0.02}


@pytest.mark.parametrize("rule", [
    {k: v for k, v in RULE.items() if k != "tie_relative"},
    {**RULE, "simplicity_order": RULE["simplicity_order"][:5]},
    {**RULE, "simplicity_order": RULE["simplicity_order"][:5] + RULE["simplicity_order"][:1]},
    {**RULE, "seed_combination": "median"},
    {**RULE, "tie_relative": -0.1},
    {**RULE, "extra": 1}])
def test_selection_rule_is_never_defaulted(rule):
    with pytest.raises(ValueError):
        grid.select(rows({c: [0.5] * 3 for c in AGGREGATOR_GRID}), rule)


def test_select_needs_every_cell_and_seed():
    full = rows({c: [0.5] * 3 for c in AGGREGATOR_GRID})
    with pytest.raises(ValueError):
        grid.select(full[:-1], RULE)
    with pytest.raises(ValueError):
        grid.select(full + full[:1], RULE)


def test_grid_end_to_end_freeze_and_uniform_reuse(ac_data, tmp_path):
    tr, va = ac_data
    b = base()
    result = grid.run_grid(b, [tr], [va])
    assert len(result) == 18 and all(r["val_mae"] >= 0 for r in result)
    out = grid.write_grid_run(b, [tr], [va], result, root=tmp_path / "runs")
    raw = load_raw(out)
    assert raw["cost_regime"] == "ac" and len(raw["rows"]) == 18 and raw["seeds"] == [0, 1, 2]
    sel = grid.select(result, RULE)
    path = tmp_path / "d28_aggregators.json"
    frozen = grid.freeze(sel, RULE, str(out), path)
    assert frozen["cost_regime"] == "ac" and grid.freeze(sel, RULE, str(out), path) == frozen  # idempotent
    other = next(c for c in AGGREGATOR_GRID if c != (sel["pre_agg"], sel["readout"]))
    with pytest.raises(FileExistsError):  # frozen once: a different choice is never written over it
        grid.freeze({**sel, "pre_agg": other[0], "readout": other[1]}, RULE, str(out), path)
    agg = grid.frozen_aggregators(path)
    assert (agg["pre_agg"], agg["readout"]) == (sel["pre_agg"], sel["readout"])
    # The uniform set reuses the frozen ac choice; no grid runs under uniform.
    utr = offline_batch("syn-train-uni", "train", "uniform", SEEDS[:3])
    uva = offline_batch("syn-val-uni", "val", "uniform", SEEDS[3:])
    cfg = make_config(kind="gnn", seed=0, cost_regime="uniform", pre_agg=agg["pre_agg"], readout=agg["readout"], **SMALL)
    res = run_onpolicy(cfg, [utr], [uva], PROTOCOL)
    assert len(res["rounds"]) == 3 and all(m.pre_agg == agg["pre_agg"] for m in res["models"])


def test_frozen_file_must_come_from_the_ac_grid(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"cost_regime": "uniform", "pre_agg": "mean", "readout": "sum"}))
    with pytest.raises(ValueError):
        grid.frozen_aggregators(p)
    p.write_text(json.dumps({"cost_regime": "ac", "pre_agg": "min", "readout": "sum"}))
    with pytest.raises(ValueError):
        grid.frozen_aggregators(p)
