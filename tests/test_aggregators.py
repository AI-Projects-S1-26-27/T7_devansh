"""GNN aggregator grid (D28 Round 11, AUDIT_4 inventory #4): precondition aggregation {mean, sum, max} x
readout {mean, sum}, configurable, saved with the checkpoint."""

import pytest
import torch

from ahg.learning.heuristic import Learned
from ahg.learning.models import AGGREGATOR_GRID, GNN, aggregate_pre, readout
from ahg.learning.runs import make_config
from ahg.learning.train import load, new_gnn, save, train_model

from test_learning import batches, no_shared_cache  # noqa: F401  (pytest fixtures)

# 3 facts, 3 actions. Action 0 has preconditions {0, 1}, action 1 has {2}, action 2 has none.
G = {"pre_src": torch.tensor([0, 1, 2]), "pre_dst": torch.tensor([0, 0, 1]),
     "inv_deg_pre": torch.tensor([[0.5], [1.0], [1.0]]), "act_x": torch.zeros(3, 7)}
F = torch.tensor([[[1.0, 4.0], [3.0, 2.0], [5.0, 6.0]]])  # [B=1, facts, H=2]


def test_grid_is_the_pre_registered_six():
    assert sorted(AGGREGATOR_GRID) == sorted((p, r) for p in ("mean", "sum", "max") for r in ("mean", "sum"))


@pytest.mark.parametrize("mode, want", [("mean", [[2.0, 3.0], [5.0, 6.0], [0.0, 0.0]]),
                                        ("sum", [[4.0, 6.0], [5.0, 6.0], [0.0, 0.0]]),
                                        ("max", [[3.0, 4.0], [5.0, 6.0], [0.0, 0.0]])])
def test_precondition_aggregation_by_hand(mode, want):
    assert torch.equal(aggregate_pre(F, G, mode), torch.tensor([want]))


def test_max_is_a_real_max_for_negative_values():
    assert torch.equal(aggregate_pre(-F, G, "max")[0, 0], torch.tensor([-1.0, -2.0]))


def test_readout_by_hand():
    assert torch.equal(readout(F, "mean"), torch.tensor([[3.0, 4.0]]))
    assert torch.equal(readout(F, "sum"), torch.tensor([[9.0, 12.0]]))


def test_invalid_aggregators_are_refused():
    with pytest.raises(ValueError):
        GNN(8, 2, pre_agg="min")
    with pytest.raises(ValueError):
        GNN(8, 2, readout="max")
    with pytest.raises(ValueError):
        make_config(kind="gnn", pooling="sum")


def test_absent_keys_are_the_d21_mean_mean(batches):
    torch.manual_seed(0)
    a = new_gnn(make_config(kind="gnn", hidden=8, layers=2))
    b = new_gnn(make_config(kind="gnn", hidden=8, layers=2, pre_agg="mean", readout="mean"))
    assert (a.pre_agg, a.readout) == (b.pre_agg, b.readout) == ("mean", "mean")


@pytest.mark.parametrize("pre_agg, ro", AGGREGATOR_GRID)
@pytest.mark.parametrize("target", ["hstar", "residual"])
def test_every_cell_trains_and_round_trips(batches, tmp_path, pre_agg, ro, target):
    train, val = batches
    cfg = make_config(kind="gnn", hidden=8, layers=2, max_epochs=1, pre_agg=pre_agg, readout=ro, target=target)
    model, _ = train_model(cfg, [train], [val])
    assert (model.pre_agg, model.readout) == (pre_agg, ro)
    save(model, cfg, tmp_path / "m.pt")
    model2, cfg2 = load(tmp_path / "m.pt")
    assert (model2.pre_agg, model2.readout) == (pre_agg, ro)
    inst = val.instances[val.examples[0].instance]
    states = [ex.state for ex in val.examples if ex.instance == val.examples[0].instance][:5]
    h1, h2 = Learned(model, cfg, "a").bind(inst), Learned(model2, cfg2, "b").bind(inst)
    assert [h1(inst.task, s) for s in states] == [h2(inst.task, s) for s in states]


def test_cells_compute_different_functions(batches):
    """The settings are wired into the forward pass: same weights, different cells, different outputs."""
    _, val = batches
    from ahg.dataset import static_graph
    from ahg.learning.encode import fact_bits
    inst = val.instances[val.examples[0].instance]
    states = [ex.state for ex in val.examples if ex.instance == val.examples[0].instance][:5]
    g, fx = static_graph(inst), fact_bits(inst.task, states)
    torch.manual_seed(0)
    base = GNN(8, 2)
    outs = []
    for p, r in AGGREGATOR_GRID:
        m = GNN(8, 2, pre_agg=p, readout=r)
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            outs.append(m(fx, g))
    assert len({tuple(o.tolist()) for o in outs}) == len(AGGREGATOR_GRID)
