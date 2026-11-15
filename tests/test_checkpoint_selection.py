"""F4 (D28 Round 11): every model keeps the checkpoint with the lowest validation MAE, macro-averaged over
validation networks, with a deterministic tie rule (earliest); WL keeps its best IRLS round, not the last."""

import pytest
import torch

from ahg.learning import encode, train as train_mod
from ahg.learning.models import output_transform, sparse_matrix, wl_transform
from ahg.learning.runs import make_config
from ahg.learning.train import _early_stop, gnn_pred, macro_mae, network_ids, train_model

from test_learning import batches, no_shared_cache  # noqa: F401  (pytest fixtures)


def test_macro_mae_by_hand():
    # Network 0: four states, |error| 1 each (mean 1). Network 1: one state, |error| 3. Macro 2; micro 1.4.
    pred, y = torch.tensor([1.0, 2.0, 3.0, 4.0, 3.0]), torch.tensor([0.0, 1.0, 2.0, 5.0, 0.0])
    assert macro_mae(pred, y, torch.tensor([0, 0, 0, 0, 1])) == pytest.approx(2.0)
    assert macro_mae(pred, y, torch.tensor([0, 0, 0, 0, 0])) == pytest.approx(1.4)


def test_network_ids_distinguish_batches(batches):
    train, val = batches
    ids = network_ids([train, val])
    n_train = len({ex.instance for ex in train.examples})
    assert int(ids.max()) + 1 == n_train + len({ex.instance for ex in val.examples})
    assert ids[: len(train.examples)].max() < n_train <= ids[len(train.examples):].min()


def test_early_stop_selects_lowest_mae_and_the_earliest_tie():
    model = torch.nn.Linear(1, 1, bias=False)
    script = iter([(1.0, 3.0), (5.0, 1.0), (0.1, 2.0), (9.0, 1.0), (0.0, 4.0)])  # (val loss, val MAE)
    e = iter(range(5))

    def epoch():
        with torch.no_grad():
            model.weight.fill_(float(next(e)))  # the weight records which epoch's state this is
        return 0.0

    history = []
    out = _early_stop(model, epoch, lambda: next(script), {"max_epochs": 5, "patience": 10}, history)
    assert float(out.weight) == 1.0  # epoch 1: lowest MAE; not epoch 2 (lowest loss), not epoch 3 (tie, later)
    assert [h["val_mae"] for h in history] == [3.0, 1.0, 2.0, 1.0, 4.0]


def test_patience_counts_epochs_without_a_lower_mae():
    model = torch.nn.Linear(1, 1)
    script = iter([(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 1.0)])
    history = []
    _early_stop(model, lambda: 0.0, lambda: next(script), {"max_epochs": 4, "patience": 2}, history)
    assert len(history) == 3  # a tie is not an improvement


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_returned_model_has_the_minimum_recorded_mae(batches, kind):
    train, val = batches
    cfg = make_config(kind=kind, max_epochs=3, patience=5, hidden=16, layers=2, loss="asym")
    model, history = train_model(cfg, [train], [val])
    best = min(h["val_mae"] for h in history if "epoch" in h)
    with torch.no_grad():
        if kind == "mlp":
            pv = output_transform(model(encode.mlp_matrix(val)), encode.goal_flags(val))
            got = macro_mae(pv, encode.targets(val), network_ids([val]))
        else:
            gs = encode.gnn_groups(val)
            got = sum(float((gnn_pred(model, g, False) - g["y"]).abs().double().mean()) for g in gs) / len(gs)
    assert got == pytest.approx(best, rel=1e-6)


def test_wl_keeps_its_best_irls_round_not_the_last(batches, monkeypatch):
    """Scripted ridge fits: zero weights and bias b_r in round r, so round r predicts max(0, b_r) everywhere."""
    train, val = batches
    y = encode.targets(val).double()
    biases = iter([50.0, float(y.median()), 40.0, 30.0])
    monkeypatch.setattr(train_mod, "fit_ridge", lambda X, yy, lam, w: (torch.zeros(X.shape[1], dtype=torch.float64), next(biases)))
    cfg = make_config(kind="wl", loss="asym", irls_rounds=4)
    model, history = train_model(cfg, [train], [val])
    maes = [h["val_mae"] for h in history if "epoch" in h]
    assert len(maes) == 4 and maes.index(min(maes)) == 1 and history[-1]["selected_round"] == 1
    assert model.bias == float(y.median())  # round 1's fit, although round 3 was the last
    pv = wl_transform(sparse_matrix(encode.wl_histograms(val), model.vocab) @ model.beta + model.bias, encode.goal_flags(val))
    assert macro_mae(pv, y, network_ids([val])) == pytest.approx(min(maes))
