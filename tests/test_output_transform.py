"""LRN7 (D28 Round 11): one output transform for training and search, h(goal) = 0.

The same checkpoint and state must give identical values through the training path (batched, as training
and validation compute them) and the search path (Learned.bind, one state per call), for positive, zero
and negative raw outputs and for goal states."""

import math
from collections import Counter

import pytest
import torch
from torch.nn.functional import softplus

from ahg.dataset import mlp_features, static_graph, wl_colours
from ahg.learning.encode import gnn_inputs
from ahg.learning.heuristic import Learned
from ahg.learning.models import MLP, WLLinear, output_transform, sparse_matrix, wl_transform
from ahg.learning.runs import make_config
from ahg.learning.train import gnn_pred, new_gnn

from test_learning import batches, no_shared_cache  # noqa: F401  (pytest fixtures)

RAW = [2.0, 0.0, -3.0]  # positive, zero and negative raw outputs
# Same function, same inputs; torch's vectorised float32 kernels (a batch in training, one state in search) can
# still differ in the last bit, so values are compared to float32 rounding (rel 1e-6), and h(goal) exactly.
F32 = dict(rtol=1e-6, atol=0.0)


LS, CS = 3.5, 2.25  # label and cost scales different from 1 and from each other, so a missing one shows


def _same(train, h, inst, states, scale=LS):
    search = torch.tensor([h(inst.task, s) for s in states], dtype=torch.float64)
    assert torch.allclose(scale * train.double(), search, **F32), (train, search)
    assert search[-1] == 0 and train[-1] == 0  # h(goal) = 0 explicitly, in both paths


def _states(val):
    """Some labelled states of one instance plus a goal state (every fact held)."""
    i = val.examples[0].instance
    inst = val.instances[i]
    states = [ex.state for ex in val.examples if ex.instance == i][:10]
    goal = (1 << len(inst.task.facts)) - 1
    assert inst.task.is_goal(goal) and not any(inst.task.is_goal(s) for s in states)
    return inst, states + [goal]


def _constant(model_last: torch.nn.Linear, c: float):
    with torch.no_grad():
        model_last.weight.zero_()
        model_last.bias.fill_(c)


def test_transform_by_hand():
    raw, goal, hm = torch.tensor([2.0, 0.0, -3.0, 5.0]), torch.tensor([False, False, False, True]), torch.tensor([1.0, 1.0, 1.0, 0.0])
    assert torch.allclose(output_transform(raw, goal), torch.tensor([math.log1p(math.e ** 2), math.log(2), math.log1p(math.e ** -3), 0.0]))
    assert torch.allclose(output_transform(raw, goal, hm), hm + torch.tensor([*softplus(raw[:3]), -1.0]).where(~goal, torch.tensor(0.0)))
    assert output_transform(raw, goal, hm)[3] == 0 and (output_transform(raw, goal)[:3] > 0).all()
    assert torch.equal(wl_transform(raw, goal), torch.tensor([2.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("c", RAW)
@pytest.mark.parametrize("residual", [False, True])
def test_gnn_training_and_search_values_are_identical(batches, c, residual):
    _, val = batches
    cfg = make_config(kind="gnn", hidden=8, layers=2, **({"target": "residual"} if residual else {}))
    torch.manual_seed(0)
    model = new_gnn(cfg, LS, CS).eval()
    _constant(model.head[-1], c)
    inst, states = _states(val)
    g = {"graph": static_graph(inst), **gnn_inputs(inst.task, states, residual)}
    with torch.no_grad():
        train = gnn_pred(model, g, residual)
    h = Learned(model, cfg, "m").bind(inst)
    _same(train, h, inst, states)
    expect = torch.tensor(softplus(torch.tensor(c)).item()) + (g["hm"][:-1] / LS if residual else 0)
    assert torch.allclose(train[:-1], expect.expand(len(states) - 1), atol=1e-6)


@pytest.mark.parametrize("residual", [False, True])
def test_gnn_parity_with_trained_like_weights(batches, residual):
    _, val = batches
    cfg = make_config(kind="gnn", hidden=8, layers=2, **({"target": "residual"} if residual else {}))
    torch.manual_seed(1)
    model = new_gnn(cfg, LS, CS).eval()
    inst, states = _states(val)
    g = {"graph": static_graph(inst), **gnn_inputs(inst.task, states, residual)}
    with torch.no_grad():
        train = gnn_pred(model, g, residual)
    _same(train, Learned(model, cfg, "m").bind(inst), inst, states)


@pytest.mark.parametrize("c", RAW)
def test_mlp_training_and_search_values_are_identical(batches, c):
    _, val = batches
    inst, states = _states(val)
    X = torch.tensor([mlp_features(inst.task, s) for s in states], dtype=torch.float32)
    model = MLP(X.shape[1], 8, X.mean(0), X.std(0).clamp_min(1e-6), LS).eval()
    _constant(model.net[-1], c)
    G = torch.tensor([inst.task.is_goal(s) for s in states])
    with torch.no_grad():
        train = output_transform(model(X), G)  # what _train_mlp computes on its val rows
    h = Learned(model, make_config(kind="mlp"), "m").bind(inst)
    _same(train, h, inst, states)
    assert torch.allclose(train[:-1], softplus(torch.tensor(c)).expand(len(states) - 1))


@pytest.mark.parametrize("c", RAW)
def test_wl_training_and_search_values_are_identical(batches, c):
    _, val = batches
    inst, states = _states(val)
    hists = [wl_colours(inst.task, s) for s in states]
    vocab = {col: k for k, col in enumerate(sorted({col for hh in hists for col in hh}))}
    beta = torch.linspace(-0.01, 0.01, len(vocab), dtype=torch.float64)
    model = WLLinear(vocab, beta, c, LS)
    G = torch.tensor([inst.task.is_goal(s) for s in states])
    train = wl_transform(sparse_matrix(hists, vocab) @ beta + c, G)  # what _train_wl computes
    h = Learned(model, make_config(kind="wl"), "m").bind(inst)
    search = torch.tensor([h(inst.task, s) for s in states], dtype=torch.float64)
    assert torch.allclose(LS * train, search, rtol=1e-12, atol=0.0) and search[-1] == 0 and (search >= 0).all()
