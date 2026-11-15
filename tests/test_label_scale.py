"""P9a2 (OQ-P9a-4): the model's unit comes from its own regime's training labels, not the D9 constant 38.
Pre-P9a2 checkpoints still load with the D9 constants they were trained with."""

import pytest
import torch

from ahg.dataset import Example, mlp_features
from ahg.learning import encode
from ahg.learning.heuristic import Learned
from ahg.learning.models import GNN, MLP, output_transform
from ahg.learning.runs import make_config
from ahg.learning.train import LEGACY_COST_SCALE, LEGACY_LABEL_SCALE, label_scale, load, save, train_model
from ahg.onpolicy import make_batch

from synthetic import instance, offline_batch, synthetic_seeds

SMALL = {"max_epochs": 2, "irls_rounds": 2, "hidden": 8, "layers": 2, "patience": 2}
SEEDS = synthetic_seeds(5)


@pytest.fixture(autouse=True)
def no_shared_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(encode, "CACHE", tmp_path / "cache")


@pytest.fixture(scope="module")
def corpus():
    return {r: (offline_batch(f"t-{r}", "train", r, SEEDS[:3]), offline_batch(f"v-{r}", "val", r, SEEDS[3:]))
            for r in ("ac", "uniform")}


def test_scale_is_the_mean_training_label(corpus):
    for r, (tr, _) in corpus.items():
        hs = [ex.hstar for ex in tr.examples]
        assert label_scale([tr]) == pytest.approx(sum(hs) / len(hs), rel=1e-7)
    tr, _ = corpus["ac"]
    with pytest.raises(ValueError):
        label_scale([make_batch("e", "train", "ac", tr.instances, [], [])])
    zero = [Example(0, tr.instances[0].task.init, 0, ["walk"])]
    with pytest.raises(ValueError):
        label_scale([make_batch("z", "train", "ac", tr.instances, zero, [])])


@pytest.mark.parametrize("kind", ["mlp", "wl", "gnn"])
def test_each_regime_trains_in_its_own_unit(corpus, kind, tmp_path):
    scales = {}
    for r, (tr, va) in corpus.items():
        cfg = make_config(kind=kind, seed=0, cost_regime=r, **SMALL)
        model, _ = train_model(cfg, [tr], [va])
        scales[r] = float(model.label_scale)
        assert scales[r] == label_scale([tr])  # from this regime's training labels only, never the val labels
        save(model, cfg, tmp_path / f"{r}.pt")
        assert float(load(tmp_path / f"{r}.pt")[0].label_scale) == scales[r]
    # Every ac action costs 4 or 7 and every uniform action 1, so the units differ, and neither is D9's 38.
    assert scales["ac"] > 3 * scales["uniform"] and LEGACY_LABEL_SCALE not in scales.values()


def test_scaled_targets_have_mean_one(corpus):
    for tr, _ in corpus.values():
        assert float((encode.targets(tr) / label_scale([tr])).mean()) == pytest.approx(1.0, rel=1e-6)


def _legacy(path, kind, cfg, sd):
    sd = {k: v for k, v in sd.items() if k not in ("label_scale", "cost_scale")}
    torch.save({"kind": kind, "config": cfg, "state_dict": sd}, path)


def test_pre_p9a2_mlp_checkpoint_loads_with_its_d9_feature_scaling(tmp_path):
    inst = instance(SEEDS[0], "ac")
    states = [inst.task.init] + [s for _, s in inst.task.successors(inst.task.init)]
    raw = torch.tensor([mlp_features(inst.task, s) for s in states], dtype=torch.float32)
    old = raw.clone()
    old[:, [3, 5, 6]] /= LEGACY_LABEL_SCALE  # what mlp_features returned before P9a2
    torch.manual_seed(0)
    m = MLP(raw.shape[1], 8, old.mean(0), old.std(0).clamp_min(1e-6), 1.0)
    cfg = make_config(kind="mlp", hidden=8)
    _legacy(tmp_path / "m.pt", "mlp", cfg, m.state_dict())
    with torch.no_grad():
        want = LEGACY_LABEL_SCALE * output_transform(m(old), torch.zeros(len(states), dtype=torch.bool))
    loaded, cfg2 = load(tmp_path / "m.pt")
    h = Learned(loaded, cfg2, "old").bind(inst)
    got = torch.tensor([h(inst.task, s) for s in states])
    # Equal up to float32 rounding: (x - 38m) / 38s instead of (x/38 - m) / s, carried through the network.
    assert torch.allclose(got, want, rtol=1e-4) and float(loaded.label_scale) == LEGACY_LABEL_SCALE


def test_pre_p9a2_gnn_and_wl_checkpoints_get_the_d9_constants(tmp_path):
    cfg = make_config(kind="gnn", hidden=8, layers=2)
    _legacy(tmp_path / "g.pt", "gnn", cfg, GNN(8, 2, label_scale=1.0, cost_scale=1.0).state_dict())
    g, _ = load(tmp_path / "g.pt")
    assert (float(g.label_scale), float(g.cost_scale)) == (LEGACY_LABEL_SCALE, LEGACY_COST_SCALE) == (38, 380)
    torch.save({"kind": "wl", "config": make_config(kind="wl"),
                "state_dict": {"vocab": {}, "beta": torch.zeros(0, dtype=torch.float64), "bias": 1.0}}, tmp_path / "w.pt")
    assert load(tmp_path / "w.pt")[0].label_scale == LEGACY_LABEL_SCALE
