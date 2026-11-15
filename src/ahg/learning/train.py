"""Training (D21 §2–§3) and loading of the three model kinds.

train_model(cfg, train_batches, val_batches) -> (model, history). Train
batches are the only data the weights, standardisation statistics or WL
vocabulary are fitted on; val batches are read only to compute the
early-stopping (or, for WL, selection) loss.
"""

import random
import time

import torch
from torch.nn.functional import softplus

from ..cost import SCALE
from ..dataset import LabelBatch
from . import encode
from .models import GNN, MLP, WLLinear, fit_ridge, goal_mask, output_transform, sparse_matrix, wl_transform

KINDS = ("mlp", "wl", "gnn")
LOSSES = ("mse", "asym", "rank")  # rank: GNN only (D22 c, d)


def loss_fn(pred, y, loss: str, w_over: float):
    """MSE, or the squared error with overestimates (pred > h*) weighted w_over (D21 §2)."""
    err2 = (pred - y) ** 2
    if loss == "asym":
        err2 = torch.where(pred > y, w_over * err2, err2)
    return err2.mean()


def _cat(xs):
    return torch.cat(xs) if len(xs) > 1 else xs[0]


def network_ids(batches: list[LabelBatch]) -> torch.Tensor:
    """[states] the validation network (batch, instance) of each example row, as consecutive integers."""
    ids, seen = [], {}
    for b, batch in enumerate(batches):
        for ex in batch.examples:
            ids.append(seen.setdefault((b, ex.instance), len(seen)))
    return torch.tensor(ids, dtype=torch.long)


def macro_mae(pred: torch.Tensor, y: torch.Tensor, net: torch.Tensor) -> float:
    """Validation MAE macro-averaged over networks (D28 F4): the mean over networks of each network's mean
    |h - h*|, so a network with many labelled states counts as much as one with few."""
    err = (pred.double() - y.double()).abs()
    n = int(net.max()) + 1
    total = torch.zeros(n, dtype=torch.float64).index_add_(0, net, err)
    count = torch.zeros(n, dtype=torch.float64).index_add_(0, net, torch.ones_like(err))
    present = count > 0
    return float((total[present] / count[present]).mean())


def _train_mlp(cfg, train, val, history, deadline=None):
    X = _cat([encode.mlp_matrix(b) for b in train])
    y = _cat([encode.targets(b) for b in train])
    G = _cat([encode.goal_flags(b) for b in train])
    Xv = _cat([encode.mlp_matrix(b) for b in val])
    yv = _cat([encode.targets(b) for b in val])
    Gv = _cat([encode.goal_flags(b) for b in val])
    dev = torch.device(cfg["device"])
    mean, std = X.mean(0), X.std(0).clamp_min(1e-6)  # train-only statistics (D14 §5)
    model = MLP(X.shape[1], cfg["hidden"], mean, std).to(dev)
    X, y, G, Xv, yv, Gv = X.to(dev), y.to(dev), G.to(dev), Xv.to(dev), yv.to(dev), Gv.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    gen = torch.Generator().manual_seed(cfg["seed"])

    def epoch():
        model.train()
        total = 0.0
        for idx in torch.randperm(len(y), generator=gen).split(cfg["batch_size"]):
            idx = idx.to(dev)
            loss = loss_fn(output_transform(model(X[idx]), G[idx]), y[idx], cfg["loss"], cfg["w_over"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        return total / len(y)

    net = network_ids(val).to(dev)

    def validate():
        model.eval()
        with torch.no_grad():
            pv = output_transform(model(Xv), Gv)
            return float(loss_fn(pv, yv, cfg["loss"], cfg["w_over"])), macro_mae(pv, yv, net)

    return _early_stop(model, epoch, validate, cfg, history, deadline)


def rank_loss(h, g):
    """Chrestien et al. (2023) Eq. 4 with alpha = beta = 1: sum over pairs of log(1 + exp(f(s_i) - f(s_j))),
    f = g + h, for one instance; s_i is a plan state and s_j an off-plan state of its open list."""
    return softplus(g["dg"] + h[g["pi"]] - h[g["pj"]]).sum()


def gnn_pred(model, g, residual: bool, sl=slice(None)):
    """h for states sl of group g, through the shared output transform (D28 LRN7): the same function of the
    same inputs that Learned.bind computes in search."""
    fx = g["fx"][sl]
    raw = model(fx, g["graph"], g["glob"][sl] if residual else None)
    return output_transform(raw, goal_mask(fx), g["hm"][sl] if residual else None)


def _train_gnn(cfg, train, val, history, deadline=None):
    dev = torch.device(cfg["device"])
    residual, rank = cfg.get("target", "hstar") == "residual", cfg["loss"] == "rank"
    move = lambda gr: {k: ({kk: vv.to(dev) for kk, vv in v.items()} if k == "graph" else
                           v.to(dev) if torch.is_tensor(v) else v) for k, v in gr.items()}
    enc = encode.rank_groups if rank else encode.gnn_groups
    groups = [move(g) for b in train for g in enc(b, residual=residual)]
    vgroups = [move(g) for b in val for g in enc(b, residual=residual)]
    # Checkpoints are selected by h vs h* on every model, whatever its loss (F4): the labelled val states.
    mgroups = vgroups if not rank else [move(g) for b in val for g in encode.gnn_groups(b, residual=residual)]
    model = new_gnn(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = random.Random(f"train:{cfg['seed']}")
    chunk = cfg["batch_size"]

    def group_rank_loss(g):
        # The ranking loss orders the h search uses: h_max + softplus(r) for variant d (D22 §4, D28 LRN7).
        return rank_loss(gnn_pred(model, g, residual), g)

    def epoch():
        model.train()
        order = list(range(len(groups)))
        rng.shuffle(order)
        total = n = 0
        for gi in order:
            g = groups[gi]
            if rank:  # one instance per minibatch, all its pairs (Chrestien et al. §6)
                loss = group_rank_loss(g)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, n = total + float(loss), n + 1
                continue
            for s in range(0, len(g["y"]), chunk):
                sl = slice(s, s + chunk)
                loss = loss_fn(gnn_pred(model, g, residual, sl), g["y"][sl], cfg["loss"], cfg["w_over"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, n = total + float(loss) * len(g["y"][sl]), n + len(g["y"][sl])
        return total / n

    def validate():
        model.eval()
        total = n = 0
        with torch.no_grad():
            for g in vgroups:
                if rank:  # mean over instances of the per-instance summed loss
                    total, n = total + float(group_rank_loss(g)), n + 1
                else:
                    total += float(loss_fn(gnn_pred(model, g, residual), g["y"], cfg["loss"], cfg["w_over"])) * len(g["y"])
                    n += len(g["y"])
            # One group per validation network, so the mean of group MAEs is the macro average.
            mae = sum(float((gnn_pred(model, g, residual) - g["y"]).abs().double().mean()) for g in mgroups) / len(mgroups)
        return total / n, mae

    return _early_stop(model, epoch, validate, cfg, history, deadline)


def new_gnn(cfg: dict) -> GNN:
    """The GNN a config describes; the residual target (D22 a, d) adds the h_max inputs.

    pre_agg / readout pick the aggregator grid cell (D28 Round 11). A config without them is a pre-D28 one,
    whose only architecture was mean / mean; that is what they mean when absent."""
    agg = {"pre_agg": cfg.get("pre_agg", "mean"), "readout": cfg.get("readout", "mean")}
    if cfg.get("target", "hstar") == "residual":
        return GNN(cfg["hidden"], cfg["layers"], fact_dim=4, global_dim=1, **agg)
    return GNN(cfg["hidden"], cfg["layers"], **agg)


def _early_stop(model, epoch, validate, cfg, history, deadline=None):
    """Train up to max_epochs; keep the checkpoint with the lowest validation MAE, macro-averaged over
    networks (D28 F4); stop after `patience` epochs without a strictly lower MAE.

    Tie rule: a later epoch replaces the kept one only with a strictly lower MAE, so among equal MAEs the
    earliest epoch is kept. validate() returns (val loss, macro val MAE); the loss is recorded only.
    deadline: a time.time() past which no new epoch starts (the D21 §5 search budget).
    """
    best, best_state, bad = float("inf"), None, 0
    for e in range(cfg["max_epochs"]):
        if e > 0 and deadline is not None and time.time() >= deadline:
            history.append({"stopped": "budget"})  # a search budget ran out (D21 §5): the run is partial
            break
        t0 = time.perf_counter()
        tr = epoch()
        t_train = time.perf_counter() - t0
        vl, mae = validate()
        history.append({"epoch": e, "train_loss": tr, "val_loss": vl, "val_mae": mae, "epoch_seconds": t_train,
                        "val_seconds": time.perf_counter() - t0 - t_train})
        if mae < best:
            best, bad = mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state is None:
        raise ValueError("no finite validation MAE: no checkpoint to select")
    model.load_state_dict(best_state)
    return model.cpu().eval()


def _train_wl(cfg, train, val, history, deadline=None):
    """deadline (D24 §5, AUDIT_1 L3): checked between IRLS rounds, like the GNN/MLP epoch check, so no new
    round starts after it. A ridge fit in progress is not interrupted: the bound is soft by one fit
    (373 s for the D23 WL), and the first fit always runs, as the first epoch does in _early_stop."""
    hists = [h for b in train for h in encode.wl_histograms(b)]
    y = _cat([encode.targets(b) for b in train]).double()
    G = _cat([encode.goal_flags(b) for b in train])
    vocab: dict[str, int] = {}
    for h in hists:  # fitted on train only; colours unseen in train are dropped later
        for c in sorted(h):
            vocab.setdefault(c, len(vocab))
    X = sparse_matrix(hists, vocab)
    Xv = sparse_matrix([h for b in val for h in encode.wl_histograms(b)], vocab)
    yv = _cat([encode.targets(b) for b in val]).double()
    Gv = _cat([encode.goal_flags(b) for b in val])
    net = network_ids(val)
    w, best = None, None
    rounds = cfg["irls_rounds"] if cfg["loss"] == "asym" else 1
    for r in range(rounds):
        if r > 0 and deadline is not None and time.time() >= deadline:
            history.append({"stopped": "budget"})
            break
        t0 = time.perf_counter()
        beta, bias = fit_ridge(X, y, cfg["lam"], w)
        pred = wl_transform(X @ beta + bias, G)
        # IRLS for the asymmetric loss: reweight by which side of h* each state now falls on.
        w = torch.where(pred > y, torch.full_like(y, cfg["w_over"]), torch.ones_like(y))
        pv = wl_transform(Xv @ beta + bias, Gv)
        mae = macro_mae(pv, yv, net)
        history.append({"epoch": r, "train_loss": float(loss_fn(pred, y, cfg["loss"], cfg["w_over"])),
                        "val_loss": float(loss_fn(pv, yv, cfg["loss"], cfg["w_over"])), "val_mae": mae,
                        "epoch_seconds": time.perf_counter() - t0, "val_seconds": 0.0})
        # F4 / AUDIT_4 inventory #3: keep the IRLS round with the lowest macro val MAE, not the last one;
        # strict < keeps the earliest round among equal MAEs (deterministic tie rule).
        if best is None or mae < best[0]:
            best = (mae, r, beta.clone(), bias)
    history.append({"vocab_size": len(vocab), "selected_round": best[1]})
    return WLLinear(vocab, best[2], best[3])


def train_model(cfg: dict, train: list[LabelBatch], val: list[LabelBatch], deadline: float | None = None):
    """Train one model; returns (model, history). cfg["kind"] in KINDS, cfg["loss"] in LOSSES."""
    if any(b.manifest["split"] != "train" for b in train) or any(b.manifest["split"] != "val" for b in val):
        raise ValueError("train_model takes train batches to fit and val batches to stop on, nothing else")
    if cfg["kind"] != "gnn" and (cfg["loss"] == "rank" or cfg.get("target", "hstar") != "hstar"):
        raise ValueError("the ranking loss and the residual target are implemented for the GNN only (D22)")
    torch.manual_seed(cfg["seed"])
    history: list[dict] = []
    fit = {"mlp": _train_mlp, "gnn": _train_gnn, "wl": _train_wl}[cfg["kind"]]
    return fit(cfg, train, val, history, deadline), history


def save(model, cfg: dict, path) -> None:
    sd = model.state_dict()
    torch.save({"kind": cfg["kind"], "config": cfg, "state_dict": sd}, path)


def load(path):
    ck = torch.load(path, weights_only=False)
    cfg, sd = ck["config"], ck["state_dict"]
    if ck["kind"] == "wl":
        return WLLinear(sd["vocab"], sd["beta"], sd["bias"]), cfg
    if ck["kind"] == "mlp":
        model = MLP(sd["mean"].shape[0], cfg["hidden"], sd["mean"], sd["std"])
    else:
        model = new_gnn(cfg)
    model.load_state_dict(sd)
    return model.eval(), cfg


def effort_to_units(h: float) -> float:
    return h * SCALE
