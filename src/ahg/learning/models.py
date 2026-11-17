"""The three learned heuristics of D21 §1. Each predicts h*(s) / label_scale and is ≥ 0.

label_scale (P9a2, OQ-P9a-4) is fitted on the model's own training labels: the mean h* of its training
examples, so it is that cost regime's own unit and the training targets have mean 1. It is saved with the
model; search multiplies by it. The GNN also divides its cost-valued inputs (action cost, per-fact and
global h_max) by cost_scale, which equals label_scale for every model trained since P9a2. Checkpoints from
before P9a2 have neither and load with the D9 constants they were trained with (train.load).

The networks (MLP, GNN) return a raw output; output_transform is the one map
from raw output to h (D28 LRN7), used by training, validation and search alike:
softplus (not ReLU) keeps h_θ ≥ 0 while still passing a gradient when the raw
output is negative, the residual GNN adds it to h_max, and h(goal) = 0 is set
explicitly. The WL model is linear and fitted by ridge regression, so its map
is max(0, ·), again with h(goal) = 0 (wl_transform).
"""

from collections import Counter

import torch
from torch import nn
from torch.nn.functional import softplus

from ..dataset import ACTION_DIM

FACT_DIM = 2


def output_transform(raw: torch.Tensor, is_goal: torch.Tensor, hm: torch.Tensor | None = None) -> torch.Tensor:
    """h from a network's raw output (D28 LRN7): softplus(raw), or h_max + softplus(raw) for the residual
    GNN, and exactly 0 at goal states. The only place this map is defined."""
    h = softplus(raw) if hm is None else hm + softplus(raw)
    return torch.where(is_goal, torch.zeros_like(h), h)


def wl_transform(raw: torch.Tensor, is_goal: torch.Tensor) -> torch.Tensor:
    """h from the linear WL model's raw output: max(0, raw), and exactly 0 at goal states."""
    return torch.where(is_goal, torch.zeros_like(raw), raw.clamp_min(0))


def goal_mask(fx: torch.Tensor) -> torch.Tensor:
    """[B] True where every goal fact is held; fx [B, facts, >= 2] with channels held, is_goal."""
    return ((fx[..., 1] == 0) | (fx[..., 0] == 1)).all(-1)


class MLP(nn.Module):
    """Graph-level hand-built features -> h. Standardisation stats are buffers, saved with the weights."""

    def __init__(self, n_in: int, hidden: int, mean: torch.Tensor, std: torch.Tensor, label_scale: float):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("label_scale", torch.tensor(float(label_scale)))
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        """Raw output; h = output_transform(raw, is_goal)."""
        return self.net((x - self.mean) / self.std).squeeze(-1)


# The pre-registered aggregator grid (D28 Round 11, AUDIT_4 inventory #4): how an action combines its
# preconditions (the AND side) x how the graph is read out. mean/mean is the D21 architecture.
PRE_AGGS = ("mean", "sum", "max")
READOUTS = ("mean", "sum")
AGGREGATOR_GRID = tuple((p, r) for p in PRE_AGGS for r in READOUTS)


def aggregate_pre(f: torch.Tensor, g: dict, mode: str) -> torch.Tensor:
    """[B, actions, H] message to each action from its preconditions' fact embeddings f [B, facts, H].

    mean / sum / max over the preconditions; an action without preconditions gets 0 in every mode.
    max is an elementwise maximum (scatter_reduce "amax"), the closest smooth-free analogue of h_max's AND.
    """
    b, n_a, hdim = f.shape[0], g["act_x"].shape[0], f.shape[2]
    src = f[:, g["pre_src"]]  # [B, edges, H]
    out = f.new_zeros(b, n_a, hdim)
    if mode == "max":
        idx = g["pre_dst"].view(1, -1, 1).expand_as(src)
        return out.scatter_reduce(1, idx, src, "amax", include_self=False)
    out = out.index_add(1, g["pre_dst"], src)
    if mode == "mean":
        return out * g["inv_deg_pre"]
    if mode == "sum":
        return out
    raise ValueError(f"pre_agg must be one of {PRE_AGGS}, not {mode!r}")


def readout(x: torch.Tensor, mode: str) -> torch.Tensor:
    """[B, H] graph-level pooling of node embeddings x [B, nodes, H]."""
    if mode == "mean":
        return x.mean(1)
    if mode == "sum":
        return x.sum(1)
    raise ValueError(f"readout must be one of {READOUTS}, not {mode!r}")


class GNN(nn.Module):
    """Message passing over the fact-action graph; one instance's states per call (shared graph).

    fact_dim/global_dim > their defaults give the D22 residual model its h_max inputs (per-fact and
    global). forward returns the raw output; h = output_transform(raw, is_goal[, h_max]).
    pre_agg / readout: one cell of AGGREGATOR_GRID. action_dim: the width of act_x, the D19 encoder's
    ACTION_DIM or the D28 encoder's (ahg.learning.d28encode.ACTION_DIM, cost first in both).
    """

    def __init__(self, hidden: int, layers: int, fact_dim: int = FACT_DIM, global_dim: int = 0,
                 pre_agg: str = "mean", readout: str = "mean", *, label_scale: float, cost_scale: float,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        if (pre_agg, readout) not in AGGREGATOR_GRID:
            raise ValueError(f"(pre_agg, readout) = {(pre_agg, readout)} is not in the grid {AGGREGATOR_GRID}")
        self.pre_agg, self.readout = pre_agg, readout
        self.register_buffer("label_scale", torch.tensor(float(label_scale)))
        self.register_buffer("cost_scale", torch.tensor(float(cost_scale)))
        self.enc_f = nn.Linear(fact_dim, hidden)
        self.enc_a = nn.Linear(action_dim, hidden)
        self.up_a = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.up_f = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(3 * hidden + global_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, fx, g, glob=None):
        """fx: [B, facts, fact_dim]; g: dataset.static_graph of the instance; glob: [B, global_dim] or None."""
        b, n_f = fx.shape[0], fx.shape[1]
        # Inputs arrive in the regime's raw cost units; only the cost-valued channels are rescaled: action
        # cost (act_x column 0), the per-fact h_max cost (fx channel 3 of the residual model) and glob.
        act_x = torch.cat([g["act_x"][:, :1] / self.cost_scale, g["act_x"][:, 1:]], -1)
        if fx.shape[-1] > FACT_DIM:
            fx = torch.cat([fx[..., :3], fx[..., 3:] / self.cost_scale], -1)
        if glob is not None:
            glob = glob / self.cost_scale
        f = torch.relu(self.enc_f(fx))
        a = torch.relu(self.enc_a(act_x)).expand(b, -1, -1)
        for up_a, up_f in zip(self.up_a, self.up_f):
            # An action sees the mean / sum / max of its preconditions: the AND side of the graph.
            m_a = aggregate_pre(f, g, self.pre_agg)
            a = torch.relu(up_a(torch.cat([a, m_a], -1)))
            # A fact sees the mean of the actions that add it: the OR side.
            m_f = f.new_zeros(b, n_f, f.shape[2]).index_add_(1, g["add_dst"], a[:, g["add_src"]]) * g["inv_deg_add"]
            f = torch.relu(up_f(torch.cat([f, m_f], -1)))
        pooled = [readout(f, self.readout), readout(a, self.readout), readout(f[:, g["goal"]], self.readout)]
        pooled += [glob] if glob is not None else []
        return self.head(torch.cat(pooled, -1)).squeeze(-1)


class WLLinear:
    """Linear model on WL colour counts (WL-GOOSE style). vocab: colour -> column, fitted on train only."""

    def __init__(self, vocab: dict[str, int], beta: torch.Tensor, bias: float, label_scale: float):
        self.vocab, self.beta, self.bias, self.label_scale = vocab, beta, bias, float(label_scale)
        self._w = {c: float(beta[k]) for c, k in vocab.items()}  # dict lookup is faster than a sparse dot here

    def raw_one(self, hist: Counter) -> float:
        return self.bias + sum(n * self._w.get(c, 0.0) for c, n in hist.items())

    def state_dict(self) -> dict:
        return {"vocab": self.vocab, "beta": self.beta, "bias": self.bias, "label_scale": self.label_scale}


def sparse_matrix(hists: list[Counter], vocab: dict[str, int]) -> torch.Tensor:
    """CSR [states, |vocab|] of colour counts; colours outside the vocabulary are dropped."""
    crow, col, val = [0], [], []
    for h in hists:
        for c, n in sorted((vocab[c], n) for c, n in h.items() if c in vocab):
            col.append(c)
            val.append(float(n))
        crow.append(len(col))
    return torch.sparse_csr_tensor(torch.tensor(crow), torch.tensor(col, dtype=torch.long),
                                   torch.tensor(val, dtype=torch.float64), size=(len(hists), len(vocab)))


def fit_ridge(X: torch.Tensor, y: torch.Tensor, lam: float, w: torch.Tensor | None = None, iters: int = 300,
              tol: float = 1e-10) -> tuple[torch.Tensor, float]:
    """Weighted ridge regression, min sum w_i (x_i b + c - y_i)^2 + lam |b|^2, by conjugate gradients.

    The bias c is unpenalised: it is solved exactly by centring with the weights,
    then CG solves the centred normal equations (X'WX + lam I) b = X'W y on the
    sparse X without ever forming X'X.
    """
    y = y.double()
    w = torch.ones_like(y) if w is None else w.double()
    sw = w.sum()
    Xt = X.t().to_sparse_csr()
    xbar = (Xt @ w) / sw  # weighted column means
    ybar = float((w * y).sum() / sw)
    yc = y - ybar

    def A(v):
        xv = X @ v - float(xbar @ v)  # centred columns applied implicitly
        r = Xt @ (w * xv) - xbar * float((w * xv).sum())
        return r + lam * v

    b_rhs = Xt @ (w * yc) - xbar * float((w * yc).sum())
    beta = torch.zeros(X.shape[1], dtype=torch.float64)
    r = b_rhs - A(beta)
    p, rs = r.clone(), float(r @ r)
    for _ in range(iters):
        if rs <= tol * max(1.0, float(b_rhs @ b_rhs)):
            break
        ap = A(p)
        alpha = rs / float(p @ ap)
        beta += alpha * p
        r -= alpha * ap
        rs_new = float(r @ r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return beta, ybar - float(xbar @ beta)
