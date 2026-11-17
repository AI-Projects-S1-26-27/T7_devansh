"""D28 learned inputs (Round 11 LRN6; P7c, OQ-P7b-1, decisions §19): one encoder for the GNN, the MLP and the
WL baseline, used unchanged by training and by search.

It reads only the planner-visible part of a generated task document (planner_view): "task" (facts, actions
with both regime costs, init, goal) and "network". Of the network it reads only each host's OS family
(network.hosts[].os) and each principal index's identity class (network.principal_classes); the generator
writes both (decided 17 Nov 2026: the generator exposes them, d28-gen). The "metadata" block (CVEs,
products, versions, CPE names, principals by account name), "params" (seeds) and "stats" are dropped before
anything is encoded, so no feature can come from them.

Per action (ACTION_FEATURES; one row of the GNN's act_x, cost first):
  cost             the action's cost in the active regime, raw (the GNN divides it by its cost_scale)
  type             remote_exploit | local_escalation   from the action name "remote a<i>" / "local a<i>"
  grants           foothold | admin                    admin if the action adds an admin(h) fact
  av               N | L                               the pool admits remote units only at AV:N and local
                                                       units only at AV:L (ahg.gen.pool), so AV follows type
  ac               L | H                               from the action's ac-regime cost (AC:L = 4, AC:H = 7);
                                                       read in both regimes, since AC is a feature, not a cost
  effect           RCE_CONFIRMED | ADMIN_CONFIRMED     the pool's effect label of every remote / local unit
  os               linux | windows                     OS family of the host the action acts on
  principal_class  admin | service | UNKNOWN           the runtime principal context, never the account name:
                                                       a remote action's foothold runs as the identity of the
                                                       exploited service; a local escalation starts from the
                                                       identity of its precondition foothold
Every categorical group is one-hot with exactly one bit set per action; a value outside its vocabulary is
refused (EncodingError), so no group can silently encode as all zeros (the OQ-P7b-1 defect of the D19
encoder on D28 actions).

Per fact and state: held, is_goal (and for the residual GNN h_max-reachable and h_max cost), as for D19
(encode.gnn_inputs). The graph structure is the precondition and add edges (dataset.graph_of).
MLP: the 7 D21 graph-level features plus, per categorical action column, the number of applicable actions
with it (DECIDED-BY-AGENT, OQ-P7c-2). WL: the D19 refinement with an action's initial colour built from its
categorical values and cost instead of its name's kind and cost.

Leakage audit (LRN6): audit_docs encodes D28 tasks, collects the feature names the encoding exposes (the
fixed names plus every categorical value the documents actually carry) and runs ahg.learning.leakage
against configs/learning/d28_features.json; a forbidden or unknown name fails it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from ..cost import AC_COSTS, check_regime
from ..dataset import graph_of, mlp_features as d21_mlp_features, wl_colours as d21_wl_colours
from ..task import Task
from . import leakage
from .encode import gnn_inputs

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / "configs" / "learning" / "d28_features.json"
FEATURE_SET = "d28"  # training config key "features"; absent means the D19 encoder (dataset.py)

PLANNER_KEYS = ("task", "network")  # the only parts of a document the encoder may see
OS_FAMILIES = ("linux", "windows")
PRINCIPAL_CLASSES = ("attacker", "admin", "service", "UNKNOWN")  # network.principal_classes vocabulary
# Categorical action groups, in column order; values are what the one-hot columns are named after.
GROUPS: dict[str, tuple[str, ...]] = {
    "type": ("remote_exploit", "local_escalation"),
    "grants": ("foothold", "admin"),
    "av": ("N", "L"),
    "ac": ("L", "H"),
    "effect": ("RCE_CONFIRMED", "ADMIN_CONFIRMED"),
    "os": OS_FAMILIES,
    "principal_class": ("admin", "service", "UNKNOWN"),  # an action never runs as the attacker's own p0
}
ONE_HOT = tuple(f"{g}={v}" for g, vs in GROUPS.items() for v in vs)
ACTION_FEATURES = ("cost",) + ONE_HOT
ACTION_DIM = len(ACTION_FEATURES)
FACT_FEATURES = ("held", "is_goal")
RESIDUAL_FACT_FEATURES = ("hmax_reachable", "hmax_cost")
GRAPH_FEATURES = ("edge=precondition", "edge=add")
MLP_FEATURES = ("n_facts", "n_actions", "n_held", "h_sp", "n_applicable", "applicable_cost_mean",
                "applicable_cost_min") + tuple(f"applicable:{c}" for c in ONE_HOT)

_KIND = {"remote": "remote_exploit", "local": "local_escalation"}
_AV = {"remote_exploit": "N", "local_escalation": "L"}  # ahg.gen.pool: remote units AV:N, local units AV:L
_EFFECT = {"remote_exploit": "RCE_CONFIRMED", "local_escalation": "ADMIN_CONFIRMED"}
_AC_OF_COST = {c: ac for ac, c in AC_COSTS.items()}
_FOOTHOLD = re.compile(r"foothold\(h(\d+),p(\d+)\)")
_ADMIN = re.compile(r"admin\(h(\d+)\)")


class EncodingError(ValueError):
    """A task document the D28 encoder cannot encode without guessing."""


def feature_names(residual: bool = False) -> list[str]:
    """Every fixed feature name the D28 encoding exposes (GNN facts, actions, graph; MLP; WL uses the same
    fact and action values as its initial colours)."""
    fact = FACT_FEATURES + (RESIDUAL_FACT_FEATURES if residual else ())
    return list(dict.fromkeys(fact + ACTION_FEATURES + GRAPH_FEATURES + MLP_FEATURES))


def planner_view(doc: dict) -> dict:
    """The planner-visible part of a task document; everything else (metadata, params, stats, flags) is gone."""
    missing = [k for k in PLANNER_KEYS if k not in doc]
    if missing:
        raise EncodingError(f"task document has no {missing}")
    return {k: doc[k] for k in PLANNER_KEYS}


def _fact(name: str) -> tuple[str, int, int | None]:
    """(kind, host, principal index or None) of a D28 fact name."""
    if m := _FOOTHOLD.fullmatch(name):
        return "foothold", int(m[1]), int(m[2])
    if m := _ADMIN.fullmatch(name):
        return "admin", int(m[1]), None
    raise EncodingError(f"fact {name!r} is neither foothold(h,p) nor admin(h)")


def action_values(view: dict) -> list[dict[str, str]]:
    """Per action, its raw categorical values (group -> value), read from the planner view only and not yet
    checked against the vocabularies (the audit sees exactly what a document carries)."""
    net = view["network"]
    if "principal_classes" not in net or any("os" not in h for h in net["hosts"]):
        raise EncodingError("task document has no planner-visible host OS or principal classes "
                            "(network.hosts[].os, network.principal_classes); the generator writes them (d28-gen, "
                            "decided 17 Nov 2026, OQ-P7c-1)")
    os_of = {h["id"]: h["os"] for h in net["hosts"]}
    classes = net["principal_classes"]
    facts = [_fact(f) for f in view["task"]["facts"]]
    out = []
    for a in view["task"]["actions"]:
        kind = _KIND.get(a["name"].split()[0])
        if kind is None:
            raise EncodingError(f"action {a['name']!r} is neither remote nor local")
        added, pre = [facts[i] for i in a["add"]], [facts[i] for i in a["pre"]]
        hosts = {h for _, h, _ in added}
        if len(hosts) != 1:
            raise EncodingError(f"action {a['name']!r} acts on hosts {sorted(hosts)}, not one")
        # The identity the action's code runs as: the granted foothold's (remote), the precondition foothold's
        # (local escalation). Each remote action adds one foothold; each local one has one foothold precondition.
        side = added if kind == "remote_exploit" else pre
        ks = [k for kind_, _, k in side if kind_ == "foothold"]
        if len(ks) != 1:
            raise EncodingError(f"action {a['name']!r} has {len(ks)} footholds where one is expected")
        if not 0 <= ks[0] < len(classes):
            raise EncodingError(f"action {a['name']!r}: principal p{ks[0]} has no class")
        out.append({"type": kind, "grants": "admin" if any(k == "admin" for k, _, _ in added) else "foothold",
                    "av": _AV[kind], "ac": _AC_OF_COST.get(a["cost"]["ac"], f"cost_{a['cost']['ac']}"),
                    "effect": _EFFECT[kind], "os": str(os_of.get(hosts.pop())),
                    "principal_class": str(classes[ks[0]])})
    return out


@dataclass(frozen=True)
class D28Instance:
    """One generated task under one cost regime, encoded: the planner's Task, the action rows (act_x) and the
    actions' categorical values (the WL initial colours). Built only by instance()."""
    task: Task
    regime: str
    action_x: tuple[tuple[float, ...], ...]
    action_labels: tuple[tuple, ...]


def instance(doc: dict, regime: str) -> D28Instance:
    from ..gen.generate import to_task  # the planner's Task, the same view the search uses
    check_regime(regime)
    view = planner_view(doc)
    task = to_task(view, regime)
    rows, labels = [], []
    for a, vals in zip(task.actions, action_values(view)):
        bits = []
        for g, vocab in GROUPS.items():
            if vals[g] not in vocab:
                raise EncodingError(f"action {a.name!r}: {g} {vals[g]!r} is not one of {vocab}")
            bits += [float(vals[g] == v) for v in vocab]
        rows.append((float(a.cost),) + tuple(bits))
        labels.append(tuple(f"{g}={vals[g]}" for g in GROUPS) + (a.cost,))
    return D28Instance(task, regime, tuple(rows), tuple(labels))


# ------------------------------------------------------------------ model inputs (training and search alike)

def static_graph(inst: D28Instance, device="cpu") -> dict[str, torch.Tensor]:
    """The GNN's state-independent input: the D19 graph structure with the D28 action rows."""
    return graph_of(inst.task, [list(r) for r in inst.action_x], device)


def mlp_features(inst: D28Instance, state: int) -> list[float]:
    """MLP_FEATURES of one state: the D21 features and the applicable actions' categorical counts."""
    counts = [0.0] * len(ONE_HOT)
    for ai, _ in inst.task.successors(state):
        for j, x in enumerate(inst.action_x[ai][1:]):
            counts[j] += x
    return d21_mlp_features(inst.task, state) + counts


def wl_colours(inst: D28Instance, state: int) -> Counter:
    return d21_wl_colours(inst.task, state, action_labels=inst.action_labels)


def gnn_group(inst: D28Instance, states: list[int], residual: bool = False) -> dict:
    """Training inputs of some states of one task: its static graph and the per-state fact inputs."""
    return {"graph": static_graph(inst), **gnn_inputs(inst.task, states, residual)}


def mlp_matrix(inst: D28Instance, states: list[int]) -> torch.Tensor:
    return torch.tensor([mlp_features(inst, s) for s in states], dtype=torch.float32)


def wl_histograms(inst: D28Instance, states: list[int]) -> list[Counter]:
    return [wl_colours(inst, s) for s in states]


# ------------------------------------------------------------------ leakage audit (LRN6)

def exposed_names(docs: list[dict], residual: bool = False) -> list[str]:
    """The feature names an encoding of these documents exposes: the fixed names plus every categorical value
    the documents carry (so a value outside the vocabularies, e.g. an account name, shows up by name)."""
    seen = {f"{g}={v}" for d in docs for vals in action_values(planner_view(d)) for g, v in vals.items()}
    seen |= {f"principal_class={c}" for d in docs for c in planner_view(d)["network"]["principal_classes"]
             if c != "attacker"}  # p0, the attacker's own foothold, is a class no action runs as
    seen |= {f"os={h['os']}" for d in docs for h in planner_view(d)["network"]["hosts"] if h["id"] != 0}
    fixed = feature_names(residual)
    return fixed + sorted(seen - set(fixed))


def load_policy(path: Path = POLICY) -> leakage.FeaturePolicy:
    return leakage.FeaturePolicy.from_json(path)


def audit_docs(docs: list[dict], split: str, report_path: Path | None = None, policy: leakage.FeaturePolicy | None = None,
               residual: bool = False) -> dict:
    """The LRN6 audit of one split's encoded D28 tasks; raises LeakageError on any violation. With report_path
    the report is written there (kept); either way it is returned."""
    if not docs:
        raise leakage.LeakageError(f"split {split!r}: no task documents to audit")
    names = exposed_names(docs, residual)
    policy = policy or load_policy()
    if report_path is not None:
        return leakage.run_audit(names, policy, split, report_path)
    report = leakage.audit(names, policy, split)
    if not report["ok"]:
        raise leakage.LeakageError(f"split {split!r}: {len(report['violations'])} feature violation(s): "
                                   f"{report['violations'][:5]}")
    return report
