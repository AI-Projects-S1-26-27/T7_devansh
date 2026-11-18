"""AL2: structural metrics (ahg.agmetrics) on hand-built tasks, the new aggregate keys, the mechanical go/stop
rule (ahg.al1gate.decision), the pinned al2_diag.json thresholds and the chain document selection.

Every expected metric value below is derived by hand in the comments, not read off the code.
"""

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ahg import agmetrics, al1gate, al1pilot, refsolver
from ahg.task import Task

from test_al1_deletes import REACQ, SIG, bits, build

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "configs" / "gen" / "al2_diag.json"


# ---------------------------------------------------------------- SIG (signature block)
# facts s, sig, a, b, g; init {s, sig}.
#   xA  pre {s, sig}  add {a}    del {sig}   edges s->a, sig->a
#   xB  pre {a, sig}  add {b}    del {sig}   edges a->b, sig->b
#   yB  pre {a}       add {b}                edge  a->b (already there)
#   evade pre {}      add {sig}              no edge (no precondition)
#   esc pre {b}       add {g}                edge  b->g
# Distinct edges: s->a, sig->a, a->b, sig->b, b->g = 5; n = 5, n(n-1) = 20 -> density 0.25.
# BFS from {s, sig}: a = 1, b = 1 (sig->b), g = 2 -> depth 2.
# No cycle: 5 singleton components, none non-trivial.
# AND: xA and xB have 2 preconditions -> 2/5 = 0.4.
# Deletes: xA deletes sig, which xB needs; xB deletes sig, which xA needs -> 2.

def test_sig_structure_by_hand():
    t = build(SIG)
    assert agmetrics.fact_edges(t) == {(0, 2), (1, 2), (2, 3), (1, 3), (3, 4)}
    assert agmetrics.edge_density(t) == pytest.approx(5 / 20)
    assert agmetrics.depth(t) == 2
    assert agmetrics.sccs(t) == (5, 0)
    assert agmetrics.and_share(t) == pytest.approx(0.4)
    assert agmetrics.delete_interactions(t) == 2


# States {s, sig} (init, h* 15) and {s, a} (after xA, h* 11: yB 7 + esc 4; evade 5 + xB 4 + esc 4 = 13 is worse).
#   init: xA applies -> {s, a}; evade re-adds the held sig (self-loop, skipped); the rest lack a or b -> 1 successor.
#   {s, a}: yB -> {s, a, b}; evade -> {s, a, sig}; xA and xB lack sig; esc lacks b -> 2 successors. Mean 1.5.
# First moves from init: only xA, to {s, a}; 4 + h*({s, a}) 11 = 15 = h*(s0) -> 1.
# LM-cut: h_max <= LM-cut <= h+ (optimal delete-relaxed cost).
#   init: h_max = 12 (a 4, b 8 via xB, g 12) and h+ = xA + xB + esc = 12 -> LM-cut 12 < 15: a gap.
#   {s, a}: h_max = 11 (sig 5 via evade, b = min(5 + 4, 0 + 7) = 7, g 11) and h+ = yB + esc = 11 = h*: exact.
# -> 1 of 2 states has LM-cut < h*.

def test_sig_state_metrics_by_hand():
    t = build(SIG)
    hs = {bits(t, ["s", "sig"]): 15, bits(t, ["s", "a"]): 11}
    assert agmetrics.branching(t, hs) == pytest.approx(1.5)
    assert agmetrics.optimal_first_moves(t, 15) == 1
    assert agmetrics.lmcut_gap_states(t, hs) == 1
    assert agmetrics.lmcut_gap_share(t, hs) == pytest.approx(0.5)


# ---------------------------------------------------------------- REACQ (re-acquisition)
# facts s, f, q, g; init {s}.
#   x1  pre {s}     add {f}               edge s->f
#   x2  pre {f}     add {q}   del {f}     edge f->q
#   fin pre {q, f}  add {g}               edges q->g, f->g
# 4 edges; n = 4, n(n-1) = 12 -> density 1/3. BFS from {s}: f 1, q 2, g 2 (f->g) -> depth 2.
# No cycle: 4 singletons. AND: fin only -> 1/3.
# Deletes: x2 deletes f; fin (another action) needs f -> 1 (x2's own precondition f does not count).

def test_reacq_structure_by_hand():
    t = build(REACQ)
    assert len(agmetrics.fact_edges(t)) == 4
    assert agmetrics.edge_density(t) == pytest.approx(4 / 12)
    assert agmetrics.depth(t) == 2
    assert agmetrics.sccs(t) == (4, 0)
    assert agmetrics.and_share(t) == pytest.approx(1 / 3)
    assert agmetrics.delete_interactions(t) == 1


# Reachable non-goal states and h*:
#   {s}       11  (x1 3 + x2 4 + x1 3 + fin 1)      successors: x1 -> {s, f}                       -> 1
#   {s, f}     8  (x2 4 + x1 3 + fin 1)            successors: x2 -> {s, q} (x1 is a self-loop)   -> 1
#   {s, q}     4  (x1 3 + fin 1)                   successors: x1 -> {s, f, q}                    -> 1
#   {s, f, q}  1  (fin)                            successors: x2 -> {s, q}, fin -> goal          -> 2
# Branching (1 + 1 + 1 + 2) / 4 = 1.25. First moves: x1 only, 3 + 8 = 11 -> 1.
# LM-cut, by h_max <= LM-cut <= h+:
#   {s}: h_max 8 (max(f 3, q 7) + 1), h+ = x1 + x2 + fin = 8 -> 8 < 11: gap.
#   {s, f}: h_max 5 (q 4, g 5), h+ = x2 + fin = 5 -> 5 < 8: gap.
#   {s, q}: h_max 4, h+ = x1 + fin = 4 = h*.   {s, f, q}: 1 = h*.
# -> 2 of 4 states: share 0.5.

def reacq_hs(t):
    return {bits(t, ["s"]): 11, bits(t, ["s", "f"]): 8, bits(t, ["s", "q"]): 4, bits(t, ["s", "f", "q"]): 1}


def test_reacq_state_metrics_by_hand():
    t = build(REACQ)
    hs = reacq_hs(t)
    st = refsolver.serialize(t)  # the hand list is every reachable non-goal state, with the reference h*
    assert {refsolver.to_bits(st, s): h for s, h in refsolver.hstar_all(st).items() if h > 0} == hs
    assert agmetrics.branching(t, hs) == pytest.approx(1.25)
    assert agmetrics.optimal_first_moves(t, 11) == 1
    assert agmetrics.lmcut_gap_states(t, hs) == 2
    assert agmetrics.lmcut_gap_share(t, hs) == pytest.approx(0.5)


def test_metrics_dict_collects_everything():
    t = build(REACQ)
    m = agmetrics.metrics(t, 11, reacq_hs(t))
    assert m == {"depth": 2, "edge_density": pytest.approx(1 / 3), "scc_count": 4, "scc_nontrivial": 0,
                 "and_share": pytest.approx(1 / 3), "delete_interactions": 1, "branching": pytest.approx(1.25),
                 "optimal_first_moves": 1, "lmcut_gap_states": 2, "lmcut_gap_share": pytest.approx(0.5)}


def test_empty_state_set_gives_none():
    t = build(REACQ)
    assert agmetrics.branching(t, {}) is None and agmetrics.lmcut_gap_share(t, {}) is None
    assert agmetrics.metrics(t, 11, {})["lmcut_gap_share"] is None


# ---------------------------------------------------------------- alternative routes, cap

def test_optimal_first_moves_counts_ties_and_caps():
    # 12 parallel actions s -> g, cost 1 each: every one is an optimal first move (1 + h*(goal) 0 = 1).
    acts = [(f"a{i}", ["s"], ["g"], 1) for i in range(12)] + [("slow", ["s"], ["g"], 2)]
    t = Task.from_names(["s", "g"], ["s"], ["g"], acts)
    assert agmetrics.optimal_first_moves(t, 1) == 10  # capped at 10
    assert agmetrics.optimal_first_moves(t, 1, cap=50) == 12  # "slow" (2 + 0 != 1) is not counted


# ---------------------------------------------------------------- SCCs with cycles

def test_sccs_two_cycle_and_self_loop():
    # p -> q (a), q -> p (b): one 2-cycle {p, q}; q -> r (c): r alone. 2 components, 1 non-trivial.
    acts = [("a", ["p"], ["q"], 1), ("b", ["q"], ["p"], 1), ("c", ["q"], ["r"], 1)]
    t = Task.from_names(["p", "q", "r"], ["p"], ["r"], acts)
    assert agmetrics.sccs(t) == (2, 1)
    assert agmetrics.depth(t) == 2  # p 0, q 1, r 2
    # d: r and p -> r adds r back to itself: a self-loop on r (and an edge p -> r), so {r} becomes non-trivial.
    t2 = Task.from_names(["p", "q", "r"], ["p"], ["r"], acts + [("d", ["r", "p"], ["r"], 1)])
    assert agmetrics.sccs(t2) == (2, 2)
    assert agmetrics.edge_density(t2) == pytest.approx(4 / 6)  # p->q, q->p, q->r, p->r; the self-loop not counted
    assert agmetrics.depth(t2) == 1  # r now 1 hop from p


def _brute_sccs(n, edges):
    reach = [{v} for v in range(n)]
    changed = True
    while changed:
        changed = False
        for p, q in edges:
            for v in range(n):
                if p in reach[v] and q not in reach[v]:
                    reach[v].add(q)
                    changed = True
    comps = {frozenset(w for w in range(n) if w in reach[v] and v in reach[w]) for v in range(n)}
    return len(comps), sum(len(c) > 1 or (next(iter(c)),) * 2 in edges for c in comps)


def test_tarjan_matches_mutual_reachability_on_random_graphs():
    rng = random.Random(11)
    for _ in range(200):
        n = rng.randint(1, 9)
        facts = [f"f{i}" for i in range(n)]
        acts = [(f"a{k}", rng.sample(facts, rng.randint(1, min(2, n))), rng.sample(facts, 1), 1)
                for k in range(rng.randint(0, 12))]
        t = Task.from_names(facts, [facts[0]], [facts[-1]], acts)
        assert agmetrics.sccs(t) == _brute_sccs(n, agmetrics.fact_edges(t))


def test_tarjan_is_iterative_on_a_long_chain():
    n = 5000  # far past Python's default recursion limit (1000)
    facts = [f"f{i}" for i in range(n)]
    acts = [(f"a{i}", [facts[i]], [facts[i + 1]], 1) for i in range(n - 1)] + [("back", [facts[-1]], [facts[0]], 1)]
    t = Task.from_names(facts, [facts[0]], [facts[-1]], acts)
    assert agmetrics.sccs(t) == (1, 1)
    assert agmetrics.depth(t) == n - 1


# ---------------------------------------------------------------- aggregate: the new keys

def row(gap=0, lgap=0, ev=10, solvable=True, metrics=None):
    return {"solvable": solvable, "gap_states": gap, "lmcut_gap_states": lgap, "eval_states": ev, "best_ratio": 1.0,
            "recovery_used": 0, "revisit": False, "loss_actions_on_path": 0, "eval_method": "exhaustive",
            "eval_skipped": 0, "oracle_s0": "agree", "loss": {"monitored_share": None}, "ratios": {"astar_hmax": 1.0},
            "route": {k: False for k in ("a_cost_up", "b_recovery", "c_revisit", "d_not_m0_optimal")},
            "metrics": metrics if metrics is not None else {"depth": 3, "branching": 2.0}}


def test_aggregate_lmcut_share_and_metric_medians():
    rows = [row(gap=1, lgap=2, ev=10, metrics={"depth": 2, "branching": 1.0}),
            row(gap=0, lgap=4, ev=30, metrics={"depth": 4, "branching": None}),  # no states: None is left out
            row(gap=0, lgap=0, ev=0, metrics={"depth": 9, "branching": 3.0}),
            row(solvable=False, metrics={"depth": 100, "branching": 100.0})]  # unsolvable rows never count
    a = al1pilot.aggregate(rows)
    assert a["lmcut_gap_state_share"] == pytest.approx(6 / 40)  # pooled: (2 + 4 + 0) / (10 + 30 + 0)
    assert a["gap_state_share"] == pytest.approx(1 / 40)
    assert a["metric_medians"] == {"depth": 4, "branching": 2.0}  # median(2, 4, 9); median(1.0, 3.0)


def test_aggregate_new_keys_without_solvable_rows():
    a = al1pilot.aggregate([row(solvable=False)])
    assert a["lmcut_gap_state_share"] is None and a["metric_medians"] == {}


# ---------------------------------------------------------------- the go/stop rule

RULE = {"archetype": "chain", "config": "all|medium",
        "any_of": {"gap_state_share_min": 0.05, "route_change_instance_share_min": 0.10},
        "all_of": {"lmcut_gap_state_share_min": 0.05}}


def summ(gap=0.0, route=0.0, lm=0.0, key="all|medium"):
    return {key: {"gap_state_share": gap, "route_change_instance_share": route, "lmcut_gap_state_share": lm},
            "M0": {"gap_state_share": 1.0, "route_change_instance_share": 1.0, "lmcut_gap_state_share": 1.0}}


def test_decision_go():
    assert al1gate.decision(summ(gap=0.2, lm=0.2), RULE)["go"]       # (a) via the h_max gap
    assert al1gate.decision(summ(route=0.5, lm=0.2), RULE)["go"]     # (a) via route change
    d = al1gate.decision(summ(gap=0.2, route=0.5, lm=0.2), RULE)
    assert d["go"] and d["config"] == "all|medium"
    assert d["any_of"]["gap_state_share"] == {"value": 0.2, "min": 0.05, "ok": True}


def test_decision_stop():
    assert not al1gate.decision(summ(gap=0.2, route=0.5, lm=0.01), RULE)["go"]  # (b) fails
    assert not al1gate.decision(summ(gap=0.01, route=0.05, lm=0.9), RULE)["go"]  # (a) fails both ways
    assert not al1gate.decision(summ(gap=0.2, lm=0.2, key="all|low"), RULE)["go"]  # only all|medium is read
    assert not al1gate.decision(summ(gap=None, route=None, lm=0.2), RULE)["go"]  # None fails its check
    assert not al1gate.decision(summ(gap=0.2, lm=None), RULE)["go"]
    assert not al1gate.decision({}, RULE)["go"]


def test_decision_boundaries_are_inclusive():
    assert al1gate.decision(summ(gap=0.05, lm=0.05), RULE)["go"]
    assert al1gate.decision(summ(route=0.10, lm=0.05), RULE)["go"]
    assert not al1gate.decision(summ(gap=0.0499, route=0.0999, lm=0.05), RULE)["go"]
    assert not al1gate.decision(summ(gap=0.05, lm=0.0499), RULE)["go"]


# ---------------------------------------------------------------- pinned config

def test_al2_diag_config_is_pinned():
    c = json.loads(DIAG.read_text())
    assert c["decision_rule"]["archetype"] == "chain" and c["decision_rule"]["config"] == "all|medium"
    assert c["decision_rule"]["any_of"] == {"gap_state_share_min": 0.05, "route_change_instance_share_min": 0.10}
    assert c["decision_rule"]["all_of"] == {"lmcut_gap_state_share_min": 0.05}
    assert c["sizes"] == list(range(8, 21)) and c["per_size"] == 10
    assert c["decision_rule"]["config"].split("|")[0] in c["mechanism_sets"]
    assert c["decision_rule"]["config"].split("|")[1] in c["levels"]
    # every other key is the AL1 pilot's, unchanged
    al1 = json.loads((ROOT / "configs" / "gen" / "al1_pilot.json").read_text())
    own = {"_doc", "sizes", "per_size", "decision_rule", "chain_max_attempts_per_size"}
    assert {k: v for k, v in c.items() if k not in own} == {k: v for k, v in al1.items() if k not in own}


# ---------------------------------------------------------------- chain document selection (gap --archetype chain)

def _load_pipe():
    spec = importlib.util.spec_from_file_location("d28_pipe", ROOT / "scripts" / "d28_pipe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["d28_pipe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_chain_docs_keep_solvable_count_unsolvable_and_cap(monkeypatch):
    from ahg.gen import generate as gen
    calls = []

    def fake(ctx, arch, seed, n, **kw):
        calls.append((arch, seed, n, kw))
        k = int(seed.rsplit("/", 1)[1])
        ok = n == 8 and k != 1  # n = 8: index 1 unsolvable; n = 9: never solvable
        return {"params": {"archetype": arch, "seed": seed, "n": n}, "solvable": ok,
                "unsolvable_reason": None if ok else "goal unreachable"}

    monkeypatch.setattr(gen, "generate", fake)
    training = SimpleNamespace(os_mix={"linux": 1.0})
    reg = SimpleNamespace(training=training, excluded=lambda c: frozenset({"exim"}),
                          excluded_cves=lambda c: frozenset({"CVE-X"}))
    pipe = _load_pipe()
    sample, sel = pipe._chain_docs(reg, None, {"sizes": [8, 9], "chain_max_attempts_per_size": 4}, per=2)
    assert [d["params"]["seed"] for d in sample] == ["al2-chain/n8/0", "al2-chain/n8/2"]
    assert sel[8] == {"generated": 3, "kept": 2, "unsolvable": {"goal unreachable": 1}, "shortfall": 0}
    assert sel[9] == {"generated": 4, "kept": 0, "unsolvable": {"goal unreachable": 4}, "shortfall": 2}
    arch, _, _, kw = calls[0]  # generated as cmd_n4 does: chain, core cohort, training's mix and exclusions, p = 1
    assert arch == "chain" and kw == {"cohort": "core", "os_mix": {"linux": 1.0}, "p": 1.0,
                                      "excluded": frozenset({"exim"}), "excluded_cves": frozenset({"CVE-X"})}
