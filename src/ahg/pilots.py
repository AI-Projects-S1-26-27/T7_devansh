"""D28 training-only pilots (P7): the budget feasibility pilot (Round 9) and the landscape pilot (Round 12).

Both read training-cohort tasks only (the scripts refuse any other role). Both are descriptive: the budget
pilot proposes a per-state budget that Devansh freezes; the landscape pilot never gates learning.

Budget pilot (Round 9). For each sampled task and regime: the states labelling would see (the five offline
sources, labels.collect_states, with the pilot's cap as the initial budget), then A*+LM-cut from each state up
to the cap, and the independent oracle on each label up to its pop limit. A state certifies under budget B if
its search expanded at most B nodes without hitting the cap. The proposal is the smallest candidate budget
under which at least `target` of all sampled states certify, in both regimes (OQ-P7-8); None if no candidate
reaches it. Oracle checks that hit the pop limit are counted: Round 9 makes infeasible per-state oracle
checking a BLOCKED item, so the count is reported with the proposal.

Landscape pilot (Round 12, P0b/P0c/P0d), regime `ac`:
- canonical optimal plan = the A*+LM-cut plan under the D3 tie rule (f, h, FIFO);
- R_h = expansions of A*+h from s0 / the number of actions in the canonical plan, for h in
  h0, h_max, h_add, h_FF, LM-cut, h_sp (P0d); tasks whose plan is empty are counted and left out;
- LM-cut absolute and relative gap (h* - LM-cut) / h* at s0 and at the labelled states (h* > 0);
- AND-arity distribution: the number of preconditions of each action;
- local escalation: the canonical plan has an action generated from an AV:L CVE (a "local" action).
Aggregation: per archetype x size, median and IQR (P0b); per task, its sampled-state relative gaps are
summarised by their median first (OQ-P7-9). Near-exact = median LM-cut relative gap at s0 <= 0.05 AND median
R_LM-cut <= 1.5 (P0b, P0d).
"""

from __future__ import annotations

import statistics
import time
from collections import Counter

from .heuristics import HEURISTICS, lmcut
from .labels import collect_states
from .oracle import optimal_cost
from .search import astar

LANDSCAPE_HEURISTICS = ("h0", "hmax", "hadd", "hff", "lmcut", "hsp")
NEAR_EXACT_GAP, NEAR_EXACT_R = 0.05, 1.5
LANDSCAPE_REGIME = "ac"  # Round 5 amendment: the pilot runs under the primary regime


def quantiles(xs: list[float]) -> dict | None:
    """Median and IQR (25th and 75th percentiles, statistics.quantiles 'inclusive' method)."""
    if not xs:
        return None
    if len(xs) == 1:
        return {"n": 1, "median": xs[0], "q1": xs[0], "q3": xs[0]}
    q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return {"n": len(xs), "median": statistics.median(xs), "q1": q1, "q3": q3}


# ------------------------------------------------------------------ budget feasibility pilot

def budget_task(doc: dict, regime: str, cap: int, label_cfg: dict) -> dict:
    """Per-state LM-cut expansions (capped) and oracle feasibility for one solvable training task."""
    from .gen.generate import to_task
    task = to_task(doc, regime)
    prm = doc["params"]
    key = f"{prm['archetype']}:{prm['n']}:{prm['seed']}"
    cfg = {**label_cfg, "init_budget": cap}
    r_lm, _, sources, _, _ = collect_states(task, key, cfg)
    if sources is None:  # even s0 needs more than the cap
        return {"n": prm["n"], "regime": regime, "init_capped": True, "states": [{"expanded": r_lm.expanded,
                "capped": True, "source": ["init"], "oracle": None, "seconds": None}]}
    states = []
    for s in sorted(sources):
        t0 = time.perf_counter()
        r = astar(task, lmcut, start=s, max_expansions=cap)
        secs = time.perf_counter() - t0
        oracle = None
        if not r.budget_hit:
            o = optimal_cost(task, s, max_pops=label_cfg["oracle_max_pops"])
            oracle = "skipped" if o is None else ("agrees" if o == r.cost else "MISMATCH")
        states.append({"expanded": r.expanded, "capped": r.budget_hit, "source": sorted(sources[s]),
                       "oracle": oracle, "seconds": round(secs, 6), "init": s == task.init})
    return {"n": prm["n"], "regime": regime, "init_capped": False, "states": states}


def budget_summary(rows: list[dict], candidates: list[int], cap: int, target: float) -> dict:
    """Coverage per candidate budget and regime, oracle feasibility, and the proposed budget."""
    if any(b > cap for b in candidates):
        raise ValueError("a candidate budget above the pilot's cap cannot be assessed")
    out = {"cap": cap, "target": target, "candidates": candidates, "by_regime": {}}
    ok_all = {b: True for b in candidates}
    for regime in sorted({r["regime"] for r in rows}):
        st = [s for r in rows if r["regime"] == regime for s in r["states"]]
        init = [s for r in rows if r["regime"] == regime for s in r["states"] if s.get("init") or r["init_capped"]]
        cov = {b: sum(not s["capped"] and s["expanded"] <= b for s in st) / len(st) for b in candidates}
        icov = {b: sum(not s["capped"] and s["expanded"] <= b for s in init) / len(init) for b in candidates} if init else {}
        for b in candidates:
            ok_all[b] = ok_all[b] and cov[b] >= target
        out["by_regime"][regime] = {
            "tasks": sum(r["regime"] == regime for r in rows), "states": len(st),
            "capped_states": sum(s["capped"] for s in st),
            "coverage": {str(b): cov[b] for b in candidates}, "init_coverage": {str(b): v for b, v in icov.items()},
            "expanded": quantiles([s["expanded"] for s in st if not s["capped"]]),
            "oracle": dict(Counter(str(s["oracle"]) for s in st)),
            "seconds_total": sum(s["seconds"] or 0 for s in st)}
    if any(v["oracle"].get("MISMATCH") for v in out["by_regime"].values()):
        raise RuntimeError("the oracle disagrees with A*+LM-cut on a pilot state (rule 6): abort")
    passing = [b for b in candidates if ok_all[b]]
    out["proposed_state_budget"] = min(passing) if passing else None
    return out


# ------------------------------------------------------------------ landscape pilot

def landscape_task(doc: dict, label: dict, budget: int) -> dict:
    """The Round 12 metrics of one solvable, labelled training task under `ac`."""
    from .gen.generate import to_task
    task = to_task(doc, LANDSCAPE_REGIME)
    if label["regime"] != LANDSCAPE_REGIME or not label["certified"]:
        raise ValueError("the landscape pilot uses certified ac labels only")
    hstar = label["hstar_init"]
    canon = astar(task, lmcut, max_expansions=budget, tie_break="fifo")
    if canon.budget_hit or canon.cost != hstar:
        raise ValueError(f"canonical plan cost {canon.cost} != certified h* {hstar}")
    n_act = len(canon.plan)
    runs = {}
    for name in LANDSCAPE_HEURISTICS:
        r = canon if name == "lmcut" else astar(task, HEURISTICS[name], max_expansions=budget, tie_break="fifo")
        runs[name] = {"expanded": r.expanded, "budget_hit": r.budget_hit}
    R = {h: (None if n_act == 0 or v["budget_hit"] else v["expanded"] / n_act) for h, v in runs.items()}
    lm0 = lmcut(task, task.init)
    gaps = [(h - lmcut(task, int(s, 16))) / h for s, h, _ in label["rows"] if h > 0]
    return {"n": doc["params"]["n"], "archetype": doc["params"]["archetype"], "hstar": hstar, "plan_actions": n_act,
            "runs": runs, "R": R, "gap_abs_init": hstar - lm0, "gap_rel_init": (hstar - lm0) / hstar if hstar else None,
            "gap_rel_states_median": statistics.median(gaps) if gaps else None, "gap_states": len(gaps),
            "and_arity": dict(Counter(len(a.pre) for a in task.actions)),
            "local_escalation": any(task.actions[i].name.startswith("local ") for i in canon.plan)}


def landscape_summary(rows: list[dict]) -> dict:
    out = {}
    for key in sorted({(r["archetype"], r["n"]) for r in rows}):
        rs = [r for r in rows if (r["archetype"], r["n"]) == key]
        live = [r for r in rs if r["plan_actions"] > 0]
        arity = Counter()
        for r in rs:
            arity.update({int(k): v for k, v in r["and_arity"].items()})
        R = {h: quantiles([r["R"][h] for r in live if r["R"][h] is not None]) for h in LANDSCAPE_HEURISTICS}
        gap0 = quantiles([r["gap_rel_init"] for r in live])
        out[f"{key[0]}|{key[1]}"] = {
            "tasks": len(rs), "trivial_tasks": len(rs) - len(live),
            "R": R, "budget_hits": {h: sum(r["runs"][h]["budget_hit"] for r in rs) for h in LANDSCAPE_HEURISTICS},
            "gap_abs_init": quantiles([r["gap_abs_init"] for r in live]), "gap_rel_init": gap0,
            "gap_rel_states": quantiles([r["gap_rel_states_median"] for r in live if r["gap_rel_states_median"] is not None]),
            "and_arity": dict(sorted(arity.items())),
            "local_escalation_fraction": sum(r["local_escalation"] for r in live) / len(live) if live else None,
            "near_exact": (gap0 is not None and R["lmcut"] is not None and gap0["median"] <= NEAR_EXACT_GAP
                           and R["lmcut"]["median"] <= NEAR_EXACT_R)}
    return out
