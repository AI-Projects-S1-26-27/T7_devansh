"""AL1 classical-only gap pilot: per-instance metrics and per-level aggregates for the gate (decisions.md §21).

Per instance:
- the optimal plan of the M3 task (A*+LM-cut, checked at s0 against the independent oracle);
- expansions per plan action for A* and GBFS with h_max, h_add, h_FF and LM-cut;
- the evaluated states, their h* and h_max, with the h_max < h* count;
- the route-change components (a) to (d) against the M0 task;
- the AL2 structural metrics (ahg.agmetrics) and the LM-cut < h* count over the evaluated states.

Evaluated states (plan §3b): every reachable state when the reachable set fits under enum_cap. Otherwise,
the states A*+h_max expands from s0, plus walk_states seeded random-walk states.
A mismatch with the oracle raises HardFailure. An oracle budget stop is counted as unchecked.
"""

import math
import random
import statistics

from ahg import agmetrics, oracle, refsolver
from ahg.gen.generate import doc_digest, neutral, to_task
from ahg.heuristics import hadd, hff, hmax, lmcut
from ahg.measure import HardFailure
from ahg.search import astar, gbfs
from ahg.task import plan_cost

CLASSICAL = {"hmax": hmax, "hadd": hadd, "hff": hff, "lmcut": lmcut}
INF = float("inf")


def _hstar(task, s, cfg) -> float | None:
    r = astar(task, lmcut, start=s, max_expansions=cfg["search_budget"])
    return None if r.budget_hit else r.cost


def evaluated_states(task, doc, cfg) -> tuple[str, dict[int, float], int]:
    """(method, {state: h*} over non-goal states with finite h*, states skipped at the budget)."""
    st = neutral(doc, cfg["regime"])
    full = refsolver.hstar_all(st, cap=cfg["enum_cap"])
    if full is not None:
        out = {refsolver.to_bits(st, s): h for s, h in full.items() if 0 < h < INF}
        return "exhaustive", out, 0
    trace = []
    astar(task, hmax, max_expansions=cfg["search_budget"], trace=trace)
    states = set(trace)
    rng = random.Random(f"al1:walk:{doc_digest(doc)}")
    for _ in range(cfg["walk_states"]):
        s = task.init
        for _ in range(rng.randint(1, 12)):
            succ = [s2 for _, s2 in task.successors(s)]
            if not succ or task.is_goal(s):
                break
            s = rng.choice(succ)
        states.add(s)
    out, skipped = {}, 0
    for s in sorted(states):
        if task.is_goal(s):
            continue
        h = _hstar(task, s, cfg)
        if h is None:
            skipped += 1
            continue
        o = oracle.optimal_cost(task, s, max_pops=cfg["oracle_max_pops"])
        if o is not None and o != h:
            raise HardFailure(f"oracle {o} != A*+LM-cut {h} at {s:#x}")
        if h < INF:
            out[s] = h
    return "sampled", out, skipped


def _targets(doc, plan) -> list[int]:
    """Host entered by each remote plan action. Local escalation is excluded: escalating on a host just entered
    is the normal two-step pattern, not a revisit (smoke-run fix, 18 Nov 2026)."""
    out = []
    for i in plan:
        m = doc["metadata"]["actions"][i]
        if m.get("kind") == "remote":
            out.append(m["target"][0] if "target" in m else m["matches"][0][0])
    return out


def instance(m0: dict, m3: dict, cfg: dict) -> dict:
    reg = cfg["regime"]
    t0, t3 = to_task(m0, reg), to_task(m3, reg)
    r0 = astar(t0, lmcut, max_expansions=cfg["search_budget"])
    r3 = astar(t3, lmcut, max_expansions=cfg["search_budget"])
    row = {"seed": m0["params"]["seed"], "n": m0["params"]["n"], "archetype": m0["params"]["archetype"],
           "hstar_m0": r0.cost, "hstar_m3": r3.cost,
           "budget_hit": r3.budget_hit, "solvable": r3.plan is not None and r3.cost < INF,
           "loss": {k: m3.get("loss", {}).get(k) for k in ("monitored_share", "loss_actions_total",
                                                           "deleted_facts_total", "recovery_actions")}}
    o = oracle.optimal_cost(t3, max_pops=cfg["oracle_max_pops"])
    if o is not None and o != r3.cost and not r3.budget_hit:
        raise HardFailure(f"{row['seed']}: oracle {o} != A*+LM-cut {r3.cost} at s0")
    row["oracle_s0"] = "agree" if o is not None else "unchecked"
    if not row["solvable"]:
        return row
    plan = r3.plan
    L = len(plan)
    row["plan_len"] = L
    row["hmax_s0"] = hmax(t3, t3.init)
    if row["hmax_s0"] > r3.cost:
        raise HardFailure(f"{row['seed']}: h_max {row['hmax_s0']} > h* {r3.cost} at s0")
    ratios = {}
    for sname, search in (("astar", astar), ("gbfs", gbfs)):
        for hname, h in CLASSICAL.items():
            r = search(t3, h, max_expansions=cfg["search_budget"])
            ratios[f"{sname}_{hname}"] = r.expanded / L if L else 1.0
    row["ratios"] = ratios
    row["best_ratio"] = min(ratios.values())
    method, hs, skipped = evaluated_states(t3, m3, cfg)
    gap = sum(1 for s, h in hs.items() if hmax(t3, s) < h)
    row.update({"eval_method": method, "eval_states": len(hs), "eval_skipped": skipped, "gap_states": gap})
    meta = m3["metadata"]["actions"]
    on_path = [i for i in plan if t3.actions[i].dele]
    tg = _targets(m3, plan)
    recovery = sum(meta[i].get("kind") == "recovery" for i in plan)
    revisit = len(tg) != len(set(tg))
    # M0 rows (no transform) carry no m0_action: the plan is already an M0 plan
    proj = [meta[i].get("m0_action", i) for i in plan if meta[i].get("kind") != "recovery"]
    try:
        proj_cost = plan_cost(t0, proj)
    except ValueError:
        proj_cost = INF
    row.update({"loss_actions_on_path": len(on_path),
                "deleted_facts_on_path": sum(len(t3.actions[i].dele) for i in plan),
                "recovery_used": recovery, "revisit": revisit,
                "route": {"a_cost_up": r3.cost > r0.cost, "b_recovery": recovery > 0, "c_revisit": revisit,
                          "d_not_m0_optimal": proj_cost > r0.cost}})
    met = agmetrics.metrics(t3, r3.cost, hs)
    row["lmcut_gap_states"] = met.pop("lmcut_gap_states")
    row["metrics"] = met
    return row


def aggregate(rows: list[dict]) -> dict:
    sol = [r for r in rows if r["solvable"]]
    ev = sum(r["eval_states"] for r in sol)
    out = {
        "instances": len(rows), "solvable": len(sol),
        "solvable_share": len(sol) / len(rows) if rows else None,
        "gap_state_share": sum(r["gap_states"] for r in sol) / ev if ev else None,
        "lmcut_gap_state_share": sum(r["lmcut_gap_states"] for r in sol) / ev if ev else None,
        "gap_instance_share": sum(r["gap_states"] > 0 for r in sol) / len(sol) if sol else None,
        "median_best_classical_ratio": statistics.median(r["best_ratio"] for r in sol) if sol else None,
        "recovery_or_revisit_plan_share": (sum(r["recovery_used"] > 0 or r["revisit"] for r in sol) / len(sol)
                                           if sol else None),
        "route_change_instance_share": sum(any(r["route"].values()) for r in sol) / len(sol) if sol else None,
        "route_components": {k: sum(r["route"][k] for r in sol) for k in
                             ("a_cost_up", "b_recovery", "c_revisit", "d_not_m0_optimal")} if sol else {},
        "loss_on_path_instances": sum(r["loss_actions_on_path"] > 0 for r in sol),
        "eval_methods": {m: sum(r["eval_method"] == m for r in sol) for m in ("exhaustive", "sampled")},
        "eval_skipped": sum(r["eval_skipped"] for r in sol),
        "oracle_s0_unchecked": sum(r["oracle_s0"] == "unchecked" for r in rows),
        "monitored_share_median": statistics.median(r["loss"]["monitored_share"] for r in rows
                                                    if r["loss"]["monitored_share"] is not None) if rows and
        rows[0]["loss"]["monitored_share"] is not None else None,
        "median_ratio_by": {k: statistics.median(r["ratios"][k] for r in sol) for k in sol[0]["ratios"]} if sol else {},
    }
    # per-metric medians over solvable instances; a metric that is None on a row (no evaluated states) is left out
    out["metric_medians"] = {k: _median([r["metrics"][k] for r in sol]) for k in sol[0]["metrics"]} if sol else {}
    if sol:
        out["geomean_best_ratio"] = math.exp(statistics.fmean(math.log(r["best_ratio"]) for r in sol))
    return out


def _median(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
