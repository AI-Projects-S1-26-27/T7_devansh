"""Measurement helpers shared by the landscape scripts (measure_landscape.py, measure_depth_coupling.py).

h* of a state is computed by A*+LM-cut and cross-checked against A*+h_max.
Both are admissible, so the two optimal costs must be equal; a mismatch, or
an admissible heuristic above h*, raises HardFailure, and the calling script
aborts before writing anything. enumerate_exact gives a third, independent
h* from the reference solver (refsolver.py) where the reachable space is
small enough.

The C1 admissibility check (D28 Round 8) is audit_admissibility: every
heuristic claimed admissible against the reference solver's h*, on every
reachable state. check_admissible below is only a consistency check against
a caller-supplied h*; when that h* comes from A*+LM-cut it is circular for
LM-cut (AUDIT_4 inventory #1), so it never counts as the C1 check.
"""

from . import refsolver
from .heuristics import ADMISSIBLE, hmax, lmcut
from .search import INF, astar


class HardFailure(Exception):
    pass


def exact_cost(task, start, budget, traces=None):
    """h*(start) by A*+LM-cut, cross-checked by A*+h_max. None if a search hit the budget.

    traces: optional (lmcut_list, hmax_list) receiving each search's expanded states.
    """
    t_lm, t_hm = traces if traces is not None else (None, None)
    r_lm = astar(task, lmcut, start=start, max_expansions=budget, trace=t_lm)
    r_hm = astar(task, hmax, start=start, max_expansions=budget, trace=t_hm)
    if r_lm.budget_hit or r_hm.budget_hit:
        # Not silently dropped: the caller counts it as "h* unknown".
        return None, r_lm, r_hm
    if r_lm.cost != r_hm.cost:
        raise HardFailure(f"A*+LM-cut {r_lm.cost} != A*+h_max {r_hm.cost} from state {start:#x}")
    if r_lm.cost == INF:
        # Delete-free with a reachable goal: every reachable state reaches it (D14 §1).
        raise HardFailure(f"reachable state {start:#x} cannot reach the goal")
    return r_lm.cost, r_lm, r_hm


def check_admissible(task, s, hstar):
    """h_max and LM-cut at s against a caller-supplied h* (consistency check only; see the module docstring)."""
    vals = (hmax(task, s), lmcut(task, s))
    for name, v in zip(("h_max", "LM-cut"), vals):
        if v > hstar:
            raise HardFailure(f"{name}={v} > h*={hstar} at state {s:#x}")
    return vals


def sample_states(task, rng, cap, walks_per_sample):
    """Distinct reachable non-goal states from random-depth walks (D14 §2 off-optimal rollouts).

    Each walk applies uniformly chosen applicable, useful actions from the
    initial state for a uniformly drawn number of steps in [1, max depth],
    where max depth = facts not held initially (each useful action adds at
    least one fact). The walk stops early at a goal or dead end; goal states
    are skipped. Returns (states, cap_bound): cap_bound is True if cap was
    reached, False if the walk budget (cap * walks_per_sample walks) ran out
    first (few distinct states).
    """
    max_depth = len(task.facts) - bin(task.init).count("1")
    seen = set()
    for _ in range(cap * walks_per_sample):
        s = task.init
        for _ in range(rng.randint(1, max_depth)):
            succ = [s2 for _, s2 in task.successors(s)]
            if not succ or task.is_goal(s):
                break
            s = rng.choice(succ)
        if s != task.init and not task.is_goal(s):
            seen.add(s)
            if len(seen) == cap:
                return sorted(seen), True
    return sorted(seen), False


def enumerate_exact(task, cap):
    """h* of every reachable state, keyed by bitmask; None if more than cap states.

    Computed by the independent reference solver (exhaustive backward
    Dijkstra on fact-name sets), so it shares no transition code with
    task.py or search.py (AUDIT_4 inventory #1).
    """
    st = refsolver.serialize(task)
    h = refsolver.hstar_all(st, cap=cap)
    return None if h is None else {refsolver.to_bits(st, s): v for s, v in h.items()}


def _reference_values(task, heuristics: dict, cap: int):
    """(state bitmask, reference h*, {name: h(state)}) for every reachable state. Refuses any heuristic
    not claimed admissible: a learned heuristic is never treated as admissible (D28 Round 8, rule 8)."""
    unclaimed = set(heuristics) - ADMISSIBLE
    if unclaimed:
        raise ValueError(f"not claimed admissible, so never audited as admissible: {sorted(unclaimed)}")
    exact = enumerate_exact(task, cap)
    if exact is None:
        raise HardFailure(f"more than {cap} reachable states: too large for the exhaustive check")
    for s, hs in sorted(exact.items()):
        yield s, hs, {name: h(task, s) for name, h in heuristics.items()}


def audit_admissibility(task, heuristics: dict, cap: int = 100_000) -> dict:
    """C1: h(s) <= h*(s) for every reachable state s and every heuristic claimed admissible.

    h* comes from the reference solver, never from a search that uses one of
    the heuristics checked. Raises HardFailure on the first violation; returns
    per-heuristic counts (states, states with h < h*) otherwise.
    """
    out = {name: {"states": 0, "below": 0} for name in heuristics}
    for s, hs, vals in _reference_values(task, heuristics, cap):
        for name, v in vals.items():
            if v > hs:
                raise HardFailure(f"{name}={v} > h*={hs} at state {s:#x}")
            out[name]["states"] += 1
            out[name]["below"] += v < hs
    return out


def audit_dead_end_detection(task, heuristics: dict, cap: int = 100_000) -> int:
    """Unreachable-goal detection, tested separately from admissibility (D28 Round 8, C1).

    A heuristic claimed admissible must return INF exactly at the reachable
    states whose reference h* is INF: INF where h* is finite would prune a
    solvable state, and a finite value where h* is INF would let the common
    reachability pre-check (h_max = INF, Round 7) pass an unsolvable task.
    In a delete-free task h_max and LM-cut are exact on this question.
    Returns the number of dead-end states seen; raises HardFailure otherwise.
    """
    dead = 0
    for s, hs, vals in _reference_values(task, heuristics, cap):
        dead += hs == INF
        for name, v in vals.items():
            if (v == INF) != (hs == INF):
                raise HardFailure(f"{name}={v} but h*={hs} at state {s:#x}")
    return dead


def gap_stats(triples):
    """triples: (h*, h_max, LM-cut). Per heuristic: count with h < h*, sum and max of h* - h."""
    out = {"n": len(triples)}
    for i, name in ((1, "hmax"), (2, "lmcut")):
        gaps = [t[0] - t[i] for t in triples]
        out[name] = {"lt": sum(g > 0 for g in gaps), "gap_sum": sum(gaps), "gap_max": max(gaps, default=0)}
    return out
