"""Search policies of D28 Round 7: reopening (S1), peak open/closed list sizes and tie rules (S2/S4)."""

import pytest

from ahg import search
from ahg.heuristics import HEURISTICS, hadd
from ahg.search import astar, best_first_search, gbfs, uniform_cost, weighted_astar
from ahg.task import Task, plan_cost

from helpers import random_task

# A closed node reached again by a cheaper path. States: S0={s}, A={s,a}, B={s,b}, AB={s,a,b},
# ABC={s,a,b,c}, goal {d}. Greedy by H: S0, A (h 1), AB (h 2, g 11 via the expensive xa), then B (h 3),
# which finds AB at g 2 while AB is closed. ABC has h 4, so B is expanded before it.
REOPEN_TASK = Task.from_names(
    ["s", "a", "b", "c", "d"], ["s"], ["d"],
    [("xa", ["s"], ["a"], 10), ("xb", ["s"], ["b"], 1), ("ab", ["a"], ["b"], 1), ("ba", ["b"], ["a"], 1),
     ("abc", ["a", "b"], ["c"], 1), ("cd", ["c"], ["d"], 1)])
_H = {frozenset("s"): 5, frozenset("sa"): 1, frozenset("sb"): 3, frozenset("sab"): 2, frozenset("sabc"): 4}


def H(task, state):
    facts = frozenset(task.facts[i] for i in range(len(task.facts)) if state >> i & 1)
    return _H.get(facts, 0)


AB = 0b00111


def test_gbfs_never_reopens_but_takes_the_cheaper_parent_like_fast_downward():
    trace = []
    r = gbfs(REOPEN_TASK, H, trace=trace)
    assert trace.count(AB) == 1 and r.reopened == 0 and r.expanded == 5
    # FD update_closed_node_parent: AB's parent is now B, so the extracted plan is the cheap one, and its
    # cost, not the stale g of the goal (13), is what is reported.
    assert [REOPEN_TASK.actions[i].name for i in r.plan] == ["xb", "ba", "abc", "cd"]
    assert r.cost == plan_cost(REOPEN_TASK, r.plan) == 4


def test_the_same_greedy_order_with_reopening_reexpands():
    trace = []
    r = best_first_search(REOPEN_TASK, H, w_g=0.0, reopen=True, trace=trace)
    assert trace.count(AB) == 2 and r.reopened == 1 and r.expanded == 6
    assert r.cost == plan_cost(REOPEN_TASK, r.plan) == 4


def test_reopen_is_required():
    with pytest.raises(TypeError):
        best_first_search(REOPEN_TASK, H)


def test_wrappers_pass_the_fast_downward_defaults(monkeypatch):
    seen = {}
    real = search.best_first_search

    def spy(task, h, **kw):
        seen["reopen"] = kw["reopen"]
        return real(task, h, **kw)

    monkeypatch.setattr(search, "best_first_search", spy)
    for fn, want in ((lambda: search.astar(REOPEN_TASK, H), True), (lambda: search.uniform_cost(REOPEN_TASK), True),
                     (lambda: search.weighted_astar(REOPEN_TASK, H, 2.0), True), (lambda: search.gbfs(REOPEN_TASK, H), False)):
        fn()
        assert seen.pop("reopen") is want


def test_weighted_astar_reopens():
    """f = g + 100h orders A, AB, then B as greedy does; the cheaper path to closed AB reopens it."""
    trace = []
    r = weighted_astar(REOPEN_TASK, H, 100.0, trace=trace)
    assert trace.count(AB) == 2 and r.reopened == 1
    assert r.cost == plan_cost(REOPEN_TASK, r.plan) == 4


def test_random_tasks_reopen_only_where_allowed():
    """GBFS never expands a state twice; any repeat in weighted A* is a counted reopening."""
    for seed in range(60):
        t = random_task(seed)
        for name in ("hadd", "hff", "lmcut"):
            h = HEURISTICS[name]
            tr = []
            r = gbfs(t, h, trace=tr)
            assert len(tr) == len(set(tr)) and r.reopened == 0
            if r.solved:
                assert r.cost == plan_cost(t, r.plan)
            tr = []
            r = weighted_astar(t, h, 2.0, trace=tr)
            assert len(tr) - len(set(tr)) <= r.reopened
            if r.solved:
                assert r.cost == plan_cost(t, r.plan)


def test_evaluation_searches_use_the_same_policy():
    from ahg.evaluate import HADD_SEARCHES
    assert HADD_SEARCHES["gbfs_hadd"]["reopen"] is False
    assert all(v["reopen"] for k, v in HADD_SEARCHES.items() if k.startswith("wastar"))
    t = random_task(3)
    a = best_first_search(t, hadd, **HADD_SEARCHES["gbfs_hadd"])
    b = gbfs(t, hadd)
    assert (a.plan, a.expanded, a.generated) == (b.plan, b.expanded, b.generated)
    assert uniform_cost(t).cost == astar(t, HEURISTICS["lmcut"]).cost


# Two cost-1 branches from a, each one step from the goal: every node on a level ties on (f, h).
TIE_TASK = Task.from_names(["a", "p", "q", "g"], ["a"], ["g"],
                           [("ap", ["a"], ["p"], 1), ("aq", ["a"], ["q"], 1), ("pg", ["p"], ["g"], 1),
                            ("qg", ["q"], ["g"], 1)])
A, P, Q, PQ = 0b0001, 0b0011, 0b0101, 0b0111

# By hand, UCS (f = g, h = 0), heap key (f, h, tie):
# FIFO: expand A -> P(1), Q(1). Expand P (older) -> PQ(2), PG(2) [open: Q, PQ, PG = 3]. Expand Q -> QG(2)
#   (PQ via ap is not cheaper). f = 2 ties: PQ is oldest, expanded; then PG, the goal.
#   Trace A, P, Q, PQ; peak open 3; closed 4.
# LIFO: expand A -> P, Q. Expand Q (newer) -> PQ(2), QG(2) [open: P, PQ, QG = 3]. Expand P -> PG(2), the
#   newest f = 2 node, popped next: the goal. Trace A, Q, P; peak open 3; closed 3.


def test_tie_rules_order_equal_keys_by_generation():
    fifo, lifo = [], []
    rf = uniform_cost(TIE_TASK, tie_break="fifo", trace=fifo)
    rl = uniform_cost(TIE_TASK, tie_break="lifo", trace=lifo)
    assert fifo == [A, P, Q, PQ] and lifo == [A, Q, P]
    assert rf.cost == rl.cost == 2


def test_default_is_the_d3_rule():
    a, b = uniform_cost(TIE_TASK, trace=(ta := [])), uniform_cost(TIE_TASK, tie_break="fifo", trace=(tb := []))
    assert ta == tb and a.plan == b.plan


def test_unknown_tie_rule_is_refused():
    with pytest.raises(ValueError):
        astar(TIE_TASK, HEURISTICS["hmax"], tie_break="random")


@pytest.mark.parametrize("seed", range(40))
def test_both_tie_rules_are_optimal_with_admissible_h(seed):
    t = random_task(seed)
    for name in ("hmax", "lmcut"):
        assert astar(t, HEURISTICS[name], tie_break="fifo").cost == astar(t, HEURISTICS[name], tie_break="lifo").cost


def test_peak_sizes_by_hand():
    """The derivation above TIE_TASK."""
    f, l = uniform_cost(TIE_TASK, tie_break="fifo"), uniform_cost(TIE_TASK, tie_break="lifo")
    assert (f.peak_open, f.peak_closed) == (3, 4)
    assert (l.peak_open, l.peak_closed) == (3, 3)


@pytest.mark.parametrize("seed", range(40))
def test_peak_sizes_are_consistent_with_the_trace(seed):
    t = random_task(seed)
    for run in (lambda **kw: astar(t, HEURISTICS["lmcut"], **kw), lambda **kw: gbfs(t, hadd, **kw),
                lambda **kw: weighted_astar(t, hadd, 2.0, **kw)):
        for tie in ("fifo", "lifo"):
            tr = []
            r = run(tie_break=tie, trace=tr)
            assert r.peak_closed == len(set(tr))
            assert 1 <= r.peak_open <= r.generated + 1


def test_evaluation_records_peaks_and_passes_the_tie_rule():
    from ahg.evaluate import _run
    f, l = _run(TIE_TASK, lambda t, s: 0, None), _run(TIE_TASK, lambda t, s: 0, None, tie_break="lifo")
    assert (f["peak_open"], f["peak_closed"], f["expanded"]) == (3, 4, 4)
    assert (l["peak_open"], l["peak_closed"], l["expanded"]) == (3, 3, 3)
