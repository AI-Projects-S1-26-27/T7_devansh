"""The independent reference solver (D28 Round 8, C2): hand-derived optima, independence, generated tasks."""

import ast
from pathlib import Path

import pytest

from ahg import refsolver
from ahg.attack_graph import make_instance
from ahg.heuristics import lmcut
from ahg.search import astar, uniform_cost
from ahg.task import Task

from helpers import brute_force_optimal, random_task, reachable_states
from ref_fixtures import FIXTURES, INF

SRC = Path(refsolver.__file__)


def _replay(stask, plan):
    """Execute a plan by hand on fact-name sets; return its cost (raises if a step is inapplicable)."""
    by_name = {a["name"]: a for a in stask["actions"]}
    s, cost = set(stask["init"]), 0
    for n in plan:
        a = by_name[n]
        assert set(a["pre"]) <= s, n
        s |= set(a["add"])
        cost += a["cost"]
    assert set(stask["goal"]) <= s
    return cost


def test_imports_nothing_from_ahg():
    """C2: no import of task.py, search.py or any other ahg module, at any level."""
    tree = ast.parse(SRC.read_text())
    mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert all(not m.startswith("ahg") for m in mods), mods
    assert not any(isinstance(n, ast.ImportFrom) and n.level > 0 for n in ast.walk(tree))
    assert set(mods) <= {"heapq", "itertools"}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_optimum(name):
    stask, opt, _ = FIXTURES[name]
    cost, plan = refsolver.solve(stask)
    assert cost == opt
    if opt == INF:
        assert plan is None
    else:
        assert _replay(stask, plan) == opt
    assert refsolver.hstar_all(stask)[frozenset(stask["init"])] == opt


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_hand_derived_state_values(name):
    stask, _, per_state = FIXTURES[name]
    h = refsolver.hstar_all(stask)
    for facts, v in per_state.items():
        assert h[frozenset(facts)] == v, facts
        assert refsolver.solve(stask, start=facts)[0] == v, facts


def test_forward_and_backward_dijkstra_agree_on_every_reachable_state():
    for stask, _, _ in FIXTURES.values():
        h = refsolver.hstar_all(stask)
        assert set(h) == refsolver.reachable(stask)
        for s, v in h.items():
            assert refsolver.solve(stask, start=s)[0] == v


def test_cap_and_validation():
    stask = FIXTURES["cycle"][0]
    assert refsolver.hstar_all(stask, cap=1) is None and refsolver.reachable(stask, cap=2) is None
    bad = {**stask, "actions": [*stask["actions"], {"name": "x", "pre": ["nope"], "add": ["a"], "cost": 1}]}
    with pytest.raises(refsolver.ReferenceTaskError):
        refsolver.solve(bad)
    with pytest.raises(refsolver.ReferenceTaskError):
        refsolver.solve({**stask, "facts": ["a", "a", "b", "c", "g"]})
    neg = {**stask, "actions": [{"name": "x", "pre": [], "add": ["a"], "cost": -1}]}
    with pytest.raises(refsolver.ReferenceTaskError):
        refsolver.solve(neg)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_serializer_round_trips_the_fixture(name):
    stask = FIXTURES[name][0]
    t = Task.from_names(stask["facts"], stask["init"], stask["goal"],
                        [(a["name"], a["pre"], a["add"], a["cost"]) for a in stask["actions"]])
    assert refsolver.serialize(t) == stask


@pytest.mark.parametrize("seed", range(40))
def test_generated_random_tasks_agree_with_brute_force_and_search(seed):
    t = random_task(seed)
    st = refsolver.serialize(t)
    h = refsolver.hstar_all(st)
    assert {refsolver.to_bits(st, s) for s in h} == reachable_states(t)
    for s, v in h.items():
        assert v == brute_force_optimal(t, start=refsolver.to_bits(st, s))
    assert h[frozenset(st["init"])] == uniform_cost(t).cost == astar(t, lmcut).cost


@pytest.mark.parametrize("family", ["flat", "tiered"])
@pytest.mark.parametrize("seed", range(5))
def test_generated_attack_graph_tasks_agree_with_search(family, seed):
    from test_attack_graph import POOL
    t = make_instance(family, seed, 5, POOL).task
    cost, plan = refsolver.solve(refsolver.serialize(t))
    assert cost == uniform_cost(t).cost == astar(t, lmcut).cost
    assert _replay(refsolver.serialize(t), plan) == cost
