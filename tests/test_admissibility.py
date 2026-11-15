"""C1 (D28 Round 8): heuristics claimed admissible against the independent h*, on every reachable state of
tiny instances; unreachable-goal detection tested separately; learned heuristics never audited as admissible."""

import pytest

from ahg.attack_graph import make_instance
from ahg.heuristics import ADMISSIBLE, HEURISTICS, INF, hmax, lmcut
from ahg.measure import HardFailure, audit_admissibility, audit_dead_end_detection, enumerate_exact
from ahg.task import Task

from helpers import brute_force_optimal, random_task, reachable_states
from ref_fixtures import FIXTURES

CLAIMED = {name: HEURISTICS[name] for name in sorted(ADMISSIBLE)}  # includes h_max and LM-cut
# Dead-end exactness is claimed only for h_max (the common pre-check, Round 7) and LM-cut; h0 and h_sp are
# admissible but cannot see that an AND precondition is unreachable.
DEAD_END_EXACT = {"hmax": hmax, "lmcut": lmcut}


def _task(stask):
    return Task.from_names(stask["facts"], stask["init"], stask["goal"],
                           [(a["name"], a["pre"], a["add"], a["cost"]) for a in stask["actions"]])


def test_hmax_and_lmcut_are_among_the_audited():
    assert {"hmax", "lmcut"} <= set(CLAIMED)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_admissible_on_every_reachable_state_of_the_fixtures(name):
    out = audit_admissibility(_task(FIXTURES[name][0]), CLAIMED)
    assert all(v["states"] > 0 for v in out.values())


@pytest.mark.parametrize("seed", range(40))
def test_admissible_on_every_reachable_state_of_random_tasks(seed):
    audit_admissibility(random_task(seed), CLAIMED)


@pytest.mark.parametrize("seed", range(5))
def test_admissible_on_tiny_attack_graphs(seed):
    from test_attack_graph import POOL
    audit_admissibility(make_instance("tiered", seed, 4, POOL).task, CLAIMED)


def test_the_audit_is_not_vacuous():
    """The fixtures make LM-cut or h_max strictly below h* somewhere, so the check compares real values."""
    t = _task(FIXTURES["shared_subgoal"][0])
    out = audit_admissibility(t, CLAIMED)
    assert out["hmax"]["below"] > 0
    assert hmax(t, t.init) == 3 and lmcut(t, t.init) <= 4


def test_an_overestimate_is_caught():
    t = _task(FIXTURES["cycle"][0])
    with pytest.raises(HardFailure):
        audit_admissibility(t, {"hmax": lambda task, s: hmax(task, s) + 1})


@pytest.mark.parametrize("name", ["hadd", "hff", "learned_gnn"])
def test_unclaimed_heuristics_are_refused(name):
    """A learned (or any inadmissible) heuristic is never treated as admissible (rule 8)."""
    t = _task(FIXTURES["cycle"][0])
    with pytest.raises(ValueError):
        audit_admissibility(t, {name: lambda task, s: 0})
    with pytest.raises(ValueError):
        audit_dead_end_detection(t, {name: lambda task, s: 0})


def test_dead_end_detection_on_impossible_fixtures():
    for name in ("impossible", "no_achiever"):
        t = _task(FIXTURES[name][0])
        assert audit_dead_end_detection(t, DEAD_END_EXACT) == len(reachable_states(t))  # every state is a dead end


@pytest.mark.parametrize("seed", range(40))
def test_dead_end_detection_on_random_tasks(seed):
    t = random_task(seed)
    dead = audit_dead_end_detection(t, DEAD_END_EXACT)
    assert dead == sum(brute_force_optimal(t, s) == INF for s in reachable_states(t))


def test_h0_is_admissible_but_not_dead_end_exact():
    with pytest.raises(HardFailure):
        audit_dead_end_detection(_task(FIXTURES["impossible"][0]), {"h0": HEURISTICS["h0"]})


def test_dead_end_detection_catches_both_errors():
    solvable, impossible = _task(FIXTURES["cycle"][0]), _task(FIXTURES["impossible"][0])
    with pytest.raises(HardFailure):  # INF where h* is finite: would prune a solvable state
        audit_dead_end_detection(solvable, {"hmax": lambda task, s: INF})
    with pytest.raises(HardFailure):  # finite where h* is INF: the pre-check would pass an unsolvable task
        audit_dead_end_detection(impossible, {"hmax": lambda task, s: 0})


def test_enumerate_exact_matches_brute_force_and_respects_cap():
    for seed in range(10):
        t = random_task(seed)
        ex = enumerate_exact(t, 10_000)
        assert set(ex) == reachable_states(t)
        assert all(v == brute_force_optimal(t, s) for s, v in ex.items())
    assert enumerate_exact(random_task(0), 1) is None
