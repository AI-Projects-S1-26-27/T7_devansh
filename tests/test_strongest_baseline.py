"""Strongest classical baseline (D28 Round 11, P0c; M1): every A* baseline records expansions, so any of
h_max, h_add, h_FF, LM-cut, h_sp can be picked; the inadmissible ones are not checked against h*."""

from types import SimpleNamespace

import pytest

import ahg.evaluate as ev
from ahg import refsolver
from ahg.evaluate import STRONGEST_CANDIDATES, evaluate, strongest_baseline
from ahg.heuristics import hadd, hff, lmcut
from ahg.measure import HardFailure
from ahg.timing import InvalidMeasurement

from helpers import random_task


def _batch(tasks):
    """A minimal opened batch: one instance per task, h*(init) from the reference solver."""
    insts, recs, exs = [], [], []
    for i, t in enumerate(tasks):
        cost, _ = refsolver.solve(refsolver.serialize(t))
        insts.append(SimpleNamespace(task=t))
        recs.append({"stats": {"hstar_init": cost},
                     "spec": {"seed": i, "variant": "v", "n": 0, "D": 0, "rho": 0, "m": 0}})
        exs.append(SimpleNamespace(instance=i, state=t.init, hstar=cost, sources=("walk",)))
    return SimpleNamespace(manifest={"batch_id": "b", "instances": recs}, instances=insts, examples=exs)


def _solvable(k):
    out, seed = [], 0
    while len(out) < k:
        t = random_task(seed)
        cost, _ = refsolver.solve(refsolver.serialize(t))
        if 0 < cost < float("inf"):
            out.append(t)
        seed += 1
    return out


def test_every_astar_candidate_records_expansions_under_the_default_protocol():
    recs = evaluate([_batch(_solvable(3))], [])
    assert {c for c in STRONGEST_CANDIDATES} == {"hmax", "hadd", "hff", "lmcut", "hsp"}
    for r in recs:
        assert (r["repeats"], r["warmup"]) == (5, 1)
        for c in STRONGEST_CANDIDATES:
            run = r["runs"][c]
            assert isinstance(run["expanded"], int) and run["expanded"] > 0 and run["cost"] is not None
            assert len(run["time"]["ns"]) == 5
    assert strongest_baseline(recs)["baseline"] in STRONGEST_CANDIDATES


def test_inadmissible_baselines_are_not_checked_against_hstar(monkeypatch):
    b = _batch(_solvable(1))
    b.manifest["instances"][0]["stats"]["hstar_init"] += 1  # a wrong label: only an h*-check notices
    monkeypatch.setattr(ev, "HADD_SEARCHES", {})
    monkeypatch.setattr(ev, "BASELINES", {"hadd": hadd, "hff": hff})
    evaluate([b], [], repeats=1, warmup=0)  # no HardFailure
    monkeypatch.setattr(ev, "BASELINES", {"lmcut": lmcut})
    with pytest.raises(HardFailure):
        evaluate([b], [], repeats=1, warmup=0)


def _recs(exp: dict, hstar=5, n=2, unsolved=(), ns=None):
    """n identical records; ns maps a method to its per-task repeat times (default 1 ms for all)."""
    ns = ns or {}
    run = lambda c, e: {"expanded": e, "cost": None if c in unsolved else hstar,
                        "time": {"ns": ns.get(c, [10**6]), "resolution_ns": 1000}}
    return [{"hstar": hstar, "runs": {c: run(c, e) for c, e in exp.items()}} for _ in range(n)]


BASE = {"hmax": 50, "hadd": 50, "hff": 50, "lmcut": 50, "hsp": 50}


@pytest.mark.parametrize("winner", ["hmax", "hadd", "hff", "lmcut", "hsp"])
def test_any_of_the_five_can_be_picked(winner):
    r = strongest_baseline(_recs({**BASE, winner: 7}))
    assert r["baseline"] == winner and r["tied"] == [winner] and r["common"] == 2 and r["tie_break"] is None
    assert r["geo_mean_expansions"][winner] == pytest.approx(7)


def test_ties_go_to_lmcut_even_when_lmcut_is_slower():
    r = strongest_baseline(_recs({**BASE, "hff": 7, "lmcut": 7}, ns={"hff": [10], "lmcut": [10**9]}))
    assert r["baseline"] == "lmcut" and r["tie_break"] == "lmcut" and r["tied"] == ["hff", "lmcut"]


def test_tie_without_lmcut_goes_to_lowest_median_time():
    # Per-task medians: hff 300 (repeats 100/300/900), hadd 400, hmax 200 on one task and 900 on the other
    # -> median 550. hff has the lowest median time though hmax is fastest on one task and alphabetically first.
    recs = _recs({**BASE, "hff": 7, "hadd": 7, "hmax": 7}, n=1,
                 ns={"hff": [100_000, 300_000, 900_000], "hadd": [400_000], "hmax": [200_000]}) \
        + _recs({**BASE, "hff": 7, "hadd": 7, "hmax": 7}, n=1,
                ns={"hff": [300_000], "hadd": [400_000], "hmax": [900_000]})
    r = strongest_baseline(recs)
    assert r["baseline"] == "hff" and r["tie_break"] == "time" and r["time_censored"] == 0
    assert r["median_time_ns"] == {"hmax": 550_000, "hadd": 400_000, "hff": 300_000}


def test_tie_without_lmcut_and_equal_time_goes_alphabetically():
    r = strongest_baseline(_recs({**BASE, "hsp": 7, "hff": 7, "hmax": 7}))  # all 1 ms
    assert r["baseline"] == "hff" and r["tie_break"] == "alphabetical"
    r = strongest_baseline(_recs({**BASE, "hsp": 7, "hmax": 7, "hff": 7}, ns={"hff": [9 * 10**6]}))
    assert r["baseline"] == "hmax" and r["tie_break"] == "alphabetical"  # hmax and hsp tie on time


def test_censored_times_are_left_out_paired_and_never_raise():
    # Task 1: hadd censored (below 1000 ns) -> task 1 dropped for both; task 2 decides: hff faster.
    recs = _recs({**BASE, "hff": 7, "hadd": 7}, n=1, ns={"hadd": [5], "hff": [9 * 10**6]}) \
        + _recs({**BASE, "hff": 7, "hadd": 7}, n=1, ns={"hadd": [2 * 10**6]})
    r = strongest_baseline(recs)
    assert r["baseline"] == "hff" and r["tie_break"] == "time" and r["time_censored"] == 1
    # Every task censored: no time comparison is possible -> alphabetical, still no error.
    r = strongest_baseline(_recs({**BASE, "hff": 7, "hadd": 7}, ns={"hff": [5]}))
    assert r["baseline"] == "hadd" and r["tie_break"] == "alphabetical"
    assert r["median_time_ns"] is None and r["time_censored"] == 2


def test_common_tasks_trivial_tasks_and_bad_inputs():
    recs = _recs({**BASE, "hff": 7}) + _recs({**BASE, "hmax": 1}, unsolved=("hsp",)) \
        + _recs({c: 0 for c in BASE}, hstar=0, n=3)
    r = strongest_baseline(recs)
    assert r["baseline"] == "hff" and r["common"] == 2 and r["trivial"] == 3
    with pytest.raises(ValueError):
        strongest_baseline([{"hstar": 5, "runs": {"hmax": {"expanded": 1, "cost": 5}}}])
    with pytest.raises(InvalidMeasurement):
        strongest_baseline(_recs(BASE, unsolved=("hff",)))
    with pytest.raises(InvalidMeasurement):
        strongest_baseline(_recs({**BASE, "hsp": 0}))
