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


def _recs(exp: dict, hstar=5, n=2, unsolved=()):
    run = lambda c, e: {"expanded": e, "cost": None if c in unsolved else hstar}
    return [{"hstar": hstar, "runs": {c: run(c, e) for c, e in exp.items()}} for _ in range(n)]


BASE = {"hmax": 50, "hadd": 50, "hff": 50, "lmcut": 50, "hsp": 50}


@pytest.mark.parametrize("winner", ["hmax", "hadd", "hff", "lmcut", "hsp"])
def test_any_of_the_five_can_be_picked(winner):
    r = strongest_baseline(_recs({**BASE, winner: 7}))
    assert r["baseline"] == winner and r["tied"] == [winner] and r["common"] == 2
    assert r["geo_mean_expansions"][winner] == pytest.approx(7)


def test_ties_go_to_lmcut_and_other_ties_are_not_decided():
    assert strongest_baseline(_recs({**BASE, "hff": 7, "lmcut": 7}))["baseline"] == "lmcut"
    with pytest.raises(ValueError):
        strongest_baseline(_recs({**BASE, "hff": 7, "hadd": 7}))


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
