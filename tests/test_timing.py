"""Timing (D28 Round 12, AUDIT_4 inventory #5): perf_counter_ns repeats, censoring, guarded log summaries."""

import math
import time

import pytest

from ahg import timing
from ahg.evaluate import _geo_ci, _run, summarise
from ahg.timing import InvalidMeasurement, Timing, geometric_mean, log_ratio, timed

from helpers import random_task


def test_resolution_comes_from_the_clock():
    assert timing.RESOLUTION_NS == max(1, math.ceil(time.get_clock_info("perf_counter").resolution * 1e9))


def test_timed_repeats_and_uses_perf_counter_ns(monkeypatch):
    ticks = iter([100, 1100, 2000, 2500, 3000, 6000])
    monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: next(ticks))
    calls = []
    results, t = timed(lambda: calls.append(1) or len(calls), 3)
    assert results == [1, 2, 3] and t.ns == (1000, 500, 3000)
    assert t.median_ns == 1000


def test_sub_resolution_is_censored_not_replaced():
    t = Timing((5, 500), resolution_ns=42)
    assert t.censored and t.ns == (5, 500)  # kept as measured
    assert t.record() == {"ns": [5, 500], "resolution_ns": 42, "censored": True}
    with pytest.raises(InvalidMeasurement):
        t.median_ns
    with pytest.raises(InvalidMeasurement):
        log_ratio(t, Timing((900,), resolution_ns=42))
    assert not Timing((42, 43), resolution_ns=42).censored  # exactly the resolution is one full tick


@pytest.mark.parametrize("ns", [(0,), (-3,), (10, 0), (1.5,), ()])
def test_non_positive_or_non_integer_durations_are_invalid(ns):
    with pytest.raises(InvalidMeasurement):
        Timing(ns)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), None, True])
def test_log_ratio_and_geometric_mean_refuse_invalid_values(bad):
    with pytest.raises(InvalidMeasurement):
        log_ratio(bad, 1.0)
    with pytest.raises(InvalidMeasurement):
        log_ratio(1.0, bad)
    with pytest.raises(InvalidMeasurement):
        geometric_mean([2.0, bad])


def test_geometric_mean_and_log_ratio_values():
    assert geometric_mean([1, 4, 16]) == pytest.approx(4.0)
    assert log_ratio(Timing((200, 200, 900), 1), Timing((100,), 1)) == pytest.approx(math.log(2))
    with pytest.raises(InvalidMeasurement):
        geometric_mean([])
    with pytest.raises(InvalidMeasurement):
        _geo_ci([0.0, float("nan")], "k")


def test_run_repeats_rebind_and_must_agree():
    t = random_task(4)
    made = []
    from ahg.heuristics import hmax

    def make():
        made.append(1)
        return hmax

    rec = _run(t, make, None, 3)
    assert len(made) == 3 and len(rec["time"]["ns"]) == 3 and all(x > 0 for x in rec["time"]["ns"])
    flip = iter([hmax, lambda task, s: 0, hmax])
    from ahg.measure import HardFailure
    with pytest.raises(HardFailure):
        _run(t, lambda: next(flip), None, 3)


def _rec(exp_m, exp_b, ns_m, ns_b, res=10):
    run = lambda e, ns: {"expanded": e, "cost": None if e is None else 5, "budget_hit": e is None,
                         "time": {"ns": list(ns), "resolution_ns": res, "censored": min(ns) < res}}
    return {"hstar": 5, "runs": {"m": run(exp_m, ns_m), "b": run(exp_b, ns_b)},
            "over": {"m": {"walk": [], "astar_expanded": [], "successor": []}}, "order": {}}


def test_summarise_uses_common_tasks_and_counts_censored(monkeypatch):
    import ahg.evaluate as ev
    monkeypatch.setattr(ev, "COMPARATORS", ("b",))
    recs = [_rec(10, 20, (100,), (200,)), _rec(None, 20, (100,), (200,)), _rec(30, 30, (1,), (200,))]
    s = summarise(recs, ["m"], 0)["m"]["vs"]["b"]
    assert s["common"] == 2 and s["time_censored"] == 1
    assert s["expansions"][0] == pytest.approx(math.sqrt(0.5))  # exp(mean(log 0.5, log 1))
    assert s["time"][0] == pytest.approx(0.5)  # only the uncensored task


def test_summarise_raises_on_a_zero_count(monkeypatch):
    import ahg.evaluate as ev
    monkeypatch.setattr(ev, "COMPARATORS", ("b",))
    with pytest.raises(InvalidMeasurement):
        summarise([_rec(0, 20, (100,), (200,))], ["m"], 0)
