"""Timing protocol (decided 15 Nov 2026, M1; OQ-P7-1): 5 timed repeats after 1 discarded warm-up, median."""

import inspect

import pytest

from ahg import fd_timing, timing
from ahg.evaluate import _run, evaluate
from ahg.heuristics import hmax
from ahg.timing import timed

from helpers import random_task
from test_fd_timing import LOG


def test_protocol_constants_and_defaults(monkeypatch, capsys):
    assert (timing.REPEATS, timing.WARMUP) == (5, 1)
    p = inspect.signature(evaluate).parameters
    assert (p["repeats"].default, p["warmup"].default) == (5, 1)
    p = inspect.signature(fd_timing.run).parameters
    assert (p["repeats"].default, p["warmup"].default) == (5, 1)
    seen = []
    monkeypatch.setattr(fd_timing, "run", lambda d, p, h, repeats, fd_dir, *, warmup: seen.append((repeats, warmup)))
    fd_timing.main(["d.pddl", "p.pddl", "--heuristic", "lmcut"])  # the CLI defaults to the same protocol
    assert seen == [(5, 1)]


def test_warmup_runs_are_discarded_and_untimed(monkeypatch):
    ticks = iter([0, 10_000, 10_000, 30_000, 30_000, 60_000])  # consumed only by the 3 timed repeats
    monkeypatch.setattr(timing.time, "perf_counter_ns", lambda: next(ticks))
    calls = []
    results, t = timed(lambda: calls.append(1) or len(calls), 3, warmup=2)
    assert len(calls) == 5 and results == [3, 4, 5] and t.ns == (10_000, 20_000, 30_000) and t.median_ns == 20_000
    with pytest.raises(ValueError):
        timed(lambda: 0, 1, warmup=-1)


def test_run_makes_a_fresh_heuristic_for_warmup_too():
    made = []
    rec = _run(random_task(4), lambda: made.append(1) or hmax, None, 5, warmup=1)
    assert len(made) == 6 and len(rec["time"]["ns"]) == 5


def test_fd_run_discards_warmup(monkeypatch, tmp_path):
    calls = []

    def fake(args, cwd):
        calls.append(args)
        return 0, LOG, 1000 + len(calls)

    monkeypatch.setattr(fd_timing, "_run", fake)
    monkeypatch.setattr(fd_timing, "fd_version", lambda fd_dir=None: fd_timing.FD_VERSION)
    (tmp_path / "d.pddl").write_text("")
    (tmp_path / "p.pddl").write_text("")
    r = fd_timing.run(tmp_path / "d.pddl", tmp_path / "p.pddl", "lmcut")
    assert len(calls) == 3 * 6  # (end-to-end, translate, search) x (1 warm-up + 5 timed)
    assert r["repeats"] == 5 and r["warmup"] == 1 and len(r["fd_reported"]) == 5
    assert r["wall"]["end_to_end"]["ns"] == [1004, 1007, 1010, 1013, 1016]  # warm-up call 1 not recorded
    with pytest.raises(ValueError):
        fd_timing.run(tmp_path / "d.pddl", tmp_path / "p.pddl", "lmcut", warmup=-1)
