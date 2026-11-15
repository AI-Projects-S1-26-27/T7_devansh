"""Fast Downward timing harness (D28 Round 7): separate translate / search / end-to-end timings, repeats."""

import pytest

from ahg import fd_timing, refsolver
from ahg.pddl import domain_pddl, problem_pddl
from ahg.task import Task

from ref_fixtures import FIXTURES

FD_OK = (fd_timing.FD_DIR / "fast-downward.py").exists() and fd_timing.fd_version() == fd_timing.FD_VERSION
needs_fd = pytest.mark.skipif(not FD_OK, reason="Fast Downward release-24.06.1 not set up (scripts/setup_fast_downward.sh)")

LOG = """Done! [0.000s CPU, 0.002s wall-clock]
[t=0.001893s, 410597136 KB] Actual search time: 0.000034s
[t=0.001896s, 410597136 KB] Plan cost: 4
[t=0.001896s, 410597136 KB] Expanded 5 state(s).
[t=0.001896s, 410597136 KB] Evaluated 6 state(s).
[t=0.001896s, 410597136 KB] Expanded until last jump: 0 state(s).
[t=0.001896s, 410597136 KB] Search time: 0.000078s
[t=0.001896s, 410597136 KB] Total time: 0.001896s
"""


def test_parse_log():
    assert fd_timing.parse_log(LOG) == {"cost": 4, "expanded": 5, "evaluated": 6, "translate_wall_s": 0.002,
                                        "search_s": 0.000078, "total_s": 0.001896}
    assert fd_timing.parse_log("")["cost"] is None


def _pddl(tmp_path, name):
    st = FIXTURES[name][0]
    t = Task.from_names(st["facts"], st["init"], st["goal"], [(a["name"], a["pre"], a["add"], a["cost"]) for a in st["actions"]])
    (tmp_path / "d.pddl").write_text(domain_pddl(t))
    (tmp_path / "p.pddl").write_text(problem_pddl(t))
    return tmp_path / "d.pddl", tmp_path / "p.pddl", refsolver.solve(st)[0]


def test_refuses_other_versions_and_heuristics(tmp_path, monkeypatch):
    d, p, _ = _pddl(tmp_path, "cycle")
    with pytest.raises(ValueError):
        fd_timing.run(d, p, "hff", 1)
    monkeypatch.setattr(fd_timing, "fd_version", lambda fd_dir=None: "release-23.06.0")
    with pytest.raises(fd_timing.FDError):
        fd_timing.run(d, p, "lmcut", 1)


@needs_fd
@pytest.mark.parametrize("heuristic", ["lmcut", "hmax"])
def test_solved_task_times_each_phase_per_repeat(tmp_path, heuristic):
    d, p, opt = _pddl(tmp_path, "shared_subgoal")
    r = fd_timing.run(d, p, heuristic, 2)
    assert r["status"] == "solved" and r["cost"] == opt == 4 and r["fd_version"] == "release-24.06.1"
    for phase in ("translate", "search", "end_to_end"):
        assert len(r["wall"][phase]["ns"]) == 2 and all(x > 0 for x in r["wall"][phase]["ns"])
    assert all(rep["end_to_end"]["search_s"] is not None and rep["phases"]["translate_wall_s"] is not None
               for rep in r["fd_reported"])


@needs_fd
def test_unsolvable_task_is_reported_not_solved(tmp_path):
    d, p, opt = _pddl(tmp_path, "impossible")
    r = fd_timing.run(d, p, "hmax", 1)
    assert opt == float("inf") and r["status"] != "solved" and r["cost"] is None
