"""Fast Downward 24.06.1 timing harness (D28 Round 7: compiled end-to-end timing baselines).

Runs FD's A*+LM-cut and A*+h_max on any PDDL domain/problem pair it is given
(the exporter, pddl.py, is updated separately). Per repeat it measures three
wall-clock durations with time.perf_counter_ns() around the FD driver process:

    translate    fast-downward.py --translate   domain problem -> output.sas
    search       fast-downward.py output.sas --search ...
    end_to_end   fast-downward.py domain problem --search ...   (one process, both phases)

so each phase and the whole run are timed separately, every repeat in a fresh
temporary directory. FD's own reported times (translator "Done! [.. wall-clock]",
"Search time", "Total time") are parsed and kept beside them; they exclude
process start-up, ours do not. Python-loop timings and these compiled timings
are always reported separately (D28 Rounds 7 and 12).

Each run starts with WARMUP discarded runs of the whole per-repeat sequence,
then REPEATS timed ones, summarised by the median (decided 15 Nov 2026, M1).

Durations go through timing.Timing: below-resolution repeats are censored,
never replaced. The FD checkout must be release-24.06.1, the pinned version;
anything else is refused.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .timing import REPEATS, WARMUP, Timing

ROOT = Path(__file__).resolve().parents[2]
FD_DIR = ROOT / "third_party" / "downward"
FD_VERSION = "release-24.06.1"
SEARCHES = {"lmcut": "astar(lmcut())", "hmax": "astar(hmax())"}
# FD driver exit codes (driver/returncodes.py): 0 plan found; 10 the translator proved the task unsolvable;
# 11 the search proved it unsolvable; 12 the search ended without a plan and without a proof.
STATUS = {0: "solved", 10: "unsolvable_translate", 11: "unsolvable_search", 12: "unsolved_incomplete"}


class FDError(RuntimeError):
    pass


def fd_version(fd_dir: Path = FD_DIR) -> str:
    out = subprocess.run(["git", "-C", str(fd_dir), "describe", "--tags", "--exact-match"],
                         capture_output=True, text=True, check=False)
    return out.stdout.strip()


def _grab(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse_log(text: str) -> dict:
    """The values FD prints: plan cost, expansions, evaluations, and its own phase times in seconds."""
    return {"cost": _grab(r"Plan cost: (\d+)", text, int),
            "expanded": _grab(r"Expanded (\d+) state\(s\)\.", text, int),
            "evaluated": _grab(r"Evaluated (\d+) state\(s\)\.", text, int),
            "translate_wall_s": _grab(r"Done! \[[\d.]+s CPU, ([\d.]+)s wall-clock\]", text),
            "search_s": _grab(r"\] Search time: ([\d.]+)s", text),
            "total_s": _grab(r"\] Total time: ([\d.]+)s", text)}


def _run(args: list[str], cwd: Path) -> tuple[int, str, int]:
    """(exit code, stdout, wall ns) of one FD driver process."""
    t0 = time.perf_counter_ns()
    p = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, check=False)
    ns = time.perf_counter_ns() - t0
    if p.returncode not in STATUS:
        raise FDError(f"FD exit code {p.returncode}: {p.stderr[-2000:]}")
    return p.returncode, p.stdout, ns


def run(domain: Path, problem: Path, heuristic: str, repeats: int = REPEATS, fd_dir: Path = FD_DIR, *,
        warmup: int = WARMUP) -> dict:
    """Time FD A*+<heuristic> on one PDDL task: `warmup` discarded runs, then `repeats` timed ones.

    heuristic in SEARCHES. Warm-up runs are not timed, parsed or compared; they only take first-call
    costs (disk cache, Python start-up of the driver) out of the timed repeats."""
    if heuristic not in SEARCHES:
        raise ValueError(f"heuristic must be one of {sorted(SEARCHES)}")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    version = fd_version(fd_dir)
    if version != FD_VERSION:
        raise FDError(f"Fast Downward at {fd_dir} is {version!r}, not the pinned {FD_VERSION}")
    fd = str(fd_dir / "fast-downward.py")
    domain, problem = Path(domain).resolve(), Path(problem).resolve()
    search = SEARCHES[heuristic]
    ns = {"translate": [], "search": [], "end_to_end": []}
    reports, outcomes = [], set()
    for i in range(warmup + repeats):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            code_e, log_e, t_e = _run([fd, "--plan-file", "e2e_plan", domain, problem, "--search", search], d)
            code_t, log_t, t_t = _run([fd, "--sas-file", "task.sas", "--translate", domain, problem], d)
            if code_t == 10:  # proved unsolvable in translation: there is no search phase to time
                code_s, log_s, t_s = 10, "", None
            else:
                code_s, log_s, t_s = _run([fd, "--plan-file", "plan", "task.sas", "--search", search], d)
        if i < warmup:
            continue  # discarded warm-up run
        e2e, phase = parse_log(log_e), parse_log(log_t + log_s)
        if code_e != code_s or (e2e["cost"], e2e["expanded"]) != (phase["cost"], phase["expanded"]):
            raise FDError("the end-to-end run and the phase-by-phase run disagree")
        outcomes.add((code_e, e2e["cost"], e2e["expanded"]))
        ns["end_to_end"].append(t_e)
        ns["translate"].append(t_t)
        if t_s is not None:
            ns["search"].append(t_s)
        reports.append({"end_to_end": e2e, "phases": phase})
    if len(outcomes) != 1:
        raise FDError(f"repeats disagree: {sorted(outcomes, key=str)}")
    code, cost, expanded = outcomes.pop()
    return {"fd_version": version, "heuristic": heuristic, "search": search, "repeats": repeats, "warmup": warmup,
            "status": STATUS[code], "exit_code": code, "cost": cost, "expanded": expanded,
            "wall": {k: Timing(tuple(v)).record() if v else None for k, v in ns.items()},
            "fd_reported": reports}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("domain", type=Path)
    ap.add_argument("problem", type=Path)
    ap.add_argument("--heuristic", choices=sorted(SEARCHES), action="append", required=True)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--fd-dir", type=Path, default=FD_DIR)
    a = ap.parse_args(argv)
    print(json.dumps({h: run(a.domain, a.problem, h, a.repeats, a.fd_dir, warmup=a.warmup) for h in a.heuristic}, indent=1))


if __name__ == "__main__":
    main()
