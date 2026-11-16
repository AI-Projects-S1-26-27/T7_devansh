"""D28 report builder (P7): every table and figure from the results directories (Round 12, §13.1).

Input: evaluation runs under results/d28_eval/<split>/<run>/raw.json.gz written by runlog.write_run. raw is
{"split", "regime", "tie_break", "records": [...]}, one record per task:

    {"task": digest, "network": id, "n": int, "solvable": bool, "unsolvable_reason": str|None,
     "certified": bool, "hstar": number|None,
     "runs": {method: {"expanded": int|None, "cost": number|None, "budget_hit": bool,
                       "python_time": Timing record|None, "fd_time": Timing record|None,
                       "peak_memory_bytes": int|None,
                       "overestimation": {"states": int, "violations": int}|None}}}

Learned methods run once per D23 seed and are named "<model>@<seed>" (gnn@0, mlp@1, wl@2); every classical
method has one run. build_report() reads nothing else, so one command regenerates all of it.

Rules applied (Round 12): primary regime `ac`; expansions compared on tasks solved by every method of the
comparison; tasks with h*(s0) = 0 are counted and left out; the per-task value is the mean over seeds of
log E; Holm across 3 comparisons x every primary split, `ac` only; `uniform` tables are separate and never
enter a verdict. Missing or non-positive values raise (Timing / log_ratio), never get replaced.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from . import runlog
from .stats import coverage, decide_family
from .timing import InvalidMeasurement, Timing, geometric_mean

CLASSICAL = ("h0", "hmax", "hadd", "hff", "lmcut", "hsp")
LEARNED = ("gnn", "mlp", "wl")
COMPARISONS = ("BASE", "mlp", "wl")  # GNN against the strongest classical baseline, the MLP, the WL baseline
STRONGEST_POOL = ("hmax", "hadd", "hff", "lmcut", "hsp")
PRIMARY_REGIME = "ac"


def model_of(method: str) -> str:
    return method.split("@")[0]


def load_runs(root: Path) -> list[dict]:
    """Every evaluation raw under root/<split>/<run>/, tagged with its run folder."""
    out = []
    for raw_path in sorted(root.glob("*/*/raw.json.gz")):
        raw = runlog.load_raw(raw_path.parent)
        out.append(raw | {"run": str(raw_path.parent.relative_to(root))})
    return out


def _methods(records: list[dict]) -> list[str]:
    seen = {m for r in records for m in r["runs"]}
    return [m for m in CLASSICAL if m in seen] + sorted(m for m in seen if model_of(m) in LEARNED)


def _ns(rec: dict | None) -> float | None:
    return None if rec is None else Timing.from_record(rec).median_ns


def _fmt(x, spec=".3g") -> str:
    return "–" if x is None else format(x, spec)


def split_table(raw: dict) -> list[dict]:
    """One row per method for one (split, regime) raw: the Round 12 secondary quantities."""
    recs = raw["records"]
    solvable = [r for r in recs if r["solvable"]]
    rows = []
    for m in _methods(recs):
        runs = [(r, r["runs"][m]) for r in solvable if m in r["runs"]]
        solved = [(r, x) for r, x in runs if x["cost"] is not None]
        cert = [(r, x) for r, x in solved if r["certified"] and r["hstar"]]  # h* > 0: a ratio exists
        py = [_ns(x.get("python_time")) for _, x in solved]
        fd = [_ns(x.get("fd_time")) for _, x in solved]
        mem = [x["peak_memory_bytes"] for _, x in solved if x.get("peak_memory_bytes") is not None]
        over = [x["overestimation"] for _, x in runs if x.get("overestimation")]
        states = sum(o["states"] for o in over)
        rows.append({
            "method": m, "solvable": len(solvable), "solved": len(solved),
            "coverage": len(solved) / len(solvable) if solvable else None,
            "geo_expansions": geometric_mean([x["expanded"] for _, x in solved if x["expanded"]]) if any(
                x["expanded"] for _, x in solved) else None,
            "cost_ratio": geometric_mean([x["cost"] / r["hstar"] for r, x in cert]) if cert else None,
            "cost_ratio_tasks": len(cert),
            "python_ms": statistics.median(py) / 1e6 if py and None not in py else None,
            "fd_ms": statistics.median(fd) / 1e6 if fd and None not in fd else None,
            "peak_mem_mb": max(mem) / 2**20 if mem else None,
            "overestimation_rate": sum(o["violations"] for o in over) / states if states else None})
    return rows


def certification(raw: dict) -> dict:
    """Label-certification coverage (Round 9) and the unsolvable count with reasons (Round 7)."""
    recs = raw["records"]
    sol = [r for r in recs if r["solvable"]]
    reasons: dict[str, int] = defaultdict(int)
    for r in recs:
        if not r["solvable"]:
            reasons[r["unsolvable_reason"] or "unknown"] += 1
    return {"tasks": len(recs), "solvable": len(sol), "unsolvable": len(recs) - len(sol),
            "unsolvable_reasons": dict(sorted(reasons.items())),
            "certified": sum(r["certified"] for r in sol),
            "certification_coverage": sum(r["certified"] for r in sol) / len(sol) if sol else None}


def _log_e(run: dict) -> float:
    if run["expanded"] is None or run["cost"] is None:
        raise InvalidMeasurement("an unsolved run has no expansion count")
    if run["expanded"] <= 0:
        raise InvalidMeasurement(f"expansions {run['expanded']} is not positive")
    return math.log(run["expanded"])


def pick_base(val_records: list[dict]) -> str:
    """The strongest classical baseline on validation records (Round 11): lowest geometric-mean expansions on
    the non-trivial tasks all candidates solved; ties go to LM-cut, else the alphabetically first."""
    sol = [r for r in val_records if r["solvable"] and r.get("hstar") != 0
           and all(r["runs"].get(c, {}).get("cost") is not None for c in STRONGEST_POOL)]
    if not sol:
        raise InvalidMeasurement("no validation task solved by every baseline candidate")
    mean = {c: math.fsum(_log_e(r["runs"][c]) for r in sol) / len(sol) for c in STRONGEST_POOL}
    best = min(mean.values())
    tied = [c for c in STRONGEST_POOL if mean[c] == best]
    return "lmcut" if "lmcut" in tied else sorted(tied)[0]


def comparison(raw: dict, other: str) -> dict | None:
    """Inputs of one GNN-vs-`other` verdict on one split: logs[network][seed][task] and paired task values.
    `other` is a classical method or a learned model name. None if no task survives."""
    recs = [r for r in raw["records"] if r["solvable"]]
    g_seeds = sorted(m for m in _methods(recs) if model_of(m) == "gnn")
    o_methods = [other] if other in CLASSICAL else sorted(m for m in _methods(recs) if model_of(m) == other)
    if not g_seeds or not o_methods:
        return None
    need = g_seeds + o_methods
    common, trivial = [], 0
    for r in recs:
        if not all(m in r["runs"] and r["runs"][m]["cost"] is not None for m in need):
            continue
        if r.get("hstar") == 0:
            trivial += 1
            continue
        common.append(r)
    if not common:
        return None
    g_n = len(g_seeds)
    per_net: dict[str, list] = defaultdict(lambda: [[] for _ in range(g_n)])
    values = []
    for r in common:
        lg = [_log_e(r["runs"][m]) for m in g_seeds]
        lo = [_log_e(r["runs"][m]) for m in o_methods]
        base = sum(lo) / len(lo)
        values.append(sum(lg) / len(lg) - base)  # mean over seeds of log E_gnn minus that of the other
        for j, x in enumerate(lg):
            per_net[r["network"]][j].append(x - (lo[j] if len(lo) == g_n else base))
    solv = [r["task"] for r in recs]
    cov = lambda ms: min(coverage({t["task"]: t["runs"][m]["expanded"] if t["runs"].get(m, {}).get("cost") is not None
                                   else None for t in recs}, solv) for m in ms)
    return {"logs": list(per_net.values()), "task_values": values, "common": len(common), "trivial": trivial,
            "coverage_gnn": cov(g_seeds), "coverage_other": cov(o_methods)}


def verdict_table(by_split: dict[str, dict], base: str, n_boot: int) -> list[dict]:
    """Verdicts for the whole confirmatory family: 3 comparisons x every primary split, Holm across all."""
    family, meta = [], []
    for split, raw in sorted(by_split.items()):
        for other in COMPARISONS:
            c = comparison(raw, base if other == "BASE" else other)
            if c is None:
                meta.append({"name": f"{split}|GNN vs {other}", "split": split, "other": other, "verdict": "NOT RUN",
                             "common": 0})
                continue
            family.append({"name": f"{split}|GNN vs {other}", **c})
            meta.append({"name": family[-1]["name"], "split": split, "other": other, "common": c["common"],
                         "trivial": c["trivial"]})
    decided = {d["name"]: d for d in decide_family(family, n=n_boot)} if family else {}
    return [m | ({k: decided[m["name"]][k] for k in ("ratio", "ci", "wilcoxon", "p_holm", "coverage_gnn",
                                                      "coverage_other", "verdict")} if m["name"] in decided else {})
            for m in meta]


def _md_table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def size_curves(by_split: dict[str, dict]) -> dict:
    """geometric-mean expansions by network size n, per method, over the solved solvable tasks of all splits."""
    pts: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for raw in by_split.values():
        for r in raw["records"]:
            if not r["solvable"]:
                continue
            for m, x in r["runs"].items():
                if x["cost"] is not None and x["expanded"]:
                    pts[model_of(m)][r["n"]].append(x["expanded"])
    return {m: {n: geometric_mean(v) for n, v in sorted(d.items())} for m, d in pts.items()}


def plot_sizes(curves: dict, path: Path, title: str) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not curves:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, d in sorted(curves.items()):
        ax.plot(list(d), list(d.values()), marker="o", label=m)
    ax.set_xlabel("hosts n")
    ax.set_ylabel("geometric-mean expansions (solved tasks)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


PRIMARY_SPLITS_NOTE = "primary splits only enter the verdict family; sensitivity and exploratory splits are tables only"


def build_report(eval_root: Path, out_dir: Path, *, primary: set[str], val_split: str | None = None,
                 n_boot: int = 10_000) -> Path:
    """Write report.md and PNG figures into out_dir; return report.md. primary: names of primary splits.

    The strongest baseline is chosen on the `ac` records of val_split (Round 11); without one, the verdict
    table is omitted and says so."""
    runs = load_runs(eval_root)
    if not runs:
        raise FileNotFoundError(f"no evaluation runs under {eval_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    keyed: dict[tuple[str, str], dict] = {}
    for raw in runs:
        k = (raw["split"], raw["regime"])
        if (k, raw["tie_break"]) in keyed:
            raise ValueError(f"two runs for {k} under tie rule {raw['tie_break']}: pick one run per split")
        keyed[(k, raw["tie_break"])] = raw
    L = ["# D28 results report", "", f"Source: `{eval_root}`. {PRIMARY_SPLITS_NOTE}.", ""]
    base = None
    if val_split and ((val_split, PRIMARY_REGIME), "fifo") in keyed:
        base = pick_base(keyed[((val_split, PRIMARY_REGIME), "fifo")]["records"])
    for regime in sorted({k[0][1] for k in keyed}, key=lambda r: r != PRIMARY_REGIME):
        L += [f"## Regime `{regime}`" + (" (primary)" if regime == PRIMARY_REGIME else " (secondary comparison, no verdict)"), ""]
        for (split, rg), tie in sorted(k for k in keyed if k[0][1] == regime):
            raw = keyed[((split, rg), tie)]
            cert = certification(raw)
            L += [f"### {split} / tie rule {tie}", "",
                  f"tasks {cert['tasks']}, solvable {cert['solvable']}, unsolvable {cert['unsolvable']} "
                  f"{cert['unsolvable_reasons']}, certification coverage {_fmt(cert['certification_coverage'], '.3f')}"
                  f" ({cert['certified']}/{cert['solvable']})", ""]
            rows = [[r["method"], r["solved"], _fmt(r["coverage"], ".3f"), _fmt(r["geo_expansions"]),
                     _fmt(r["cost_ratio"], ".4f") + f" ({r['cost_ratio_tasks']})", _fmt(r["python_ms"]),
                     _fmt(r["fd_ms"]), _fmt(r["peak_mem_mb"]), _fmt(r["overestimation_rate"], ".3f")]
                    for r in split_table(raw)]
            L += [_md_table(["method", "solved", "coverage", "geo expansions", "cost / optimum (certified tasks)",
                             "Python ms", "FD ms", "peak MB", "overestimation rate"], rows), ""]
            png = f"sizes_{split}_{regime}_{tie}.png"
            if plot_sizes(size_curves({split: raw}), out_dir / png, f"{split} ({regime}, {tie})"):
                L += [f"![{split}]({png})", ""]
    L += ["## Verdict rule (primary regime `ac`, tie rule fifo)", ""]
    prim = {s: keyed[((s, PRIMARY_REGIME), "fifo")] for s in sorted(primary) if ((s, PRIMARY_REGIME), "fifo") in keyed}
    if base is None or not prim:
        L += ["Not built: needs the validation run (strongest baseline) and at least one primary split run.", ""]
    else:
        L += [f"Strongest classical baseline on `{val_split}`: `{base}`. Family size {3 * len(prim)} "
              f"(3 comparisons x {len(prim)} primary splits).", ""]
        rows = [[v["name"], v.get("common"), v.get("trivial", "–"), _fmt(v.get("ratio"), ".4f"),
                 "–" if "ci" not in v else f"[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]",
                 _fmt(v.get("p_holm")), _fmt(v.get("coverage_gnn"), ".3f"), _fmt(v.get("coverage_other"), ".3f"),
                 v["verdict"]] for v in verdict_table(prim, base, n_boot)]
        L += [_md_table(["comparison", "common tasks", "trivial", "geo ratio", "95% CI", "Holm p", "cov GNN", "cov other",
                         "verdict"], rows), ""]
    sizes = size_curves({f"{k[0][0]}": v for k, v in keyed.items() if k[0][1] == PRIMARY_REGIME and k[1] == "fifo"})
    if plot_sizes(sizes, out_dir / "sizes_all.png", "expansions against size (ac, fifo)"):
        L += ["## Against size", "", "![sizes](sizes_all.png)", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(L) + "\n")
    (out_dir / "tables.json").write_text(json.dumps(
        {f"{k[0][0]}|{k[0][1]}|{k[1]}": {"table": split_table(v), "certification": certification(v)}
         for k, v in sorted(keyed.items())}, indent=1, default=str) + "\n")
    return path
