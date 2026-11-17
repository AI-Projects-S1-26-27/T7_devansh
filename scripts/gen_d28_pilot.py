"""P6 development pilot (decisions.md §18 item 5, 17 Nov 2026): solvable rate of every cohort cell and the
generator's throughput, under the seed namespace "p6-pilot", which no split uses.

Usage: uv run python scripts/gen_d28_pilot.py [--config configs/gen/p6_pilot.json] [--tasks N]

Every candidate is generated in seed order (k = 0, 1, ...) and logged: seed, cell, size, solvability and
reason (or the infeasibility), generation seconds, actions, document digest. Cells and their exclusions
(held-out products, the unseen-CVE holdout) come from the pilot config. With 2,401 tasks per cell every
95% Wilson interval is at most +-2 percentage points wide.

From these rates alone the report proposes (Devansh decides; nothing here is frozen):
- the training candidate budget per size: the smallest N with P(Binomial(N, r_low) >= quota) >= 1 - eps,
  r_low the cell's Wilson 95% lower bound (rejection sampling in seed order stops at the quota, so the
  budget is a cap, not the number generated);
- each test cohort's fixed size per size: ceil(200 / r).

Writes one runlog run under results/d28_gen/p6_pilot/<UTC>-<commit>/ (untracked). Generation runs in
worker processes; throughput is reported per task (worker seconds) and as wall-clock tasks per second.
Diagnostics on development seeds, not results.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ahg import runlog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d28_gen"
CONFIG = ROOT / "configs" / "gen" / "p6_pilot.json"
Z95 = 1.959963984540054


def seed_of(namespace: str, cell: str, n: int, k: int) -> str:
    return f"{namespace}/{cell}/n{n}/{k}"


def wilson(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for x successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # at x = 0 (x = n) the lower (upper) bound is exactly 0 (1); rounding would leave ~1e-18 instead
    return (0.0 if x == 0 else max(0.0, c - h)), (1.0 if x == n else min(1.0, c + h))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) >= k), exact, summed in log space (pure Python; no SciPy in the lock file)."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    lc = math.lgamma(n + 1)
    terms = [lc - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def budget(quota: int, rate: float, eps: float) -> int | None:
    """Smallest N with P(Binomial(N, rate) >= quota) >= 1 - eps (None if the rate is 0)."""
    if rate <= 0:
        return None
    lo, hi = quota, max(quota, math.ceil(quota / rate))
    while binom_sf(quota, hi, rate) < 1 - eps:
        hi *= 2
    while lo < hi:  # the tail grows with N, so bisect
        mid = (lo + hi) // 2
        if binom_sf(quota, mid, rate) >= 1 - eps:
            hi = mid
        else:
            lo = mid + 1
    return lo


def exclusions(cell: dict, cfg: dict, holdout: list[str], training_cves: list[str]) -> tuple[frozenset, frozenset]:
    prods = {"heldout": frozenset(cfg["heldout_products"]), "none": frozenset()}[cell["products"]]
    cves = {"holdout": frozenset(holdout), "none": frozenset(), "training": frozenset(training_cves)}[cell["cves"]]
    return prods, cves


_CTX = None


def _init() -> None:
    global _CTX
    from ahg.gen import generate as gen

    _CTX = gen.Context.load()


def _run(job: tuple) -> dict:
    """One candidate in a worker. Seconds cover generation only (the rate is the pilot's object)."""
    from ahg.gen import generate as gen
    from ahg.gen.hosts import InfeasibleCell

    cell, arch, n, k, seed, cohort, linux, p, prods, cves = job
    row = {"cell": cell, "n": n, "k": k, "seed": seed}
    t0 = time.perf_counter()
    try:
        doc = gen.generate(_CTX, arch, seed, n, cohort=cohort, os_mix={"linux": linux, "windows": 1 - linux}, p=p,
                           excluded=prods, excluded_cves=cves)
    except InfeasibleCell as e:
        return row | {"infeasible": str(e), "seconds": round(time.perf_counter() - t0, 4)}
    sec = time.perf_counter() - t0
    return row | {"solvable": doc["solvable"], "reason": doc["unsolvable_reason"], "seconds": round(sec, 4),
                  "actions": doc["stats"]["actions"], "digest": gen.doc_digest(doc)}


def summarise(rows: list[dict], cfg: dict, wall: float, workers: int) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["n"])].append(r)
    cells = {}
    for (cell, n), rs in sorted(by.items()):
        x = sum(r.get("solvable", False) for r in rs)
        lo, hi = wilson(x, len(rs))
        secs = [r["seconds"] for r in rs]
        cells[f"{cell}|n={n}"] = {
            "cell": cell, "n": n, "tasks": len(rs), "solvable": x, "rate": x / len(rs), "wilson_low": lo, "wilson_high": hi,
            "half_width_max": max(x / len(rs) - lo, hi - x / len(rs)),
            "infeasible": sum("infeasible" in r for r in rs),
            "reasons": dict(Counter(r["reason"] for r in rs if r.get("solvable") is False)),
            "seconds_mean": sum(secs) / len(secs), "seconds_max": max(secs),
            "actions_median": sorted(r["actions"] for r in rs if "actions" in r)[len(rs) // 2] if rs else None}
    tr = cfg["training"]
    train = {}
    for key, c in cells.items():
        if c["cell"] == tr["cell"]:
            train[key] = {"quota": tr["solvable_per_size"], "rate": c["rate"], "rate_low": c["wilson_low"],
                          "expected_candidates": math.ceil(tr["solvable_per_size"] / c["rate"]) if c["rate"] else None,
                          "budget": budget(tr["solvable_per_size"], c["wilson_low"], tr["shortfall_probability"])}
    want = cfg["test_expected_solvable"]
    tests = {}
    for spec in cfg["cells"]:
        for co in spec["cohorts"]:
            if not co.startswith("d28_test_"):
                continue
            for n in spec["sizes"]:
                c = cells[f"{spec['name']}|n={n}"]
                size = math.ceil(want / c["rate"]) if c["rate"] else None
                tests[f"{co}|{spec['name']}|n={n}"] = {
                    "cohort": co, "cell": spec["name"], "n": n, "rate": c["rate"], "fixed_size": size,
                    "expected_solvable": size * c["rate"] if size else 0,
                    "expected_solvable_low": size * c["wilson_low"] if size else 0,
                    "expected_solvable_high": size * c["wilson_high"] if size else 0}
    secs = [r["seconds"] for r in rows]
    return {"cells": cells, "training_budget": train, "test_sizes": tests,
            "throughput": {"tasks": len(rows), "workers": workers, "wall_seconds": round(wall, 1),
                           "tasks_per_second_wall": len(rows) / wall if wall else None,
                           "worker_seconds_total": round(sum(secs), 1),
                           "tasks_per_worker_second": len(rows) / sum(secs) if sum(secs) else None}}


def fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def report(raw: dict) -> str:
    s, cfg = raw["summary"], raw["config"]
    t = s["throughput"]
    L = ["# P6 development pilot (seed namespace p6-pilot)", "",
         "Diagnostics on development seeds, not results. Every number below is read from raw.json.gz of this run.",
         f"Held-out products {cfg['heldout_products']}; unseen-CVE holdout {raw['holdout']['count']} IDs, "
         f"SHA-256 {raw['holdout']['sha256']}.", "",
         "## Solvable rate per cell", "",
         "| cell | n | tasks | solvable | rate | Wilson 95% | max half-width | infeasible | unsolvable reasons | s/task mean (max) | actions median |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in s["cells"].values():
        L.append(f"| {c['cell']} | {c['n']} | {c['tasks']} | {c['solvable']} | {fmt(c['rate'])} | "
                 f"{fmt(c['wilson_low'])}-{fmt(c['wilson_high'])} | {fmt(c['half_width_max'], 4)} | {c['infeasible']} | "
                 f"{c['reasons']} | {fmt(c['seconds_mean'], 4)} ({fmt(c['seconds_max'], 3)}) | {c['actions_median']} |")
    L += ["", "Cells: " + "; ".join(f"{c['name']} = {', '.join(c['cohorts'])} ({c['archetype']}, {c['cohort']}, "
                                   f"Linux {c['os_linux']}, p {c['p']}, products {c['products']}, CVEs {c['cves']}"
                                   + (f"; {c['variant']}" if c.get('variant') else "") + ")" for c in cfg["cells"]), "",
          "## Throughput", "",
          f"- {t['tasks']} candidates in {t['wall_seconds']} s wall on {t['workers']} worker processes: "
          f"{fmt(t['tasks_per_second_wall'], 1)} tasks/s wall; {fmt(t['tasks_per_worker_second'], 1)} tasks per worker-second "
          f"({t['worker_seconds_total']} worker-seconds, generation only).", "",
          "## Proposal: training candidate budget (rejection sampling in seed order, every candidate logged)", "",
          f"Rule: {cfg['training']['rule']}.", "",
          "| cell | quota | rate | Wilson low | expected candidates (quota / rate) | proposed cap |", "|---|---|---|---|---|---|"]
    for k, v in s["training_budget"].items():
        L.append(f"| {k} | {v['quota']} | {fmt(v['rate'])} | {fmt(v['rate_low'])} | {v['expected_candidates']} | {v['budget']} |")
    L += ["", "## Proposal: fixed test cohort sizes (about 200 expected solvable per size)", "",
          f"Rule: {cfg['test_rule']}.", "",
          "| cohort | cell | n | rate | fixed size | expected solvable (Wilson range) |", "|---|---|---|---|---|---|"]
    for v in s["test_sizes"].values():
        L.append(f"| {v['cohort']} | {v['cell']} | {v['n']} | {fmt(v['rate'])} | {v['fixed_size']} | "
                 f"{fmt(v['expected_solvable'], 1)} ({fmt(v['expected_solvable_low'], 1)}-{fmt(v['expected_solvable_high'], 1)}) |")
    return "\n".join(L) + "\n"


def jobs(cfg: dict, tasks: int, holdout: list[str], training_cves: list[str]) -> list[tuple]:
    out = []
    for cell in cfg["cells"]:
        prods, cves = exclusions(cell, cfg, holdout, training_cves)
        for n in cell["sizes"]:
            for k in range(tasks):  # seed order
                out.append((cell["name"], cell["archetype"], n, k, seed_of(cfg["namespace"], cell["name"], n, k),
                            cell["cohort"], cell["os_linux"], cell["p"], prods, cves))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--tasks", type=int, default=None, help="tasks per cell (default: the config's)")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    tasks = a.tasks or cfg["tasks_per_cell"]
    ho = json.loads((ROOT / cfg["cve_holdout"]).read_text())
    from ahg.gen import generate as gen
    from ahg.gen import pool

    ctx = gen.Context.load()
    if pool.cve_set_sha256(ho["cves"]) != ho["sha256"]:
        raise SystemExit("cve_holdout.json: IDs do not match their SHA-256")
    pool_ids = {u.cve_id for u in ctx.run_units("core", frozenset(cfg["heldout_products"]))}
    training_cves = sorted(pool_ids - set(ho["cves"]), key=pool.cve_key)
    js = jobs(cfg, tasks, ho["cves"], training_cves)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init) as ex:
        rows = list(ex.map(_run, js, chunksize=64))
    wall = time.perf_counter() - t0
    raw = {"config": cfg, "tasks_per_cell": tasks, "inputs": ctx.input_hashes,
           "holdout": {"count": len(ho["cves"]), "sha256": ho["sha256"]},
           "training_cves": {"count": len(training_cves), "sha256": pool.cve_set_sha256(training_cves)},
           "candidates": rows, "summary": summarise(rows, cfg, wall, cfg["workers"])}
    seeds = [cfg["namespace"]]
    run = runlog.write_run("p6_pilot", raw, report(raw), seeds, cfg, root=OUT)
    print(run)


if __name__ == "__main__":
    main()
