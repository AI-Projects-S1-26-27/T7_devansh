"""P6 development pilot (decisions.md §18 item 5, 17 Nov 2026): solvable rate of every cohort cell and the
generator's throughput, under the seed namespace "p6-pilot", which no split uses.

Usage: uv run python scripts/gen_d28_pilot.py [--config configs/gen/p6_pilot_v2.json] [--tasks N]

P8a (training-size pilot, §19 P7d): configs/gen/p8_pilot.json runs the training cell at every size n = 8..20 under
the seed namespace "p8-pilot" and proposes the training and the validation cap per size (quotas 231 and 31); the
config's "run_name" and "title" name its run folder (results/d28_gen/p8_pilot/) and report.

Configs: p6_pilot.json (P6c, CVE holdout v1; run 2026-11-17T073113Z-024d17d) and p6_pilot_v2.json (§18 P6d
item 4: CVE holdout v2). A config may name a "carried" earlier run and the cells whose pool the change left
untouched: those cells are not regenerated; their logged candidates are read from that run, after checking
that the run had the same generator inputs, tasks per cell and cell spec, and that each carried cell's units
are identical under the old and the new exclusions. Throughput counts the new candidates only.

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
CONFIG = ROOT / "configs" / "gen" / "p6_pilot_v2.json"
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


def summarise(rows: list[dict], cfg: dict, wall: float, workers: int, carried: list[dict] = (),
              previous: dict | None = None) -> dict:
    """Per-cell rates over the new rows plus the carried ones (marked with their source run), the proposals,
    and throughput over the new rows only. `previous`: an earlier run's cell summaries, shown beside each rate."""
    by = defaultdict(list)
    for r in [*rows, *carried]:
        by[(r["cell"], r["n"])].append(r)
    carried_keys = {(r["cell"], r["n"]) for r in carried}
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
            "actions_median": sorted(r["actions"] for r in rs if "actions" in r)[len(rs) // 2] if rs else None,
            "source": cfg["carried"]["run"] if (cell, n) in carried_keys else "this run",
            "previous_rate": ((previous or {}).get(f"{cell}|n={n}") or {}).get("rate")}
    def caps(spec: dict) -> dict:
        """Per size of the spec's cell: the cap at the spec's per-size shortfall probability, and (P8a) the cap
        at that probability divided by the number of sizes, so that by the union bound no size falls short with
        probability >= 1 - eps over all sizes together."""
        keys = [k for k, c in cells.items() if c["cell"] == spec["cell"]]
        out = {}
        for key in keys:
            c, q, eps = cells[key], spec["solvable_per_size"], spec["shortfall_probability"]
            out[key] = {"quota": q, "rate": c["rate"], "rate_low": c["wilson_low"],
                        "expected_candidates": math.ceil(q / c["rate"]) if c["rate"] else None,
                        "budget": budget(q, c["wilson_low"], eps),
                        "budget_all_sizes": budget(q, c["wilson_low"], eps / len(keys))}
        return out

    train = caps(cfg["training"])
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
    extra = {"validation_budget": caps(cfg["validation"])} if "validation" in cfg else {}
    return {"cells": cells, "training_budget": train, **extra, "test_sizes": tests,
            "throughput": {"tasks": len(rows), "workers": workers, "wall_seconds": round(wall, 1),
                           "tasks_per_second_wall": len(rows) / wall if wall else None,
                           "worker_seconds_total": round(sum(secs), 1),
                           "tasks_per_worker_second": len(rows) / sum(secs) if sum(secs) else None}}


def fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def report(raw: dict) -> str:
    s, cfg = raw["summary"], raw["config"]
    t = s["throughput"]
    L = [f"# {cfg.get('title', 'P6 development pilot')} (seed namespace {cfg['namespace']})", "",
         "Diagnostics on development seeds, not results. Every number below is read from raw.json.gz of this run.",
         f"Held-out products {cfg['heldout_products']}; unseen-CVE holdout {raw['holdout']['count']} IDs, "
         f"SHA-256 {raw['holdout']['sha256']}.", "",
         "## Solvable rate per cell", "",
         "| cell | n | tasks | solvable | rate | Wilson 95% | max half-width | infeasible | unsolvable reasons | s/task mean (max) | actions median | source | previous rate |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in s["cells"].values():
        L.append(f"| {c['cell']} | {c['n']} | {c['tasks']} | {c['solvable']} | {fmt(c['rate'])} | "
                 f"{fmt(c['wilson_low'])}-{fmt(c['wilson_high'])} | {fmt(c['half_width_max'], 4)} | {c['infeasible']} | "
                 f"{c['reasons']} | {fmt(c['seconds_mean'], 4)} ({fmt(c['seconds_max'], 3)}) | {c['actions_median']} | "
                 f"{c.get('source', 'this run')} | {fmt(c.get('previous_rate'))} |")
    if raw.get("carried"):
        cr = raw["carried"]
        L += ["", f"Carried cells {cr['cells']} from run {cr['run']} (not regenerated; checks: {cr['checks']}). "
              f"Previous rates are from that run."]
    L += ["", "Cells: " + "; ".join(f"{c['name']} = {', '.join(c['cohorts'])} ({c['archetype']}, {c['cohort']}, "
                                   f"Linux {c['os_linux']}, p {c['p']}, products {c['products']}, CVEs {c['cves']}"
                                   + (f"; {c['variant']}" if c.get('variant') else "") + ")" for c in cfg["cells"]), "",
          "## Throughput", "",
          f"- {t['tasks']} candidates in {t['wall_seconds']} s wall on {t['workers']} worker processes: "
          f"{fmt(t['tasks_per_second_wall'], 1)} tasks/s wall; {fmt(t['tasks_per_worker_second'], 1)} tasks per worker-second "
          f"({t['worker_seconds_total']} worker-seconds, generation only).", "",
          ]
    for role, key in (("training", "training_budget"), ("validation", "validation_budget")):
        if key not in s:
            continue
        L += [f"## Proposal: {role} candidate budget (rejection sampling in seed order, every candidate logged)", "",
              f"Rule: {cfg[role]['rule']}.", "",
              "| cell | quota | rate | Wilson low | expected candidates (quota / rate) | proposed cap | cap, all sizes together |",
              "|---|---|---|---|---|---|---|"]
        for k, v in s[key].items():
            L.append(f"| {k} | {v['quota']} | {fmt(v['rate'])} | {fmt(v['rate_low'])} | {v['expected_candidates']} | "
                     f"{v['budget']} | {v['budget_all_sizes']} |")
        tot = [sum(v[f] for v in s[key].values()) for f in ("quota", "expected_candidates", "budget", "budget_all_sizes")]
        L += [f"| total | {tot[0]} | | | {tot[1]} | {tot[2]} | {tot[3]} |", ""]
    if not s["test_sizes"]:
        return "\n".join(L) + "\n"
    L += ["## Proposal: fixed test cohort sizes (about 200 expected solvable per size)", "",
          f"Rule: {cfg['test_rule']}.", "",
          "| cohort | cell | n | rate | fixed size | expected solvable (Wilson range) |", "|---|---|---|---|---|---|"]
    for v in s["test_sizes"].values():
        L.append(f"| {v['cohort']} | {v['cell']} | {v['n']} | {fmt(v['rate'])} | {v['fixed_size']} | "
                 f"{fmt(v['expected_solvable'], 1)} ({fmt(v['expected_solvable_low'], 1)}-{fmt(v['expected_solvable_high'], 1)}) |")
    return "\n".join(L) + "\n"


def jobs(cfg: dict, tasks: int, holdout: list[str], training_cves: list[str]) -> list[tuple]:
    out = []
    skip = set(cfg.get("carried", {}).get("cells", ()))
    for cell in cfg["cells"]:
        if cell["name"] in skip:  # read from the carried run instead (load_carried)
            continue
        prods, cves = exclusions(cell, cfg, holdout, training_cves)
        for n in cell["sizes"]:
            for k in range(tasks):  # seed order
                out.append((cell["name"], cell["archetype"], n, k, seed_of(cfg["namespace"], cell["name"], n, k),
                            cell["cohort"], cell["os_linux"], cell["p"], prods, cves))
    return out


def holdout_sets(cfg: dict, ctx, root: Path = ROOT) -> tuple[list[str], list[str]]:
    """(the config's CVE holdout, the training CVEs = training pool minus it), the holdout checked against its SHA-256."""
    from ahg.gen import pool

    ho = json.loads((root / cfg["cve_holdout"]).read_text())
    if pool.cve_set_sha256(ho["cves"]) != ho["sha256"]:
        raise SystemExit(f"{cfg['cve_holdout']}: IDs do not match their SHA-256")
    pool_ids = {u.cve_id for u in ctx.run_units("core", frozenset(cfg["heldout_products"]))}
    return ho["cves"], sorted(pool_ids - set(ho["cves"]), key=pool.cve_key)


def load_carried(cfg: dict, ctx, tasks: int, holdout: list[str], training_cves: list[str],
                 root: Path = ROOT) -> tuple[list[dict], dict, dict]:
    """Candidates of the carried cells, read from the earlier run (§18 P6d item 4: only cells whose pool changed
    are regenerated). Refuses unless that run had the same generator inputs, tasks per cell and cell spec, its
    own holdout file still hashes as it recorded, and each carried cell's units are identical under the old and
    the new exclusions (so its candidates would be generated again unchanged).
    Returns (rows, that run's cell summaries, the checks made)."""
    import gzip

    from ahg.gen import pool

    spec = cfg.get("carried")
    if not spec:
        return [], {}, {}
    old = json.loads(gzip.decompress((root / spec["run"] / "raw.json.gz").read_bytes()))
    if old["inputs"] != ctx.input_hashes:
        raise SystemExit(f"carried run {spec['run']}: generator inputs differ")
    if old["tasks_per_cell"] != tasks:
        raise SystemExit(f"carried run {spec['run']}: {old['tasks_per_cell']} tasks per cell, not {tasks}")
    ocfg = old["config"]
    old_ho, old_tr = holdout_sets(ocfg, ctx, root)
    if pool.cve_set_sha256(old_ho) != old["holdout"]["sha256"] or pool.cve_set_sha256(old_tr) != old["training_cves"]["sha256"]:
        raise SystemExit(f"carried run {spec['run']}: its holdout no longer hashes as recorded")
    new_cells, old_cells = {c["name"]: c for c in cfg["cells"]}, {c["name"]: c for c in ocfg["cells"]}
    rows = []
    for name in spec["cells"]:
        cell = new_cells[name]
        if old_cells.get(name) != cell:
            raise SystemExit(f"carried cell {name}: spec differs from run {spec['run']}")
        before = ctx.run_units(cell["cohort"], *exclusions(cell, ocfg, old_ho, old_tr))
        after = ctx.run_units(cell["cohort"], *exclusions(cell, cfg, holdout, training_cves))
        if before != after:
            raise SystemExit(f"carried cell {name}: its pool changed; it must be regenerated")
        rs = [r for r in old["candidates"] if r["cell"] == name]
        if len(rs) != tasks * len(cell["sizes"]):
            raise SystemExit(f"carried cell {name}: {len(rs)} candidates in run {spec['run']}")
        rows += rs
    checks = {"same_inputs": True, "same_tasks_per_cell": tasks, "same_cell_specs": True,
              "same_units_under_old_and_new_exclusions": True, "old_holdout_sha256": old["holdout"]["sha256"]}
    return rows, old["summary"]["cells"], checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--tasks", type=int, default=None, help="tasks per cell (default: the config's)")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    tasks = a.tasks or cfg["tasks_per_cell"]
    from ahg.gen import generate as gen
    from ahg.gen import pool

    ctx = gen.Context.load()
    holdout, training_cves = holdout_sets(cfg, ctx)
    carried, previous, checks = load_carried(cfg, ctx, tasks, holdout, training_cves)
    js = jobs(cfg, tasks, holdout, training_cves)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init) as ex:
        rows = list(ex.map(_run, js, chunksize=64))
    wall = time.perf_counter() - t0
    raw = {"config": cfg, "tasks_per_cell": tasks, "inputs": ctx.input_hashes,
           "holdout": {"count": len(holdout), "sha256": pool.cve_set_sha256(holdout), "path": cfg["cve_holdout"]},
           "training_cves": {"count": len(training_cves), "sha256": pool.cve_set_sha256(training_cves)},
           "carried": cfg.get("carried", {}) | {"checks": checks} if carried else None,
           "candidates": rows, "summary": summarise(rows, cfg, wall, cfg["workers"], carried, previous)}
    seeds = [cfg["namespace"]]
    run = runlog.write_run(cfg.get("run_name", "p6_pilot"), raw, report(raw), seeds, cfg, root=OUT)
    print(run)


if __name__ == "__main__":
    main()
