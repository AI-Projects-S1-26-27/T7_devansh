"""D28 experiment pipeline CLI (P7). Nothing here runs on a real cohort until the registry is frozen
(configs/d28_cohorts.json) and configs/d28_prereg.json is committed; --smoke uses the p7-smoke cohorts.

  build     <cohort> [--smoke] [--n4-pass 12,20]   generate a cohort into the task store; writes its record
  label     <cohort> [--smoke] --state-budget B [--shard k/K]   exact labels, both regimes, resumable
  n4        [--smoke]                               N4 pilot (training seeds only)
  landscape [--smoke]                               landscape pilot (training tasks, certified ac labels)
  budget    [--smoke]                               budget feasibility pilot; proposes the frozen budget
  ac-counts                                         eligible AC:L / AC:H counts per cohort (environment check)
  report    --eval-root R --out O --primary a,b --val-split v   regenerate every table and figure

Cohort records live in data/d28/cohorts/<name>.json (untracked); runs go through runlog under results/d28_pipe/.
Test cohorts: build and label first log a "label generation" read in results/test_access_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ahg import cohorts as co, d28labels, n4, pilots, runlog  # noqa: E402
from ahg.splits import LABEL_GENERATION  # noqa: E402

RUNS = ROOT / "results" / "d28_pipe"
RECORDS = ROOT / "data" / "d28" / "cohorts"


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def record_path(c: co.Cohort, root: Path = RECORDS) -> Path:
    return root / f"{c.name}.json"


def load_record(c: co.Cohort, root: Path = RECORDS) -> dict:
    p = record_path(c, root)
    if not p.exists():
        raise FileNotFoundError(f"{p}: build the cohort first")
    return json.loads(p.read_text())


def require_buildable(reg, c: co.Cohort) -> None:
    """Refuse before anything is read or logged (a refused cohort never writes a test-access entry)."""
    why = reg.problems(c)
    if why:
        raise co.BlockedCohort(f"{c.name}: " + "; ".join(why))


def access_for(reg, c: co.Cohort, store_log=None) -> dict | None:
    """A logged label-generation token for a test cohort; None for the others."""
    if not c.is_test:
        return None
    kw = {} if store_log is None else {"log": store_log}
    return co.record_test_read(reg, c.name, LABEL_GENERATION, commit(), dataset_sha256=None, **kw)


def cmd_build(a, reg, ctx) -> None:
    c = reg[a.cohort]
    require_buildable(reg, c)
    access = access_for(reg, c)
    n4_pass = None if a.n4_pass is None else {int(x) for x in a.n4_pass.split(",") if x}
    rec = co.build(ctx, reg, c.name, access=access, n4_pass=n4_pass)
    RECORDS.mkdir(parents=True, exist_ok=True)
    record_path(c).write_text(json.dumps(rec, sort_keys=True) + "\n")
    print(c.name, rec["by_size"], "shortfall", rec["shortfall"], "dataset", rec["dataset_sha256"][:12])


def cmd_label(a, reg, ctx) -> None:
    c = reg[a.cohort]
    require_buildable(reg, c)
    rec = load_record(c)
    cfg = d28labels.with_budget(d28labels.load_config(), a.state_budget)
    d28labels.check_config(cfg, max_n=max(c.sizes))
    access = access_for(reg, c)
    docs = [d for _, d in co.open_members(rec, reg, access=access) if d is not None]
    shard = None if a.shard is None else tuple(map(int, a.shard.split("/")))
    summary = d28labels.label_docs(docs, cfg, shard=shard, commit=commit())
    name = f"label_{c.name}" + (f"_s{shard[0]}of{shard[1]}" if shard else "")
    out = runlog.write_run(name, summary, f"# labels {c.name}\n\n```\n{json.dumps(summary['counts'], indent=1)}\n```\n",
                           [], {"cohort": c.name, "state_budget": cfg["state_budget"], "shard": shard}, root=RUNS)
    print(out, summary["counts"])


def _training_docs(reg, ctx, smoke: bool) -> tuple[co.Cohort, dict, list[dict]]:
    c = next(x for x in reg.cohorts.values() if x.role == "training" and x.smoke == smoke)
    path = record_path(c)
    rec = json.loads(path.read_text()) if path.exists() else co.build(ctx, reg, c.name)
    return c, rec, [d for _, d in co.open_members(rec, reg) if d is not None]


def cmd_n4(a, reg, ctx) -> None:
    import gzip  # noqa: F401
    cfg = json.loads(co.REGISTRY.read_text())["n4"]
    per = cfg["smoke_per_family_size"] if a.smoke else cfg["per_family_size"]
    why = [] if a.smoke else reg.problems(reg.training)
    if why:
        raise co.BlockedCohort("n4: " + "; ".join(why))
    from ahg.gen import generate as gen
    rows, disconnected = [], {}
    for n in cfg["sizes"]:
        for arch in cfg["archetypes"]:
            for k in range(per):
                doc = gen.generate(ctx, arch, f"{cfg['smoke_namespace' if a.smoke else 'namespace']}/n{n}/{k}", n,
                                   cohort="core", os_mix=reg.training.os_mix, p=1.0, excluded=reg.excluded(reg.training)
                                   if not a.smoke else frozenset())
                try:
                    rows.append({"archetype": arch, "n": n, "seed": doc["params"]["seed"], **n4.doc_stats(doc)})
                except n4.DisconnectedNetwork:
                    # OQ-P7-6: the generator's random firewall can cut hosts off from the internet; Round 6 says
                    # such a network "raises an error". The pilot counts and leaves it out instead of aborting.
                    disconnected[f"{arch}|{n}"] = disconnected.get(f"{arch}|{n}", 0) + 1
    res = n4.classify(rows)
    md = "# N4\n\n" + f"Reach-disconnected networks left out (OQ-P7-6): {disconnected}\n\n" + "\n".join(f"- n={n}: {v['verdict']} (mean balanced accuracy {v['mean_balanced_accuracy']})"
                                  for n, v in res.items()) + "\n"
    print(runlog.write_run("n4", {"results": res, "rows": rows, "disconnected": disconnected}, md, [], {"per_family_size": per, "smoke": a.smoke}, root=RUNS))


def cmd_landscape(a, reg, ctx) -> None:
    c, rec, docs = _training_docs(reg, ctx, a.smoke)
    cfg = d28labels.with_budget(d28labels.load_config(), a.state_budget)
    per = json.loads(co.REGISTRY.read_text())["landscape"]["smoke_per_stratum" if a.smoke else "per_stratum"]
    sample, seen = [], {}
    for d in docs:
        key = (d["params"]["archetype"], d["params"]["n"])
        if seen.get(key, 0) < per and d["solvable"]:
            seen[key] = seen.get(key, 0) + 1
            sample.append(d)
    d28labels.label_docs(sample, {**cfg, "regimes": ["ac"]}, commit=commit())
    labels = d28labels.load_labels(sample, {**cfg, "regimes": ["ac"]})
    from ahg.gen.generate import doc_digest
    rows = [pilots.landscape_task(d, labels[(doc_digest(d), "ac")], a.search_budget) for d in sample
            if labels[(doc_digest(d), "ac")]["certified"]]
    summ = pilots.landscape_summary(rows)
    md = "# Landscape pilot (ac)\n\n```\n" + json.dumps(summ, indent=1, default=str) + "\n```\n"
    print(runlog.write_run("landscape", {"summary": summ, "rows": rows}, md, [], {"per_stratum": per, "smoke": a.smoke}, root=RUNS))


def cmd_budget(a, reg, ctx) -> None:
    c, rec, docs = _training_docs(reg, ctx, a.smoke)
    base = json.loads(co.REGISTRY.read_text())["budget_pilot"]
    cfg = d28labels.load_config()
    lcfg = {k: v for k, v in cfg.items() if k not in ("regimes", "state_budget")}
    sample = [d for d in docs if d["solvable"]][: base["smoke_tasks" if a.smoke else "tasks"]]
    rows = [pilots.budget_task(d, r, base["cap"], lcfg) for d in sample for r in cfg["regimes"]]
    summ = pilots.budget_summary(rows, base["candidates"], base["cap"], base["target"])
    md = f"# Budget feasibility pilot\n\nProposed per-state budget: {summ['proposed_state_budget']}\n"
    print(runlog.write_run("budget", {"summary": summ, "rows": rows}, md, [], {**base, "smoke": a.smoke}, root=RUNS))


def cmd_ac_counts(a, reg, ctx) -> None:
    from collections import Counter
    cnt = Counter((u.cohort, u.os, u.kind, u.ac) for u in ctx.units)
    print(json.dumps({"|".join(k): v for k, v in sorted(cnt.items())}, indent=1))


def cmd_report(a, reg, ctx) -> None:
    from ahg import report
    print(report.build_report(Path(a.eval_root), Path(a.out), primary=set(a.primary.split(",")),
                              val_split=a.val_split, n_boot=a.boot))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "label"):
        p = sub.add_parser(name)
        p.add_argument("cohort")
        p.add_argument("--smoke", action="store_true")
        if name == "build":
            p.add_argument("--n4-pass")
        else:
            p.add_argument("--state-budget", type=int)
            p.add_argument("--shard")
    for name in ("n4", "landscape", "budget"):
        p = sub.add_parser(name)
        p.add_argument("--smoke", action="store_true")
        if name == "landscape":
            p.add_argument("--state-budget", type=int)
            p.add_argument("--search-budget", type=int, default=200_000)
    sub.add_parser("ac-counts")
    p = sub.add_parser("report")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--primary", required=True)
    p.add_argument("--val-split")
    p.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    reg = co.load_registry()
    ctx = None
    if a.cmd != "report":
        from ahg.gen.generate import Context
        ctx = Context.load()
    {"build": cmd_build, "label": cmd_label, "n4": cmd_n4, "landscape": cmd_landscape, "budget": cmd_budget,
     "ac-counts": cmd_ac_counts, "report": cmd_report}[a.cmd](a, reg, ctx)


if __name__ == "__main__":
    main()
