"""D28 experiment pipeline CLI (P7). Nothing here runs on a real cohort until the registry is frozen
(configs/d28_cohorts.json) and configs/d28_prereg.json is committed; --smoke uses the p7-smoke cohorts.

  build     <cohort> [--smoke] [--n4-pass 12,20]   generate a cohort into the task store; writes its record
  label     <cohort> [--smoke] --state-budget B [--shard k/K]   exact labels, both regimes, resumable
  n4        [--smoke]                               N4 pilot (training seeds only)
  landscape [--smoke]                               landscape pilot (training tasks, certified ac labels)
  budget    [--smoke]                               budget feasibility pilot; proposes the frozen budget
  ac-counts                                         eligible AC:L / AC:H counts per cohort (environment check)
  evaluate  <cohort> --state-budget B --search-budget S --checkpoint P [...] [--eval-root R] [--permitted-by D]
                                                    every baseline and model, both regimes and tie rules, FD
  report    --eval-root R --out O --primary a,b --val-split v   regenerate every table and figure

Cohort records live in data/d28/cohorts/<name>.json (untracked); runs go through runlog under results/d28_pipe/.
Test cohorts: build and label first log a "label generation" read in results/test_access_log.jsonl;
evaluate logs the single "evaluation" read (Round 10) after every check that needs no test data has passed.
Evaluation runs go to results/d28_eval/<cohort>.<regime>.<tie>/ (ahg.d28eval), which `report` reads.
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
from ahg.splits import EVALUATION, LABEL_GENERATION  # noqa: E402

RUNS = ROOT / "results" / "d28_pipe"
RECORDS = ROOT / "data" / "d28" / "cohorts"


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def record_path(c: co.Cohort, root: Path | None = None) -> Path:
    return (RECORDS if root is None else root) / f"{c.name}.json"  # RECORDS looked up now (OQ-P7b-4)


def load_record(c: co.Cohort, root: Path | None = None) -> dict:
    p = record_path(c, root)
    if not p.exists():
        raise FileNotFoundError(f"{p}: build the cohort first")
    return json.loads(p.read_text())


def require_buildable(reg, c: co.Cohort) -> None:
    """Refuse before anything is read or logged (a refused cohort never writes a test-access entry)."""
    why = reg.problems(c)
    if why:
        raise co.BlockedCohort(f"{c.name}: " + "; ".join(why))


def access_for(reg, c: co.Cohort) -> dict | None:
    """A logged label-generation token for a test cohort; None for the others."""
    if not c.is_test:
        return None
    return co.record_test_read(reg, c.name, LABEL_GENERATION, commit(), dataset_sha256=None, log=co.TEST_ACCESS_LOG)


# The task store, label root and test-access log are looked up when a command runs, never bound when a module is
# imported (a default argument is bound then), so a redirect (a test, a scratch run) reaches every command
# (OQ-P7b-4, P7c).


def cmd_build(a, reg, ctx) -> None:
    c = reg[a.cohort]
    require_buildable(reg, c)
    access = access_for(reg, c)
    n4_pass = None if a.n4_pass is None else {int(x) for x in a.n4_pass.split(",") if x}
    rec = co.build(ctx, reg, c.name, store=co.TASK_STORE, access=access, n4_pass=n4_pass, log=co.TEST_ACCESS_LOG)
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
    docs = [d for _, d in co.open_members(rec, reg, store=co.TASK_STORE, access=access, log=co.TEST_ACCESS_LOG)
            if d is not None]
    shard = None if a.shard is None else tuple(map(int, a.shard.split("/")))
    summary = d28labels.label_docs(docs, cfg, root=d28labels.LABEL_ROOT, shard=shard, commit=commit())
    name = f"label_{c.name}" + (f"_s{shard[0]}of{shard[1]}" if shard else "")
    out = runlog.write_run(name, summary, f"# labels {c.name}\n\n```\n{json.dumps(summary['counts'], indent=1)}\n```\n",
                           [], {"cohort": c.name, "state_budget": cfg["state_budget"], "shard": shard}, root=RUNS)
    print(out, summary["counts"])


def _training_docs(reg, ctx, smoke: bool) -> tuple[co.Cohort, dict, list[dict]]:
    c = next(x for x in reg.cohorts.values() if x.role == "training" and x.smoke == smoke)
    path = record_path(c)
    rec = json.loads(path.read_text()) if path.exists() else co.build(ctx, reg, c.name, store=co.TASK_STORE)
    return c, rec, [d for _, d in co.open_members(rec, reg, store=co.TASK_STORE) if d is not None]


def cmd_n4(a, reg, ctx) -> None:
    import gzip  # noqa: F401
    cfg = json.loads(co.REGISTRY.read_text())["n4"]
    per = cfg["smoke_per_family_size"] if a.smoke else cfg["per_family_size"]
    # P8 (DECIDED-BY-AGENT, OQ-P8-2): N4 generates no labels, so neither the registry freeze nor the hypotheses
    # freeze (§13.1: before any label generation) gates it (Round 6: measured before training). It needs only
    # training's exclusions; reg.excluded / reg.excluded_cves raise BlockedCohort while either set is unknown.
    from ahg.gen import generate as gen
    rows, unreachable = [], {}
    for n in cfg["sizes"]:
        for arch in cfg["archetypes"]:
            for k in range(per):
                # Training's exclusions (held-out products and, since M4, the held-out CVEs) unless smoke.
                doc = gen.generate(ctx, arch, f"{cfg['smoke_namespace' if a.smoke else 'namespace']}/n{n}/{k}", n,
                                   cohort="core", os_mix=reg.training.os_mix, p=1.0,
                                   excluded=frozenset() if a.smoke else reg.excluded(reg.training),
                                   excluded_cves=frozenset() if a.smoke else reg.excluded_cves(reg.training))
                rows.append({"archetype": arch, "n": n, "seed": doc["params"]["seed"], **n4.doc_stats(doc)})
                # OQ-P7-6 (decided 17 Nov 2026): networks with hosts unreachable from the internet stay in; counted.
                if rows[-1]["unreachable_frac"] > 0:
                    unreachable[f"{arch}|{n}"] = unreachable.get(f"{arch}|{n}", 0) + 1
    res = n4.classify(rows)
    # Per family x size: each statistic's mean, median and IQR (inclusive quartiles, OQ-P7-9), descriptive.
    summary = {}
    for key in sorted({f"{r['archetype']}|{r['n']}" for r in rows}):
        rs = [r for r in rows if f"{r['archetype']}|{r['n']}" == key]
        summary[key] = {f: {"mean": sum(r[f] for r in rs) / len(rs), **pilots.quantiles([r[f] for r in rs])}
                        for f in n4.FEATURES}
    md = "# N4\n\n" + f"Networks with hosts unreachable from the internet (kept, OQ-P7-6): {unreachable}\n\n" + "\n".join(f"- n={n}: {v['verdict']} (mean balanced accuracy {v['mean_balanced_accuracy']}; confusion {v.get('confusion')})"
                                  for n, v in res.items()) + "\n"
    print(runlog.write_run("n4", {"results": res, "rows": rows, "with_unreachable_hosts": unreachable,
                                  "feature_summary": summary}, md, [cfg["smoke_namespace" if a.smoke else "namespace"]],
                           {"per_family_size": per, "smoke": a.smoke}, root=RUNS))


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
    d28labels.label_docs(sample, {**cfg, "regimes": ["ac"]}, root=d28labels.LABEL_ROOT, commit=commit())
    labels = d28labels.load_labels(sample, {**cfg, "regimes": ["ac"]}, root=d28labels.LABEL_ROOT)
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


def cmd_evaluate(a, reg, ctx) -> None:
    from ahg import d28eval
    c = reg[a.cohort]
    require_buildable(reg, c)
    rec = load_record(c)
    cfg = d28eval.with_budget(d28eval.load_config(), a.search_budget)
    cfg |= {k: getattr(a, k) for k in ("repeats", "warmup") if getattr(a, k) is not None}
    lcfg = d28labels.with_budget(d28labels.load_config(), a.state_budget)
    ckpts = d28eval.load_checkpoints(a.checkpoint)
    # Looked up at call time (not import time), so tests can point them at a temporary directory.
    labels_root, store, log = d28labels.LABEL_ROOT, co.TASK_STORE, co.TEST_ACCESS_LOG
    d28eval.preflight(c, rec, cfg, ckpts, lcfg, labels_root)  # before a test read is logged
    access = None
    if c.is_test:
        access = co.record_test_read(reg, c.name, EVALUATION, commit(), dataset_sha256=rec["dataset_sha256"],
                                     eval_config_sha256=d28eval.config_sha(cfg),
                                     checkpoint_sha256=d28eval.checkpoint_hashes(ckpts),
                                     permitted_by=a.permitted_by, log=log)
    raws = d28eval.evaluate_cohort(rec, reg, cfg, ckpts, lcfg, labels_root=labels_root, store=store, access=access,
                                   log=log)
    for d in d28eval.write_runs(raws, Path(a.eval_root) if a.eval_root else d28eval.EVAL_ROOT):
        print(d)


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
    p = sub.add_parser("evaluate")
    p.add_argument("cohort")
    p.add_argument("--state-budget", type=int, help="the label config's per-state budget (finds the label records)")
    p.add_argument("--search-budget", type=int, help="per-task A* expansion budget, until frozen in configs/eval/d28.json")
    p.add_argument("--checkpoint", action="append", default=[], help="a trained checkpoint; repeat for each")
    p.add_argument("--repeats", type=int, help="timed repeats (smoke cohorts only; others must use 5)")
    p.add_argument("--warmup", type=int, help="discarded warm-up runs (smoke cohorts only; others must use 1)")
    p.add_argument("--eval-root")
    p.add_argument("--permitted-by", help="the D-entry permitting a second evaluation read of a test cohort")
    p = sub.add_parser("report")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--primary", required=True)
    p.add_argument("--val-split")
    p.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    reg = co.load_registry()
    ctx = None
    if a.cmd not in ("report", "evaluate"):
        from ahg.gen.generate import Context
        ctx = Context.load()
    {"build": cmd_build, "label": cmd_label, "n4": cmd_n4, "landscape": cmd_landscape, "budget": cmd_budget,
     "ac-counts": cmd_ac_counts, "evaluate": cmd_evaluate, "report": cmd_report}[a.cmd](a, reg, ctx)


if __name__ == "__main__":
    main()
