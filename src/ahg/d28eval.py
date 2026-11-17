"""D28 evaluation runner (P7b, OQ-P7-11; decisions Rounds 7, 10, 12): every classical baseline and learned
model on one cohort, under both cost regimes and both tie rules, with the Fast Downward runs, written as the
per-task records ahg.report.build_report reads.

Per member task of the cohort and per cost regime:
- the common exact reachability pre-check (Round 7): h_max(s0) finite must agree with the generator's flag.
  An unsolvable task, or a generator cell that failed, is kept with its reason and gets no search;
- its label record (ahg.d28labels, which must exist): the certified h*(s0) when LABELLED, else uncertified;
- the D28 learned inputs (ahg.learning.d28encode, P7c): the task's planner view encoded by the same functions
  training uses, inside every timed repeat (like the static graph, it is the model's per-task set-up). Before
  any learned model runs, the cohort's encoded tasks pass the LRN6 leakage audit, whose report goes into every
  raw;
- per tie rule, A* with h0, h_max, h_add, h_FF, LM-cut, h_sp and every learned checkpoint of that regime,
  all through evaluate._run: the same best_first_search, reopening (A* reopens, Round 7), per-task expansion
  budget, warm-up and timed repeats with a determinism check (Round 12, M1). Recorded: expansions, plan cost,
  Python time (the learned model's per-task set-up inside it), peak open and closed list sizes (Round 7);
- an admissible baseline that did not hit the budget must return the certified h*, else HardFailure; h_add,
  h_FF and the learned models are inadmissible, so their cost is measured, never bounded (CLAUDE.md rule 8);
- overestimation of every method on the task's labelled states: h(s) > h*(s) + tolerance, in regime units;
- Fast Downward 24.06.1 A*+LM-cut and A*+h_max on the PDDL export (Round 7, fd_timing.run): end-to-end wall
  time, the peak resident memory of the FD process tree, and FD's plan cost, which must equal a certified h*.
  FD breaks ties its own way, so its record does not depend on our tie rule: it runs once per task and regime
  and the same record goes into both tie-rule raws. Python and FD numbers stay in separate fields (Round 12).

Output: one runlog run per (regime, tie rule) at <eval root>/<split>.<regime>.<tie>/<stamp>-<commit>/, whose
raw is {"split", "regime", "tie_break", "records", ...the inputs' hashes}, one record per member:

    {"task": digest|None, "network": seed, "seed", "n", "solvable", "unsolvable_reason", "label_outcome",
     "certified", "hstar", "unverified_release",
     "runs": {method: {"expanded", "generated", "evaluations", "cost", "budget_hit", "python_time",
                       "peak_open", "peak_closed", "reopened", "fd_time", "peak_memory_bytes", "fd",
                       "overestimation": {"states", "violations", "max_violation"} | None}}}

One task per generated network, so "network" is the task's seed. A test cohort is evaluated only complete
(both regimes, both tie rules, FD, GNN/MLP/WL x D23 seeds 0-2 per regime, the M1 repeats) and only through a
logged "evaluation" read that names the dataset, this config and every checkpoint (Round 10); every check that
needs no test data runs before that read is logged.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import cohorts as co, d28labels, fd_timing, runlog
from .cost import REGIMES, check_regime
from .evaluate import BASELINES, _run
from .heuristics import ADMISSIBLE, INF, hmax
from .measure import HardFailure
from .search import TIE_RULES
from .splits import EVALUATION, LABEL_GENERATION
from .task import Task
from .timing import REPEATS, WARMUP

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "eval" / "d28.json"
EVAL_ROOT = ROOT / "results" / "d28_eval"
SCHEMA = "ahg.d28eval/1"
CONFIG_KEYS = {"regimes", "tie_rules", "search_budget", "repeats", "warmup", "fast_downward",
               "overestimation_tolerance"}
MODELS = ("gnn", "mlp", "wl")  # the confirmatory models (Round 11)
D23_SEEDS = (0, 1, 2)  # Round 11 "Seeds"


# ------------------------------------------------------------------ config

def load_config(path: Path = CONFIG) -> dict:
    cfg = {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}  # "_note" documents
    check_config(cfg)
    return cfg


def check_config(cfg: dict) -> None:
    if set(cfg) != CONFIG_KEYS:
        raise ValueError(f"evaluation config keys {sorted(set(cfg) ^ CONFIG_KEYS)} missing or unknown")
    for r in cfg["regimes"]:
        check_regime(r)
    if not cfg["tie_rules"] or set(cfg["tie_rules"]) - set(TIE_RULES):
        raise ValueError(f"tie_rules must be drawn from {TIE_RULES}")
    if set(cfg["fast_downward"]) - set(fd_timing.SEARCHES):
        raise ValueError(f"fast_downward must be drawn from {sorted(fd_timing.SEARCHES)}")


def with_budget(cfg: dict, search_budget: int | None) -> dict:
    """The config with its per-task search budget. Round 12: set from validation, frozen before any test read;
    while the config holds null, a run needs an explicit value (and records it)."""
    budget = cfg["search_budget"] if search_budget is None else search_budget
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("the search budget is not frozen yet (search_budget null); pass one explicitly")
    return {**cfg, "search_budget": budget}


def config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ learned checkpoints

@dataclass(frozen=True)
class Checkpoint:
    name: str  # "<kind>@<seed>", the report's method name
    regime: str
    path: str
    sha256: str
    learned: object  # learning.heuristic.Learned


def load_checkpoints(paths: list[Path]) -> dict[str, list[Checkpoint]]:
    """regime -> its checkpoints, named from their own training config (kind, seed, cost_regime).

    Round 11: a model is used only in the regime it was trained in, so a checkpoint without a cost regime
    (pre-D28) is refused, as is a second checkpoint with the same regime and name. D28 tasks are encoded only
    by the D28 encoder, so a checkpoint trained on other features is refused too (P7c, OQ-P7b-1)."""
    from .learning import d28encode
    from .learning.heuristic import Learned
    from .learning.train import load
    out: dict[str, list[Checkpoint]] = {}
    for p in paths:
        p = Path(p)
        model, cfg = load(p)
        regime = cfg.get("cost_regime")
        if regime is None:
            raise ValueError(f"{p}: the checkpoint names no cost regime (pre-D28); D28 evaluates per regime")
        check_regime(regime)
        if cfg["kind"] not in MODELS:
            raise ValueError(f"{p}: unknown model kind {cfg['kind']!r}")
        if cfg.get("features") != d28encode.FEATURE_SET:
            raise ValueError(f"{p}: the checkpoint was not trained on the D28 encoder (features "
                             f"{cfg.get('features')!r}, not 'd28'); D28 tasks are encoded only by ahg.learning.d28encode")
        name = f"{cfg['kind']}@{cfg['seed']}"
        if any(c.name == name for c in out.get(regime, [])):
            raise ValueError(f"two checkpoints for {regime}/{name}")
        out.setdefault(regime, []).append(Checkpoint(name, regime, str(p), hashlib.sha256(p.read_bytes()).hexdigest(),
                                                     Learned(model, cfg, name)))
    return out


def checkpoint_hashes(ckpts: dict[str, list[Checkpoint]]) -> dict[str, str]:
    return {f"{r}/{c.name}": c.sha256 for r in sorted(ckpts) for c in sorted(ckpts[r], key=lambda c: c.name)}


# ------------------------------------------------------------------ checks before anything is read

def completeness_problems(c: co.Cohort, cfg: dict, ckpts: dict[str, list[Checkpoint]]) -> list[str]:
    """Why this run would not be the one complete test batch of Round 10; [] if it would be."""
    out = []
    if sorted(cfg["regimes"]) != sorted(REGIMES):
        out.append(f"regimes {cfg['regimes']} are not both {list(REGIMES)}")
    if sorted(cfg["tie_rules"]) != sorted(TIE_RULES):
        out.append(f"tie rules {cfg['tie_rules']} are not both {list(TIE_RULES)}")
    if sorted(cfg["fast_downward"]) != sorted(fd_timing.SEARCHES):
        out.append("the Fast Downward runs are not all there")
    want = sorted(f"{m}@{s}" for m in MODELS for s in D23_SEEDS)
    for r in REGIMES:
        got = sorted(x.name for x in ckpts.get(r, []))
        if got != want:
            out.append(f"{r} checkpoints {got} are not {want}")
    return out


def missing_labels(record: dict, label_cfg: dict, root: Path) -> int:
    """Members with a task but no label record on disk, over every regime. Looks at file names only, so it
    reads no test data and can run before a test read is logged."""
    return sum(not d28labels._path(root, r, d28labels.record_key(m["digest"], r, label_cfg)).exists()
               for m in record["members"] if m["digest"] is not None for r in label_cfg["regimes"])


def preflight(c: co.Cohort, record: dict, cfg: dict, ckpts: dict[str, list[Checkpoint]], label_cfg: dict,
              labels_root: Path) -> None:
    """Every check that needs no test data, so a run that would fail stops before a test read is logged."""
    check_config(cfg)
    with_budget(cfg, None)  # the run's budget must be set
    if not c.smoke and (cfg["repeats"], cfg["warmup"]) != (REPEATS, WARMUP):  # smoke tasks are never data
        raise ValueError(f"timing protocol {cfg['repeats']}+{cfg['warmup']} is not the M1 {REPEATS}+{WARMUP}")
    if c.is_test:
        why = completeness_problems(c, cfg, ckpts)
        if why:
            raise co.TestAccessError(f"{c.name}: not the complete test batch: " + "; ".join(why))
    missing = missing_labels(record, label_cfg, labels_root)
    if missing:
        raise FileNotFoundError(f"{c.name}: {missing} (task, regime) label records missing; label the cohort first")


# ------------------------------------------------------------------ one task

def _overestimation(h, task: Task, rows: list[list], tol: float) -> dict | None:
    """Counts of labelled states where h exceeds the certified h*; None if the task has no labelled states."""
    if not rows:
        return None
    over = [h(task, int(s, 16)) - hs for s, hs, _ in rows]
    viol = [x for x in over if x > tol]
    return {"states": len(over), "violations": len(viol), "max_violation": max(viol, default=0.0)}


def _fd(doc: dict, regime: str, cfg: dict, hstar, certified: bool) -> dict:
    """FD A*+<h> per cfg["fast_downward"] on the task's PDDL export; checked against a certified h*."""
    from .gen.generate import export_pddl
    domain, problem = export_pddl(doc, regime)
    out = {}
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "domain.pddl").write_text(domain)
        (Path(d) / "problem.pddl").write_text(problem)
        for h in cfg["fast_downward"]:
            r = fd_timing.run(Path(d) / "domain.pddl", Path(d) / "problem.pddl", h, cfg["repeats"],
                              warmup=cfg["warmup"])
            if r["status"] != "solved":  # FD has no limits here, so a solvable task must be solved
                raise HardFailure(f"FD A*+{h} did not solve a solvable task: {r['status']}")
            if certified and r["cost"] != hstar:
                raise HardFailure(f"FD A*+{h} cost {r['cost']} != certified h* {hstar}")
            out[h] = r
    return out


def evaluate_task(doc: dict | None, member: dict, regime: str, label: dict | None, cfg: dict,
                  ckpts: list[Checkpoint]) -> dict[str, dict]:
    """tie rule -> the task's record under `regime` (module docstring)."""
    from .gen.generate import to_task
    base = {"task": member["digest"], "network": member["seed"], "seed": member["seed"], "n": member["n"],
            "solvable": False, "unsolvable_reason": member["reason"], "label_outcome": None, "certified": False,
            "hstar": None, "unverified_release": None if doc is None else doc["flags"]["unverified_release"],
            "runs": {}}
    if doc is None:  # an infeasible generator cell: a failed member, kept and counted (Round 7)
        return {t: base for t in cfg["tie_rules"]}
    task = to_task(doc, regime)
    reachable = hmax(task, task.init) < INF  # the common exact pre-check (Round 7)
    if reachable != doc["solvable"]:
        raise HardFailure(f"{member['digest']}: generator solvable={doc['solvable']} but h_max(s0) finite={reachable}")
    if label is None:
        raise FileNotFoundError(f"{member['digest']}/{regime}: no label record; label the cohort first")
    if (label["outcome"] == d28labels.DEAD_END) == reachable:
        raise HardFailure(f"{member['digest']}/{regime}: label outcome {label['outcome']} vs h_max(s0) finite={reachable}")
    base |= {"label_outcome": label["outcome"]}
    if not reachable:
        return {t: base for t in cfg["tie_rules"]}
    certified, hstar = label["certified"], label["hstar_init"]
    base |= {"solvable": True, "unsolvable_reason": None, "certified": certified, "hstar": hstar}
    from .learning import d28encode
    inst = d28encode.instance(doc, regime) if ckpts else None  # untimed: the overestimation statistics only
    tol = cfg["overestimation_tolerance"]
    rows = label["rows"] if certified else []
    over = {name: _overestimation(h, task, rows, tol) for name, h in BASELINES.items()}
    for c in ckpts:
        bound = c.learned.bind(inst)
        over[c.name] = _overestimation(bound, task, rows, tol)
    fd = _fd(doc, regime, cfg, hstar, certified)
    out = {}
    for tie in cfg["tie_rules"]:
        makers = [(name, lambda h=h: h) for name, h in BASELINES.items()]
        # A fresh encoding and bind per repeat: the model's per-task set-up is inside every timed run (Round 12).
        makers += [(c.name, lambda c=c: c.learned.bind(d28encode.instance(doc, regime))) for c in ckpts]
        runs = {}
        for name, make in makers:
            r = _run(task, make, cfg["search_budget"], cfg["repeats"], tie_break=tie, warmup=cfg["warmup"])
            if name in ADMISSIBLE and certified and not r["budget_hit"] and r["cost"] != hstar:
                raise HardFailure(f"{member['digest']}/{regime}/{tie}: A*+{name} cost {r['cost']} != certified h* {hstar}")
            f = fd.get(name)
            runs[name] = {**{k: v for k, v in r.items() if k != "time"}, "python_time": r["time"],
                          "fd_time": None if f is None else f["wall"]["end_to_end"],
                          "peak_memory_bytes": None if f is None else max(f["peak_rss_bytes"]["end_to_end"]),
                          "fd": f, "overestimation": over[name]}
        out[tie] = base | {"runs": runs}
    return out


# ------------------------------------------------------------------ one cohort

def evaluate_cohort(record: dict, registry: co.Registry, cfg: dict, ckpts: dict[str, list[Checkpoint]],
                    label_cfg: dict, *, labels_root: Path, store: Path, access: dict | None = None,
                    log: Path = co.TEST_ACCESS_LOG) -> dict[tuple[str, str], dict]:
    """(regime, tie rule) -> raw. A test cohort needs `access`, the logged evaluation read (Round 10)."""
    c = registry[record["cohort"]]
    preflight(c, record, cfg, ckpts, label_cfg, labels_root)
    if c.is_test:
        if access is None or access.get("purpose") != EVALUATION:
            raise co.TestAccessError(f"{c.name} is a test cohort: evaluation needs its logged evaluation read")
        if (access.get("eval_config_sha256"), access.get("checkpoint_sha256")) != (config_sha(cfg), checkpoint_hashes(ckpts)):
            raise co.TestAccessError("the evaluation read names a different config or different checkpoints")
    purposes = (EVALUATION,) if c.is_test else (LABEL_GENERATION, EVALUATION)
    members = co.open_members(record, registry, store=store, access=access, purposes=purposes, log=log)
    docs = [d for _, d in members if d is not None]
    audit = None
    if any(ckpts.get(r) for r in cfg["regimes"]) and docs:
        from .learning import d28encode  # LRN6: the encoded split passes the leakage audit before any model runs
        audit = d28encode.audit_docs(docs, c.name)
    labels = d28labels.load_labels(docs, label_cfg, root=labels_root)
    raws = {}
    for regime in cfg["regimes"]:
        recs = {t: [] for t in cfg["tie_rules"]}
        for m, doc in members:
            lab = None if doc is None else labels.get((m["digest"], regime))
            for t, rec in evaluate_task(doc, m, regime, lab, cfg, ckpts.get(regime, [])).items():
                recs[t].append(rec)
        for t in cfg["tie_rules"]:
            raws[(regime, t)] = {
                "schema": SCHEMA, "split": c.name, "regime": regime, "tie_break": t, "role": c.role,
                "analysis": c.analysis, "dataset_sha256": record["dataset_sha256"],
                "registry_sha256": registry.sha256, "config": cfg, "config_sha256": config_sha(cfg),
                "label_config_sha256": d28labels.config_sha(label_cfg),
                "checkpoints": {x.name: {"path": x.path, "sha256": x.sha256} for x in ckpts.get(regime, [])},
                "test_access": access, "fd_version": fd_timing.FD_VERSION, "leakage_audit": audit,
                "records": recs[t]}
    return raws


def summary_md(raw: dict) -> str:
    recs = raw["records"]
    sol = [r for r in recs if r["solvable"]]
    return (f"# Evaluation {raw['split']} ({raw['regime']}, tie rule {raw['tie_break']})\n\n"
            f"tasks {len(recs)}, solvable {len(sol)}, certified {sum(r['certified'] for r in sol)}; "
            f"methods {sorted({m for r in sol for m in r['runs']})}. Tables: ahg.report.build_report.\n")


def write_runs(raws: dict[tuple[str, str], dict], root: Path = EVAL_ROOT) -> list[Path]:
    """One runlog run per (regime, tie rule): <root>/<split>.<regime>.<tie>/<stamp>-<commit>/ (append-only)."""
    out = []
    for (regime, tie), raw in sorted(raws.items()):
        seeds = sorted({int(n.split("@")[1]) for n in raw["checkpoints"]})
        out.append(runlog.write_run(f"{raw['split']}.{regime}.{tie}", raw, summary_md(raw), seeds,
                                    {"evaluation": raw["config"], "split": raw["split"], "regime": regime,
                                     "tie_break": tie, "checkpoints": raw["checkpoints"]}, root=root))
    return out
