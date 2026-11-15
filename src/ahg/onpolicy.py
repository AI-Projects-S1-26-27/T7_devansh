"""On-policy data aggregation (D28 Round 9 "On-policy rounds"), one run per (model kind, seed, cost regime).

A run starts from the five-source offline corpus and does exactly ROUNDS rounds:

  round 0   train on the offline corpus (train batches) with checkpoint selection on val batches;
  round k   run this model's own learned A* on every training network of the offline corpus, label the
            expanded states the corpus lacks (exact A*+LM-cut labeller; every label cross-checked against
            the independent oracle, abort on mismatch), add them to this run's corpus only, retrain.

Isolation: a run's corpus is local to the run, so on-policy data is never shared across models, seeds or
regimes. Collection reads only train batches (refuse_non_training); val batches reach nothing but
train_model's checkpoint selection, and test batches nothing at all. Predictions are never labels: a
state whose exact label runs out of budget gets no label (BUDGET_EXHAUSTED), and a proven dead end
(h* = inf) is recorded apart from the regression labels.

The search and labelling budgets are explicit protocol keys with no defaults; Round 9 fixes them by a
training-only pilot, and they are the same for every model and seed.
"""

import gzip
import hashlib
import json
import time
from pathlib import Path

from .cost import REGIMES, check_priced, check_regime
from .dataset import Example, LabelBatch
from .heuristics import INF, lmcut
from .labels import SOURCES
from .learning.heuristic import Learned
from .learning.train import save, train_model
from .measure import HardFailure, check_admissible, enumerate_exact
from .oracle import optimal_cost
from .runlog import ROOT, write_run
from .search import astar
from .task import task_digest

ROUNDS = 2  # exactly 2 on-policy rounds (Round 9)
MODEL_KINDS = ("gnn", "mlp", "wl")  # the confirmatory models (Round 11)
SEEDS = (0, 1, 2)  # the D23 seeds (Round 11 "Seeds")
PROTOCOL_KEYS = {"search_budget", "state_budget", "oracle_max_pops", "enum_cap"}
ONPOLICY = ROOT / "results" / "training" / "onpolicy"


def source_name(k: int) -> str:
    """The source tag of states first collected in on-policy round k."""
    return f"onpolicy_r{k}"


def plan() -> list[tuple[str, str, int]]:
    """Every (regime, model kind, seed) run, in a fixed order (an sbatch array index picks one)."""
    return [(r, k, s) for r in REGIMES for k in MODEL_KINDS for s in SEEDS]


def run_config(index: int, base: dict, aggregators: dict | None) -> dict:
    """The training config of plan()[index]: `base` (a full training config without kind, seed or regime)
    with the run's kind, seed and cost regime. A GNN takes the frozen `ac` grid choice (aggregator_grid.
    frozen_aggregators) under both regimes; the MLP and WL take none."""
    regime, kind, seed = plan()[index]
    if {"kind", "seed", "cost_regime", "pre_agg", "readout"} & set(base):
        raise ValueError("kind, seed, cost_regime and the aggregators come from the plan and the frozen grid")
    if (kind == "gnn") != (aggregators is not None):
        raise ValueError(f"the frozen aggregator choice is required for the GNN and only for it (run {index}: {kind})")
    agg = {"pre_agg": aggregators["pre_agg"], "readout": aggregators["readout"]} if aggregators else {}
    return {**base, "kind": kind, "seed": seed, "cost_regime": regime, **agg}


def check_protocol(protocol: dict) -> dict:
    """The protocol, or ValueError if a key is missing or unknown (no budget is ever defaulted)."""
    if set(protocol) != PROTOCOL_KEYS:
        raise ValueError(f"protocol keys must be exactly {sorted(PROTOCOL_KEYS)}, got {sorted(protocol)}")
    return protocol


def refuse_non_training(batches: list[LabelBatch]) -> None:
    """On-policy states are collected on training networks only (Round 9): anything else is refused."""
    for b in batches:
        if b.manifest["split"] != "train":
            raise PermissionError(f"on-policy collection runs on training networks only; refused batch "
                                  f"{b.manifest.get('batch_id')} of split {b.manifest['split']!r}")


def check_regime_of(batches: list[LabelBatch], regime: str) -> None:
    """Every batch is labelled under `regime`, and every task in it is priced under `regime`."""
    for b in batches:
        if b.manifest.get("cost_regime") != regime:
            raise ValueError(f"batch {b.manifest.get('batch_id')} has cost regime "
                             f"{b.manifest.get('cost_regime')!r}, not {regime!r}")
        for inst in b.instances:
            if inst is not None:
                check_priced(inst.task, regime)


def make_batch(batch_id: str, split: str, regime: str, instances: list, examples: list[Example],
               failures: list[dict], parent: str | None = None) -> LabelBatch:
    """An in-memory LabelBatch with a content hash over everything its encodings depend on.

    The encoder caches by manifest["content_sha256"], so the hash covers the regime, every task digest
    and every labelled state: two batches can never share a cache entry unless they hold the same data.
    """
    check_regime(regime)
    digests = [None if inst is None else task_digest(inst.task) for inst in instances]
    rows = [[ex.instance, hex(ex.state), ex.hstar, ex.sources] for ex in examples]
    blob = json.dumps({"regime": regime, "digests": digests, "rows": rows, "failures": failures},
                      sort_keys=True).encode()
    manifest = {"batch_id": batch_id, "split": split, "cost_regime": regime, "parent": parent,
                "content_sha256": hashlib.sha256(blob).hexdigest(),
                "instances": [{"task_digest": d} for d in digests]}
    return LabelBatch(manifest, instances, examples, failures)


def label_states(task, states: list[int], protocol: dict) -> dict:
    """Exact h* of each state by A*+LM-cut, every label cross-checked; HardFailure on any mismatch.

    Returns {"labels": {state: h*}, "budget": [states], "dead_ends": [states], "stats": {...}}.
    A budget stop gives no label; a search that exhausts its open list without the goal is a proven
    dead end (h* = inf). Cross-checks: h_max and LM-cut must not exceed the label; full enumeration by
    the reference solver where the task has at most enum_cap reachable states; and the independent
    oracle on every label and dead end (a pop-budget stop is counted as skipped, never as agreement).
    """
    t0 = time.perf_counter()
    known, budget, dead, expansions = {}, [], [], 0
    for s in sorted(set(states)):
        if s in known:
            continue
        r = astar(task, lmcut, start=s, max_expansions=protocol["state_budget"])
        expansions += r.expanded
        if r.budget_hit:
            budget.append(s)
            continue
        if not r.solved:
            dead.append(s)
            continue
        # Every state on an optimal plan gets its suffix cost: a suffix of an optimal plan is optimal.
        st, rest = s, r.cost
        for ai in r.plan:
            known.setdefault(st, rest)
            st, rest = st | task.actions[ai].add_mask, rest - task.actions[ai].cost
        known.setdefault(st, 0)
    labels = {s: known[s] for s in sorted(set(states)) if s in known}
    label_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    for s, h in labels.items():
        check_admissible(task, s, h)
    exact = enumerate_exact(task, protocol["enum_cap"])
    if exact is not None:
        for s, h in labels.items():
            if exact[s] != h:
                raise HardFailure(f"enumeration h*={exact[s]} != label {h} at {s:#x}")
        for s in dead:
            if exact.get(s, INF) != INF:
                raise HardFailure(f"enumeration h*={exact[s]} at {s:#x}, which A*+LM-cut called a dead end")
    checked = skipped = 0
    for s, h in [*labels.items(), *((s, INF) for s in dead)]:
        o = optimal_cost(task, s, max_pops=protocol["oracle_max_pops"])
        if o is None:
            skipped += 1
        elif o != h:
            raise HardFailure(f"oracle {o} != label {h} at {s:#x}")
        else:
            checked += 1
    stats = {"label_expansions": expansions, "label_seconds": round(label_s, 4),
             "check_seconds": round(time.perf_counter() - t1, 4), "enumerated": exact is not None,
             "oracle_checked": checked, "oracle_skipped": skipped}
    return {"labels": labels, "budget": budget, "dead_ends": dead, "stats": stats}


def _seen(batches: list[LabelBatch]) -> set[tuple[str, int]]:
    """(task digest, state) of every example in a corpus: the per-task deduplication key."""
    return {(b.manifest["instances"][ex.instance]["task_digest"], ex.state) for b in batches for ex in b.examples}


def collect(model, cfg: dict, offline: list[LabelBatch], regime: str, protocol: dict, k: int,
            seen: set[tuple[str, int]]) -> tuple[list[LabelBatch], dict]:
    """Round k: this model's learned A* on every training network; label the new expanded states.

    `seen` is this run's dedup set and is extended in place. Returns one on-policy batch per offline
    batch (same instances, new examples only) and the round's statistics.
    """
    refuse_non_training(offline)
    learned = Learned(model, cfg, f"{cfg['kind']}-s{cfg['seed']}-{regime}-r{k - 1}")
    tot = dict.fromkeys(("networks", "solved", "search_budget_hits", "search_expansions", "states_expanded",
                         "states_new", "labelled", "label_budget_hits", "dead_ends", "label_expansions",
                         "oracle_checked", "oracle_skipped"), 0)
    tot.update(label_seconds=0.0, check_seconds=0.0, search_seconds=0.0)
    out = []
    for b in offline:
        examples, failures = [], []
        for i, inst in enumerate(b.instances):
            if inst is None:
                continue
            task, dig = inst.task, b.manifest["instances"][i]["task_digest"]
            trace: list[int] = []
            t0 = time.perf_counter()
            r = astar(task, learned.bind(inst), max_expansions=protocol["search_budget"], trace=trace)
            tot["search_seconds"] += time.perf_counter() - t0
            expanded = list(dict.fromkeys(trace))
            new = [s for s in expanded if (dig, s) not in seen]
            lab = label_states(task, new, protocol)
            for s, h in lab["labels"].items():
                examples.append(Example(i, s, h, [source_name(k)]))
                seen.add((dig, s))
            failures += [{"kind": "label_budget", "instance": i, "state": hex(s), "round": k} for s in lab["budget"]]
            failures += [{"kind": "dead_end", "instance": i, "state": hex(s), "round": k} for s in lab["dead_ends"]]
            tot["networks"] += 1
            tot["solved"] += r.solved
            tot["search_budget_hits"] += r.budget_hit
            tot["search_expansions"] += r.expanded
            tot["states_expanded"] += len(expanded)
            tot["states_new"] += len(new)
            tot["labelled"] += len(lab["labels"])
            tot["label_budget_hits"] += len(lab["budget"])
            tot["dead_ends"] += len(lab["dead_ends"])
            for key in ("label_expansions", "oracle_checked", "oracle_skipped", "label_seconds", "check_seconds"):
                tot[key] += lab["stats"][key]
        bid = f"{b.manifest['batch_id']}+{cfg['kind']}-s{cfg['seed']}-{regime}-{source_name(k)}"
        out.append(make_batch(bid, "train", regime, b.instances, examples, failures, parent=b.manifest["batch_id"]))
    # Coverage: the share of new states that got an exact label (Round 9 reporting).
    tot["label_coverage"] = tot["labelled"] / tot["states_new"] if tot["states_new"] else None
    for key in ("label_seconds", "check_seconds", "search_seconds"):
        tot[key] = round(tot[key], 4)
    return out, tot


def _train(cfg: dict, corpus: list[LabelBatch], val: list[LabelBatch]):
    model, history = train_model(cfg, corpus, val)
    epochs = [e for e in history if "epoch" in e]
    best = min(epochs, key=lambda e: (e["val_mae"], e["epoch"]))  # F4: the earliest of equal MAEs
    return model, {"history": history, "selected_epoch": best["epoch"], "val_mae": best["val_mae"],
                   "train_examples": sum(len(b.examples) for b in corpus)}


def run_onpolicy(cfg: dict, offline: list[LabelBatch], val: list[LabelBatch], protocol: dict) -> dict:
    """One on-policy run. cfg: a training config with kind, seed and cost_regime set (the GNN also with its
    frozen pre_agg / readout). Returns {"models": [round 0..ROUNDS model], "rounds": [...], "batches": [...]}."""
    regime = check_regime(cfg.get("cost_regime"))
    if cfg["kind"] not in MODEL_KINDS or cfg["seed"] not in SEEDS:
        raise ValueError(f"on-policy runs are for kinds {MODEL_KINDS} and seeds {SEEDS}, "
                         f"not {cfg['kind']!r} seed {cfg['seed']!r}")
    if cfg["kind"] == "gnn" and not {"pre_agg", "readout"} <= set(cfg):
        raise ValueError("a GNN on-policy run needs the frozen aggregator choice (pre_agg, readout) in its config")
    check_protocol(protocol)
    refuse_non_training(offline)
    check_regime_of(offline + val, regime)
    for b in offline:
        extra = {s for ex in b.examples for s in ex.sources} - set(SOURCES)
        if extra:
            raise ValueError(f"offline batch {b.manifest['batch_id']} has non-offline sources {sorted(extra)}")

    corpus, seen = list(offline), _seen(offline)
    model, info = _train(cfg, corpus, val)
    models, rounds, collected = [model], [{"round": 0, "train": info}], []
    for k in range(1, ROUNDS + 1):
        new, stats = collect(model, cfg, offline, regime, protocol, k, seen)
        corpus += new  # this run's corpus only; the offline examples stay throughout
        collected += new
        model, info = _train(cfg, corpus, val)
        models.append(model)
        rounds.append({"round": k, "collect": stats, "train": info})
    return {"models": models, "rounds": rounds, "batches": collected}


def _batch_lines(b: LabelBatch) -> bytes:
    lines = [json.dumps({"i": ex.instance, "s": hex(ex.state), "h": ex.hstar, "src": ex.sources}) for ex in b.examples]
    return gzip.compress(("\n".join(lines) + "\n").encode() if lines else b"", mtime=0)


def render(raw: dict) -> str:
    c = raw["config"]
    L = [f"# on-policy run: {c['kind']} seed {c['seed']}, cost regime `{raw['cost_regime']}`", "",
         f"Offline batches: {', '.join(raw['offline_batches'])}. Val batches: {', '.join(raw['val_batches'])}.",
         f"Protocol: `{json.dumps(raw['protocol'], sort_keys=True)}`. Exactly {ROUNDS} rounds (D28 Round 9).", "",
         "| round | train examples | selected epoch | macro val MAE | networks | search expansions | search budget hits "
         "| states expanded | new states | labelled | label coverage | label budget hits | dead ends "
         "| labelling expansions | labelling s | oracle checked / skipped |",
         "|---" * 16 + "|"]
    for r in raw["rounds"]:
        t, s = r["train"], r.get("collect")
        cov = "—" if not s or s["label_coverage"] is None else f"{s['label_coverage']:.4f}"
        col = "— | " * 11 + "— |" if not s else (  # round 0 trains on the offline corpus and collects nothing
            f"{s['networks']} | {s['search_expansions']} | {s['search_budget_hits']} | {s['states_expanded']} | "
            f"{s['states_new']} | {s['labelled']} | {cov} | {s['label_budget_hits']} | {s['dead_ends']} | "
            f"{s['label_expansions']} | {s['label_seconds']:.2f} | {s['oracle_checked']} / {s['oracle_skipped']} |")
        L.append(f"| {r['round']} | {t['train_examples']} | {t['selected_epoch']} | {t['val_mae']:.4f} | " + col)
    return "\n".join(L) + "\n"


def write_onpolicy_run(cfg: dict, offline: list[LabelBatch], val: list[LabelBatch], protocol: dict,
                       result: dict, root: Path | None = None, extra: dict | None = None) -> Path:
    """Write the run through runlog: raw.json.gz (regime, protocol, inputs, per-round statistics), report.md,
    provenance.json, one checkpoint per round (round<k>.pt; the last is the run's model) and the collected
    on-policy labels (onpolicy-<j>.jsonl.gz, hashed in raw). extra: more raw fields (e.g. the frozen grid file)."""
    raw = {"cost_regime": cfg["cost_regime"], "config": cfg, "protocol": protocol, "rounds_planned": ROUNDS,
           "offline_batches": [b.manifest["batch_id"] for b in offline],
           "offline_content_sha256": [b.manifest["content_sha256"] for b in offline],
           "val_batches": [b.manifest["batch_id"] for b in val],
           "val_content_sha256": [b.manifest["content_sha256"] for b in val],
           "rounds": result["rounds"],
           "onpolicy_batches": [{"batch_id": b.manifest["batch_id"], "parent": b.manifest["parent"],
                                 "content_sha256": b.manifest["content_sha256"], "examples": len(b.examples),
                                 "failures": b.failures} for b in result["batches"]], **(extra or {})}
    name = f"{cfg['cost_regime']}-{cfg['kind']}-s{cfg['seed']}"
    out = write_run(name, raw, render(raw), [cfg["seed"]], cfg, root=root or ONPOLICY)
    for k, model in enumerate(result["models"]):
        save(model, cfg, out / f"round{k}.pt")
    for j, b in enumerate(result["batches"]):
        (out / f"onpolicy-{j}.jsonl.gz").write_bytes(_batch_lines(b))
    return out
