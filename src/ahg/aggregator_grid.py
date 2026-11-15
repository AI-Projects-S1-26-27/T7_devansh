"""GNN aggregator grid (D28 Round 11 "GNN aggregators"; regime: CA2).

The pre-registered grid is precondition aggregation {mean, sum, max} x readout {mean, sum}
(models.AGGREGATOR_GRID). It is one selection procedure, not six models:

  run_grid   trains every cell with the same offline training batches, validation batches, protocol and
             seeds, and records every cell's selected macro validation MAE (all are reported);
  select     picks the cell with the lowest validation MAE, ties broken by a deterministic simplicity order;
  freeze     writes the choice once, to a tracked config file; the frozen file is the record.

The grid runs once, under the `ac` regime, on offline data only, before any on-policy round. The `uniform`
models reuse the frozen `ac` choice; no grid runs under `uniform` (CA2). The selection rule's simplicity
order, how the seeds of a cell combine and what counts as a tie are not fixed by D28 (open question
OQ-P9a-1), so select() takes them as an explicit rule and never defaults them. Decided 15 Nov 2026 (P9a2):
seeds combine by the mean; a cell within 2% (relative) of the best mean MAE ties with it; the simplest tied
cell wins, in the order of configs/d28_grid_rule.json.
"""

import hashlib
import json
import statistics
from pathlib import Path

from .cost import PRIMARY_REGIME
from .labels import SOURCES
from .learning.models import AGGREGATOR_GRID
from .learning.train import train_model
from .onpolicy import SEEDS, check_regime_of
from .runlog import ROOT, write_run

GRID_REGIME = PRIMARY_REGIME  # "ac" (CA2)
GRID = ROOT / "results" / "training" / "grid"
COMBINE = {"mean": statistics.fmean}  # the decided seed combination (P9a2)
RULE_KEYS = {"simplicity_order", "seed_combination", "tie_relative"}
RULE_FILE = ROOT / "configs" / "d28_grid_rule.json"


def grid_configs(base: dict) -> list[dict]:
    """One config per (cell, seed): the base GNN config with that cell's pre_agg / readout and seed."""
    if base["kind"] != "gnn":
        raise ValueError("the aggregator grid is for the GNN only")
    if {"pre_agg", "readout"} & set(base):
        raise ValueError("the base config must leave pre_agg and readout to the grid")
    if base.get("cost_regime") != GRID_REGIME:
        raise ValueError(f"the grid runs once, under {GRID_REGIME!r}; no grid runs under "
                         f"{base.get('cost_regime')!r} (D28 Round 11, CA2)")
    return [{**base, "pre_agg": p, "readout": r, "seed": s} for p, r in AGGREGATOR_GRID for s in SEEDS]


def run_grid(base: dict, train: list, val: list) -> list[dict]:
    """Train every (cell, seed) on offline training data; one row per run with its selected val MAE."""
    configs = grid_configs(base)
    check_regime_of(train + val, GRID_REGIME)
    for b in train:
        extra = {s for ex in b.examples for s in ex.sources} - set(SOURCES)
        if extra:  # the grid runs before any on-policy round: offline sources only
            raise ValueError(f"grid training batch {b.manifest['batch_id']} has non-offline sources {sorted(extra)}")
    rows = []
    for cfg in configs:
        _, history = train_model(cfg, train, val)
        epochs = [e for e in history if "epoch" in e]
        best = min(epochs, key=lambda e: (e["val_mae"], e["epoch"]))  # F4: the earliest of equal MAEs
        rows.append({"pre_agg": cfg["pre_agg"], "readout": cfg["readout"], "seed": cfg["seed"],
                     "val_mae": best["val_mae"], "selected_epoch": best["epoch"], "history": history})
    return rows


def render(raw: dict) -> str:
    L = [f"# GNN aggregator grid, cost regime `{raw['cost_regime']}` (D28 Round 11)", "",
         f"Offline train batches: {', '.join(raw['train_batches'])}. Val batches: {', '.join(raw['val_batches'])}.",
         "Every cell and seed is reported; the selection is made from these rows by the frozen rule.", "",
         "| pre_agg | readout | seed | selected epoch | macro val MAE |", "|---|---|---|---|---|"]
    L += [f"| {r['pre_agg']} | {r['readout']} | {r['seed']} | {r['selected_epoch']} | {r['val_mae']:.6f} |"
          for r in raw["rows"]]
    return "\n".join(L) + "\n"


def write_grid_run(base: dict, train: list, val: list, rows: list[dict], root: Path | None = None) -> Path:
    """The grid's rows (every validation result) through runlog, under results/training/grid/<regime>/."""
    raw = {"cost_regime": GRID_REGIME, "base_config": base, "seeds": list(SEEDS),
           "train_batches": [b.manifest["batch_id"] for b in train],
           "train_content_sha256": [b.manifest["content_sha256"] for b in train],
           "val_batches": [b.manifest["batch_id"] for b in val],
           "val_content_sha256": [b.manifest["content_sha256"] for b in val], "rows": rows}
    return write_run(GRID_REGIME, raw, render(raw), list(SEEDS), base, root=root or GRID)


def check_rule(rule: dict) -> dict:
    """The selection rule, or ValueError. simplicity_order lists all six cells, simplest first."""
    if set(rule) != RULE_KEYS:
        raise ValueError(f"selection rule keys must be exactly {sorted(RULE_KEYS)}, got {sorted(rule)}")
    order = [tuple(c) for c in rule["simplicity_order"]]
    if sorted(order) != sorted(AGGREGATOR_GRID) or len(order) != len(AGGREGATOR_GRID):
        raise ValueError(f"simplicity_order must list each of the six grid cells once, got {order}")
    if rule["seed_combination"] not in COMBINE:
        raise ValueError(f"seed_combination must be one of {sorted(COMBINE)}")
    if not (isinstance(rule["tie_relative"], (int, float)) and rule["tie_relative"] >= 0):
        raise ValueError("tie_relative must be a number >= 0")
    return rule


def load_rule(path: Path = RULE_FILE) -> dict:
    """The selection rule from its config file (default: the decided rule). The file may carry a "decided"
    provenance note next to the rule keys, and nothing else."""
    raw = json.loads(Path(path).read_text())
    if set(raw) - {"decided"} != RULE_KEYS:
        raise ValueError(f"{path}: rule keys must be {sorted(RULE_KEYS)} (plus an optional 'decided'), got {sorted(raw)}")
    return check_rule({k: raw[k] for k in RULE_KEYS})


def select(rows: list[dict], rule: dict) -> dict:
    """The selected cell: lowest combined val MAE; among cells within tie_relative of it (score <= best *
    (1 + tie_relative)), the first in
    simplicity_order. Every cell must have exactly one row per seed, so no cell is judged on fewer runs."""
    check_rule(rule)
    by_cell: dict[tuple, dict[int, float]] = {}
    for r in rows:
        cell = by_cell.setdefault((r["pre_agg"], r["readout"]), {})
        if r["seed"] in cell:
            raise ValueError(f"two rows for cell {(r['pre_agg'], r['readout'])} seed {r['seed']}")
        cell[r["seed"]] = r["val_mae"]
    if sorted(by_cell) != sorted(AGGREGATOR_GRID) or any(sorted(v) != list(SEEDS) for v in by_cell.values()):
        raise ValueError(f"the grid needs every cell with seeds {SEEDS}; got "
                         f"{ {c: sorted(v) for c, v in sorted(by_cell.items())} }")
    combine = COMBINE[rule["seed_combination"]]
    score = {c: combine([v[s] for s in SEEDS]) for c, v in by_cell.items()}
    best = min(score.values())
    tied = [c for c in map(tuple, rule["simplicity_order"]) if score[c] <= best * (1 + rule["tie_relative"])]
    choice = tied[0]
    return {"pre_agg": choice[0], "readout": choice[1], "tied": [list(c) for c in tied],
            "scores": [{"pre_agg": c[0], "readout": c[1], "score": score[c],
                        "val_mae_by_seed": [by_cell[c][s] for s in SEEDS]} for c in AGGREGATOR_GRID]}


def freeze(selection: dict, rule: dict, grid_run: str, path: Path) -> dict:
    """Write the frozen choice to `path` once. An existing file with other contents is never overwritten."""
    frozen = {"cost_regime": GRID_REGIME, "pre_agg": selection["pre_agg"], "readout": selection["readout"],
              "rule": check_rule(rule), "grid_run": grid_run, "tied": selection["tied"],
              "scores": selection["scores"]}
    text = json.dumps(frozen, indent=1, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"{path} exists with a different choice; the grid choice is frozen once")
    path.write_text(text)
    return frozen


def frozen_aggregators(path: Path) -> dict:
    """{"pre_agg", "readout", "sha256"} from a frozen grid file, for configs of either regime."""
    text = Path(path).read_text()
    frozen = json.loads(text)
    if frozen.get("cost_regime") != GRID_REGIME:
        raise ValueError(f"{path}: the grid choice must come from the {GRID_REGIME!r} grid")
    if (frozen["pre_agg"], frozen["readout"]) not in AGGREGATOR_GRID:
        raise ValueError(f"{path}: {(frozen['pre_agg'], frozen['readout'])} is not a grid cell")
    return {"pre_agg": frozen["pre_agg"], "readout": frozen["readout"],
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
