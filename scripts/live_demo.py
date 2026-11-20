"""Live demo: build a FRESH attack-graph network and run A* with every heuristic on it, at the frozen tag's code.

Run from the repository root (the checkpoints are tracked under results/training/d23t1):
  uv run python scripts/live_demo.py --family hub_and_spoke --seed 90129
  uv run python scripts/live_demo.py --family tiered --seed 90214
  uv run python scripts/live_demo.py --family flat --random

Seeds >= 90000 lie outside every split (train 1000-1599, 10000-12399; val/test 1600-2199), so no test data is read.
The instance is built exactly like a split instance, with the train CVE pool. The optimal cost is certified by
A*+LM-cut (admissible), and every learned run reports its cost relative to that optimum.
"""

import argparse
import random
import time
from pathlib import Path

import torch

from ahg.attack_graph import KnobInfeasible, UnreachableGoal, make_instance
from ahg.cve import load_table
from ahg.heuristics import HEURISTICS
from ahg.labels import CVES
from ahg.learning.heuristic import Learned
from ahg.learning.train import load
from ahg.search import best_first_search
from ahg.splits import COUPLINGS, cve_pool

MODELS = {"A": "results/training/d23t1/gnnres-64x3-lr0.001-rank-s0/2026-10-31T084318Z-0126a7d",
          "B": "results/training/d23t1/gnnres-64x3-lr0.001-mse-s0/2026-10-31T094018Z-0126a7d"}
LABEL = {"h0": "normal A* (h0)", "hsp": "A* + h_sp", "hmax": "A* + h_max", "hadd": "A* + h_add (inadm.)",
         "lmcut": "A* + LM-cut", "A": "A* + GNN Model A", "B": "A* + GNN Model B"}
BUDGET = 200_000


def build(family: str, seed: int, n: int, D: int, pool):
    """Same knobs as trace_demo.py; returns None when the seed gives no coupled, solvable instance."""
    rho, m = COUPLINGS[seed % len(COUPLINGS)]
    if rho == 0.0:
        return None
    try:
        return make_instance(family, seed, n, pool, cred_levels=D, coupling=rho, share_pool=m, exposed_level0=True)
    except (UnreachableGoal, KnobInfeasible):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="hub_and_spoke", choices=["flat", "tiered", "hub_and_spoke"])
    ap.add_argument("--seed", type=int, default=90129)
    ap.add_argument("--random", action="store_true", help="pick a fresh seed >= 90000 at random")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--D", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(1)
    pool = cve_pool(load_table(CVES), "train")
    learned = {k: Learned(*load(Path.cwd() / p / "model.pt"), k) for k, p in MODELS.items()}

    seed = random.randint(90000, 99999) if args.random else args.seed
    inst = build(args.family, seed, args.n, args.D, pool)
    while inst is None:  # skip seeds with no coupling or no reachable goal (counted, not hidden)
        print(f"seed {seed}: no coupled solvable instance, trying {seed + 1}")
        seed += 1
        inst = build(args.family, seed, args.n, args.D, pool)
    task = inst.task
    print(f"\nnetwork: {args.family}, seed {seed}, {args.n} hosts, {args.D} credential tiers, "
          f"{len(task.actions)} actions, {len(inst.coupled)} coupled exploits, goal host {inst.target}\n")

    rows = []
    for name in ("h0", "hsp", "hmax", "hadd", "lmcut", "A", "B"):
        h = HEURISTICS[name] if name in HEURISTICS else learned[name].bind(inst)
        t0 = time.perf_counter()
        r = best_first_search(task, h, max_expansions=BUDGET)
        dt = time.perf_counter() - t0
        rows.append((name, r, dt))
    opt = next(r.cost for n, r, _ in rows if n == "lmcut")  # LM-cut is admissible: A* returns the optimum

    print(f"{'search':22s} {'expanded':>9s} {'cost':>6s} {'cost/opt':>9s} {'time ms':>9s}")
    for name, r, dt in rows:
        cost = r.cost if r.solved else None
        ratio = f"{cost / opt:.3f}" if cost is not None else "-"
        print(f"{LABEL[name]:22s} {r.expanded:>9,d} {str(cost):>6s} {ratio:>9s} {dt * 1000:>9.1f}")

    plan = next(r for n, r, _ in rows if n == "A").plan
    print("\nattack found by A* + GNN Model A:")
    for i, a in enumerate(plan, 1):
        print(f"  {i:2d}. {task.actions[a].name}")


if __name__ == "__main__":
    main()
