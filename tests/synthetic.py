"""A tiny synthetic task set for testing the D28 training plumbing (P9a) without the new attack model.

Each task is a small random delete-free task whose actions carry an explicit CVSS AC ("L" or "H"), priced
by cost.regime_cost. The structure and the ACs come from one rng stream that ignores the regime, so the
`ac` and `uniform` versions of a seed are the same instance with different costs (D28 Round 5).
"""

import random

from ahg.attack_graph import Instance
from ahg.cost import regime_cost
from ahg.dataset import Example
from ahg.labels import label_instance
from ahg.oracle import optimal_cost
from ahg.onpolicy import make_batch
from ahg.task import Task

from test_labels import CFG as LABEL_CFG

PROTOCOL = {"search_budget": 2_000, "state_budget": 20_000, "oracle_max_pops": 200_000, "enum_cap": 20_000}


def synthetic_task(seed: int, regime: str, n_facts: int = 14, n_actions: int = 26, n_goal: int = 2) -> Task:
    rng = random.Random(f"synthetic:{seed}")
    facts = [f"f{i}" for i in range(n_facts)]
    actions = []
    for i in range(n_actions):
        pre = rng.sample(facts[: n_facts - n_goal], rng.randint(1, 2))  # goal facts are never preconditions
        add = rng.sample(facts[1:], rng.randint(1, 2))
        actions.append((f"act{i}", pre, add, regime_cost(regime, rng.choice("LH"))))
    return Task.from_names(facts, ["f0"], facts[-n_goal:], actions)


def synthetic_seeds(count: int, start: int = 0, min_steps: int = 3) -> list[int]:
    """The first `count` seeds from `start` whose task is solvable in at least `min_steps` actions, so the
    searches on it expand more than a handful of states (the regime does not change solvability)."""
    out, seed = [], start
    while len(out) < count:
        h = optimal_cost(synthetic_task(seed, "uniform"))  # uniform h* = the fewest actions to the goal
        if h not in (None, float("inf")) and h >= min_steps:
            out.append(seed)
        seed += 1
    return out


def instance(seed: int, regime: str) -> Instance:
    # Action names carry no CVE kind, so the GNN's action features hold only the cost (no PR one-hot).
    return Instance(network=None, cves={}, target=0, task=synthetic_task(seed, regime))


def offline_batch(batch_id: str, split: str, regime: str, seeds: list[int]):
    """The five-source offline labels (labels.label_instance) of the given seeds, as an in-memory batch."""
    instances, examples, failures = [], [], []
    for i, seed in enumerate(seeds):
        inst = instance(seed, regime)
        out = label_instance(inst.task, f"synthetic:{seed}", 1, LABEL_CFG)  # n = 1: every label oracle-checked
        instances.append(inst)
        examples += [Example(i, s, h, src) for s, h, src in out["rows"]]
        failures += out["failures"]
    return make_batch(batch_id, split, regime, instances, examples, failures)
