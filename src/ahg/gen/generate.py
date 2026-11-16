"""D28 generator entry point (P6): one seeded call gives one task document, byte-identical across runs.

    ctx = Context.load()                      # pinned inputs, SHA-256-checked
    doc = generate(ctx, "tree", "p6-smoke-0", 12, cohort="core", os_mix={"linux": .5, "windows": .5}, p=1.0)
    text = dumps(doc)                         # canonical JSON (sorted keys, no spaces)

Seeding: one rng stream "archetype:seed" draws the host configurations and then the firewall, in a fixed
order; the goal host (if the sensitive zone has several) comes from its own stream "goal:archetype:seed".
Training uses "tree"; "chain" is the held-out structure (Round 10).

The document has a planner-visible part ("task", "network") and a "metadata" block (CVEs, products,
versions, CPE names, match records, principals by name) that the planner and the features never read.
Unsolvable tasks are returned like any other, flagged with their reason; solvability is never forced.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ahg import pddl, refsolver
from ahg.cost import check_regime
from ahg.gen import build, hosts as hosts_mod, inventory, pool, topology
from ahg.task import Action, Task, to_mask

CONFIG = pool.ROOT / "configs" / "gen" / "p6.json"


def dumps(doc: dict) -> str:
    """Canonical JSON: the byte-identity the determinism tests compare."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


@dataclass
class Context:
    """Pinned inputs, loaded once; inventories are cached per (cohort, p, exclusions)."""

    cfg: dict
    units: list[pool.Unit]
    funnel: dict
    master: dict[str, dict]
    catalogue: dict
    principals: dict
    input_hashes: dict
    _inv: dict = field(default_factory=dict)

    @classmethod
    def load(cls, config: Path = CONFIG, root: Path = pool.ROOT) -> "Context":
        cfg = json.loads(config.read_text())
        units, funnel, master = pool.load_pool(cfg, root)
        cat = inventory.load_catalogue(cfg, root)
        ppath = root / cfg["principals"]
        table = json.loads(ppath.read_text())
        hashes = {"config": pool.sha256_file(config), "master": cfg["master"]["sha256"],
                  "catalogue": cfg["catalogue"]["sha256"], "labels": cfg["labels"],
                  "principals": pool.sha256_file(ppath)}
        return cls(cfg, units, funnel, master, cat, table, hashes)

    def run_units(self, cohort: str, excluded: frozenset[str]) -> list[pool.Unit]:
        return pool.restrict(self.units, self.master, cohort, excluded)

    def inventory(self, cohort: str, p: float, excluded: frozenset[str]) -> inventory.Inventory:
        key = (cohort, p, excluded)
        if key not in self._inv:
            self._inv[key] = inventory.build_inventory(self.cfg, self.catalogue, self.run_units(cohort, excluded),
                                                       self.master, cohort, p, excluded)
        return self._inv[key]


def check_params(os_mix: dict[str, float], p: float, cfg: dict) -> None:
    if set(os_mix) != {"linux", "windows"} or abs(sum(os_mix.values()) - 1) > 1e-9 or min(os_mix.values()) < 0:
        raise ValueError(f"os_mix must give linux and windows probabilities summing to 1, not {os_mix}")
    if p not in cfg["p_values"]:
        raise ValueError(f"p must be one of {cfg['p_values']}, not {p}")


def generate(ctx: Context, archetype: str, seed: str, n: int, cohort: str = "core",
             os_mix: dict[str, float] | None = None, p: float = 1.0, excluded: frozenset[str] = frozenset()) -> dict:
    cfg = ctx.cfg
    os_mix = dict(os_mix or cfg["os_mix_default"])
    check_params(os_mix, p, cfg)
    excluded = frozenset(excluded)
    lay = topology.layout(archetype, n, cfg["topology"])
    rng = random.Random(f"{archetype}:{seed}")
    inv = ctx.inventory(cohort, p, excluded)
    roles = [(h, s.zone) for s in lay.subnets if s.zone != "internet" for h in s.hosts]
    configs = hosts_mod.configure(rng, inv, roles, cfg["roles"], os_mix, tuple(cfg["families_per_host"]),
                                  cfg["reuse_alpha"])
    families_of = {0: []} | {h: c.families for h, c in configs.items()}
    firewall = topology.draw_firewall(lay, families_of, rng, cfg["topology"]["firewall_allow_probability"])
    matches = build.match_records(configs, ctx.run_units(cohort, excluded), ctx.master)
    sensitive = sorted(h for s in lay.subnets if s.zone == "sensitive" for h in s.hosts)
    goal_host = sensitive[0] if len(sensitive) == 1 else random.Random(f"goal:{archetype}:{seed}").choice(sensitive)
    built = build.build_task(lay, configs, firewall, matches, ctx.principals, set(cfg["admin_principals"]), goal_host)
    held = build.reachable_facts(len(built.facts), built.actions, [0])
    reason = build.unsolvable_reason(built, goal_host, held)
    zone = lay.zone_of()
    remote_m = [m for m in matches if m["kind"] == "remote"]
    unknown = sum(build.principal_of(ctx.principals, m["unit_product"], m["family"], m["cve"]) == build.UNKNOWN
                  for m in remote_m)
    unverified = any(x.fixture.unverified for c in configs.values() for x in c.components)
    return {
        "schema": cfg["schema"],
        "params": {"archetype": archetype, "seed": seed, "n": n, "cohort": cohort, "os_mix": os_mix, "p": p,
                   "excluded": sorted(excluded)},
        "network": {
            "hosts": [{"id": 0, "zone": "internet", "subnet": 0, "families": []}]
                     + [{"id": h, "zone": zone[h], "subnet": lay.subnet_of()[h], "families": configs[h].families}
                        for h in sorted(configs)],
            "subnets": [{"name": s.name, "zone": s.zone, "hosts": s.hosts} for s in lay.subnets],
            "links": [list(l) for l in lay.links],
            "firewall": [{"from": a, "to": b, "allow": fams} for (a, b), fams in sorted(firewall.items())]},
        "task": {"facts": built.facts, "init": [0], "goal": [built.goal_fact], "actions": built.actions},
        "solvable": reason is None,
        "unsolvable_reason": reason,
        "flags": {"unverified_release": unverified},
        "stats": {"hosts_per_zone": dict(sorted(Counter(zone[h] for h in configs).items())),
                  "actions": len(built.actions),
                  "remote_actions": sum(a["name"].startswith("remote ") for a in built.actions),
                  "local_actions": sum(a["name"].startswith("local ") for a in built.actions),
                  "merged_transitions": built.merges, "matches": len(matches),
                  "remote_matches": len(remote_m), "remote_matches_unknown_principal": unknown,
                  "reused_configs": sum(c.copied_from is not None for c in configs.values())},
        "metadata": {
            "inputs": ctx.input_hashes,
            "goal_host": goal_host,
            "principals": built.principals,
            "hosts": [{"id": h, "os": c.os, "copied_from": c.copied_from,
                       "components": [{"family": x.family, "product": x.product, "version": x.fixture.version,
                                       "release_line": x.fixture.release_line, "kbs": list(x.fixture.kbs),
                                       "cpe_names": list(x.fixture.cpe_names),
                                       "status": x.status, "unverified_release": x.fixture.unverified,
                                       "builtin": x.builtin} for x in c.components]}
                      for h, c in sorted(configs.items())],
            "matches": matches,
            "actions": [{**m, "matches": [list(t) for t in m["matches"]]} for m in built.action_meta]},
    }


# ---------------------------------------------------------------- views of a finished document

def to_task(doc: dict, regime: str) -> Task:
    """The planner's Task under one cost regime (reads only the planner-visible "task" block)."""
    check_regime(regime)
    t = doc["task"]
    acts = [Action(a["name"], tuple(a["pre"]), tuple(a["add"]), a["cost"][regime]) for a in t["actions"]]
    return Task(list(t["facts"]), acts, to_mask(t["init"]), tuple(t["goal"]))


def neutral(doc: dict, regime: str) -> dict:
    """The reference solver's neutral format, built from the document directly (no task.py involved)."""
    t = doc["task"]
    f = t["facts"]
    return {"facts": list(f), "init": [f[i] for i in t["init"]], "goal": [f[i] for i in t["goal"]],
            "actions": [{"name": a["name"], "pre": [f[i] for i in a["pre"]], "add": [f[i] for i in a["add"]],
                         "cost": a["cost"][regime]} for a in t["actions"]]}


def export_pddl(doc: dict, regime: str) -> tuple[str, str]:
    """(domain, problem) PDDL of the finished task under one regime."""
    task = to_task(doc, regime)
    return pddl.domain_pddl(task), pddl.problem_pddl(task)


def doc_digest(doc: dict) -> str:
    return hashlib.sha256(dumps(doc).encode()).hexdigest()


def reference_cost(doc: dict, regime: str) -> float:
    """Optimal cost by the independent exhaustive reference solver (small tasks only)."""
    return refsolver.solve(neutral(doc, regime))[0]
