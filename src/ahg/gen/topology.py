"""D28 topology archetypes (P6, decided 16 Nov 2026). Replaces the D10 families on the D28 path only;
ahg.topology and its goldens are untouched.

Two archetypes, both from published benchmark generators:

- "tree": the network generator of Speicher, Steinmetz, Hoffmann et al. (2019), arXiv:1705.05088,
  Appendix A [VERIFY title]; NASim (Schwartz and Kurniawati) generates its scenarios after it. Zones: internet (one host,
  the attacker's start), DMZ, sensitive, users. DMZ and sensitive get max(1, round(n/40)) hosts each; the
  remaining hosts are users, in subnets of at most 5 hosts arranged as a binary tree. Links: internet-DMZ,
  DMZ-sensitive, every user subnet to its parent subnet and to sensitive, the root user subnet to DMZ.
- "chain": the user / enterprise / operational subnet layout of the CAGE Challenge 2 network
  (github.com/cage-challenge/cage-challenge-2, 2022), adapted to an internet start: internet - DMZ (1 host) - users - enterprise - sensitive
  (1 host). Enterprise gets ceil(remaining/3) hosts, users the rest.

n counts the internal hosts (the internet host is extra). round() rounds halves up (OQ-P6-1).

Firewall (Speicher et al.'s "fraction of ports"): traffic inside a subnet is open. Every link between two
subnets has two directions; each direction allows each service family present on its far side
independently with probability 0.5, drawn once per link and direction. Routing is not transitive: a host
reaches a service only in its own subnet or across one link that allows the service's family (OQ-P6-2).
The internet host runs nothing, so nothing is drawn towards it. Solvability is never forced.

Every number above is a fixed declared parameter (configs/gen/p6.json), never swept.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

ARCHETYPES = ("tree", "chain")


@dataclass
class Subnet:
    name: str
    zone: str  # "internet" | "dmz" | "sensitive" | "users" | "enterprise"
    hosts: list[int] = field(default_factory=list)


@dataclass
class Layout:
    """Zones, subnets and undirected links before any host is configured."""

    archetype: str
    n: int
    subnets: list[Subnet]
    links: list[tuple[int, int]]  # subnet index pairs (a < b), sorted

    def subnet_of(self) -> dict[int, int]:
        return {h: i for i, s in enumerate(self.subnets) for h in s.hosts}

    def zone_of(self) -> dict[int, str]:
        return {h: s.zone for s in self.subnets for h in s.hosts}


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def _split(total: int, cap: int) -> list[int]:
    """`total` hosts in ceil(total/cap) subnets, sizes as equal as possible, larger ones first."""
    k = math.ceil(total / cap)
    return [total // k + (1 if i < total % k else 0) for i in range(k)]


def layout(archetype: str, n: int, params: dict) -> Layout:
    """The deterministic part of a topology: zone sizes, subnets, links (no randomness here)."""
    if archetype not in ARCHETYPES:
        raise ValueError(f"archetype must be one of {ARCHETYPES}, not {archetype!r}")
    subnets = [Subnet("internet", "internet", [0])]
    nxt = 1

    def add(name: str, zone: str, size: int) -> int:
        nonlocal nxt
        subnets.append(Subnet(name, zone, list(range(nxt, nxt + size))))
        nxt += size
        return len(subnets) - 1

    links: list[tuple[int, int]] = []
    if archetype == "tree":
        p = params["tree"]
        per = max(1, round_half_up(n / p["zone_divisor"]))
        users = n - 2 * per
        if users < 1:
            raise ValueError(f"tree needs at least one user host; n={n} is too small")
        dmz, sens = add("dmz", "dmz", per), add("sensitive", "sensitive", per)
        user_ids = [add(f"users{i}", "users", size) for i, size in enumerate(_split(users, p["user_subnet_max"]))]
        links += [(0, dmz), (dmz, sens), (dmz, user_ids[0])]
        for i, s in enumerate(user_ids):
            if i > 0:
                links.append((user_ids[(i - 1) // 2], s))  # binary tree: subnet i's parent is (i-1)//2
            links.append((sens, s) if sens < s else (s, sens))
    else:
        p = params["chain"]
        rest = n - p["dmz_hosts"] - p["sensitive_hosts"]
        ent = math.ceil(rest / p["enterprise_share_divisor"])
        if rest - ent < 1:
            raise ValueError(f"chain needs at least one user and one enterprise host; n={n} is too small")
        dmz = add("dmz", "dmz", p["dmz_hosts"])
        usr = add("users", "users", rest - ent)
        ent_i = add("enterprise", "enterprise", ent)
        sens = add("sensitive", "sensitive", p["sensitive_hosts"])
        links += [(0, dmz), (dmz, usr), (usr, ent_i), (ent_i, sens)]
    if nxt - 1 != n:
        raise AssertionError(f"layout placed {nxt - 1} hosts, expected {n}")
    return Layout(archetype, n, subnets, sorted(set(links)))


def draw_firewall(lay: Layout, families_of: dict[int, list[str]], rng: random.Random, prob: float
                  ) -> dict[tuple[int, int], list[str]]:
    """Allowed families per directed subnet pair (src_subnet, dst_subnet), drawn once per link and direction.

    Draw order is fixed: links in sorted order, direction a->b then b->a, families sorted. A direction whose
    far side runs nothing (the internet) consumes no draws.
    """
    out = {}
    for a, b in lay.links:
        for src, dst in ((a, b), (b, a)):
            far = sorted({f for h in lay.subnets[dst].hosts for f in families_of[h]})
            out[(src, dst)] = [f for f in far if rng.random() < prob]
    return out


def can_reach(lay: Layout, firewall: dict[tuple[int, int], list[str]], src: int, dst: int, family: str) -> bool:
    """Can host src open a connection to host dst's service of `family`? (same subnet, or one allowing link)"""
    sub = lay.subnet_of()
    a, b = sub[src], sub[dst]
    if src == dst:
        return False
    if a == b:
        return True
    return family in firewall.get((a, b), ())
