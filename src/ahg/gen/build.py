"""From a configured network to a delete-free planning task (P6, decided 16 Nov 2026).

Facts (the attacker's privileges; no trust, logins, credential levels, coupling or PR-gating on this path):
    foothold(h0,p0)   the attacker on the internet host; the only initial fact (p0 = "attacker")
    foothold(h,pK)    code execution on host h as principal K (a cited run-as identity, or UNKNOWN)
    admin(h)          root / SYSTEM on host h
Principals appear in fact names only as indices; their names live in the metadata block.

Actions (every one tied to CVEs; Round 5 "Actions not tied to a CVE"):
- remote: a foothold on a host that can reach the service through the firewall rules (ahg.gen.topology)
  -> foothold(target, principal of the vulnerable service), plus admin(target) only if that principal is
  root or SYSTEM by citation (configs/gen/principals.json). One transition per source foothold fact.
- local: any foothold on the host -> admin(host). One transition per foothold fact on the host (OQ-CA-1).
  The pool data records no CVE-specific prerequisite beyond the foothold, so none is added (OQ-P6-5).
Identical transitions (same preconditions and effect) are merged: the merged action keeps the cheapest AC
of its CVEs, ties to the lowest numeric CVE ID, and keeps every CVE as metadata (Round 5 "Duplicates").
Each action carries both regime costs: ac (AC:L = 4, AC:H = 7) and uniform (1).

Action names are "<kind> a<index>" and never derive from CVE IDs; CVE, product, version and CPE data are
kept in the metadata block, which the planner and the features never read.

Every action traces to match records: (host, component, CVE) found by the primary CPE evaluator
(ahg.gen.cpe) and confirmed by the independent one (ahg.gen.cpe_check); a disagreement aborts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ahg.cost import AC_COSTS
from ahg.gen import cpe as cpe_eval
from ahg.gen import cpe_check
from ahg.gen.hosts import Component, HostConfig
from ahg.gen.pool import Unit, cve_key
from ahg.gen.topology import Layout, can_reach

ATTACKER = "attacker"
UNKNOWN = "UNKNOWN"


class EvaluatorDisagreement(AssertionError):
    """The two CPE evaluators disagree on a (host, component, CVE): generation must stop."""


def label_product(host: HostConfig, comp: Component) -> str:
    """The pool's product name for a component (the OS component is 'windows_os' on Windows)."""
    return "windows_os" if comp.family == "os" and host.os == "windows" else comp.product


_CACHE: dict[tuple, list[dict]] = {}


def _criteria_products(configs: list[dict]) -> set[tuple[str, ...]] | None:
    """(part, vendor, product) of every criteria in a CVE's configurations; None if one is a wildcard."""
    out = set()
    stack = [n for c in configs for n in c.get("nodes", [])]
    while stack:
        node = stack.pop()
        for m in node.get("cpeMatch", []):
            key = tuple(m["criteria"].split(":")[2:5])
            if "*" in key:
                return None
            out.add(key)
        stack += [*node.get("nodes", []), *node.get("children", [])]
    return out


def evaluate(cve_id: str, configs: list[dict], host: list[str], comp: list[str]) -> list[dict]:
    """Both evaluators on one (host, component, CVE); raises EvaluatorDisagreement if they differ.

    Host names whose (part, vendor, product) no criteria of the CVE names cannot satisfy any cpeMatch, so
    both evaluators are given only the others (unless a criteria wildcards one of those fields). Results are
    cached on that reduced input, which is all either evaluator reads.
    """
    keys = _criteria_products(configs)
    rel = tuple(host) if keys is None else tuple(n for n in host if tuple(n.split(":")[2:5]) in keys)
    key = (cve_id, rel, tuple(comp))
    if key not in _CACHE:
        ev = cpe_eval.component_matches(configs, list(rel), list(comp))
        ev2 = cpe_check.applies(configs, list(rel), list(comp))
        if [e["configuration"] for e in ev] != ev2:
            raise EvaluatorDisagreement(f"{cve_id} on {comp}: primary {[e['configuration'] for e in ev]} vs second {ev2}")
        _CACHE[key] = ev
    return _CACHE[key]


def match_records(hosts: dict[int, HostConfig], units: list[Unit], master: dict[str, dict]) -> list[dict]:
    """Every (host, component, CVE) match, checked by both evaluators. Sorted by host, component, CVE."""
    by: dict[tuple[str, str], list[Unit]] = defaultdict(list)
    for u in units:
        by[(u.product, u.family)].append(u)
    out = []
    for h in sorted(hosts):
        cfg = hosts[h]
        names = cfg.cpe_names()
        for ci, comp in enumerate(cfg.components):
            for u in by.get((label_product(cfg, comp), comp.family), []):
                ev = evaluate(u.cve_id, master[u.cve_id]["configurations"], names, list(comp.fixture.cpe_names))
                if ev:
                    out.append({"host": h, "component": ci, "family": comp.family, "product": comp.product,
                                "version": comp.fixture.version, "cve": u.cve_id, "kind": u.kind, "ac": u.ac,
                                "unit_product": u.product, "evidence": ev})
    return out


@dataclass
class Built:
    facts: list[str]
    actions: list[dict]  # {"name", "kind", "pre": [i], "add": [i], "cost": {"ac", "uniform"}}
    action_meta: list[dict]  # per action: the merged CVEs and the match records behind it
    principals: list[str]
    goal_fact: int
    merges: int  # transitions folded into another one with the same pre and add


def principal_of(table: dict, product: str, family: str) -> str:
    entry = table["entries"].get(f"{product}|{family}")
    return entry["principal"] if entry else UNKNOWN


def build_task(lay: Layout, hosts: dict[int, HostConfig], firewall: dict, matches: list[dict], table: dict,
               admin_principals: set[str], goal_host: int) -> Built:
    """Facts, merged actions and the goal fact for one configured network."""
    remote = [m for m in matches if m["kind"] == "remote"]
    local = [m for m in matches if m["kind"] == "local"]
    principal = {(m["host"], m["component"]): principal_of(table, m["unit_product"], m["family"]) for m in remote}
    principals = [ATTACKER] + sorted({UNKNOWN} | set(principal.values()))  # fixed: p0 attacker, then by name
    pidx = {p: i for i, p in enumerate(principals)}
    # foothold facts that can exist: the attacker's, and one per (host, principal) some remote match gives
    foot = sorted({(0, 0)} | {(h, pidx[p]) for (h, _), p in principal.items()})
    admin_hosts = sorted({goal_host} | {m["host"] for m in local}
                         | {h for (h, _), p in principal.items() if p in admin_principals})
    facts = [f"foothold(h{h},p{k})" for h, k in foot] + [f"admin(h{h})" for h in admin_hosts]
    fid = {f: i for i, f in enumerate(facts)}
    foot_on = defaultdict(list)
    for h, k in foot:
        foot_on[h].append(fid[f"foothold(h{h},p{k})"])
    trans: dict[tuple, list[dict]] = defaultdict(list)  # (kind, pre, add) -> match records
    for m in remote:
        t, comp = m["host"], m["component"]
        fam = hosts[t].components[comp].family
        p = principal[(t, comp)]
        add = [fid[f"foothold(h{t},p{pidx[p]})"]] + ([fid[f"admin(h{t})"]] if p in admin_principals else [])
        for src in sorted(foot_on):
            if not can_reach(lay, firewall, src, t, fam):
                continue
            for f in foot_on[src]:
                trans[("remote", (f,), tuple(sorted(add)))].append(m)
    for m in local:
        h = m["host"]
        for f in foot_on.get(h, []):
            trans[("local", (f,), (fid[f"admin(h{h})"],))].append(m)
    actions, meta, merges = [], [], 0
    for i, ((kind, pre, add), ms) in enumerate(sorted(trans.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1]))):
        cves = sorted({m["cve"] for m in ms}, key=cve_key)
        ac_of = {m["cve"]: m["ac"] for m in ms}
        best_ac = "L" if "L" in ac_of.values() else "H"
        chosen = min((c for c in cves if ac_of[c] == best_ac), key=cve_key)
        merges += len(ms) - 1
        actions.append({"name": f"{kind} a{i}", "pre": list(pre), "add": list(add),
                        "cost": {"ac": AC_COSTS[best_ac], "uniform": 1}})
        meta.append({"action": i, "kind": kind, "ac": best_ac, "cost_cve": chosen, "cves": cves,
                     "matches": sorted({(m["host"], m["component"], m["cve"]) for m in ms}, key=lambda x: (x[0], x[1], cve_key(x[2])))})
    return Built(facts, actions, meta, principals, fid[f"admin(h{goal_host})"], merges)


def reachable_facts(n_facts: int, actions: list[dict], init: list[int]) -> set[int]:
    """Delete-free reachability fixpoint (the exact pre-check, h_max < infinity)."""
    held = set(init)
    changed = True
    while changed:
        changed = False
        for a in actions:
            if all(p in held for p in a["pre"]) and not set(a["add"]) <= held:
                held |= set(a["add"])
                changed = True
    return held


def unsolvable_reason(built: Built, goal_host: int, held: set[int]) -> str | None:
    """None if the goal is reachable; otherwise why not, from the most to the least basic cause."""
    g = built.goal_fact
    if g in held:
        return None
    if not any(g in a["add"] for a in built.actions):
        return "no_action_gives_admin_on_goal_host"
    on_goal = [i for i, f in enumerate(built.facts) if f.startswith(f"foothold(h{goal_host},")]
    if not any(i in held for i in on_goal):
        return "no_foothold_on_goal_host_reachable"
    return "admin_on_goal_host_needs_an_unreached_foothold"
