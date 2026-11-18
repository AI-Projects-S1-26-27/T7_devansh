"""M3 typed access loss (AL1, decisions.md §21): an exploratory, deterministic, non-monotonic extension of D28.

M3 removes *access monotonicity* (attacker footholds can disappear). It keeps deterministic classical
planning: fixed positive costs, one goal fact, no time, no probabilities and no adaptive defender. The frozen
D28 model (M0) is untouched: `apply_m3` turns a finished D28 document into a new document. M0 is the
document itself, so loss-off generation stays byte-identical.

Every delete comes from one row of MECHANISMS. Each row names the attacker action, the trigger, the
defender response, the exact deleted facts, the recovery action and its cost. Nothing is deleted at random
or on a timer.

Semantics (decisions.md §21, answers to the AL1 semantic questions):
- process_kill: a remote exploit whose target is a monitored host is detected. The defender kills the
  attacking process on the *source* host, deleting the source foothold that was the action's precondition
  (unless it is the attacker's own start, which is outside the network). The new target foothold is kept, so
  the attacker can always continue forward. Recovery: re-exploit the source from a foothold still held.
- service_restart: a crash-labelled CVE (CRASH_RULE matches its NVD description) is exploited through
  service c on host t. The action succeeds, then the service dies, deleting service_up(t, c), which every later
  exploit through (t, c) needs. Recovery: restart_service(t, c), the supervisor restart the attacker waits
  for. It adds service_up(t, c) and deletes the attacker's footholds that service gave (in-process access
  dies with the restart).
- signature_block: a detected exploit of product X into a monitored host makes the defender block that
  product's exploits on every monitored host, deleting sig_free(X). Every exploit of X into a monitored host
  needs it; unmonitored hosts do not. Recovery: retool(X), an abstract fixed-cost step that makes X usable
  again (the model says nothing about how).
admin(h) and the attacker's start foothold are never deleted (no reboot in v1).

Why the signature block is needed: in a single-goal task whose actions each have one fluent precondition,
every optimal plan is a simple path that uses each fact once, right after gaining it. A delete can raise h*
only if it removes a fact needed *later*. A kill or a crash never does, so with those two alone
h*_M3(s0) = h*_M0(s0) (decisions.md §21, proposition; tested). sig_free(X) is shared by every exploit of X,
so losing it can matter later.

Cascading deletion goes only along edges declared in a DependencyRegistry. v1 declares none: the restart's
foothold delete is part of its own declared delete set. The registry exists so that a cascade can never
come from anywhere else, and it is tested on hand-built credential and session facts.
"""

import copy
import re
from collections import defaultdict, deque

from ahg.cost import AC_COSTS
from ahg.gen.pool import cve_key

RULES_VERSION = "al1-r1"

# The co-effect wording that marks an exploit as crashing its service ("cause a denial of service (crash) or
# possibly execute arbitrary code"). Read from the NVD description of the CVE actually used. Bump
# RULES_VERSION on any change.
CRASH_RULE = re.compile(r"denial[- ]of[- ]service|\bDoS\b|\bcrash|\bhangs?\b|\bhanging\b", re.I)

# Target monitored share of internal hosts per level (approved 18 Nov 2026), with a hard cap.
LEVELS = {"low": 0.20, "medium": 0.40, "high": 0.60}
MONITOR_CAP = 0.70
MECHANISM_NAMES = ("process_kill", "service_restart", "signature_block")

MECHANISMS = {
    "process_kill": {
        "attacker_action": "remote exploit from a volatile foothold on src into host t",
        "trigger": "t is monitored (zone-derived monitored set, LEVELS)",
        "defender_response": "kill the attacking process on src (IDS/EDR containment of the traffic's origin)",
        "deleted_facts": ["foothold(src, p) used as the action's precondition, if src is not the attacker's start"],
        "recovery_action": "re-exploit src from a foothold still held",
        "recovery_cost": "the re-exploit's own cost (ac: 4 or 7; uniform: 1)",
    },
    "service_restart": {
        "attacker_action": "remote exploit of a crash-labelled CVE through service c on host t",
        "trigger": "CRASH_RULE matches the CVE's NVD description",
        "defender_response": "the service crashes; it stays down until restarted",
        "deleted_facts": ["service_up(t, c)"],
        "recovery_action": "restart_service(t, c): adds service_up(t, c), deletes the footholds service c gave",
        "recovery_cost": "config restart_cost (fixed, > 0)",
    },
    "signature_block": {
        "attacker_action": "remote exploit of product X into a monitored host",
        "trigger": "t is monitored",
        "defender_response": "block exploits of X on every monitored host",
        "deleted_facts": ["sig_free(X)"],
        "recovery_action": "retool(X): adds sig_free(X)",
        "recovery_cost": "config retool_cost (fixed, > 0)",
    },
}


# ---------------------------------------------------------------- dependency registry (declared cascades only)

class DependencyRegistry:
    """Declared dependency edges between facts. Deleting a fact cascades only along these edges."""

    def __init__(self):
        self._dependents: dict[str, set[str]] = defaultdict(set)

    def declare(self, fact: str, depends_on: str) -> None:
        if fact == depends_on:
            raise ValueError(f"{fact} cannot depend on itself")
        self._dependents[depends_on].add(fact)

    def dependents(self, fact: str) -> set[str]:
        return set(self._dependents.get(fact, ()))

    def closure(self, deleted) -> set[str]:
        """The deleted facts plus everything that transitively depends on one of them (declared edges only)."""
        out, todo = set(deleted), deque(deleted)
        while todo:
            for d in self._dependents.get(todo.popleft(), ()):
                if d not in out:
                    out.add(d)
                    todo.append(d)
        return out


# ---------------------------------------------------------------- configuration

def check_config(cfg: dict) -> None:
    """Refuse a loss config that could make recovery free or monitor nearly every host."""
    if cfg["level"] not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}")
    unknown = set(cfg["mechanisms"]) - set(MECHANISM_NAMES)
    if unknown or not cfg["mechanisms"]:
        raise ValueError(f"mechanisms must be a non-empty subset of {MECHANISM_NAMES}, not {sorted(cfg['mechanisms'])}")
    for key in ("restart_cost", "retool_cost"):
        for regime, v in cfg[key].items():
            if not (isinstance(v, int) and v > 0):
                raise ValueError(f"{key}[{regime}] must be a positive integer (recovery is never free), not {v!r}")


def crashes(description) -> bool:
    text = description if isinstance(description, str) else " ".join(map(str, description or []))
    return bool(CRASH_RULE.search(text))


# ---------------------------------------------------------------- monitored hosts

def monitored_hosts(network: dict, level: str) -> list[int]:
    """DMZ and sensitive hosts always. Then single user/enterprise hosts in a fixed order: subnet-link distance to
    the sensitive subnet (BFS), subnet index, host id. Hosts are added until the share of internal hosts (internet
    excluded) reaches the level's target, never above MONITOR_CAP. Deterministic: no draw anywhere. Defenders
    monitor closest to the crown jewels first. Per host, not per subnet: whole subnets (3-5 hosts) overshoot at
    small n (decided 18 Nov 2026 after the smoke run)."""
    subnets, target = network["subnets"], LEVELS[level]
    internal = [h for s in subnets if s["zone"] != "internet" for h in s["hosts"]]
    chosen = {h for s in subnets if s["zone"] in ("dmz", "sensitive") for h in s["hosts"]}
    adj = defaultdict(set)
    for a, b in network["links"]:
        adj[a].add(b)
        adj[b].add(a)
    start = [i for i, s in enumerate(subnets) if s["zone"] == "sensitive"]
    dist = {i: 0 for i in start}
    todo = deque(start)
    while todo:
        i = todo.popleft()
        for j in sorted(adj[i]):
            if j not in dist:
                dist[j] = dist[i] + 1
                todo.append(j)
    order = sorted(((dist.get(i, 1 << 30), i, h) for i, s in enumerate(subnets)
                    if s["zone"] in ("users", "enterprise") for h in s["hosts"]))
    for _, _, h in order:
        if len(chosen) / len(internal) >= target or (len(chosen) + 1) / len(internal) > MONITOR_CAP:
            break
        chosen.add(h)
    return sorted(chosen)


# ---------------------------------------------------------------- the M0 -> M3 document transform

def apply_m3(doc: dict, cfg: dict, master: dict) -> dict:
    """A new document: `doc`'s task with the configured loss mechanisms, plus a "loss" block.

    `master` maps CVE IDs to master rows (for the crash rule's NVD description). Existing fact indices and
    the goal are unchanged; new facts (service_up, sig_free) are appended and held initially. Remote actions are
    split by (target component, crash) so the attacker can choose a non-crashing CVE where one exists. A
    crashing variant that costs at least as much as a non-crashing one is dominated and dropped.
    """
    check_config(cfg)

    mech = set(cfg["mechanisms"])
    out = copy.deepcopy(doc)
    t = out["task"]
    facts = list(t["facts"])
    fid = {f: i for i, f in enumerate(facts)}
    monitored = set(monitored_hosts(doc["network"], cfg["level"]))
    comps = {h["id"]: h["components"] for h in doc["metadata"]["hosts"]}
    match_of = {(m["host"], m["component"], m["cve"]): m for m in doc["metadata"]["matches"]}
    start_fact = 0  # foothold(h0,p0), the attacker's start (generate: init = [0])

    def new_fact(name: str) -> int:
        if name not in fid:
            fid[name] = len(facts)
            facts.append(name)
        return fid[name]

    # Pass 1: split every remote action by (target component, crash flag).
    variants = []  # (orig index, pre, add, comp, crash, ac, cves)
    for i, (a, meta) in enumerate(zip(doc["task"]["actions"], doc["metadata"]["actions"])):
        if meta["kind"] != "remote":
            variants.append((i, a["pre"], a["add"], None, False, meta["ac"], meta["cves"]))
            continue
        groups = defaultdict(list)
        for h, c, cve in meta["matches"]:
            crash = "service_restart" in mech and crashes(master[cve].get("description"))
            groups[(h, c, crash)].append(cve)
        best = {}
        for (h, c, crash), cves in groups.items():
            ac = "L" if any(match_of[(h, c, x)]["ac"] == "L" for x in cves) else "H"
            best[(h, c, crash)] = (ac, sorted(cves, key=cve_key))
        for (h, c, crash), (ac, cves) in sorted(best.items()):
            safe = best.get((h, c, False))
            if crash and safe is not None and AC_COSTS[safe[0]] <= AC_COSTS[ac]:
                continue  # dominated: same effect, extra delete, no cheaper
            variants.append((i, a["pre"], a["add"], (h, c), crash, ac, cves))

    crash_services = sorted({v[3] for v in variants if v[4]})
    up = {hc: new_fact(f"service_up(h{hc[0]},c{hc[1]})") for hc in crash_services}
    products = sorted({comps[v[3][0]][v[3][1]]["product"] for v in variants
                       if v[3] is not None and "signature_block" in mech and v[3][0] in monitored})
    sig = {x: new_fact(f"sig_free({x})") for x in products}

    actions, ameta = [], []
    for i, pre, add, hc, crash, ac, cves in variants:
        a, meta = doc["task"]["actions"][i], doc["metadata"]["actions"][i]
        if hc is None:  # local escalation: unchanged
            actions.append(copy.deepcopy(a))
            ameta.append({**copy.deepcopy(meta), "action": len(ameta), "m0_action": i, "loss": []})
            continue
        h, c = hc
        pre, dele, losses = list(pre), [], []
        if hc in up:
            pre.append(up[hc])
            if crash:
                dele.append(up[hc])
                losses.append("service_restart")
        if h in monitored:
            # never the attacker's own start; never a fact the action itself re-adds (an intra-host move)
            if "process_kill" in mech and pre[0] != start_fact and pre[0] not in add:
                dele.append(pre[0])
                losses.append("process_kill")
            if "signature_block" in mech:
                x = sig[comps[h][c]["product"]]
                pre.append(x)
                dele.append(x)
                losses.append("signature_block")
        suffix = f":c{c}" + (":crash" if crash else "")
        actions.append({"name": f"{a['name']}{suffix}", "pre": pre, "add": list(add),
                        **({"del": dele} if dele else {}),
                        "cost": {"ac": AC_COSTS[ac], "uniform": 1}})
        ameta.append({"action": len(ameta), "m0_action": i, "kind": "remote", "ac": ac, "cves": cves,
                      "cost_cve": min((x for x in cves if match_of[(h, c, x)]["ac"] == ac), key=cve_key),
                      "target": [h, c], "crash": crash, "loss": losses,
                      "matches": [[h, c, x] for x in cves]})

    # Recovery actions.
    for (h, c), f in up.items():
        gives = sorted({x for a in actions if f in a["pre"] for x in a["add"] if facts[x].startswith("foothold(")})
        actions.append({"name": f"restart_service h{h} c{c}", "pre": [], "add": [f], "del": gives,
                        "cost": dict(cfg["restart_cost"])})
        ameta.append({"action": len(ameta), "kind": "recovery", "mechanism": "service_restart", "target": [h, c]})
    for x, f in sig.items():
        actions.append({"name": f"retool {x}", "pre": [], "add": [f], "cost": dict(cfg["retool_cost"])})
        ameta.append({"action": len(ameta), "kind": "recovery", "mechanism": "signature_block", "product": x})

    t["facts"] = facts
    t["actions"] = actions
    t["init"] = sorted(set(t["init"]) | set(up.values()) | set(sig.values()))
    out["metadata"]["actions"] = ameta
    internal = [h["id"] for h in doc["network"]["hosts"] if h["zone"] != "internet"]
    out["loss"] = {
        "rules_version": RULES_VERSION, "level": cfg["level"], "mechanisms": sorted(mech),
        "restart_cost": cfg["restart_cost"], "retool_cost": cfg["retool_cost"],
        "monitored_hosts": sorted(monitored), "monitored_share": len(monitored) / len(internal),
        "crash_services": [list(hc) for hc in crash_services], "signature_products": products,
        "loss_actions_total": sum(bool(a.get("del")) for a in actions if not a["name"].startswith(("restart", "retool"))),
        "deleted_facts_total": sum(len(a.get("del", ())) for a in actions),
        "recovery_actions": sum(a["name"].startswith(("restart", "retool")) for a in actions),
    }
    return out
