"""P6 generator: pool, inventory, host configuration, matching, task build, goal, determinism, exports.

Needs the pinned untracked inputs (master, frozen labels, P5 catalogue); skipped in a checkout without them.
"""

import ast
import inspect
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from ahg import pddl, refsolver
from ahg.gen import build, cpe_check, generate as gen, hosts as hosts_mod, pool
from ahg.gen.topology import layout
from ahg.heuristics import lmcut
from ahg.learning.leakage import FeaturePolicy, audit
from ahg.search import astar

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())
HAVE = all((ROOT / p).exists() for p in (CFG["master"]["path"], CFG["catalogue"]["path"],
                                         "data/d28/effect/p3-labels-v2.1/p3-labels-v2.1.jsonl.gz"))
pytestmark = pytest.mark.skipif(not HAVE, reason="pinned P6 inputs (master, labels, catalogue) not in this checkout")

HALF = {"linux": 0.5, "windows": 0.5}


@cache
def ctx():
    return gen.Context.load()


@cache
def doc(arch, seed, n, cohort="core", p=1.0, linux=0.5, excluded=()):
    return gen.generate(ctx(), arch, seed, n, cohort=cohort, os_mix={"linux": linux, "windows": 1 - linux}, p=p,
                        excluded=frozenset(excluded))


def sample():
    """A fixed spread of documents: both archetypes, small and mid sizes, both cohorts, p = 1 and 0.5."""
    out = []
    for arch in ("tree", "chain"):
        for n in (6, 12):
            for s in range(3):
                out.append(doc(arch, f"t{s}", n))
                out.append(doc(arch, f"t{s}", n, p=0.5))
        out.append(doc(arch, "t0", 6, "temporal"))
        out.append(doc(arch, "t1", 8, "temporal", p=0.75))
    return out


# ---------------------------------------------------------------- pool and funnel

def test_pool_size_and_funnel():
    c = ctx()
    f = c.funnel
    assert (f["pool"], f["remote"], f["local"]) == (1778, 115, 1663)
    assert f["pool"] + sum(f["excluded"].values()) == f["label_rows"]
    assert all(v > 0 for v in f["excluded"].values())


def test_pool_units_satisfy_every_rule():
    c = ctx()
    for u in c.units:
        vr = c.master[u.cve_id]["vector_resolution"]
        assert vr["status"] != "REVIEW_REQUIRED" and u.ac in ("L", "H")
        v = vr["values"]
        if u.kind == "remote":
            assert (v["AV"], v["PR"], v["UI"]) == ("N", "N", "N")
        else:
            assert v["AV"] == "L" and v["PR"] in ("N", "L") and v["UI"] == "N"


def test_cohort_and_exclusion_isolation():
    c = ctx()
    for cohort in ("core", "temporal"):
        assert {u.cohort for u in c.run_units(cohort, frozenset())} == {cohort}
    ex = frozenset({"apache:http_server"})
    kept = c.run_units("core", ex)
    assert kept and not any(pool.touches(c.master[u.cve_id], set(ex)) for u in kept)
    assert any(u.product == "apache:http_server" for u in c.run_units("core", frozenset()))
    for s in range(4):
        d = doc("tree", f"ex{s}", 12, excluded=tuple(ex))
        assert not any(x["product"] in ex for h in d["metadata"]["hosts"] for x in h["components"])
        assert not any(pool.touches(c.master[m["cve"]], set(ex)) for m in d["metadata"]["matches"])
        assert all(c.master[m["cve"]]["cohort"] == "core" for m in d["metadata"]["matches"])
    t = doc("tree", "t0", 6, "temporal")
    assert all(c.master[m["cve"]]["cohort"] == "temporal" for m in t["metadata"]["matches"])


def test_mysql_mysql_is_never_offered():
    from ahg.gen import inventory
    inv = ctx().inventory("core", 1.0, frozenset())
    assert not any(q.name == "mysql:mysql" for prods in inv.products.values() for q in prods)
    cat = json.loads(json.dumps(ctx().catalogue))  # a copy that does list mysql:mysql with a fixture
    cat["linux"]["mysql:mysql"] = dict(cat["linux"]["oracle:mysql"], product="mysql:mysql")
    prods, notes = inventory.products(CFG, cat)
    assert "mysql:mysql" not in {q.name for q in prods}
    assert {"product": "mysql:mysql", "dropped": CFG["never_products"]["mysql:mysql"]} in notes


# ---------------------------------------------------------------- matching

def test_every_action_traces_to_match_records():
    for d in sample():
        ms = {(m["host"], m["component"], m["cve"]): m for m in d["metadata"]["matches"]}
        facts = d["task"]["facts"]
        for a, meta in zip(d["task"]["actions"], d["metadata"]["actions"]):
            assert meta["matches"] and a["name"].split()[0] == meta["kind"]
            target = {int(re.match(r"\w+\(h(\d+)", facts[i]).group(1)) for i in a["add"]}
            for h, comp, cve in meta["matches"]:
                m = ms[(h, comp, cve)]
                assert m["kind"] == meta["kind"] and {h} == target and cve in meta["cves"]
            assert set(meta["cves"]) == {t[2] for t in meta["matches"]}


def test_match_set_equals_an_independent_recheck():
    """Every pool unit of every component re-evaluated with the second evaluator alone, uncached: the
    matches are exactly the document's (no CVE on a non-matching host, none missing)."""
    c = ctx()
    for d in sample()[:12]:
        units = c.run_units(d["params"]["cohort"], frozenset(d["params"]["excluded"]))
        want = set()
        for h in d["metadata"]["hosts"]:
            names = [n for x in h["components"] for n in x["cpe_names"]]
            names = list(dict.fromkeys(names))
            for ci, x in enumerate(h["components"]):
                lp = "windows_os" if x["family"] == "os" and h["os"] == "windows" else x["product"]
                for u in units:
                    if (u.product, u.family) != (lp, x["family"]):
                        continue
                    row = c.master[u.cve_id]
                    msrc = None
                    if x["release_line"]:  # Microsoft component: the raw MSRC records of this unit
                        ids = [i for y in row["units"] if (y["product"], y["family"]) == (u.product, u.family)
                               for i in y.get("msrc_products", [])]
                        line = x["release_line"]
                        msrc = {"products": (row.get("msrc") or {}).get("products") or {}, "unit_ids": ids,
                                "name": line, "build": x["version"], "kbs": x["kbs"], "cve": u.cve_id,
                                "releases": (c.catalogue["windows"].get(line) or {}).get("fixtures"),
                                "order": "date" if line in CFG["rollup_lines"] else "build"}
                    if cpe_check.applies(row["configurations"], names, x["cpe_names"], msrc):
                        want.add((h["id"], ci, u.cve_id))
        assert want == {(m["host"], m["component"], m["cve"]) for m in d["metadata"]["matches"]}


def test_firewall_respected_by_every_remote_action():
    for d in sample():
        net = d["network"]
        sub = {h["id"]: h["subnet"] for h in net["hosts"]}
        allow = {(f["from"], f["to"]): set(f["allow"]) for f in net["firewall"]}
        facts = d["task"]["facts"]
        ms = {(m["host"], m["component"], m["cve"]): m for m in d["metadata"]["matches"]}
        for a, meta in zip(d["task"]["actions"], d["metadata"]["actions"]):
            if meta["kind"] != "remote":
                continue
            (pre,) = a["pre"]
            src = int(re.match(r"foothold\(h(\d+),", facts[pre]).group(1))
            for h, comp, cve in meta["matches"]:
                fam = ms[(h, comp, cve)]["family"]
                assert src != h
                assert sub[src] == sub[h] or fam in allow.get((sub[src], sub[h]), set()), (src, h, fam)


def test_local_actions_need_a_foothold_on_the_same_host_and_give_admin_there():
    for d in sample():
        facts = d["task"]["facts"]
        for a in d["task"]["actions"]:
            if a["name"].startswith("local "):
                (pre,), (add,) = a["pre"], a["add"]
                h = re.match(r"foothold\(h(\d+),", facts[pre]).group(1)
                assert facts[add] == f"admin(h{h})"


# ---------------------------------------------------------------- hosts

def test_os_compatibility_and_role_family_sets():
    c = ctx()
    linux_products = set(c.catalogue["linux"])
    for d in sample():
        zone = {h["id"]: h["zone"] for h in d["network"]["hosts"]}
        for h in d["metadata"]["hosts"]:
            os_comp, *svcs = h["components"]
            assert os_comp["family"] == "os"
            fams = [x["family"] for x in svcs]
            assert len(fams) == len(set(fams)) and 1 <= len(fams) <= 3 or not fams
            if zone[h["id"]] == "dmz":  # P6b: only DMZ hosts keep a role list
                assert set(fams) <= set(CFG["roles"]["dmz"])
            else:
                inv = c.inventory(d["params"]["cohort"], d["params"]["p"], frozenset(d["params"]["excluded"]))
                assert set(fams) <= {f for (o, f) in inv.products if o == h["os"]}
            for x in svcs:
                if h["os"] == "linux":
                    assert x["product"] in linux_products
                else:
                    assert x["product"] not in linux_products
                if x["builtin"]:
                    assert x["version"] == os_comp["version"] and x["status"] == os_comp["status"]
                    version = c.catalogue["windows"][os_comp["product"]]
                    assert x["product"] in version["units"]
            if h["os"] == "linux":
                assert os_comp["product"] == "linux:linux_kernel"
            else:
                assert os_comp["product"] in CFG["windows_versions"]
        for m in d["metadata"]["matches"]:
            comp = d["metadata"]["hosts"][m["host"] - 1]["components"][m["component"]]
            assert m["family"] == comp["family"]  # an IIS CVE applies to the http or the ftp service by its own unit


def test_configuration_reuse_follows_the_seed():
    copies = Counter()
    trials = Counter()
    for s in range(40):
        d = doc("chain", f"r{s}", 20)
        again = gen.generate(ctx(), "chain", f"r{s}", 20)
        assert gen.dumps(again) == gen.dumps(d)
        hosts = {h["id"]: h for h in d["metadata"]["hosts"]}
        zone = {h["id"]: h["zone"] for h in d["network"]["hosts"]}
        seen = Counter()
        for hid in sorted(hosts):
            k = seen[zone[hid]]
            seen[zone[hid]] += 1
            trials[k] += 1
            src = hosts[hid]["copied_from"]
            if src is not None:
                copies[k] += 1
                assert src < hid and zone[src] == zone[hid] and hosts[src]["components"] == hosts[hid]["components"]
    assert copies[0] == 0
    for k in (1, 2, 3):  # P(copy) = k/(k+2): 1/3, 1/2, 3/5 (loose, 40 networks)
        assert abs(copies[k] / trials[k] - k / (k + 2)) < 0.2, (k, copies[k], trials[k])


def test_p_statuses():
    c = ctx()
    for d in sample():
        p = d["params"]["p"]
        comps = [x for h in d["metadata"]["hosts"] for x in h["components"]]
        matched = {(m["host"], m["component"]) for m in d["metadata"]["matches"]}
        for h in d["metadata"]["hosts"]:
            for ci, x in enumerate(h["components"]):
                if x["status"] == "unaffected":
                    assert (h["id"], ci) not in matched
        if p == 1.0:
            assert all(x["status"] == "vulnerable" for x in comps)
        else:
            assert not any(x["unverified_release"] for x in comps)
    inv = c.inventory("core", 0.5, frozenset())
    draws = Counter()
    rng = random.Random(3)
    for _ in range(400):
        draws[hosts_mod._status(rng, 0.5)] += 1
    assert 150 < draws["vulnerable"] < 250
    for prods in inv.products.values():
        for q in prods:
            if not q.builtin:
                assert any(inv.status[(q.name, f.version)] == "unaffected" for f in q.fixtures)


def test_unverified_release_tasks_are_flagged_and_only_at_p1():
    """The flag equals "some host component is unverified"; unverified fixtures are never drawn at p < 1.
    (The 49500af catalogue holds none: P5v verified them. The flag is checked on a catalogue copy that does.)"""
    import copy

    c = ctx()
    for p in (0.75, 0.5, 0.25):
        inv = c.inventory("core", p, frozenset())
        assert not any(f.unverified for prods in inv.products.values() for q in prods for f in q.fixtures
                       if inv.status.get((q.name, f.version)) == "unaffected")
    cat = copy.deepcopy(c.catalogue)
    for f in cat["linux"]["apache:http_server"]["fixtures"]:
        f["unverified_release"] = True  # every httpd fixture becomes unverified
    c2 = gen.Context(c.cfg, c.units, c.funnel, c.master, cat, c.principals, c.input_hashes)
    seen = {True: 0, False: 0}
    for s in range(40):
        d = gen.generate(c2, "tree", f"u{s}", 12, os_mix={"linux": 1.0, "windows": 0.0})
        uses = any(x["unverified_release"] for h in d["metadata"]["hosts"] for x in h["components"])
        assert d["flags"]["unverified_release"] == uses
        seen[uses] += 1
        d5 = gen.generate(c2, "tree", f"u{s}", 12, p=0.5, os_mix={"linux": 1.0, "windows": 0.0})
        assert not d5["flags"]["unverified_release"]
    assert seen[True] and seen[False]


def test_fixtures_with_unknown_status_are_never_drawn_in_a_cohort_whose_pool_has_the_cve():
    """Devansh, 17 Nov 2026: affected_unknown_standalone fixtures are removed from draws; counted in the report."""
    c = ctx()
    for cohort in ("core", "temporal"):
        inv = c.inventory(cohort, 1.0, frozenset())
        pool_cves = {u.cve_id for u in c.run_units(cohort, frozenset())}
        removed = 0
        for prods in list(inv.products.values()) + [inv.windows_versions]:
            for q in prods:
                for f in q.fixtures:
                    if set(f.unknown) & pool_cves:
                        removed += 1
                        assert inv.status[(q.name, f.version)] == "unknown"
        assert removed == sum(inv.report["unknown_fixtures"].values()) or removed <= sum(
            inv.report["unknown_fixtures"].values())
        for s in range(6):
            d = doc("chain", f"unk{s}", 12, cohort)
            for h in d["metadata"]["hosts"]:
                for x in h["components"]:
                    assert inv.status.get((x["product"], x["version"])) != "unknown"
    assert any(c.inventory(co, 1.0, frozenset()).report["unknown_fixtures"] for co in ("core", "temporal"))


# ---------------------------------------------------------------- actions, merge, cost

def _built(matches, table=None, n=6, arch="chain", goal=None):
    lay = layout(arch, n, CFG["topology"])
    fams = {h: [] for s in lay.subnets for h in s.hosts}
    comps = {}
    for m in matches:
        comps.setdefault(m["host"], {})[m["component"]] = m["family"]
    configs = {}
    for h in fams:
        if h == 0:
            continue
        cs = [hosts_mod.Component("os", "linux:linux_kernel", _fx("k"), "vulnerable")]
        for ci in sorted(comps.get(h, {})):
            cs.append(hosts_mod.Component(comps[h][ci], "p", _fx("v"), "vulnerable"))
        configs[h] = hosts_mod.HostConfig("linux", tuple(cs))
    fw = {(a, b): ["http", "smb"] for x, y in lay.links for a, b in ((x, y), (y, x))}
    return build.build_task(lay, configs, fw, matches, table or {"entries": {}}, {"root", "SYSTEM"}, goal or n)


def _fx(v):
    from ahg.gen.inventory import Fixture
    return Fixture("p", v, (f"cpe:2.3:a:x:p:{v}:-:*:*:*:*:*:*",))


def _m(host, comp, cve, kind, ac, family="http", product="x:p"):
    return {"host": host, "component": comp, "family": family, "product": product, "version": "v", "cve": cve,
            "kind": kind, "ac": ac, "unit_product": product, "evidence": []}


def test_merge_keeps_cheapest_ac_and_lowest_numeric_cve_in_both_regimes():
    # chain n=6: DMZ is host 1; three CVEs on its http service from the internet are one transition
    b = _built([_m(1, 1, "CVE-2021-10000", "remote", "L"), _m(1, 1, "CVE-2021-9999", "remote", "L"),
                _m(1, 1, "CVE-2020-5", "remote", "H")])
    (a,), (meta,) = b.actions, b.action_meta
    assert a["cost"] == {"ac": 4, "uniform": 1} and b.merges == 2
    assert meta["cost_cve"] == "CVE-2021-9999" and meta["cves"] == ["CVE-2020-5", "CVE-2021-9999", "CVE-2021-10000"]
    b = _built([_m(1, 1, "CVE-2020-7", "remote", "H"), _m(1, 1, "CVE-2020-6", "remote", "H")])
    assert b.actions[0]["cost"] == {"ac": 7, "uniform": 1} and b.action_meta[0]["cost_cve"] == "CVE-2020-6"
    # local CVEs on two components of one host merge too; one copy per foothold fact
    b = _built([_m(1, 1, "CVE-2020-1", "remote", "L"), _m(1, 0, "CVE-2019-1", "local", "H", "os"),
                _m(1, 1, "CVE-2019-2", "local", "L")])
    local = [x for x in b.actions if x["name"].startswith("local ")]
    assert len(local) == 1 and local[0]["cost"]["ac"] == 4


def test_admin_from_a_remote_exploit_only_with_a_cited_admin_principal():
    table = {"entries": {"x:p|http": {"principal": "root"}, "x:q|smb": {"principal": "www"}}}
    b = _built([_m(1, 1, "CVE-2020-1", "remote", "L")], table)
    facts = [b.facts[i] for i in b.actions[0]["add"]]
    assert sorted(facts) == ["admin(h1)", f"foothold(h1,p{b.principals.index('root')})"]
    b = _built([_m(1, 1, "CVE-2020-1", "remote", "L", "smb", "x:q")], table)
    assert [b.facts[i] for i in b.actions[0]["add"]] == [f"foothold(h1,p{b.principals.index('www')})"]
    b = _built([_m(1, 1, "CVE-2020-1", "remote", "L", "smb", "x:unlisted")], table)
    assert [b.facts[i] for i in b.actions[0]["add"]] == [f"foothold(h1,p{b.principals.index('UNKNOWN')})"]


def test_unsolvable_reasons():
    b = _built([_m(1, 1, "CVE-2020-1", "remote", "L")])  # goal host 6 (sensitive): nothing gives admin there
    held = build.reachable_facts(len(b.facts), b.actions, [0])
    assert build.unsolvable_reason(b, 6, held) == "no_action_gives_admin_on_goal_host"
    b = _built([_m(6, 1, "CVE-2020-2", "local", "L")])  # admin via local, but no foothold on 6
    held = build.reachable_facts(len(b.facts), b.actions, [0])
    assert build.unsolvable_reason(b, 6, held) == "no_action_gives_admin_on_goal_host"  # no foothold fact at all
    b = _built([_m(6, 1, "CVE-2020-3", "remote", "L"), _m(6, 1, "CVE-2020-2", "local", "L")])
    held = build.reachable_facts(len(b.facts), b.actions, [0])
    assert build.unsolvable_reason(b, 6, held) == "no_foothold_on_goal_host_reachable"  # no path to host 6
    b = _built([_m(1, 1, "CVE-2020-3", "remote", "L"), _m(1, 1, "CVE-2020-2", "local", "L")], goal=1)
    held = build.reachable_facts(len(b.facts), b.actions, [0])
    assert build.unsolvable_reason(b, 1, held) is None


def test_goal_is_admin_on_a_sensitive_host_drawn_from_the_goal_stream():
    for d in sample():
        g = d["metadata"]["goal_host"]
        assert d["task"]["facts"][d["task"]["goal"][0]] == f"admin(h{g})"
        assert next(h for h in d["network"]["hosts"] if h["id"] == g)["zone"] == "sensitive"
    goals = set()
    for s in range(12):
        lay = layout("tree", 60, CFG["topology"])
        sens = sorted(h for x in lay.subnets if x.zone == "sensitive" for h in x.hosts)
        assert len(sens) == 2
        want = random.Random(f"goal:tree:g{s}").choice(sens)
        d = gen.generate(ctx(), "tree", f"g{s}", 60)
        assert d["metadata"]["goal_host"] == want
        goals.add(want)
    assert len(goals) == 2


def test_unsolvable_tasks_are_kept_and_agree_with_the_reference():
    seen = Counter()
    for d in sample():
        seen[d["solvable"]] += 1
        st = gen.neutral(d, "ac")
        reach = refsolver.reachable(st, cap=200000)
        if reach is not None:
            assert any(set(st["goal"]) <= s for s in reach) == d["solvable"]
        if not d["solvable"]:
            assert d["unsolvable_reason"] in ("no_action_gives_admin_on_goal_host", "no_foothold_on_goal_host_reachable",
                                              "admin_on_goal_host_needs_an_unreached_foothold")
    assert seen[True] and seen[False]


# ---------------------------------------------------------------- determinism and exports

def test_determinism_across_processes():
    d = doc("tree", "det", 12)
    code = ("from ahg.gen.generate import Context, generate, doc_digest;"
            "print(doc_digest(generate(Context.load(), 'tree', 'det', 12, os_mix={'linux': .5, 'windows': .5})))")
    for hashseed in ("0", "12345"):
        env = os.environ | {"PYTHONHASHSEED": hashseed}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, check=True)
        assert out.stdout.strip() == gen.doc_digest(d)
    assert gen.doc_digest(doc("tree", "det2", 12)) != gen.doc_digest(d)
    assert gen.doc_digest(doc("chain", "det", 12)) != gen.doc_digest(d)


@pytest.mark.parametrize("regime", ["ac", "uniform"])
def test_pddl_round_trip(regime):
    for d in sample()[:10]:
        dom, prob = gen.export_pddl(d, regime)
        back = pddl.parse_grounded(dom, prob)
        t = d["task"]
        assert back["facts"] == len(t["facts"]) and back["init"] == t["init"] and back["goal"] == t["goal"]
        assert back["actions"] == [(a["pre"], a["add"], a["cost"][regime]) for a in t["actions"]]


@pytest.mark.parametrize("regime", ["ac", "uniform"])
def test_reference_solver_agrees_with_the_planner_on_small_tasks(regime):
    compared = 0
    for arch, n in (("tree", 3), ("tree", 4), ("chain", 4), ("chain", 5)):
        for s in range(8):
            d = doc(arch, f"small{s}", n)
            st = gen.neutral(d, regime)
            if refsolver.reachable(st, cap=50000) is None:
                continue
            ref, _ = refsolver.solve(st)
            task = gen.to_task(d, regime)
            res = astar(task, lmcut)
            got = res.cost if res.plan is not None else float("inf")
            assert got == ref, (arch, n, s, got, ref)
            compared += 1
    assert compared >= 20


# ---------------------------------------------------------------- leakage and removed knobs

def _planner_text(d):
    return json.dumps({"task": d["task"], "network": d["network"]})


def test_planner_visible_blocks_carry_no_cve_product_version_or_cpe():
    c = ctx()
    products = set(c.catalogue["linux"]) | {u.product for u in c.units}
    tokens = {t for p in products for t in re.split(r"[:\s]", p.lower()) if len(t) > 3}
    tokens -= {"server", "services", "windows", "remote", "desktop"}  # generic words, not product names
    for d in sample():
        text = _planner_text(d).lower()
        assert not re.search(r"cve-\d{4}-\d+", text) and "cpe:" not in text
        for h in d["metadata"]["hosts"]:
            for x in h["components"]:
                assert f'"{x["version"].lower()}"' not in text
        hits = [t for t in tokens if re.search(rf"\b{re.escape(t)}\b", text)]
        assert not hits, hits


def task_feature_names(d):
    """The per-fact and per-action feature names a D28 encoder can read off a task (kinds and costs)."""
    t = d["task"]
    kinds = {f.split("(")[0] for f in t["facts"]}
    acts = {a["name"].split()[0] for a in t["actions"]}
    return (["held", "is_goal"] + [f"fact_kind:{k}" for k in sorted(kinds)]
            + [f"action_kind:{k}" for k in sorted(acts)] + ["cost_ac", "cost_uniform"])


def test_leakage_audit_passes_on_task_features():
    policy = FeaturePolicy(frozenset({"held", "is_goal", "fact_kind:foothold", "fact_kind:admin", "action_kind:remote",
                                      "action_kind:local", "cost_ac", "cost_uniform"}),
                           (r"cve", r"cpe", r"\d+\.\d+", r"apache|samba|exim|microsoft|windows|linux|kernel|iis",
                            r"vendor|product|version|seed|date"))
    for d in sample():
        rep = audit(task_feature_names(d), policy, "p6_smoke")
        assert rep["ok"], rep["violations"]


def test_no_trust_login_credential_coupling_or_pr_gating_on_the_d28_path():
    params = set(inspect.signature(gen.generate).parameters)
    assert params == {"ctx", "archetype", "seed", "n", "cohort", "os_mix", "p", "excluded"}
    imported = set()
    for f in sorted((ROOT / "src" / "ahg" / "gen").glob("*.py")):
        for nd in ast.walk(ast.parse(f.read_text())):
            if isinstance(nd, ast.ImportFrom):
                imported |= {f"{nd.module}.{a.name}" for a in nd.names} | {nd.module or ""}
            elif isinstance(nd, ast.Import):
                imported |= {a.name for a in nd.names}
    for old in ("ahg.attack_graph", "ahg.network", "ahg.topology", "ahg.assign", "ahg.cve"):
        assert old not in imported, old  # the D10/D15 generator and its knobs are never reached
    for d in sample():
        assert {f.split("(")[0] for f in d["task"]["facts"]} <= {"foothold", "admin"}
        assert {a["name"].split()[0] for a in d["task"]["actions"]} <= {"remote", "local"}
        text = _planner_text(d).lower()
        for word in ("trust", "login", "cred", "coupl", "capability", "pr_gat", "level"):
            assert word not in text


# ---------------------------------------------------------------- P6b

def test_non_dmz_hosts_draw_from_all_compatible_families():
    """P6b: only the DMZ keeps a role list; e.g. a sensitive host may run http (outside the old lists)."""
    old = {"sensitive": {"sql", "smb", "ssh", "rdp"}, "users": {"smb", "rdp", "ssh"},
           "enterprise": {"http", "smb", "ssh", "rdp", "dns", "sql"}}
    outside = 0
    for s in range(20):
        d = doc("chain", f"roles{s}", 12)
        zone = {h["id"]: h["zone"] for h in d["network"]["hosts"]}
        for h in d["network"]["hosts"]:
            if zone[h["id"]] in old:
                outside += bool(set(h["families"]) - old[zone[h["id"]]])
    assert outside > 0


def test_msrc_only_matches_are_stored_and_traced():
    """OQ-P6-10: SQL Server CVEs have no NVD configuration; they apply through MSRC records only."""
    found = 0
    for s in range(30):
        d = doc("tree", f"sql{s}", 12, "temporal", linux=0.25)
        for m in d["metadata"]["matches"]:
            if [e["configuration"] for e in m["evidence"]] == ["msrc"]:
                found += 1
                assert m["evidence"][0]["fixed_by"] and m["evidence"][0]["product"]
                # a match need not have an action (no foothold on the host, or no reaching source);
                # every action's matches are checked in test_every_action_traces_to_match_records
    assert found


def test_a_fixture_kb_list_reaches_both_evaluators():
    from ahg.gen.inventory import Fixture

    row = {"configurations": [], "units": [{"product": "SQL Server", "family": "sql", "msrc_products": ["9"]}],
           "msrc": {"products": {"9": {"name": "SQL X", "status_type": 3,
                                       "fixes": [{"fixed_build": "15.0.2", "kb": "5040000"}]}}}}
    unit = pool.Unit("CVE-2099-1", "SQL Server", "sql", "windows", "remote", "L", "temporal")
    name = ("cpe:2.3:a:microsoft:sql_server_2019:15.0.1:-:*:*:*:*:x64:*",)
    plain = hosts_mod.Component("sql", "SQL Server", Fixture("SQL Server", "15.0.1", name, False, "SQL X"), "vulnerable")
    patched = hosts_mod.Component("sql", "SQL Server", Fixture("SQL Server", "15.0.1", name, False, "SQL X", (),
                                                                ("KB5040000",)), "vulnerable")
    assert [e["configuration"] for e in build.evaluate(unit, row, list(name), plain)] == ["msrc"]
    assert build.evaluate(unit, row, list(name), patched) == []
    # the catalogue route: a KB-only record placed by the release that ships it (OQ-P6-14, 17 Nov 2026)
    row["msrc"]["products"]["9"]["fixes"] = [{"fixed_build": None, "kb": "5040001"}]
    windows = {"SQL X": {"fixtures": [{"version": "15.0.1", "release_date": "2024-01-09", "kb": "KB1", "fixed_for": []},
                                      {"version": "15.0.3", "release_date": "2024-02-13", "kb": "KB5040001",
                                       "fixed_for": [{"cve": "CVE-2099-1", "evidence": {}}]}]}}
    assert [e["configuration"] for e in build.evaluate(unit, row, list(name), plain, windows)] == ["msrc"]
    assert build.evaluate(unit, row, list(name), plain, {}) == []  # KB in none of the sources: not applied


def test_cve_specific_principal_overrides_the_product_entry():
    table = {"entries": {"x:p|http": {"principal": "UNKNOWN", "reason": "r",
                                      "cves": {"CVE-2020-1": {"principal": "root"}}}}}
    assert build.principal_of(table, "x:p", "http", "CVE-2020-1") == "root"
    assert build.principal_of(table, "x:p", "http", "CVE-2020-2") == "UNKNOWN"
    b = _built([_m(1, 1, "CVE-2020-1", "remote", "L"), _m(1, 1, "CVE-2020-2", "remote", "L")], table)
    adds = sorted(tuple(sorted(b.facts[i] for i in a["add"])) for a in b.actions)
    root, unk = b.principals.index("root"), b.principals.index("UNKNOWN")
    assert adds == sorted([("admin(h1)", f"foothold(h1,p{root})"), (f"foothold(h1,p{unk})",)])
