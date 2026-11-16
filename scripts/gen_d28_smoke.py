"""P6 smoke generation: every archetype x size x cohort x OS mix x p cell under the seed namespace "p6-smoke".

Usage: uv run python scripts/gen_d28_smoke.py

Writes one runlog run under results/d28_gen/p6_smoke/<UTC>-<commit>/ (raw.json.gz, report.md, provenance.json;
untracked, BUILD_RULES rule 4). Every candidate is logged (seed, archetype, size, cohort, OS mix, p,
solvability and reason). Optimal plans come from A* with LM-cut (admissible) under each regime, with an
expansion budget; a budget hit is recorded as such, never as a plan. These are smoke diagnostics on
generator seeds, not results.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from ahg import runlog
from ahg.gen import generate as gen
from ahg.gen.hosts import InfeasibleCell
from ahg.gen.topology import layout
from ahg.heuristics import lmcut
from ahg.search import astar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d28_gen"


def plan(doc: dict, regime: str, budget: int) -> dict:
    r = astar(gen.to_task(doc, regime), lmcut, max_expansions=budget)
    return {"cost": None if r.plan is None else r.cost, "length": None if r.plan is None else len(r.plan),
            "expanded": r.expanded, "budget_hit": r.budget_hit}


def candidate(ctx, arch, seed, n, cohort, linux, p, budget) -> dict:
    t0 = time.perf_counter()
    row = {"archetype": arch, "seed": seed, "n": n, "cohort": cohort, "os_linux": linux, "p": p}
    try:
        doc = gen.generate(ctx, arch, seed, n, cohort=cohort, os_mix={"linux": linux, "windows": 1 - linux}, p=p)
    except InfeasibleCell as e:
        return row | {"infeasible": str(e)}
    hosts = doc["metadata"]["hosts"]
    matched = {(m["host"], m["component"]) for m in doc["metadata"]["matches"]}
    vuln = [(h["id"], i) for h in hosts for i, x in enumerate(h["components"]) if x["status"] == "vulnerable"]
    row |= {"digest": gen.doc_digest(doc), "solvable": doc["solvable"], "reason": doc["unsolvable_reason"],
            "flag_unverified": doc["flags"]["unverified_release"], **doc["stats"],
            "windows_hosts": sum(h["os"] == "windows" for h in hosts), "hosts": len(hosts),
            "vulnerable_components": len(vuln), "vulnerable_components_without_match": sum(v not in matched for v in vuln),
            "products": sorted({x["product"] for h in hosts for x in h["components"]})}
    if doc["solvable"]:
        row["plan_ac"], row["plan_uniform"] = plan(doc, "ac", budget), plan(doc, "uniform", budget)
    row["seconds"] = round(time.perf_counter() - t0, 3)
    return row


def med(xs):
    return statistics.median(xs) if xs else None


def heldout_candidates(ctx) -> list[dict]:
    """Round 10 selection rule inputs per Linux service product (listed, never chosen here): eligible CVEs on
    the core cohort, documented vulnerable and unaffected fixtures (core, p < 1 rule), principal-table entry."""
    inv = ctx.inventory("core", 0.5, frozenset())
    units = ctx.run_units("core", frozenset())
    rows = []
    for name, row in sorted(ctx.catalogue["linux"].items()):
        fam = row["families"][0]
        if fam == "os" or name in ctx.cfg["never_products"]:
            continue
        cves = sorted({u.cve_id for u in units if u.product == name})
        st = [inv.status.get((name, f["version"])) for f in row["fixtures"] if not f.get("unverified_release")]
        keys = [k for k in ctx.principals["entries"] if k.startswith(name + "|")]
        rows.append({"product": name, "family": fam, "core_pool_cves": len(cves),
                     "verified_vulnerable_fixtures": st.count("vulnerable"), "unaffected_fixtures": st.count("unaffected"),
                     "principal_entry": [ctx.principals["entries"][k]["principal"] for k in keys] or None})
    for r in rows:
        r["meets_rule"] = (r["core_pool_cves"] >= 5 and r["verified_vulnerable_fixtures"] >= 1
                           and r["unaffected_fixtures"] >= 1 and r["principal_entry"] is not None)
    by_fam = Counter(r["family"] for r in rows if r["meets_rule"])
    for r in rows:
        r["family_has_two_eligible"] = by_fam[r["family"]] >= 2
    return rows


def unmatched_pool(ctx) -> dict:
    """Pool CVEs with no NVD configuration at all: CPE-only matching can never apply them."""
    c = Counter()
    for u in ctx.units:
        if not ctx.master[u.cve_id]["configurations"]:
            c[f"{u.cohort}|{u.os}|{u.kind}|{u.product}"] += 1
    return dict(sorted(c.items()))


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "infeasible" not in r]
    grp = defaultdict(list)
    for r in ok:
        grp[(r["archetype"], r["n"])].append(r)
    by_size = {}
    for (a, n), rs in sorted(grp.items()):
        sol = [r for r in rs if r["solvable"]]
        planned = [r for r in sol if r["plan_ac"]["length"] is not None]
        by_size[f"{a}|{n}"] = {
            "tasks": len(rs), "solvable": len(sol), "solvability": len(sol) / len(rs),
            "actions_median": med([r["actions"] for r in rs]), "actions_min": min(r["actions"] for r in rs),
            "actions_max": max(r["actions"] for r in rs),
            "plan_len_ac_median": med([r["plan_ac"]["length"] for r in planned]),
            "plan_cost_uniform_median": med([r["plan_uniform"]["cost"] for r in sol if r["plan_uniform"]["cost"] is not None]),
            "budget_hits": sum(r["plan_ac"]["budget_hit"] or r["plan_uniform"]["budget_hit"] for r in sol)}
    def rate(key):
        g = defaultdict(lambda: [0, 0])
        for r in ok:
            g[str(r[key])][0] += r["solvable"]
            g[str(r[key])][1] += 1
        return {k: {"solvable": v[0], "tasks": v[1], "rate": v[0] / v[1]} for k, v in sorted(g.items())}
    cell = defaultdict(lambda: [0, 0])
    for r in ok:
        k = f"{r['archetype']}|{r['n']}|{r['cohort']}"
        cell[k][0] += r["solvable"]
        cell[k][1] += 1
    tot = lambda k: sum(r[k] for r in ok)
    return {
        "tasks": len(ok), "infeasible": len(rows) - len(ok), "by_archetype_size": by_size,
        "solvability_by_archetype_size_cohort": {k: {"solvable": v[0], "tasks": v[1]} for k, v in sorted(cell.items())},
        "solvability_by_cohort": rate("cohort"), "solvability_by_p": rate("p"), "solvability_by_os_linux": rate("os_linux"),
        "unsolvable_reasons": dict(Counter(r["reason"] for r in ok if not r["solvable"])),
        "actions": tot("actions"), "remote_actions": tot("remote_actions"), "local_actions": tot("local_actions"),
        "merged_transitions": tot("merged_transitions"), "matches": tot("matches"), "remote_matches": tot("remote_matches"),
        "remote_matches_unknown_principal": tot("remote_matches_unknown_principal"),
        "tasks_flagged_unverified": sum(r["flag_unverified"] for r in ok),
        "tasks_p1": sum(r["p"] == 1.0 for r in ok),
        "vulnerable_components": tot("vulnerable_components"),
        "vulnerable_components_without_match": tot("vulnerable_components_without_match"),
        "reused_configs": tot("reused_configs"), "hosts": tot("hosts"), "windows_hosts": tot("windows_hosts"),
        "seconds_total": round(sum(r["seconds"] for r in ok), 1)}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def report(raw: dict) -> str:
    s, L = raw["summary"], []
    L += ["# P6 smoke generation (seed namespace p6-smoke)", "",
          "Diagnostics on generator seeds, not results. Every number below is read from raw.json.gz of this run.", "",
          "## Action pool funnel", "", f"- label rows {raw['funnel']['label_rows']}, pool {raw['funnel']['pool']} "
          f"(remote {raw['funnel']['remote']}, local {raw['funnel']['local']})", ""]
    L += ["| exclusion | units |", "|---|---|"] + [f"| {k} | {v} |" for k, v in raw["funnel"]["excluded"].items()]
    L += ["", "Pool units whose CVE has no NVD configuration (CPE-only matching never applies them):", "",
          "| cohort/os/kind/product | units |", "|---|---|"] + [f"| {k} | {v} |" for k, v in raw["unmatched_pool"].items()]
    L += ["", "## Products and families per cell (cohort, p)", ""]
    for key, inv in raw["inventories"].items():
        avail = [f"{r['os']}/{r['family']}/{r['product']}" for r in inv["products"] if r["available"]]
        drop = [f"{r['os']}/{r['family']}/{r['product']} ({r['reason']})" for r in inv["products"] if not r["available"]]
        L += [f"### {key}", f"- Windows versions: {', '.join(inv['windows_versions']) or 'none'}; kernel: {inv['kernel']}",
              f"- available ({len(avail)}): {', '.join(avail)}", f"- dropped ({len(drop)}): {', '.join(drop)}", ""]
    L += ["## Hosts per zone", "", "| archetype | n | zones |", "|---|---|---|"]
    L += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {v} |" for k, v in raw["zones"].items()]
    L += ["", "## Per archetype and size", "",
          "| archetype/n | tasks | solvable | rate | actions median (min-max) | plan length ac median | uniform optimal cost median | budget hits |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in s["by_archetype_size"].items():
        L.append(f"| {k} | {v['tasks']} | {v['solvable']} | {fmt(v['solvability'])} | {v['actions_median']} "
                 f"({v['actions_min']}-{v['actions_max']}) | {v['plan_len_ac_median']} | {v['plan_cost_uniform_median']} | {v['budget_hits']} |")
    L += ["", "Solvability by archetype, size and cohort:", "", "| cell | solvable/tasks |", "|---|---|"]
    L += [f"| {k} | {v['solvable']}/{v['tasks']} |" for k, v in s["solvability_by_archetype_size_cohort"].items()]
    for name in ("solvability_by_cohort", "solvability_by_p", "solvability_by_os_linux"):
        L += ["", f"{name}: " + "; ".join(f"{k}: {v['solvable']}/{v['tasks']} ({fmt(v['rate'])})" for k, v in s[name].items())]
    L += ["", f"Unsolvable reasons: {s['unsolvable_reasons']}", "",
          "## Totals", "",
          f"- tasks {s['tasks']}, infeasible candidates {s['infeasible']}, hosts {s['hosts']} (Windows {s['windows_hosts']})",
          f"- actions {s['actions']}: remote {s['remote_actions']}, local {s['local_actions']}",
          f"- matches {s['matches']} (remote {s['remote_matches']}); transitions merged into another {s['merged_transitions']}",
          f"- UNKNOWN-principal share of remote matches: {s['remote_matches_unknown_principal']}/{s['remote_matches']}"
          + (f" = {fmt(s['remote_matches_unknown_principal'] / s['remote_matches'])}" if s["remote_matches"] else ""),
          f"- tasks flagged unverified_release: {s['tasks_flagged_unverified']} of {s['tasks_p1']} p = 1 tasks"
          + (f" = {fmt(s['tasks_flagged_unverified'] / s['tasks_p1'])}" if s["tasks_p1"] else ""),
          f"- vulnerable components with no match on their host: {s['vulnerable_components_without_match']}/{s['vulnerable_components']}",
          f"- configurations reused: {s['reused_configs']}/{s['hosts']}",
          f"- generation and search time: {s['seconds_total']} s", "",
          "## Held-out product candidates (Round 10 rule inputs; listed, not chosen)", "",
          "| product | family | core pool CVEs | verified vulnerable fixtures | unaffected fixtures | principal entry | meets rule | family has >= 2 |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['product']} | {r['family']} | {r['core_pool_cves']} | {r['verified_vulnerable_fixtures']} | "
          f"{r['unaffected_fixtures']} | {r['principal_entry']} | {r['meets_rule']} | {r['family_has_two_eligible']} |"
          for r in raw["heldout_candidates"]]
    return "\n".join(L) + "\n"


def main() -> None:
    ctx = gen.Context.load()
    sm = ctx.cfg["smoke"]
    rows = []
    seeds = [f"{sm['namespace']}-{i}" for i in range(sm["seeds_per_cell"])]
    for cohort in ctx.cfg["cohorts"]:
        for p in ctx.cfg["p_values"]:
            for linux in sm["os_mixes"]:
                for arch in ("tree", "chain"):
                    for n in sm["sizes"]:
                        for seed in seeds:
                            rows.append(candidate(ctx, arch, seed, n, cohort, linux, p, sm["search_max_expansions"]))
                print(cohort, p, linux, len(rows), flush=True)
    invs = {f"{c}|p={p}": ctx.inventory(c, p, frozenset()).report for c in ctx.cfg["cohorts"] for p in ctx.cfg["p_values"]}
    zones = {}
    for arch in ("tree", "chain"):
        for n in sm["sizes"]:
            lay = layout(arch, n, ctx.cfg["topology"])
            zones[f"{arch}|{n}"] = dict(Counter(s.zone for s in lay.subnets for _ in s.hosts))
    raw = {"funnel": ctx.funnel, "unmatched_pool": unmatched_pool(ctx), "inventories": invs, "zones": zones,
           "candidates": rows, "summary": summarise(rows), "heldout_candidates": heldout_candidates(ctx),
           "inputs": ctx.input_hashes}
    out = runlog.write_run("p6_smoke", raw, report(raw), seeds, ctx.cfg, root=OUT)
    print(out)


if __name__ == "__main__":
    main()
