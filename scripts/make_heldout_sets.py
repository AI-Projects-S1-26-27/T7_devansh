"""Held-out sets for the D28 splits (decisions.md §18; Round 10).

Usage: uv run python scripts/make_heldout_sets.py [--version v2]

- Held-out product: exim:exim (Devansh, 17 Nov 2026). Reported: which mail-family products the training
  cohort (core, every p) keeps without it, on each OS, before and after the CVE holdout.
- Unseen-CVE holdout, versioned (each version its own config; an existing config is never overwritten):
  - v1 (§18 P6c item 6): every AC:H core-pool CVE plus random AC:L CVEs up to 30% of the pool's CVE IDs,
    seed "cve-holdout" -> configs/gen/cve_holdout.json (kept, OQ-P6-21).
  - v2 (§18 P6d item 1, 17 Nov 2026): a random half of the AC:H CVEs (seed "cve-holdout-ach") plus random
    AC:L CVEs up to 30% of the pool's CVE IDs (seed "cve-holdout") -> configs/gen/cve_holdout_v2.json.
  Both grouped by canonical CVE ID (ahg.gen.pool.cve_holdout). The pool is the training cohort's: core units
  with no CVE touching the held-out product (OQ-P6-20, accepted in P6d). Reported: AC:H counts and shares in
  training vs held out, by CVE and by unit, per action kind; the product distribution of both sides; the
  products and service families that lose every pool CVE to the holdout, and those the generator's training
  inventory loses (per p).

Writes the version's config (the IDs, their SHA-256, the counts, the run) and one runlog run under
results/d28_gen/heldout_sets/<UTC>-<commit>/ (untracked). The config is committed after the run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ahg import runlog
from ahg.gen import generate as gen
from ahg.gen import pool

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d28_gen"
HELDOUT_PRODUCTS = ("exim:exim",)
COHORT = "core"
VERSIONS = {
    "v1": {"config": "configs/gen/cve_holdout.json", "share": 0.3, "seed": "cve-holdout", "ach_share": 1.0,
           "ach_seed": None,
           "decision": "decisions.md §18 item 6 (17 Nov 2026, Devansh): the unseen-CVE split (Round 10) holds out every "
                       "AC:H pool CVE plus random AC:L CVEs up to 30% of the pool's CVE IDs, seed 'cve-holdout', grouped by "
                       "canonical CVE ID. Pool: core cohort, no CVE touching a held-out product (OQ-P6-20)."},
    "v2": {"config": "configs/gen/cve_holdout_v2.json", "share": 0.3, "seed": "cve-holdout", "ach_share": 0.5,
           "ach_seed": "cve-holdout-ach",
           "decision": "decisions.md §18 P6d item 1 (17 Nov 2026, Devansh; closes OQ-P6-21): the unseen-CVE split (Round 10) "
                       "holds out a random half of the core pool's AC:H CVEs (seed 'cve-holdout-ach', round half up) plus "
                       "random AC:L CVEs until the holdout is 30% of the pool's CVE IDs (seed 'cve-holdout'), grouped by "
                       "canonical CVE ID. Pool: core cohort, no CVE touching a held-out product (OQ-P6-20, accepted). "
                       "Supersedes v1 (configs/gen/cve_holdout.json, kept)."},
}


def ac_mix(units: list[pool.Unit]) -> dict:
    cves: dict[str, set[str]] = {}
    for u in units:
        cves.setdefault(u.cve_id, set()).add(u.ac)
    by_cve = Counter("H" if "H" in a else "L" for a in cves.values())
    return {"cve_ids": len(cves), "cves_ac_h": by_cve["H"], "cves_ac_l": by_cve["L"],
            "units": len(units), "units_ac_h": sum(u.ac == "H" for u in units),
            "units_ac_l": sum(u.ac == "L" for u in units),
            "units_by_kind_ac": dict(sorted(Counter(f"{u.kind}|AC:{u.ac}" for u in units).items())),
            "units_by_os_product": dict(sorted(Counter(f"{u.os}|{u.product}" for u in units).items()))}


def ac_h_shares(mix: dict, pool_mix: dict) -> dict:
    """AC:H share within a side, and the side's share of the pool's AC:H (by CVE ID and by unit)."""
    def div(a, b):
        return a / b if b else None
    return {"ac_h_share_of_side_cves": div(mix["cves_ac_h"], mix["cve_ids"]),
            "ac_h_share_of_side_units": div(mix["units_ac_h"], mix["units"]),
            "share_of_pool_ac_h_cves": div(mix["cves_ac_h"], pool_mix["cves_ac_h"]),
            "share_of_pool_ac_h_units": div(mix["units_ac_h"], pool_mix["units_ac_h"])}


def lost_groups(units: list[pool.Unit], held: frozenset[str]) -> dict:
    """Groups (os|product, os|family, family) whose every pool CVE is held out: training has none of them."""
    keys = {"os_product": lambda u: f"{u.os}|{u.product}", "os_family": lambda u: f"{u.os}|{u.family}",
            "family": lambda u: u.family}
    out = {}
    for name, key in keys.items():
        ids = defaultdict(set)
        for u in units:
            ids[key(u)].add(u.cve_id)
        out[name] = {k: {"cve_ids": len(v), "held_out": len(v & held)} for k, v in sorted(ids.items())}
        out[name + "_lost"] = sorted(k for k, v in ids.items() if v <= held)
    return out


def inventory_lost(ctx, excluded: frozenset[str], held: frozenset[str]) -> dict:
    """Per p: products and families the training inventory (core, held-out product excluded) has without the
    CVE holdout but not with it (the generator can no longer place them in training tasks)."""
    out = {}
    for p in ctx.cfg["p_values"]:
        rows = {}
        for tag, cves in (("before", frozenset()), ("after", held)):
            inv = ctx.inventory(COHORT, p, excluded, cves)
            rows[tag] = {(r["os"], r["family"], r["product"]): r for r in inv.report["products"]}
        av = {t: {k for k, r in rs.items() if r["available"]} for t, rs in rows.items()}
        fam = {t: {(k[0], k[1]) for k in v} for t, v in av.items()}
        out[f"p={p}"] = {
            "products_lost": [{"os": k[0], "family": k[1], "product": k[2], "reason_after": rows["after"][k].get("reason")}
                              for k in sorted(av["before"] - av["after"])],
            "products_gained": [f"{k[0]}|{k[1]}|{k[2]}" for k in sorted(av["after"] - av["before"])],
            "families_lost": [f"{o}|{f}" for o, f in sorted(fam["before"] - fam["after"])],
            "products_available_before": len(av["before"]), "products_available_after": len(av["after"])}
    return out


def mail_family(ctx, excluded: frozenset[str], excluded_cves: frozenset[str]) -> dict:
    out = {}
    for p in ctx.cfg["p_values"]:
        inv = ctx.inventory(COHORT, p, excluded, excluded_cves)
        out[f"p={p}"] = [{"os": r["os"], "product": r["product"], "available": r["available"], "reason": r.get("reason")}
                         for r in inv.report["products"] if r["family"] == "mail"]
    return out


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def report(raw: dict) -> str:
    h, s = raw["holdout"], raw["holdout"]["counts"]
    L = [f"# Held-out sets, CVE holdout {h['version']} (decisions.md §18)", "",
         "Every number below is read from raw.json.gz of this run.", "",
         "## Held-out product: exim:exim", "",
         f"- Core-pool units touching it: {raw['exim_core_units']}; CVE IDs: {raw['exim_core_cves']}",
         "", "Mail-family products in the training cohort (core) without exim, before / after the CVE holdout:", ""]
    for tag in ("without_exim", "without_exim_and_heldout_cves"):
        L += [f"### {tag}", "", "| p | os | product | available | reason |", "|---|---|---|---|---|"]
        for p, rows in raw["mail"][tag].items():
            L += [f"| {p} | {r['os']} | {r['product']} | {r['available']} | {r['reason']} |" for r in rows]
        L += [""]
    L += [f"## Unseen-CVE holdout {h['version']}", "", f"Rule: {h['decision']}", "",
          f"- Pool: {COHORT} cohort, no CVE touching {', '.join(HELDOUT_PRODUCTS)}: {s['pool_cve_ids']} CVE IDs",
          f"- Target round_half_up({h['share']} x {s['pool_cve_ids']}) = {s['target']}; AC:L seed \"{h['seed']}\""
          + (f"; AC:H share {h['ach_share']}, seed \"{h['ach_seed']}\"" if h["ach_share"] < 1 else ""),
          f"- AC:H CVEs in the pool: {s['ac_h_cves']} (of which with an AC:L unit too: {s['ac_h_cves_with_an_ac_l_unit']}); "
          f"held out: {s.get('ac_h_drawn', s['ac_h_cves'])}. AC:L-only CVEs: {s['ac_l_only_cves']}, drawn: {s['ac_l_drawn']}",
          f"- Held-out IDs: {len(h['cves'])}, SHA-256 {h['sha256']}", "",
          "AC mix, held out vs training (the pool minus the holdout):", "",
          "| side | CVE IDs | AC:H CVEs | AC:L CVEs | AC:H share of side's CVEs | side's share of pool AC:H CVEs | "
          "units | AC:H units | AC:L units | AC:H share of side's units | side's share of pool AC:H units | by kind |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for side in ("pool", "heldout", "training"):
        m, sh = raw["ac_mix"][side], raw["ac_h_shares"][side]
        L.append(f"| {side} | {m['cve_ids']} | {m['cves_ac_h']} | {m['cves_ac_l']} | {fmt(sh['ac_h_share_of_side_cves'])} | "
                 f"{fmt(sh['share_of_pool_ac_h_cves'])} | {m['units']} | {m['units_ac_h']} | {m['units_ac_l']} | "
                 f"{fmt(sh['ac_h_share_of_side_units'])} | {fmt(sh['share_of_pool_ac_h_units'])} | {m['units_by_kind_ac']} |")
    lg = raw["lost"]
    L += ["", "## Groups that lose every pool CVE to the holdout (training has none of their CVEs)", "",
          f"- Products (os|product): {lg['os_product_lost'] or 'none'}",
          f"- Service families per OS (os|family): {lg['os_family_lost'] or 'none'}",
          f"- Service families (any OS): {lg['family_lost'] or 'none'}", "",
          "Per service family and OS: pool CVE IDs / held out:", "",
          "| os|family | pool CVE IDs | held out |", "|---|---|---|"]
    L += [f"| {k} | {c['cve_ids']} | {c['held_out']} |" for k, c in lg["os_family"].items()]
    L += ["", "## Training inventory (core, exim excluded): what the holdout removes, per p", "",
          "| p | products available before / after | products lost (reason after) | families lost | products gained |",
          "|---|---|---|---|---|"]
    for p, x in raw["inventory_lost"].items():
        lost = "; ".join(f"{r['os']}|{r['family']}|{r['product']} ({r['reason_after']})" for r in x["products_lost"]) or "none"
        L.append(f"| {p} | {x['products_available_before']} / {x['products_available_after']} | {lost} | "
                 f"{', '.join(x['families_lost']) or 'none'} | {', '.join(x['products_gained']) or 'none'} |")
    L += ["", "Product distribution (units):", "", "| os/product | held out | training | pool CVE IDs held out / all |",
          "|---|---|---|---|"]
    keys = sorted(set(raw["ac_mix"]["heldout"]["units_by_os_product"]) | set(raw["ac_mix"]["training"]["units_by_os_product"]))
    L += [f"| {k} | {raw['ac_mix']['heldout']['units_by_os_product'].get(k, 0)} | "
          f"{raw['ac_mix']['training']['units_by_os_product'].get(k, 0)} | "
          f"{lg['os_product'][k]['held_out']} / {lg['os_product'][k]['cve_ids']} |" for k in keys]
    t = raw["temporal_ac_mix"]
    L += ["", f"For comparison, the temporal cohort pool (never trained on): {t['cve_ids']} CVE IDs, AC:H {t['cves_ac_h']}, "
          f"AC:L {t['cves_ac_l']}; units AC:H {t['units_ac_h']} of {t['units']}."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(VERSIONS), default="v2")
    a = ap.parse_args()
    v = VERSIONS[a.version]
    config = ROOT / v["config"]
    if config.exists():  # versioned: a stored holdout is never rewritten (§18 P6d item 1: keep the old one)
        raise SystemExit(f"{v['config']} exists; a new holdout needs a new version")
    ctx = gen.Context.load()
    ex = frozenset(HELDOUT_PRODUCTS)
    all_core = ctx.run_units(COHORT, frozenset())
    units = ctx.run_units(COHORT, ex)
    held, counts = pool.cve_holdout(units, v["share"], v["seed"], v["ach_share"], v["ach_seed"])
    held_set = frozenset(held)
    sha = pool.cve_set_sha256(held)
    mix = {"pool": ac_mix(units), "heldout": ac_mix([u for u in units if u.cve_id in held_set]),
           "training": ac_mix([u for u in units if u.cve_id not in held_set])}
    params = {"version": a.version, "share": v["share"], "seed": v["seed"], "ach_share": v["ach_share"],
              "ach_seed": v["ach_seed"], "cohort": COHORT, "heldout_products": list(HELDOUT_PRODUCTS)}
    raw = {"inputs": ctx.input_hashes, "heldout_products": list(HELDOUT_PRODUCTS),
           "exim_core_units": len(all_core) - len(units),
           "exim_core_cves": len({u.cve_id for u in all_core} - {u.cve_id for u in units}),
           "mail": {"without_exim": mail_family(ctx, ex, frozenset()),
                    "without_exim_and_heldout_cves": mail_family(ctx, ex, held_set)},
           "holdout": params | {"decision": v["decision"], "counts": counts, "cves": held, "sha256": sha},
           "ac_mix": mix, "ac_h_shares": {k: ac_h_shares(m, mix["pool"]) for k, m in mix.items()},
           "lost": lost_groups(units, held_set), "inventory_lost": inventory_lost(ctx, ex, held_set),
           "temporal_ac_mix": ac_mix(ctx.run_units("temporal", ex))}
    seeds = [v["seed"]] + ([v["ach_seed"]] if v["ach_seed"] else [])
    run = runlog.write_run("heldout_sets", raw, report(raw), seeds, params, root=OUT)
    cfg = {"decision": v["decision"], "version": a.version, "heldout_products": list(HELDOUT_PRODUCTS), "cohort": COHORT,
           "share": v["share"], "seed": v["seed"], "ach_share": v["ach_share"], "ach_seed": v["ach_seed"],
           "counts": counts, "sha256": sha, "sha256_rule": "ahg.gen.pool.cve_set_sha256: IDs in numeric order, one per line",
           "generator_inputs": ctx.input_hashes, "run": run.relative_to(ROOT).as_posix(), "cves": held}
    config.write_text(json.dumps(cfg, indent=1) + "\n")
    print(run, sha, counts)


if __name__ == "__main__":
    main()
