"""Held-out sets for the D28 splits (decisions.md §18, 17 Nov 2026; Round 10).

Usage: uv run python scripts/make_heldout_sets.py

- Held-out product: exim:exim (Devansh, 17 Nov 2026). Reported: which mail-family products the training
  cohort (core, every p) keeps without it, on each OS, before and after the CVE holdout.
- Unseen-CVE holdout (§18 item 6): every AC:H core-pool CVE plus random AC:L CVEs up to 30% of the core
  pool's CVE IDs, seed "cve-holdout", grouped by canonical CVE ID (ahg.gen.pool.cve_holdout). The pool is
  the training cohort's: core units with no CVE touching the held-out product (Round 10: such CVEs never
  enter training; OQ-P6-20). Reported: its AC mix against training's, by CVE and by unit, per action kind,
  and the product distribution of both sides.

Writes configs/gen/cve_holdout.json (the IDs, their SHA-256, the counts, the run) and one runlog run under
results/d28_gen/heldout_sets/<UTC>-<commit>/ (untracked). The config is committed after the run.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ahg import runlog
from ahg.gen import generate as gen
from ahg.gen import pool

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d28_gen"
CONFIG = ROOT / "configs" / "gen" / "cve_holdout.json"
HELDOUT_PRODUCTS = ("exim:exim",)
SHARE, SEED, COHORT = 0.3, "cve-holdout", "core"


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


def mail_family(ctx, excluded: frozenset[str], excluded_cves: frozenset[str]) -> dict:
    out = {}
    for p in ctx.cfg["p_values"]:
        inv = ctx.inventory(COHORT, p, excluded, excluded_cves)
        out[f"p={p}"] = [{"os": r["os"], "product": r["product"], "available": r["available"], "reason": r.get("reason")}
                         for r in inv.report["products"] if r["family"] == "mail"]
    return out


def report(raw: dict) -> str:
    h, s = raw["holdout"], raw["holdout"]["counts"]
    L = ["# Held-out sets (decisions.md §18, 17 Nov 2026)", "",
         "Every number below is read from raw.json.gz of this run.", "",
         "## Held-out product: exim:exim", "",
         f"- Core-pool units touching it: {raw['exim_core_units']}; CVE IDs: {raw['exim_core_cves']}",
         "", "Mail-family products in the training cohort (core) without exim, before / after the CVE holdout:", ""]
    for tag in ("without_exim", "without_exim_and_heldout_cves"):
        L += [f"### {tag}", "", "| p | os | product | available | reason |", "|---|---|---|---|---|"]
        for p, rows in raw["mail"][tag].items():
            L += [f"| {p} | {r['os']} | {r['product']} | {r['available']} | {r['reason']} |" for r in rows]
        L += [""]
    L += ["## Unseen-CVE holdout", "",
          f"- Pool: {COHORT} cohort, no CVE touching {', '.join(HELDOUT_PRODUCTS)}: {s['pool_cve_ids']} CVE IDs",
          f"- Target round_half_up({SHARE} x {s['pool_cve_ids']}) = {s['target']}; seed \"{SEED}\"",
          f"- AC:H CVEs (all held out): {s['ac_h_cves']} (of which with an AC:L unit too: {s['ac_h_cves_with_an_ac_l_unit']}); "
          f"AC:L-only CVEs: {s['ac_l_only_cves']}, drawn: {s['ac_l_drawn']}",
          f"- Held-out IDs: {len(h['cves'])}, SHA-256 {h['sha256']}", "",
          "AC mix, held out vs training (the pool minus the holdout):", "",
          "| side | CVE IDs | AC:H CVEs | AC:L CVEs | units | AC:H units | AC:L units | by kind |", "|---|---|---|---|---|---|---|---|"]
    for side in ("heldout", "training"):
        m = raw["ac_mix"][side]
        L.append(f"| {side} | {m['cve_ids']} | {m['cves_ac_h']} | {m['cves_ac_l']} | {m['units']} | {m['units_ac_h']} | "
                 f"{m['units_ac_l']} | {m['units_by_kind_ac']} |")
    L += ["", "Product distribution (units):", "", "| os/product | held out | training |", "|---|---|---|"]
    keys = sorted(set(raw["ac_mix"]["heldout"]["units_by_os_product"]) | set(raw["ac_mix"]["training"]["units_by_os_product"]))
    L += [f"| {k} | {raw['ac_mix']['heldout']['units_by_os_product'].get(k, 0)} | "
          f"{raw['ac_mix']['training']['units_by_os_product'].get(k, 0)} |" for k in keys]
    t = raw["temporal_ac_mix"]
    L += ["", f"For comparison, the temporal cohort pool (never trained on): {t['cve_ids']} CVE IDs, AC:H {t['cves_ac_h']}, "
          f"AC:L {t['cves_ac_l']}; units AC:H {t['units_ac_h']} of {t['units']}."]
    return "\n".join(L) + "\n"


def main() -> None:
    ctx = gen.Context.load()
    ex = frozenset(HELDOUT_PRODUCTS)
    all_core = ctx.run_units(COHORT, frozenset())
    units = ctx.run_units(COHORT, ex)
    held, counts = pool.cve_holdout(units, SHARE, SEED)
    held_set = frozenset(held)
    sha = pool.cve_set_sha256(held)
    raw = {"inputs": ctx.input_hashes, "heldout_products": list(HELDOUT_PRODUCTS),
           "exim_core_units": len(all_core) - len(units),
           "exim_core_cves": len({u.cve_id for u in all_core} - {u.cve_id for u in units}),
           "mail": {"without_exim": mail_family(ctx, ex, frozenset()),
                    "without_exim_and_heldout_cves": mail_family(ctx, ex, held_set)},
           "holdout": {"share": SHARE, "seed": SEED, "cohort": COHORT, "counts": counts, "cves": held, "sha256": sha},
           "ac_mix": {"heldout": ac_mix([u for u in units if u.cve_id in held_set]),
                      "training": ac_mix([u for u in units if u.cve_id not in held_set])},
           "temporal_ac_mix": ac_mix(ctx.run_units("temporal", ex))}
    run = runlog.write_run("heldout_sets", raw, report(raw), [], {"share": SHARE, "seed": SEED, "cohort": COHORT,
                           "heldout_products": list(HELDOUT_PRODUCTS)}, root=OUT)
    cfg = {"decision": "decisions.md §18 item 6 (17 Nov 2026, Devansh): the unseen-CVE split (Round 10) holds out every "
                       "AC:H pool CVE plus random AC:L CVEs up to 30% of the pool's CVE IDs, seed 'cve-holdout', grouped by "
                       "canonical CVE ID. Pool: core cohort, no CVE touching a held-out product (OQ-P6-20).",
           "heldout_products": list(HELDOUT_PRODUCTS), "cohort": COHORT, "share": SHARE, "seed": SEED,
           "counts": counts, "sha256": sha, "sha256_rule": "ahg.gen.pool.cve_set_sha256: IDs in numeric order, one per line",
           "generator_inputs": ctx.input_hashes, "run": run.relative_to(ROOT).as_posix(), "cves": held}
    CONFIG.write_text(json.dumps(cfg, indent=1) + "\n")
    print(run, sha, counts)


if __name__ == "__main__":
    main()
