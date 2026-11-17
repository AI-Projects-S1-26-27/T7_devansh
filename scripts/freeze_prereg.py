"""D28 hypotheses freeze (decisions.md §13.1; approved by Devansh 17 Nov 2026, OQ-P8-1 option (a)).

Writes configs/d28_prereg.json and the frozen values of configs/d28_cohorts.json, mechanically, from:
- docs/d28/hypotheses.md (local, never committed): its SHA-256; it must be marked FROZEN;
- the P6d development pilot run (test cohort sizes, the frozen proposal of §18 P6d item 4): each test cohort's
  fixed size per n = its cell's ceil(200 / rate); the unseen-CVE cohort uses the cve_all cell (§19), never the
  cve_only diagnostic (OQ-P6-23);
- the P8a training-size pilot run (generation caps): per size, the cap with the all-13-sizes guarantee
  ("budget_all_sizes": P(shortfall) <= 0.001 / 13 per size, so <= 0.001 over all sizes by the union bound),
  for training (quota 231) and validation (quota 31), as Devansh chose (totals 13,065 and 2,432);
- the budget rule of OQ-P7-8 (accepted, decisions.md §19): its value is left null here and is filled
  mechanically from the budget pilot into configs/labels/d28.json, with no choice.
No number is typed: each comes from a run's raw.json.gz, whose provenance must be clean.

Usage: uv run python scripts/freeze_prereg.py [--check]   (--check: compare, write nothing)
The commit of configs/d28_prereg.json is the freeze; commit_time_utc is set just before committing (P0d).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES = ROOT / "docs" / "d28" / "hypotheses.md"
PREREG = ROOT / "configs" / "d28_prereg.json"
REGISTRY = ROOT / "configs" / "d28_cohorts.json"
TEST_SIZES_RUN = "results/d28_gen/p6_pilot/2026-11-17T082305Z-b79a752"
CAPS_RUN = "results/d28_gen/p8_pilot/2026-11-17T114431Z-513aff5"
CVE_CELL = {"d28_test_cve": "cve_all"}  # §19: held-out plus the rest of the pool; cve_only is a diagnostic


def load_run(rel: str, root: Path = ROOT) -> dict:
    d = root / rel
    prov = json.loads((d / "provenance.json").read_text())
    if prov["git_dirty"]:
        raise SystemExit(f"{rel}: dirty run; its numbers may not match its commit")
    return {"raw": json.loads(gzip.decompress((d / "raw.json.gz").read_bytes())), "commit": prov["git_commit"]}


def test_sizes(run: dict, registry: dict) -> dict[str, dict[str, int]]:
    """{cohort: {n: fixed size}} for every real test cohort of the registry, from the P6d pilot's proposals."""
    props = run["raw"]["summary"]["test_sizes"]
    out = {}
    for c in registry["cohorts"]:
        if c["role"] != "test":
            continue
        got = {}
        for n in c["sizes"]:
            keys = [k for k, v in props.items() if v["cohort"] == c["name"] and v["n"] == n
                    and v["cell"] == CVE_CELL.get(c["name"], v["cell"])]
            if len(keys) != 1:
                raise SystemExit(f"{c['name']} n={n}: {len(keys)} pilot proposals, expected exactly one")
            got[str(n)] = props[keys[0]]["fixed_size"]
        out[c["name"]] = got
    return out


def caps(run: dict, role: str, sizes: list[int]) -> dict:
    key = {"training": "training_budget", "validation": "validation_budget"}[role]
    rows = run["raw"]["summary"][key]
    per = {str(n): rows[f"train|n={n}"]["budget_all_sizes"] for n in sizes}
    quota = {rows[f"train|n={n}"]["quota"] for n in sizes}
    if len(quota) != 1:
        raise SystemExit(f"{role}: quotas differ by size")
    return {"quota_per_size": quota.pop(), "max_candidates_per_size": per, "total": sum(per.values()),
            "rule": run["raw"]["config"][role]["rule"] + "; frozen with the all-sizes guarantee (P(shortfall) <= "
                    "0.001 / 13 per size, union bound over the 13 sizes), decided 17 Nov 2026 by Devansh"}


def build(root: Path = ROOT, now: datetime | None = None) -> tuple[dict, dict]:
    text = (root / "docs" / "d28" / "hypotheses.md").read_bytes()
    if b"FROZEN" not in text.splitlines()[0]:
        raise SystemExit("hypotheses.md is not marked FROZEN in its title")
    registry = json.loads((root / "configs" / "d28_cohorts.json").read_text())
    labels = json.loads((root / "configs" / "labels" / "d28.json").read_text())
    budget = registry["budget_pilot"]
    t_run, c_run = load_run(TEST_SIZES_RUN, root), load_run(CAPS_RUN, root)
    real = {c["role"]: c for c in registry["cohorts"] if c["role"] in ("training", "validation")
            and not c["namespace"].startswith("p7-smoke")}
    sizes = test_sizes(t_run, registry)
    gen_caps = {role: caps(c_run, role, real[role]["sizes"]) for role in ("training", "validation")}
    for role, c in real.items():
        if gen_caps[role]["quota_per_size"] != c["per_size"]:
            raise SystemExit(f"{role}: pilot quota {gen_caps[role]['quota_per_size']} != registry {c['per_size']}")
    now = now or datetime.now(timezone.utc)
    prereg = {
        "schema": "d28-prereg/1",
        "freeze": "decisions.md §13.1 (P0c timing, P0d record): the commit of this file freezes docs/d28/hypotheses.md "
                  "before any label generation, the landscape and budget pilots included. Approved by Devansh "
                  "17 Nov 2026 (OQ-P8-1 option (a)). The git commit timestamp is authoritative.",
        "freeze_date": now.date().isoformat(),
        "commit_time_utc": now.replace(microsecond=0).isoformat(),
        "hypotheses": {"path": "docs/d28/hypotheses.md", "sha256": hashlib.sha256(text).hexdigest(), "status": "FROZEN",
                       "note": "local file (never committed); any later change is a dated amendment outside it"},
        "test_cohort_sizes": {
            "source_run": TEST_SIZES_RUN, "source_commit": t_run["commit"],
            "rule": t_run["raw"]["config"]["test_rule"] + " (frozen proposal of decisions.md §18 P6d item 4); "
                    "d28_test_cve uses the cve_all cell",
            "per_size": sizes},
        "generation_caps": {"source_run": CAPS_RUN, "source_commit": c_run["commit"], **gen_caps},
        "state_budget": {
            "rule": "OQ-P7-8 (accepted 17 Nov 2026, decisions.md §19): the smallest candidate budget under which at "
                    "least `target` of the sampled training states certify (A*+LM-cut within the budget) in both "
                    "regimes; it also applies to s0 (OQ-P7-7)",
            "candidates": budget["candidates"], "target": budget["target"], "cap": budget["cap"],
            "sample": "the budget pilot's stratified sample of the frozen training cohort (every training size)",
            "value": None,
            "value_from": "filled mechanically from the budget pilot run into configs/labels/d28.json state_budget; "
                          "no choice is made; if no candidate meets the target the value stays null and the item is "
                          "BLOCKED (Round 9)"},
        "labels_config_state_budget_at_freeze": labels["state_budget"],
    }
    reg = json.loads(json.dumps(registry))
    reg["frozen"] = True
    reg["frozen_note"] = ("Frozen 17 Nov 2026 with configs/d28_prereg.json (Devansh, OQ-P8-1 option (a)): test cohort "
                          "sizes per n from the P6d pilot (§18 P6d item 4), training and validation generation caps "
                          "per size from the P8a pilot with the all-13-sizes guarantee. Supersedes: \"" +
                          registry["frozen_note"] + "\"")
    for c in reg["cohorts"]:
        if c["name"] in sizes:
            c["per_size"] = sizes[c["name"]]
        elif c["role"] in gen_caps and not c["namespace"].startswith("p7-smoke"):
            c["max_candidates_per_size"] = gen_caps[c["role"]]["max_candidates_per_size"]
    return prereg, reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    prereg, reg = build()
    if a.check:
        old = json.loads(PREREG.read_text())
        same = {k: v for k, v in old.items() if k not in ("freeze_date", "commit_time_utc")} == \
            {k: v for k, v in prereg.items() if k not in ("freeze_date", "commit_time_utc")}
        print("prereg matches" if same else "prereg DIFFERS")
        sys.exit(0 if same else 1)
    if PREREG.exists():
        raise SystemExit(f"{PREREG} exists: the freeze is written once")
    PREREG.write_text(json.dumps(prereg, indent=1, ensure_ascii=False) + "\n")
    REGISTRY.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
    print(PREREG, prereg["hypotheses"]["sha256"], prereg["generation_caps"]["training"]["total"],
          prereg["generation_caps"]["validation"]["total"])


if __name__ == "__main__":
    main()
