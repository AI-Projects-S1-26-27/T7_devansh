"""P6 development pilot script (§18 item 5): seeds, cells, interval and budget arithmetic, aggregation."""

import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen_d28_pilot", ROOT / "scripts" / "gen_d28_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
CFG = json.loads((ROOT / "configs" / "gen" / "p6_pilot.json").read_text())


def test_pilot_seeds_are_disjoint_from_every_split_and_other_namespace():
    """Split seeds are '<namespace>/n<n>/<k>' with namespaces 'd28.*' (P7 registry); smoke and goldens use
    'p6-smoke-<i>', 'p7-smoke.*' and 'golden-<i>'. The pilot's all start with 'p6-pilot/'."""
    js = pilot.jobs(CFG, 3, ["CVE-2020-1"], ["CVE-2020-2"])
    seeds = [j[4] for j in js]
    assert len(seeds) == len(set(seeds)) == 3 * sum(len(c["sizes"]) for c in CFG["cells"])
    assert all(s.startswith("p6-pilot/") for s in seeds)
    assert not any(s.startswith(("d28.", "p6-smoke", "p7-smoke", "golden")) for s in seeds)
    assert CFG["namespace"] == "p6-pilot" and "/" not in CFG["namespace"]
    assert [j[3] for j in js[:3]] == [0, 1, 2]  # seed order within a cell


def test_cells_cover_every_registry_cohort_but_the_removed_ac_split():
    names = {co for c in CFG["cells"] for co in c["cohorts"]}
    assert names == {"d28_train", "d28_val", "d28_test_id", "d28_test_size30", "d28_test_size60", "d28_test_structure",
                     "d28_test_temporal", "d28_test_product", "d28_test_cve", "d28_test_os25", "d28_test_os75",
                     "d28_test_p075", "d28_test_p050", "d28_test_p025"}
    assert len({c["name"] for c in CFG["cells"]}) == len(CFG["cells"])
    assert 1.96 * math.sqrt(0.25 / CFG["tasks_per_cell"]) <= 0.02


def test_exclusion_modes():
    ho, tr = ["CVE-2020-1"], ["CVE-2020-2"]
    cell = {"products": "heldout", "cves": "holdout"}
    assert pilot.exclusions(cell, CFG, ho, tr) == (frozenset({"exim:exim"}), frozenset(ho))
    assert pilot.exclusions({"products": "none", "cves": "training"}, CFG, ho, tr) == (frozenset(), frozenset(tr))
    assert pilot.exclusions({"products": "heldout", "cves": "none"}, CFG, ho, tr)[1] == frozenset()


def test_wilson_and_budget():
    lo, hi = pilot.wilson(1200, 2401)
    r = 1200 / 2401
    assert lo < r < hi and max(hi - r, r - lo) <= 0.02
    assert pilot.wilson(0, 100)[0] == 0.0 and pilot.wilson(100, 100)[1] == 1.0
    assert abs(pilot.binom_sf(1, 1, 0.3) - 0.3) < 1e-12 and abs(pilot.binom_sf(2, 2, 0.5) - 0.25) < 1e-12
    assert abs(pilot.binom_sf(1, 3, 0.5) - 0.875) < 1e-12
    b = pilot.budget(1500, 0.5, 0.001)
    assert pilot.binom_sf(1500, b, 0.5) >= 0.999 > pilot.binom_sf(1500, b - 1, 0.5)
    assert b > 3000 and pilot.budget(10, 0.0, 0.001) is None


def test_summary_and_report_on_synthetic_rows():
    cfg = {**CFG, "cells": [{"name": "id", "cohorts": ["d28_train", "d28_test_id"], "archetype": "tree", "sizes": [12],
                             "cohort": "core", "os_linux": 0.5, "p": 1.0, "products": "heldout", "cves": "holdout"}]}
    rows = [{"cell": "id", "n": 12, "k": k, "seed": f"p6-pilot/id/n12/{k}", "solvable": k % 4 == 0,
             "reason": None if k % 4 == 0 else "no_action_gives_admin_on_goal_host", "seconds": 0.01, "actions": 50}
            for k in range(400)] + [{"cell": "id", "n": 12, "k": 400, "seed": "p6-pilot/id/n12/400",
                                     "infeasible": "x", "seconds": 0.001}]
    s = pilot.summarise(rows, cfg, 2.0, 2)
    c = s["cells"]["id|n=12"]
    assert c["tasks"] == 401 and c["solvable"] == 100 and c["infeasible"] == 1
    assert s["test_sizes"]["d28_test_id|id|n=12"]["fixed_size"] == math.ceil(200 / (100 / 401))
    assert s["training_budget"]["id|n=12"]["budget"] == pilot.budget(1500, c["wilson_low"], 0.001)
    raw = {"summary": s, "config": cfg, "holdout": {"count": 1, "sha256": "x"}}
    text = pilot.report(raw)
    assert "d28_test_id" in text and "proposed cap" in text and "tasks/s wall" in text
