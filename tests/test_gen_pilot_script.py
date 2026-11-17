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


CFG2 = json.loads((ROOT / "configs" / "gen" / "p6_pilot_v2.json").read_text())


def test_v2_config_changes_only_the_holdout_and_carries_only_temporal():
    """§18 P6d item 4: same cells, seeds and precision as P6c; CVE holdout v2; only the temporal cell (whose pool
    no holdout touches) is carried from the P6c run; every other cell is regenerated."""
    same = {k: v for k, v in CFG.items() if k not in ("decision", "cve_holdout")}
    assert {k: v for k, v in CFG2.items() if k not in ("decision", "cve_holdout", "carried")} == same
    assert CFG2["cve_holdout"] == "configs/gen/cve_holdout_v2.json" and CFG["cve_holdout"] == "configs/gen/cve_holdout.json"
    assert CFG2["carried"]["cells"] == ["temporal"] and CFG2["carried"]["run"].endswith("2026-11-17T073113Z-024d17d")
    names = {j[0] for j in pilot.jobs(CFG2, 1, ["CVE-2020-1"], ["CVE-2020-2"])}
    assert names == {c["name"] for c in CFG2["cells"]} - {"temporal"}
    assert pilot.CONFIG.name == "p6_pilot_v2.json"


def test_summary_merges_carried_rows_but_times_only_new_ones():
    cfg = {**CFG2, "cells": [{"name": "a", "cohorts": ["d28_test_a"], "sizes": [12]},
                             {"name": "t", "cohorts": ["d28_test_t"], "sizes": [12]}]}
    new = [{"cell": "a", "n": 12, "k": k, "solvable": k < 5, "reason": None, "seconds": 1.0, "actions": 3} for k in range(10)]
    old = [{"cell": "t", "n": 12, "k": k, "solvable": k < 2, "reason": None, "seconds": 100.0, "actions": 3} for k in range(10)]
    s = pilot.summarise(new, cfg, 5.0, 2, old, {"a|n=12": {"rate": 0.9}})
    assert s["cells"]["t|n=12"]["rate"] == 0.2 and s["cells"]["t|n=12"]["source"] == CFG2["carried"]["run"]
    assert s["cells"]["a|n=12"]["source"] == "this run" and s["cells"]["a|n=12"]["previous_rate"] == 0.9
    assert s["test_sizes"]["d28_test_t|t|n=12"]["fixed_size"] == 1000
    assert s["throughput"]["tasks"] == 10 and s["throughput"]["worker_seconds_total"] == 10.0


class _Ctx:
    """Stand-in generator context: run_units filters units by cohort and excluded CVEs, like pool.restrict."""

    input_hashes = {"config": "c"}

    def __init__(self):
        from ahg.gen.pool import Unit

        self.units = [Unit(f"CVE-2020-{i}", "p:a", "http", "linux", "remote", "L", "core") for i in range(1, 5)]
        self.units += [Unit(f"CVE-2024-{i}", "p:a", "http", "linux", "remote", "L", "temporal") for i in range(1, 3)]

    def run_units(self, cohort, excluded, excluded_cves=frozenset()):
        return [u for u in self.units if u.cohort == cohort and u.cve_id not in excluded_cves]


def _carry_fixture(tmp_path, carried_cells, inputs=None):
    import gzip

    from ahg.gen import pool

    for name, ids in (("h1.json", ["CVE-2020-1"]), ("h2.json", ["CVE-2020-2"])):
        (tmp_path / name).write_text(json.dumps({"cves": ids, "sha256": pool.cve_set_sha256(ids)}))
    cells = [{"name": "t", "cohort": "temporal", "sizes": [12], "products": "heldout", "cves": "holdout"},
             {"name": "c", "cohort": "core", "sizes": [12], "products": "heldout", "cves": "holdout"}]
    base = {"heldout_products": ["exim:exim"], "cells": cells}
    old_cfg, new_cfg = base | {"cve_holdout": "h1.json"}, base | {"cve_holdout": "h2.json"}
    ctx = _Ctx()
    rows = [{"cell": c, "n": 12, "k": k, "solvable": True} for c in ("t", "c") for k in range(2)]
    old_tr = ["CVE-2020-2", "CVE-2020-3", "CVE-2020-4"]
    raw = {"inputs": inputs or ctx.input_hashes, "tasks_per_cell": 2, "config": old_cfg, "candidates": rows,
           "holdout": {"sha256": pool.cve_set_sha256(["CVE-2020-1"])}, "training_cves": {"sha256": pool.cve_set_sha256(old_tr)},
           "summary": {"cells": {"t|n=12": {"rate": 1.0}}}}
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "raw.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    new_cfg["carried"] = {"run": "run", "cells": carried_cells}
    ho, tr = pilot.holdout_sets(new_cfg, ctx, tmp_path)
    return new_cfg, ctx, ho, tr


def test_carry_reads_an_unchanged_cell(tmp_path):
    cfg, ctx, ho, tr = _carry_fixture(tmp_path, ["t"])
    rows, prev, checks = pilot.load_carried(cfg, ctx, 2, ho, tr, tmp_path)
    assert [r["cell"] for r in rows] == ["t", "t"] and prev == {"t|n=12": {"rate": 1.0}}
    assert checks["same_units_under_old_and_new_exclusions"] is True


def test_carry_refuses_a_cell_whose_pool_changed_or_other_inputs(tmp_path):
    import pytest

    cfg, ctx, ho, tr = _carry_fixture(tmp_path, ["c"])
    with pytest.raises(SystemExit, match="pool changed"):
        pilot.load_carried(cfg, ctx, 2, ho, tr, tmp_path)
    with pytest.raises(SystemExit, match="tasks per cell"):
        pilot.load_carried({**cfg, "carried": {"run": "run", "cells": ["t"]}}, ctx, 3, ho, tr, tmp_path)


def test_carry_refuses_other_generator_inputs(tmp_path):
    import pytest

    cfg, ctx, ho, tr = _carry_fixture(tmp_path, ["t"], inputs={"config": "other"})
    with pytest.raises(SystemExit, match="inputs differ"):
        pilot.load_carried(cfg, ctx, 2, ho, tr, tmp_path)
