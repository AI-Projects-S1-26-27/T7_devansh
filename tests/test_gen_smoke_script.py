"""P6 smoke script: aggregation and report rendering on synthetic candidate rows."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen_d28_smoke", ROOT / "scripts" / "gen_d28_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def row(arch, n, solvable, length=None, **kw):
    r = {"archetype": arch, "n": n, "cohort": "core", "os_linux": 0.5, "p": 1.0, "solvable": solvable,
         "reason": None if solvable else "no_foothold_on_goal_host_reachable", "flag_unverified": False,
         "actions": 10, "remote_actions": 7, "local_actions": 3, "merged_transitions": 2, "matches": 5,
         "remote_matches": 4, "remote_matches_unknown_principal": 3, "vulnerable_components": 6,
         "vulnerable_components_without_match": 1, "reused_configs": 2, "hosts": 12, "windows_hosts": 6, "seconds": 0.5,
         "funnel": {"hosts": 12, "hosts_with_remote_match": 3, "hosts_with_remote_action": 2, "hosts_with_foothold_reached": 1,
                    "links_with_services": 8, "links_blocked_entirely": 3, "goal_has_remote_match": True,
                    "goal_has_remote_action": True, "goal_foothold_reached": True, "goal_has_local_match": True,
                    "goal_foothold_and_local_match": True, "footholds_reached": 1, "footholds_with_local_match": 1},
         "principals_cited": 1, "goal_os": "linux", "goal_families": ["smb"], "goal_remote_matches": 1 if solvable else 0, "goal_local_matches": 1}
    if solvable:
        r["plan_ac"] = {"cost": 4 * length, "length": length, "expanded": 3, "budget_hit": False, "last_kind": "local"}
        r["plan_uniform"] = {"cost": length, "length": length, "expanded": 3, "budget_hit": False, "last_kind": "local"}
    return r | kw


def test_summary_counts_and_medians():
    rows = [row("tree", 12, True, 3), row("tree", 12, True, 5), row("tree", 12, False),
            {"archetype": "tree", "n": 12, "infeasible": "no Windows"}]
    s = smoke.summarise(rows)
    t = s["by_archetype_size"]["tree|12"]
    assert (s["tasks"], s["infeasible"]) == (3, 1)
    assert (t["tasks"], t["solvable"], t["plan_len_ac_median"], t["plan_cost_uniform_median"]) == (3, 2, 4.0, 4.0)
    assert s["remote_matches_unknown_principal"] == 9 and s["unsolvable_reasons"] == {"no_foothold_on_goal_host_reachable": 1}
    assert s["solvability_by_cohort_p_os"] == {"core|p=1.0|linux=0.5": {"solvable": 2, "tasks": 3}}
    assert s["goal_host"]["core"] == {"tasks": 3, "goal_has_remote_match": 2, "goal_has_local_match": 3,
                                      "goal_has_both": 2, "goal_os_linux": 3}


def test_report_renders_every_section():
    rows = [row("chain", 20, True, 4), row("chain", 20, False)]
    raw = {"funnel": {"label_rows": 10, "pool": 4, "remote": 1, "local": 3, "excluded": {"ac_absent": 6}},
           "unmatched_pool": {"temporal|windows|remote|SQL Server": 3},
           "inventories": {"core|p=1.0": {"windows_versions": [], "kernel": True, "unknown_fixtures": {"Windows Server 2019": 2},
                                           "unplaceable_kbs": {"Windows Server 2012 R2": ["CVE-2016-1"]},
                                           "products": [{"os": "linux", "family": "http", "product": "x", "available": True}]}},
           "zones": {"chain|20": {"internet": 1}}, "summary": smoke.summarise(rows),
           "heldout_candidates": [{"product": "x", "family": "http", "core_pool_cves": 5, "verified_vulnerable_fixtures": 1,
                                   "unaffected_fixtures": 1, "principal_entry": None, "meets_rule": False,
                                   "family_has_two_eligible": False}]}
    text = smoke.report(raw)
    for head in ("Action pool funnel", "Products and families", "Hosts per zone", "Per archetype and size", "Totals",
                 "Held-out product candidates", "Solvability by cohort, p and Linux share", "Goal host",
                 "Funnel", "Solvability per archetype x cohort", "Fixtures removed by the unknown-status rule"):
        assert head in text
