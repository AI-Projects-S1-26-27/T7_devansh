"""AL1 pilot: aggregation over instance rows and the revisit definition (remote targets only)."""

import pytest

from ahg import al1pilot


def row(gap=0, ev=10, best=1.0, rec=0, revisit=False, route=None, solvable=True, share=0.4):
    return {"solvable": solvable, "gap_states": gap, "eval_states": ev, "best_ratio": best, "recovery_used": rec,
            "revisit": revisit, "loss_actions_on_path": 0, "eval_method": "sampled", "eval_skipped": 0,
            "oracle_s0": "agree", "loss": {"monitored_share": share},
            "ratios": {"astar_hmax": best}, "route": route or {k: False for k in
                                                               ("a_cost_up", "b_recovery", "c_revisit", "d_not_m0_optimal")}}


def test_aggregate_shares_and_medians():
    rows = [row(gap=5, ev=10, best=2.0, rec=1, route={"a_cost_up": True, "b_recovery": True, "c_revisit": False,
                                                      "d_not_m0_optimal": False}),
            row(gap=0, ev=30, best=1.0), row(solvable=False)]
    a = al1pilot.aggregate(rows)
    assert a["instances"] == 3 and a["solvable"] == 2 and a["solvable_share"] == pytest.approx(2 / 3)
    assert a["gap_state_share"] == pytest.approx(5 / 40)       # pooled over evaluated states
    assert a["gap_instance_share"] == 0.5
    assert a["median_best_classical_ratio"] == 1.5
    assert a["recovery_or_revisit_plan_share"] == 0.5
    assert a["route_change_instance_share"] == 0.5
    assert a["route_components"] == {"a_cost_up": 1, "b_recovery": 1, "c_revisit": 0, "d_not_m0_optimal": 0}


def test_aggregate_with_no_solvable_rows_reports_none():
    a = al1pilot.aggregate([row(solvable=False)])
    assert a["gap_state_share"] is None and a["median_best_classical_ratio"] is None


def test_revisit_counts_remote_entries_only():
    doc = {"metadata": {"actions": [
        {"kind": "remote", "target": [3, 0]}, {"kind": "local", "matches": [[3, 0, "x"]]},
        {"kind": "remote", "target": [5, 1]}, {"kind": "recovery"}, {"kind": "remote", "target": [3, 1]}]}}
    assert al1pilot._targets(doc, [0, 1, 2]) == [3, 5]     # entering 3, then escalating on 3: no revisit
    assert al1pilot._targets(doc, [0, 1, 3, 4]) == [3, 3]  # entering 3 twice: a revisit
