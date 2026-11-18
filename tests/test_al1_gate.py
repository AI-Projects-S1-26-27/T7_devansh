"""AL1 gate config is pinned (fixed before any pilot run) and its verdict is mechanical."""


import pytest

from ahg import al1gate

PINNED = {
    "gap_state_share_min": 0.25, "gap_instance_share_min": 0.30, "median_best_classical_ratio_min": 1.5,
    "solvable_share_min": 0.90, "recovery_or_revisit_plan_share_min": 0.20, "route_change_instance_share_min": 0.10,
}


def test_gate_thresholds_are_the_pinned_ones():
    g = al1gate.load_gate()
    assert g["criteria"] == PINNED and g["primary_level"] == "medium"


def passing():
    return {"gap_state_share": 0.3, "gap_instance_share": 0.4, "median_best_classical_ratio": 1.6,
            "solvable_share": 0.95, "recovery_or_revisit_plan_share": 0.25, "route_change_instance_share": 0.2}


def test_verdict_passes_only_when_every_criterion_holds():
    g = al1gate.load_gate()
    assert al1gate.verdict(passing(), g)["pass"]
    for m in al1gate.RULES:
        bad = passing() | {m: 0.0}
        v = al1gate.verdict(bad, g)
        assert not v["pass"] and not v["checks"][m]["ok"]
        assert not al1gate.verdict({k: x for k, x in passing().items() if k != m}, g)["pass"]


def test_boundary_counts_as_pass():
    g = al1gate.load_gate()
    exact = {m: g["criteria"][k] for m, k in al1gate.RULES.items()}
    assert al1gate.verdict(exact, g)["pass"]


def test_malformed_gate_refused(tmp_path):
    p = tmp_path / "g.json"
    p.write_text('{"schema": "al1-gate/1", "criteria": {"gap_state_share_min": 0.25}}')
    with pytest.raises(ValueError):
        al1gate.load_gate(p)
