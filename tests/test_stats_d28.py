"""The D28 Round 12 verdict rule (stats.py, P0b details) on synthetic data with known answers."""

import math

import pytest

from ahg.stats import (INCONCLUSIVE, REFUTED, SUPPORTED, commonly_solved, coverage, decide_family,
                       nested_bootstrap_d28, verdict_d28, wilcoxon)

LOG_HALF = math.log(0.5)


def test_commonly_solved_and_coverage():
    exp = {"gnn": {"t1": 5, "t2": None, "t3": 7, "t4": 2}, "base": {"t1": 9, "t2": 4, "t3": None, "t4": 3}}
    assert commonly_solved(exp) == ["t1", "t4"]
    assert coverage(exp["gnn"], ["t1", "t2", "t3", "t4"]) == 0.75
    with pytest.raises(ValueError):
        commonly_solved({"gnn": {"t1": 1}, "base": {"t2": 1}})  # a task missing from one method
    with pytest.raises(ValueError):
        coverage(exp["gnn"], [])


def test_bootstrap_of_a_constant_is_that_constant():
    logs = [[[LOG_HALF] * 3] * 3] * 5
    assert nested_bootstrap_d28(logs, "k", n=1000) == pytest.approx([0.5, 0.5, 0.5])


def test_two_networks_known_percentiles():
    """Networks with log ratios a < b everywhere: a resample is all-a with probability 1/4, all-b with 1/4,
    so the 2.5% and 97.5% points are a and b; the point estimate is the mean (a + b) / 2."""
    a, b = math.log(0.4), math.log(0.8)
    logs = [[[a, a], [a, a], [a, a]], [[b, b], [b, b], [b, b]]]
    est, lo, hi = nested_bootstrap_d28(logs, "k")
    assert est == pytest.approx(math.exp((a + b) / 2)) and (lo, hi) == pytest.approx((0.4, 0.8))


def test_seeds_are_resampled_inside_networks():
    """One network, seeds a and b: the network draw is fixed, the seed draw gives the same 1/4 tails."""
    a, b = math.log(0.4), math.log(0.8)
    _, lo, hi = nested_bootstrap_d28([[[a, a], [b, b]]], "k")
    assert (lo, hi) == pytest.approx((0.4, 0.8))


def test_bootstrap_is_deterministic_and_ignores_missing():
    logs = [[[LOG_HALF, -0.1], [LOG_HALF]], [[-0.3], [-0.2, -0.4]]]
    assert nested_bootstrap_d28(logs, "k", n=500) == nested_bootstrap_d28(logs, "k", n=500)
    assert nested_bootstrap_d28(logs, "k", n=500)[0] == pytest.approx(math.exp((LOG_HALF * 2 - 0.1 - 0.3 - 0.2 - 0.4) / 6))
    with pytest.raises(ValueError):
        nested_bootstrap_d28([[[float("inf")]]], "k")
    with pytest.raises(ValueError):
        nested_bootstrap_d28([], "k")


@pytest.mark.parametrize("ci, p, cg, co, want", [
    ([0.5, 0.4, 0.6], 0.001, 1.0, 1.0, SUPPORTED),
    ([0.90, 0.8, 0.99], 0.009, 0.9, 0.9, SUPPORTED),  # boundaries: ratio = 0.90, equal coverage
    ([0.5, 0.4, 0.6], 0.001, 0.9, 0.95, INCONCLUSIVE),  # GNN covers fewer tasks
    ([0.5, 0.4, 1.0], 0.001, 1.0, 1.0, INCONCLUSIVE),  # CI upper not < 1
    ([0.5, 0.4, 0.6], 0.01, 1.0, 1.0, INCONCLUSIVE),  # Holm p not < 0.01
    ([0.91, 0.85, 0.95], 0.001, 1.0, 1.0, INCONCLUSIVE),  # ratio above 0.90
    ([1.0, 0.95, 1.05], 0.5, 1.0, 1.0, REFUTED),  # CI lower > 0.90
    ([1.0, 0.90, 1.05], 0.5, 1.0, 1.0, INCONCLUSIVE),  # lower exactly 0.90 is not > 0.90
    ([1.5, 1.2, 1.9], 0.001, 0.5, 1.0, REFUTED),  # GNN worse: refuted whatever the coverage
])
def test_verdict_table(ci, p, cg, co, want):
    assert verdict_d28(ci, p, cg, co) == want


def _win(n_tasks, name, cov=(1.0, 1.0)):
    """n_tasks networks, 3 seeds, 1 task each; every task's GNN/X log ratio is negative and distinct, so the
    exact two-sided Wilcoxon p is 2 / 2^n_tasks."""
    vals = [LOG_HALF - 0.01 * k for k in range(n_tasks)]
    return {"name": name, "logs": [[[v]] * 3 for v in vals], "task_values": vals,
            "coverage_gnn": cov[0], "coverage_other": cov[1]}


def test_wilcoxon_p_of_the_synthetic_win():
    assert wilcoxon(_win(8, "x")["task_values"])["p"] == pytest.approx(2 / 256)


def test_holm_across_the_family_changes_the_verdict():
    """n = 8: p = 2/256 = 0.0078 < 0.01 alone (SUPPORTED); Holm over 3 equal p's gives 3p = 0.0234 (INCONCLUSIVE).
    n = 10: p = 2/1024; 3p = 0.0059 < 0.01, still SUPPORTED in a family of 3."""
    alone = decide_family([_win(8, "a")], n=2000)
    assert alone[0]["verdict"] == SUPPORTED and alone[0]["p_holm"] == pytest.approx(2 / 256)
    fam8 = decide_family([_win(8, n) for n in "abc"], n=2000)
    assert [r["verdict"] for r in fam8] == [INCONCLUSIVE] * 3
    assert all(r["p_holm"] == pytest.approx(3 * 2 / 256) for r in fam8)
    fam10 = decide_family([_win(10, n) for n in "abc"], n=2000)
    assert [r["verdict"] for r in fam10] == [SUPPORTED] * 3
    assert 0.45 < fam10[0]["ratio"] < 0.5 and fam10[0]["ci"][1] < 1


def test_family_coverage_and_refutation():
    lose_vals = [0.01 * k for k in range(-5, 6)]  # log ratios around 0: GNN no better than X
    lose = {"name": "lose", "logs": [[[v]] * 3 for v in lose_vals], "task_values": lose_vals,
            "coverage_gnn": 1.0, "coverage_other": 1.0}
    res = decide_family([_win(12, "win"), _win(12, "low_cov", cov=(0.8, 0.9)), lose], n=2000)
    assert [r["verdict"] for r in res] == [SUPPORTED, INCONCLUSIVE, REFUTED]
    with pytest.raises(ValueError):
        decide_family([_win(8, "a"), _win(8, "a")])
