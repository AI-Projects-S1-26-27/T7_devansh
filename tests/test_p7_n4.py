"""N4 statistics and classifier (Round 6, P0c/P0d), on hand-computable graphs."""

import math
import random

import pytest

from ahg import n4


def edges_of(*pairs, both=()):
    e = set(pairs)
    e |= {(b, a) for a, b in both} | set(both)
    return e


def test_path_from_the_internet():
    # 0 -> 1 -> 2 (directed path). N = 3.
    s = n4.graph_stats([0, 1, 2], {(0, 1), (1, 2)})
    assert s["density"] == pytest.approx(2 / 6)
    assert (s["in_mean"], s["out_mean"]) == (pytest.approx(2 / 3),) * 2
    # in-degrees [0,1,1]: mean 2/3, m2 = 2/9, m3 = (-2/3)^3/3 + 2*(1/3)^3/3 = -8/81 + 2/81 = -6/81
    assert s["in_var"] == pytest.approx(2 / 9) and s["in_skew"] == pytest.approx((-6 / 81) / (2 / 9) ** 1.5)
    assert (s["hops_mean"], s["hops_max"]) == (1.0, 2.0) and s["hops_var"] == pytest.approx(2 / 3)
    assert s["diameter"] == 2.0 and s["clustering"] == 0.0


def test_triangle_clusters_fully_and_zero_variance_has_zero_skew():
    s = n4.graph_stats([0, 1, 2], edges_of(both=[(0, 1), (1, 2), (0, 2)]))
    assert s["density"] == 1.0 and s["clustering"] == 1.0 and s["diameter"] == 1.0
    assert (s["in_var"], s["in_skew"], s["out_skew"]) == (0.0, 0.0, 0.0)
    assert n4.moments([2, 2, 2]) == (2.0, 0.0, 0.0)


def test_nodes_below_two_neighbours_count_zero_in_clustering():
    # star: centre 0 with leaves 1..3, no leaf-leaf edge: centre clustering 0, leaves (1 neighbour) 0
    s = n4.graph_stats([0, 1, 2, 3], {(0, 1), (0, 2), (0, 3)})
    assert s["clustering"] == 0.0 and s["diameter"] == 2.0
    # add one leaf-leaf link 1-2: centre 1/3, nodes 1 and 2 have 2 neighbours each and are fully linked
    s2 = n4.graph_stats([0, 1, 2, 3], {(0, 1), (0, 2), (0, 3), (1, 2)})
    assert s2["clustering"] == pytest.approx((1 / 3 + 1 + 1 + 0) / 4)


def test_unreachable_hosts_are_kept_hops_over_reachable_hosts_and_counted():
    # OQ-P7-6 (decided 17 Nov 2026): no error; hops over the reachable hosts only, plus the unreachable fraction.
    s = n4.graph_stats([0, 1, 2], {(0, 1)})
    assert s["unreachable_frac"] == 0.5 and (s["hops_mean"], s["hops_var"], s["hops_max"]) == (0.5, 0.25, 1.0)
    assert s["diameter"] == 1.0  # host 2 is isolated: no infinite distance enters the diameter
    s = n4.graph_stats([0, 1, 2], {(0, 1), (2, 1)})  # 2 reaches 1, but nothing reaches 2 from the internet
    assert s["unreachable_frac"] == 0.5 and s["hops_max"] == 1.0 and s["diameter"] == 2.0
    full = n4.graph_stats([0, 1, 2], {(0, 1), (1, 2)})
    assert full["unreachable_frac"] == 0.0


def doc(hosts, firewall):
    return {"network": {"hosts": hosts, "firewall": firewall}}


def test_reach_graph_follows_subnets_and_firewall_directions():
    hosts = [{"id": 0, "subnet": 0, "families": []}, {"id": 1, "subnet": 1, "families": ["http"]},
             {"id": 2, "subnet": 1, "families": ["ssh"]}, {"id": 3, "subnet": 2, "families": ["smb"]}]
    fw = [{"from": 0, "to": 1, "allow": ["http"]}, {"from": 1, "to": 2, "allow": ["smb"]},
          {"from": 2, "to": 1, "allow": []}]
    nodes, edges = n4.reach_graph(doc(hosts, fw))
    assert nodes == [0, 1, 2, 3]
    # 0->1 (http allowed), 1<->2 (same subnet), 1->3 and 2->3 (smb allowed); nothing into 0; 0 cannot reach 2 (ssh)
    assert edges == {(0, 1), (1, 2), (2, 1), (1, 3), (2, 3)}


def rows(sep: float, per=40, seed=0):
    rng = random.Random(seed)
    out = []
    for n in (12, 20):
        for arch, shift in (("tree", 0.0), ("chain", sep)):
            for _ in range(per):
                out.append({"archetype": arch, "n": n, **{f: rng.gauss(shift, 1.0) for f in n4.FEATURES}})
    return out


def test_classifier_passes_separable_and_fails_identical_families():
    ok = n4.classify(rows(8.0))
    assert set(ok) == {"12", "20"} and all(v["verdict"] == "PASS" for v in ok.values())
    assert all(len(v["fold_balanced_accuracy"]) == 5 for v in ok.values())
    assert ok["12"]["mean_balanced_accuracy"] == pytest.approx(sum(ok["12"]["fold_balanced_accuracy"]) / 5)
    bad = n4.classify(rows(0.0))
    assert all(v["verdict"] == "FAIL" and v["mean_balanced_accuracy"] < 0.8 for v in bad.values())


def test_classifier_is_deterministic():
    assert n4.classify(rows(1.0)) == n4.classify(rows(1.0))


def test_threshold_and_protocol_constants_are_the_decided_ones():
    assert (n4.PASS_THRESHOLD, n4.TREES, n4.RANDOM_STATE, n4.FOLDS) == (0.80, 500, 0, 5)
    assert len(n4.FEATURES) == 13 and "unreachable_frac" in n4.FEATURES and math.isclose(n4.moments([0, 0, 3])[2], 0.7071067811865476)


def test_too_few_networks_per_class_is_not_assessable_rather_than_a_pass_or_fail():
    r = n4.classify(rows(8.0, per=4))
    assert all(v["verdict"] == "NOT_ASSESSABLE" and v["mean_balanced_accuracy"] is None for v in r.values())
