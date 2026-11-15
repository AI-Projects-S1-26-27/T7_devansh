"""LRN6 leakage-audit framework (D28 Round 11). The concrete feature lists are P9's; these use stand-ins."""

import json

import pytest

from ahg.learning.leakage import FeaturePolicy, LeakageError, audit, run_audit

POLICY = FeaturePolicy(frozenset({"held", "is_goal", "cost", "av_network", "ac_low"}),
                       (r"cve", r"vendor", r"product", r"version", r"\bdate\b", r"cpe", r"seed"))


def test_clean_features_pass_and_the_report_is_written(tmp_path):
    rep = run_audit(["held", "is_goal", "cost"], POLICY, "d28_train", tmp_path / "audit" / "train.json")
    assert rep["ok"] and rep["violations"] == []
    assert json.loads((tmp_path / "audit" / "train.json").read_text()) == rep


@pytest.mark.parametrize("name, reason", [("cve_id", "denied"), ("CVE_2021_44228", "denied"),
                                          ("product_token", "denied"), ("os_build_version", "denied"),
                                          ("row_seed", "denied"), ("hop_count", "not on the allowlist")])
def test_each_violation_fails_the_run_and_the_report_is_kept(tmp_path, name, reason):
    path = tmp_path / "r.json"
    with pytest.raises(LeakageError):
        run_audit(["held", name], POLICY, "d28_val", path)
    rep = json.loads(path.read_text())  # kept, even though the run fails
    assert not rep["ok"] and [v["feature"] for v in rep["violations"]] == [name]
    assert rep["violations"][0]["reason"].startswith(reason)


def test_deny_wins_and_a_contradictory_policy_is_refused():
    with pytest.raises(ValueError):
        FeaturePolicy(frozenset({"held", "cve_id"}), (r"cve",))
    with pytest.raises(Exception):
        FeaturePolicy(frozenset(), (r"(",))  # invalid regex


def test_duplicates_and_empty_lists_are_violations():
    assert audit(["held", "held"], POLICY, "s")["violations"] == [{"feature": "held", "reason": "duplicate feature name"}]
    with pytest.raises(LeakageError):
        audit([], POLICY, "s")


def test_policy_round_trips_through_json(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(POLICY.to_dict()))
    assert FeaturePolicy.from_json(p) == POLICY
    with pytest.raises(ValueError):
        FeaturePolicy.from_dict({"allow": [], "deny": [], "extra": []})
