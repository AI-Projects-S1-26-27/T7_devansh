"""Budget feasibility and landscape pilots (Rounds 9 and 12) on tiny documents."""

import pytest

from ahg import d28labels as dl, pilots
from p7_helpers import act, tiny_doc

LCFG = {"walk_cap": 20, "walks_per_sample": 3, "successor_cap": 50, "oracle_all_max_n": 60, "oracle_states": 20,
        "oracle_max_pops": 10000, "enum_cap": 1000}
CFG = {**LCFG, "regimes": ["ac", "uniform"], "state_budget": 1000}


def test_quantiles():
    assert pilots.quantiles([]) is None
    assert pilots.quantiles([5]) == {"n": 1, "median": 5, "q1": 5, "q3": 5}
    q = pilots.quantiles([1, 2, 3, 4, 5])
    assert (q["median"], q["q1"], q["q3"]) == (3, 2, 4)


def test_budget_task_records_states_and_oracle_agreement():
    r = pilots.budget_task(tiny_doc(), "ac", 1000, LCFG)
    assert not r["init_capped"] and any(s["init"] for s in r["states"])
    assert all(s["oracle"] == "agrees" for s in r["states"] if not s["capped"])


def test_budget_cap_marks_states_capped():
    r = pilots.budget_task(tiny_doc(), "ac", 1, LCFG)
    assert r["init_capped"] or any(s["capped"] for s in r["states"])


def rows(expansions, regime="ac", oracle="agrees"):
    return [{"n": 12, "regime": regime, "init_capped": False,
             "states": [{"expanded": e, "capped": e is None, "source": ["walk"], "oracle": oracle, "seconds": 0.1,
                         "init": i == 0} for i, e in enumerate(expansions)]}]


def test_proposal_is_the_smallest_candidate_meeting_the_target_in_every_regime():
    both = rows([5, 50, 500, 900]) + rows([5, 50, 500, 5000], "uniform")
    s = pilots.budget_summary(both, [10, 100, 1000, 10000], 10000, 0.75)
    assert s["by_regime"]["ac"]["coverage"]["100"] == 0.5 and s["by_regime"]["ac"]["coverage"]["1000"] == 1.0
    assert s["by_regime"]["uniform"]["coverage"]["1000"] == 0.75
    assert s["proposed_state_budget"] == 1000  # uniform needs 1000 as well; ac alone would allow 1000 too
    s2 = pilots.budget_summary(both, [10, 100], 10000, 0.99)
    assert s2["proposed_state_budget"] is None


def test_capped_states_never_count_as_certified_and_oracle_mismatch_aborts():
    s = pilots.budget_summary(rows([5, None]), [10], 100, 0.5)
    assert s["by_regime"]["ac"]["coverage"]["10"] == 0.5 and s["by_regime"]["ac"]["capped_states"] == 1
    with pytest.raises(RuntimeError, match="oracle"):
        pilots.budget_summary(rows([5], oracle="MISMATCH"), [10], 100, 0.5)
    with pytest.raises(ValueError, match="cap"):
        pilots.budget_summary(rows([5]), [200], 100, 0.5)


def landscape(doc):
    label = dl.label_doc(doc, "ac", CFG)
    return pilots.landscape_task(doc, label, 10000)


def test_landscape_metrics_on_the_tiny_task():
    r = landscape(tiny_doc())
    assert r["hstar"] == 12 and r["plan_actions"] == 3 and set(r["R"]) == set(pilots.LANDSCAPE_HEURISTICS)
    assert r["R"]["lmcut"] == r["runs"]["lmcut"]["expanded"] / 3
    assert r["gap_abs_init"] >= 0 and 0 <= r["gap_rel_init"] <= 1
    assert r["and_arity"] == {1: 3, 2: 1} and r["local_escalation"] is True  # "local c" is on the canonical plan


def test_remote_only_plan_has_no_local_escalation():
    d = tiny_doc()
    d["task"]["actions"] = [act("remote a", [0], [1]), act("remote z", [1], [2]), act("remote y", [1, 2], [3])]
    assert landscape(d)["local_escalation"] is False


def test_landscape_refuses_uncertified_or_wrong_regime_labels():
    d = tiny_doc()
    with pytest.raises(ValueError, match="certified ac"):
        pilots.landscape_task(d, dl.label_doc(d, "uniform", CFG), 1000)
    with pytest.raises(ValueError, match="certified ac"):
        pilots.landscape_task(d, dl.label_doc(d, "ac", {**CFG, "state_budget": 1}), 1000)


def test_summary_groups_by_stratum_and_applies_the_near_exact_rule():
    r = landscape(tiny_doc())
    s = pilots.landscape_summary([r, r])["tree|3"]
    assert s["tasks"] == 2 and s["trivial_tasks"] == 0 and s["and_arity"] == {1: 6, 2: 2}
    exact = s["gap_rel_init"]["median"] <= 0.05 and s["R"]["lmcut"]["median"] <= 1.5
    assert s["near_exact"] == exact
    trivial = {**r, "plan_actions": 0, "R": {h: None for h in r["R"]}}
    assert pilots.landscape_summary([trivial])["tree|3"]["trivial_tasks"] == 1
