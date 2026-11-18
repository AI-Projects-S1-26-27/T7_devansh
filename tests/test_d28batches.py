"""D28 cohort -> LabelBatch bridge (ahg.d28batches)."""

import pytest

from ahg import cohorts as co
from ahg import d28batches, d28labels
from ahg.learning import d28encode

REG = co.load_registry()
VAL_RECORD = d28batches.RECORDS / "d28_val.json"


def test_test_cohorts_are_refused():
    test = next(c for c in REG.cohorts.values() if c.is_test)
    with pytest.raises(PermissionError):
        d28batches.cohort_batch(test.name, "ac", reg=REG)


def test_cohort_refs():
    assert d28batches.is_cohort_ref("cohort:d28_val")
    assert not d28batches.is_cohort_ref("data/labels/val-677e9fe4-72e8821")


@pytest.mark.skipif(not VAL_RECORD.exists(), reason="d28_val not built in this checkout")
@pytest.mark.parametrize("regime", ["ac", "uniform"])
def test_val_batch_matches_label_records(regime):
    b = d28batches.cohort_batch("d28_val", regime, reg=REG)
    assert b.manifest["split"] == "val" and b.manifest["cost_regime"] == regime
    if not b.examples:
        pytest.skip("d28_val not labelled in this checkout")
    labelled = {ex.instance for ex in b.examples}
    failed = {f["instance"] for f in b.failures}
    assert not labelled & failed  # a member is either labelled or listed as a failure, never both
    for inst in b.instances:
        assert inst is None or isinstance(inst, d28encode.D28Instance) and inst.regime == regime
    # every example is a certified row of that member's own label record, h* unchanged
    from ahg.gen.generate import doc_digest
    rec = __import__("json").loads(VAL_RECORD.read_text())
    docs = [d for _, d in co.open_members(rec, REG)]
    cfg = d28labels.load_config()
    ex = b.examples[0]
    lab = d28labels.read_record(d28labels.LABEL_ROOT, regime,
                                d28labels.record_key(doc_digest(docs[ex.instance]), regime, cfg))
    assert lab["outcome"] == d28labels.LABELLED
    assert [hex(ex.state), ex.hstar] in [[r[0], r[1]] for r in lab["rows"]]
    # the initial state's label is the certified optimum h*(s0)
    s0 = b.instances[ex.instance].task.init
    hs0 = [e.hstar for e in b.examples if e.instance == ex.instance and e.state == s0]
    assert hs0 == [lab["hstar_init"]]


def test_onpolicy_protocol_follows_round9_budgets():
    import json
    from ahg.onpolicy import check_protocol
    p = check_protocol(json.loads((d28batches.ROOT / "configs/training/d28_onpolicy_protocol.json").read_text()))
    lab = d28labels.load_config()
    assert p["search_budget"] == 200_000  # Round 9 Budgets: the evaluation budget
    assert (p["state_budget"], p["oracle_max_pops"], p["enum_cap"]) == (
        lab["state_budget"], lab["oracle_max_pops"], lab["enum_cap"])
