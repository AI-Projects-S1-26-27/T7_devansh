"""Cohort registry, seed namespaces, the test-access gate and the content-addressed store (P7, Rounds 9-10)."""

import json
from dataclasses import replace

import pytest

from ahg import cohorts as co
from ahg.splits import EVALUATION, LABEL_GENERATION
from p7_helpers import tiny_doc

REG = co.load_registry()
TESTS = [c for c in REG.cohorts.values() if c.is_test and not c.smoke]


def test_registry_loads_unfrozen_with_the_decided_sets_and_sizes():
    # decided 17 Nov 2026 (decisions.md §19): exim held out; held-out CVEs not drawn yet; training 1,500 and
    # validation 200 solvable per size; test sizes and the generation limit null until the P6 pilot rates.
    assert not REG.frozen and REG.heldout_products == ("exim:exim",)
    assert REG.heldout_cves is None and REG.heldout_cves_kept_by == ("d28_test_cve",)
    assert REG.training.name == "d28_train" and REG.training.per_size * len(REG.training.sizes) == 3000
    assert REG.training.per_size == 1500 and REG.training.max_candidates_per_size is None
    v = REG["d28_val"]
    assert (v.selection, v.per_size, v.sizes, v.max_candidates_per_size) == ("solvable_quota", 200, (12, 20), None)
    assert all(c.per_size is None for c in TESTS)


def test_every_required_cohort_exists_with_its_factor():
    c = REG.cohorts
    assert c["d28_test_id"].sizes == (12, 20) and c["d28_test_size30"].sizes == (30,) and c["d28_test_size60"].sizes == (60,)
    assert c["d28_test_size60"].analysis == "exploratory" and c["d28_test_size30"].analysis == "primary"
    assert c["d28_test_structure"].archetype == "chain" and c["d28_test_structure"].requires == ("n4_pass",)
    assert c["d28_test_temporal"].gen_cohort == "temporal" and c["d28_train"].archetype == "tree"
    assert (c["d28_test_os25"].os_linux, c["d28_test_os75"].os_linux) == (0.25, 0.75)
    assert [c[f"d28_test_p{x}"].p for x in ("075", "050", "025")] == [0.75, 0.5, 0.25]
    assert c["d28_test_product"].exclude == "none" and "d28_test_ac" not in c  # no AC-shift cohort (OQ-P7-3)
    assert c["d28_test_cve"].analysis == "primary" and c["d28_test_cve"].blocked_by


def test_one_factor_at_a_time_against_in_distribution():
    # OQ-P7b-2 (decided 17 Nov 2026): only the unseen-CVE split keeps the held-out CVEs, so it differs from
    # in-distribution in exactly that factor, and no other split differs in it.
    base = REG.factors(REG["d28_test_id"])
    expect = {"d28_test_size30": {"sizes"}, "d28_test_size60": {"sizes"}, "d28_test_structure": {"archetype"},
              "d28_test_temporal": {"gen_cohort"}, "d28_test_product": {"exclude"},
              "d28_test_cve": {"heldout_cves_excluded"},
              "d28_test_os25": {"os_linux"}, "d28_test_os75": {"os_linux"}, "d28_test_p075": {"p"},
              "d28_test_p050": {"p"}, "d28_test_p025": {"p"}}
    assert set(expect) == {c.name for c in TESTS} - {"d28_test_id"}
    for name, factors in expect.items():
        diff = {k for k, v in REG.factors(REG[name]).items() if v != base[k]}
        assert diff == factors, name


def test_every_cohort_but_the_unseen_cve_split_excludes_the_heldout_cves(tmp_path):
    real = [c for c in REG.cohorts.values() if not c.smoke]
    assert {c.name for c in real if not REG.excludes_heldout_cves(c)} == {"d28_test_cve"}
    assert REG.excludes_heldout_cves(REG.training) and REG.excludes_heldout_cves(REG["d28_val"])
    assert not any(REG.excludes_heldout_cves(c) for c in REG.cohorts.values() if c.smoke)
    for bad in ([], ["d28_val"], ["d28_test_nope"], ["p7_smoke_test"]):
        cfg = json.loads(co.REGISTRY.read_text())
        cfg["heldout_cves"]["kept_by"] = bad
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="kept_by"):
            co.load_registry(path)


def test_seed_namespaces_are_disjoint_and_never_integers():
    seen = set()
    for c in REG.cohorts.values():
        for n in c.sizes:
            for k in sorted({0, 1, (c.per_size or 2) - 1}):
                s = c.seed(n, k)
                assert s not in seen and not s.isdigit()
                seen.add(s)
    assert len({c.namespace for c in REG.cohorts.values()}) == len(REG.cohorts)
    assert all(c.namespace.startswith("p7-smoke") == c.smoke for c in REG.cohorts.values())


def test_validation_rejects_shared_namespaces_and_bad_quotas():
    cs = list(REG.cohorts.values())
    a, b = cs[1], cs[2]
    with pytest.raises(ValueError, match="unique"):
        co.validate([a, replace(b, namespace=a.namespace)] + cs[3:], 3000, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="equal quotas"):
        co.validate([replace(REG.training, per_size=1400)] + [c for c in cs if c.name != "d28_train"], 3000, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="n <= 20"):
        co.validate([replace(REG.training, sizes=(12, 30), per_size=1500)] + [c for c in cs if c.name != "d28_train"], 3000,
                    [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="no solvability filtering"):
        co.validate([replace(REG["d28_test_id"], selection="solvable_quota")] + [c for c in cs if c.name != "d28_test_id"],
                    3000, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="validation selects solvable"):
        co.validate([replace(REG["d28_val"], selection="fixed")] + [c for c in cs if c.name != "d28_val"], 3000,
                    [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="positive per_size"):  # only a test cohort's size may be null
        co.validate([replace(REG["d28_val"], per_size=None)] + [c for c in cs if c.name != "d28_val"], 3000,
                    [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="generation limit"):
        co.validate([replace(REG.training, max_candidates_per_size=10)] + [c for c in cs if c.name != "d28_train"], 3000,
                    [1.0, .75, .5, .25])


def test_nothing_builds_while_unfrozen(tmp_path):
    for c in REG.cohorts.values():
        if not c.smoke:
            assert REG.problems(c, tmp_path / "d28_prereg.json")
    assert any("held-out CVEs" in w for w in REG.problems(REG["d28_val"]))
    assert any("held-out CVEs" in w for w in REG.problems(REG.training))
    assert any("generation limit" in w for w in REG.problems(REG.training))
    assert any("OQ-P7-1" in w for w in REG.problems(REG["d28_test_id"]))
    for c in TESTS:  # OQ-P7b-2: every test cohort but the unseen-CVE split waits for the exclusion too
        assert any("held-out CVEs" in w for w in REG.problems(c)) == (c.name != "d28_test_cve"), c.name
    assert any("excluded-CVEs" in w for w in REG.problems(REG["d28_test_cve"]))
    assert REG.excluded(REG["d28_val"]) == frozenset({"exim:exim"})
    assert REG.excluded(REG["d28_test_product"]) == frozenset()
    unchosen = replace(REG, heldout_products=None)
    with pytest.raises(co.BlockedCohort):
        unchosen.excluded(REG["d28_val"])
    assert any("OQ-P6-8" in w for w in unchosen.problems(REG["d28_val"]))
    assert not REG.problems(REG["p7_smoke_train"])


def test_prereg_freeze_means_a_committed_file(tmp_path):
    assert not co.prereg_frozen(tmp_path / "d28_prereg.json")
    (tmp_path / "d28_prereg.json").write_text("{}")  # present but not committed anywhere
    assert not co.prereg_frozen(tmp_path / "d28_prereg.json")


def test_store_is_content_addressed_and_idempotent(tmp_path):
    d = tiny_doc()
    h = co.store_doc(d, tmp_path)
    assert co.store_doc(d, tmp_path) == h and co.load_doc(h, tmp_path) == d
    assert co.store_doc(tiny_doc("t/9"), tmp_path) != h
    path = tmp_path / h[:2] / f"{h}.json.gz"
    import gzip
    path.write_bytes(gzip.compress(json.dumps(tiny_doc("t/9")).encode()))
    with pytest.raises(ValueError, match="does not hash"):
        co.load_doc(h, tmp_path)


# ---- the test-access gate (Round 10)

def read(split, purpose, log, **kw):
    kw = {"dataset_sha256": "d", "eval_config_sha256": "e", "checkpoint_sha256": {"gnn@0": "c"}} | kw
    return co.record_test_read(REG, split, purpose, "abc", log=log, **kw)


def test_non_test_cohort_and_bad_purpose_are_refused(tmp_path):
    log = tmp_path / "log.jsonl"
    with pytest.raises(co.TestAccessError):
        read("d28_val", LABEL_GENERATION, log)
    with pytest.raises(co.TestAccessError):
        read("d28_test_id", "peeking", log)
    assert not log.exists()


def test_second_evaluation_read_is_refused_without_a_d_entry(tmp_path):
    log = tmp_path / "log.jsonl"
    read("d28_test_id", LABEL_GENERATION, log, dataset_sha256=None)
    read("d28_test_id", LABEL_GENERATION, log, dataset_sha256=None)  # label generation may repeat (sharded runs)
    first = read("d28_test_id", EVALUATION, log)
    with pytest.raises(co.TestAccessError, match="already read"):
        read("d28_test_id", EVALUATION, log)
    with pytest.raises(co.TestAccessError, match="D-entry"):
        read("d28_test_id", EVALUATION, log, permitted_by="because I said so")
    second = read("d28_test_id", EVALUATION, log, permitted_by="D29 bug rerun")
    assert second["permitted_by"] == "D29 bug rerun" and first["checkpoint_sha256"] == {"gnn@0": "c"}
    assert len(log.read_text().splitlines()) == 4
    read("d28_test_size30", EVALUATION, log)  # another split has its own single read


def test_evaluation_read_must_name_dataset_config_and_checkpoints(tmp_path):
    for missing in ("dataset_sha256", "eval_config_sha256", "checkpoint_sha256"):
        with pytest.raises(co.TestAccessError):
            read("d28_test_id", EVALUATION, tmp_path / "l.jsonl", **{missing: None})


def test_a_test_cohort_opens_only_through_a_logged_token(tmp_path):
    c, log = REG["d28_test_id"], tmp_path / "log.jsonl"
    record = {"cohort": c.name, "members": [], "dataset_sha256": "d"}
    with pytest.raises(co.TestAccessError, match="record_test_read"):
        co.open_members(record, REG, log=log)
    tok = read(c.name, LABEL_GENERATION, log, dataset_sha256=None)
    assert co.open_members(record, REG, access=tok, log=log) == []
    forged = dict(tok, date_utc="1999-01-01T00:00:00+00:00")
    with pytest.raises(co.TestAccessError, match="not in the test-access log"):
        co.open_members(record, REG, access=forged, log=log)
    other = read("d28_test_size30", LABEL_GENERATION, log, dataset_sha256=None)
    with pytest.raises(co.TestAccessError, match="token is for"):
        co.open_members(record, REG, access=other, log=log)
    ev = read(c.name, EVALUATION, log)
    with pytest.raises(co.TestAccessError, match="token is for"):
        co.open_members(record, REG, access=ev, purposes=(LABEL_GENERATION,), log=log)
    with pytest.raises(co.TestAccessError, match="different dataset"):
        co.open_members({**record, "dataset_sha256": "x"}, REG, access=ev, log=log)


def test_training_and_validation_need_no_token():
    assert co.open_members({"cohort": "d28_val", "members": [], "dataset_sha256": "d"}, REG) == []


def test_every_test_cohort_is_gated():
    assert TESTS and all(c.is_test for c in TESTS)
    with pytest.raises(co.TestAccessError):
        for c in TESTS:
            co.check_access(c, None, (LABEL_GENERATION,))
