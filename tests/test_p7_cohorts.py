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
    # decided 17 Nov 2026 (decisions.md §19): exim held out; held-out CVEs = the pinned v2 holdout (P7e; the file
    # arrived with the d28-gen merge, M4: 89 IDs, set SHA-256 fbab7f28...); test sizes null until the P6 pilot rates. P7d: training every size n = 8..20, 3,000 solvable networks in equal quotas per size;
    # validation 400, same size mix; generation limits null until a pilot on the merged generator.
    assert not REG.frozen and REG.heldout_products == ("exim:exim",)
    from ahg.gen.pool import cve_set_sha256
    assert len(REG.heldout_cves) == len(set(REG.heldout_cves)) == 89 and REG.heldout_cves_kept_by == ("d28_test_cve",)
    assert cve_set_sha256(REG.heldout_cves) == "fbab7f28173838a9c28bcee406e160ec4fb8f36a5bc0fb18ce9cc5c0f90b8c24"
    assert REG.totals == {"training": 3000, "validation": 400}
    t, v = REG.training, REG["d28_val"]
    assert t.name == "d28_train" and t.sizes == tuple(range(8, 21)) == v.sizes
    assert (t.selection, t.per_size, t.max_candidates_per_size) == ("solvable_quota", 231, None)
    assert (v.selection, v.per_size, v.max_candidates_per_size) == ("solvable_quota", 31, None)
    assert REG.quota_rounding(t) == {"target": 3000, "sizes": 13, "per_size": 231, "total": 3003, "difference": 3,
                                     "rule": "round half up of target / sizes"}
    assert REG.quota_rounding(v)["total"] == 403 and REG.quota_rounding(v)["difference"] == 3
    assert REG.quota_rounding(REG["d28_test_id"]) is None and REG.quota_rounding(REG["p7_smoke_train"]) is None
    assert all(c.per_size is None for c in TESTS)
    # in-distribution tests stay at n = 12 and 20, size shift n = 30 (60 exploratory)
    assert REG["d28_test_id"].sizes == (12, 20) and REG["d28_test_size30"].sizes == (30,)


def test_equal_quotas_round_half_up():
    assert [co.equal_quota(t, k) for t, k in ((3000, 13), (400, 13), (3000, 2), (5, 2), (7, 4), (6, 4))] == \
        [231, 31, 1500, 3, 2, 2]


def test_every_required_cohort_exists_with_its_factor():
    c = REG.cohorts
    assert c["d28_test_id"].sizes == (12, 20) and c["d28_test_size30"].sizes == (30,) and c["d28_test_size60"].sizes == (60,)
    assert c["d28_test_size60"].analysis == "exploratory" and c["d28_test_size30"].analysis == "primary"
    assert c["d28_test_structure"].archetype == "chain" and c["d28_test_structure"].requires == ("n4_pass",)
    assert c["d28_test_temporal"].gen_cohort == "temporal" and c["d28_train"].archetype == "tree"
    assert (c["d28_test_os25"].os_linux, c["d28_test_os75"].os_linux) == (0.25, 0.75)
    assert [c[f"d28_test_p{x}"].p for x in ("075", "050", "025")] == [0.75, 0.5, 0.25]
    assert c["d28_test_product"].exclude == "none" and "d28_test_ac" not in c  # no AC-shift cohort (OQ-P7-3)
    # the d28-gen excluded-CVEs parameter that blocked it is merged (M4)
    assert c["d28_test_cve"].analysis == "primary" and c["d28_test_cve"].blocked_by == ()


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


TOTALS = {"training": 3000, "validation": 400}


def test_validation_rejects_shared_namespaces_and_bad_quotas():
    cs = list(REG.cohorts.values())
    with pytest.raises(ValueError, match="differ from training"):  # P7d: validation has the training size mix
        co.validate([replace(REG["d28_val"], sizes=(12, 20), per_size=200)] + [c for c in cs if c.name != "d28_val"],
                    TOTALS, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="equal quotas"):
        co.validate([replace(REG["d28_val"], per_size=30)] + [c for c in cs if c.name != "d28_val"], TOTALS,
                    [1.0, .75, .5, .25])
    a, b = cs[1], cs[2]
    with pytest.raises(ValueError, match="unique"):
        co.validate([a, replace(b, namespace=a.namespace)] + cs[3:], TOTALS, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="equal quotas"):
        co.validate([replace(REG.training, per_size=230)] + [c for c in cs if c.name != "d28_train"], TOTALS, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="n <= 20"):
        co.validate([replace(REG.training, sizes=(12, 30), per_size=1500)] + [c for c in cs if c.name != "d28_train"], TOTALS,
                    [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="no solvability filtering"):
        co.validate([replace(REG["d28_test_id"], selection="solvable_quota")] + [c for c in cs if c.name != "d28_test_id"],
                    TOTALS, [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="validation selects solvable"):
        co.validate([replace(REG["d28_val"], selection="fixed")] + [c for c in cs if c.name != "d28_val"], TOTALS,
                    [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="positive per_size"):  # only a test cohort's size may be null
        co.validate([replace(REG["d28_val"], per_size=None)] + [c for c in cs if c.name != "d28_val"], TOTALS,
                    [1.0, .75, .5, .25])
    with pytest.raises(ValueError, match="generation limit"):
        co.validate([replace(REG.training, max_candidates_per_size=10)] + [c for c in cs if c.name != "d28_train"], TOTALS,
                    [1.0, .75, .5, .25])


def test_nothing_builds_while_unfrozen(tmp_path):
    for c in REG.cohorts.values():
        if not c.smoke:
            assert REG.problems(c, tmp_path / "d28_prereg.json")
    assert any("generation limit" in w for w in REG.problems(REG.training))
    assert any("OQ-P7-1" in w for w in REG.problems(REG["d28_test_id"]))
    # M4: the held-out IDs are known (pinned file merged), so the exclusion no longer blocks anything ...
    assert not any("held-out CVEs" in w for c in REG.cohorts.values() for w in REG.problems(c))
    # ... but without them every excluding cohort (all but the unseen-CVE split, OQ-P7b-2) is blocked again
    unknown = replace(REG, heldout_cves=None)
    assert any("held-out CVEs" in w for w in unknown.problems(REG["d28_val"]))
    assert any("held-out CVEs" in w for w in unknown.problems(REG.training))
    for c in TESTS:
        assert any("held-out CVEs" in w for w in unknown.problems(c)) == (c.name != "d28_test_cve"), c.name
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


def test_the_registry_reads_the_heldout_cves_from_the_pinned_v2_file(tmp_path):
    # P7e (OQ-P6-26): the set is configs/gen/cve_holdout_v2.json from d28-gen, in this checkout since the merge (M4).
    spec = json.loads(co.REGISTRY.read_text())["heldout_cves"]
    assert spec["source"] == "configs/gen/cve_holdout_v2.json" and spec["version"] == "v2" and spec["count"] == 89
    assert REG.heldout_cves == co.load_heldout_cves(spec, ("exim:exim",)) and len(REG.heldout_cves) == 89
    import hashlib
    ids = [f"CVE-2020-{k}" for k in range(1, 90)]
    good = {"version": "v2", "heldout_products": ["exim:exim"], "cves": ids[::-1],
            "sha256": hashlib.sha256("".join(f"{c}\n" for c in ids).encode()).hexdigest()}
    (tmp_path / "configs" / "gen").mkdir(parents=True)
    src = tmp_path / spec["source"]
    pin = {**spec, "set_sha256": good["sha256"]}
    assert co.load_heldout_cves(pin, ("exim:exim",), tmp_path) is None  # no file yet: unknown, still blocked
    src.write_text(json.dumps(good))
    assert co.load_heldout_cves(pin, ("exim:exim",), tmp_path) == tuple(ids)  # numeric order
    for bad in ({"version": "v1"}, {"cves": ids[:-1]}, {"sha256": "0" * 64}):
        src.write_text(json.dumps(good | bad))
        with pytest.raises(ValueError, match="pin"):
            co.load_heldout_cves(pin, ("exim:exim",), tmp_path)
    src.write_text(json.dumps(good | {"heldout_products": []}))
    with pytest.raises(ValueError, match="drawn without"):
        co.load_heldout_cves(pin, ("exim:exim",), tmp_path)
    with pytest.raises(ValueError, match="keys"):
        co.load_heldout_cves({**pin, "ids": None}, ("exim:exim",), tmp_path)


def test_every_excluding_cohort_hands_the_heldout_cves_to_the_generator(tmp_path, monkeypatch):
    """M4: the generator's excluded_cves parameter (d28-gen) is merged; build passes the pinned holdout for every
    cohort but the unseen-CVE split (OQ-P7b-2) and records its count and set hash."""
    from ahg.gen import generate as gen
    from ahg.gen.pool import cve_set_sha256
    held = frozenset(REG.heldout_cves)
    for c in REG.cohorts.values():
        assert REG.excluded_cves(c) == (frozenset() if c.smoke or c.name == "d28_test_cve" else held), c.name
    with pytest.raises(co.BlockedCohort, match="unknown"):
        replace(REG, heldout_cves=None).excluded_cves(REG.training)
    seen = []

    def fake(ctx, arch, seed, n, **kw):
        seen.append(kw)
        return {"seed": seed, "solvable": True, "unsolvable_reason": None, "flags": {"unverified_release": False}}

    monkeypatch.setattr(gen, "generate", fake)
    monkeypatch.setattr(co, "store_doc", lambda doc, store: doc["seed"])
    monkeypatch.setattr(co.Registry, "problems", lambda self, c, prereg=None: [])
    reg = replace(REG, cohorts={**REG.cohorts, "d28_train": replace(REG.training, sizes=(12,), per_size=2,
                                                                       max_candidates_per_size=3)})

    class Ctx:
        input_hashes: dict = {}

    rec = co.build(Ctx(), reg, "d28_train", store=tmp_path)
    assert len(seen) == 2 and all(kw["excluded_cves"] == held and kw["excluded"] == {"exim:exim"} for kw in seen)
    assert rec["excluded_cves"] == {"count": 89, "sha256": cve_set_sha256(held)}
