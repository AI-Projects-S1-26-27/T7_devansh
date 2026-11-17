"""D28 label generation: outcomes recorded separately, both regimes, resumable, content-addressed (Round 9)."""

import gzip
import json

import pytest

from ahg import d28labels as dl
from ahg.gen.generate import doc_digest
from p7_helpers import tiny_doc

CFG = {"regimes": ["ac", "uniform"], "state_budget": 1000, "walk_cap": 20, "walks_per_sample": 3,
       "successor_cap": 50, "oracle_all_max_n": 60, "oracle_states": 20, "oracle_max_pops": 10000, "enum_cap": 1000}


def test_config_file_is_valid_and_budget_frozen_by_the_rule():
    # Frozen 17 Nov 2026 (decisions.md §20): the OQ-P7-8 rule's value from the budget pilot (checked against the
    # run in test_prereg.py). A null budget still refuses without an explicit value.
    cfg = dl.load_config()
    assert cfg["state_budget"] == 100 and cfg["regimes"] == ["ac", "uniform"]
    assert dl.with_budget(cfg, None)["state_budget"] == 100
    with pytest.raises(ValueError, match="not frozen"):
        dl.with_budget({**cfg, "state_budget": None}, None)
    assert dl.with_budget(cfg, 500)["state_budget"] == 500


def test_oracle_is_never_sampled_below_cohort_size():
    with pytest.raises(ValueError, match="sampled"):
        dl.check_config({**CFG, "oracle_all_max_n": 12}, max_n=30)
    dl.check_config(CFG, max_n=60)
    with pytest.raises(ValueError, match="unknown"):
        dl.check_config({**CFG, "bogus": 1})


def test_labelled_task_has_regime_specific_optimum():
    d = tiny_doc()
    ac, un = dl.label_doc(d, "ac", CFG), dl.label_doc(d, "uniform", CFG)
    assert ac["outcome"] == un["outcome"] == dl.LABELLED and ac["certified"]
    assert (ac["hstar_init"], un["hstar_init"]) == (12, 3)  # 4 + 4 + 4 under ac; 3 actions under uniform
    assert ac["key"] != un["key"] and ac["rows"] and all(isinstance(h, int) for _, h, _ in ac["rows"])


def test_unsolvable_task_is_a_dead_end_with_no_labels():
    rec = dl.label_doc(tiny_doc(solvable=False), "ac", CFG)
    assert rec["outcome"] == dl.DEAD_END and rec["rows"] == [] and not rec["certified"] and rec["dead_ends"]


def test_generator_flag_must_match_the_exact_precheck():
    d = tiny_doc()
    d["solvable"] = False
    with pytest.raises(Exception, match="solvable"):
        dl.label_doc(d, "ac", CFG)


def test_budget_exhaustion_is_its_own_outcome_and_labels_nothing():
    rec = dl.label_doc(tiny_doc(), "ac", {**CFG, "state_budget": 1})
    assert rec["outcome"] == dl.BUDGET_EXHAUSTED and rec["rows"] == [] and rec["hstar_init"] is None
    assert rec["budget_exhausted"] and not rec["certified"]


def test_solver_error_is_recorded_not_labelled(monkeypatch):
    def boom(*a, **k):
        raise MemoryError("out of memory")
    monkeypatch.setattr(dl, "label_instance", boom)
    rec = dl.label_doc(tiny_doc(), "ac", CFG)
    assert rec["outcome"] == dl.SOLVER_ERROR and rec["error"]["type"] == "MemoryError" and rec["rows"] == []


def test_cross_check_mismatch_aborts_the_run(monkeypatch):
    from ahg.measure import HardFailure

    def bad(*a, **k):
        raise HardFailure("oracle 1 != label 2")
    monkeypatch.setattr(dl, "label_instance", bad)
    with pytest.raises(HardFailure):
        dl.label_doc(tiny_doc(), "ac", CFG)


def test_per_state_budget_hits_are_listed_not_labelled():
    # with a budget that lets s0 through but not every state, hits appear in budget_exhausted, never in rows
    rec = dl.label_doc(tiny_doc(), "ac", {**CFG, "state_budget": 3})
    if rec["outcome"] == dl.LABELLED:
        hit = {b["state"] for b in rec["budget_exhausted"]}
        assert not hit & {r[0] for r in rec["rows"]}


def test_run_is_resumable_and_content_addressed(tmp_path):
    docs = [tiny_doc("t/0"), tiny_doc("t/1", solvable=False)]
    first = dl.label_docs(docs, CFG, root=tmp_path)
    assert first["counts"]["ac|written"] == 2 and first["counts"].get("ac|resumed", 0) == 0
    again = dl.label_docs(docs, CFG, root=tmp_path)
    assert again["counts"]["ac|resumed"] == 2 and "ac|written" not in again["counts"]
    assert [f["file_sha256"] for f in first["files"]] == [f["file_sha256"] for f in again["files"]]
    assert first["counts"]["ac|DEAD_END"] == 1 and first["counts"]["uniform|LABELLED"] == 1
    key = dl.record_key(doc_digest(docs[0]), "ac", CFG)
    assert dl.read_record(tmp_path, "ac", key)["doc_digest"] == doc_digest(docs[0])


def test_a_config_change_gives_new_keys():
    d = doc_digest(tiny_doc())
    assert dl.record_key(d, "ac", CFG) != dl.record_key(d, "ac", {**CFG, "state_budget": 999})
    assert dl.record_key(d, "ac", CFG) != dl.record_key(d, "uniform", CFG)


def test_shards_partition_the_work_and_never_overwrite(tmp_path):
    docs = [tiny_doc(f"t/{i}") for i in range(4)]
    a = dl.label_docs(docs, CFG, root=tmp_path, shard=(0, 2))
    b = dl.label_docs(docs, CFG, root=tmp_path, shard=(1, 2))
    assert {f["doc_digest"] for f in a["files"]}.isdisjoint({f["doc_digest"] for f in b["files"]})
    assert len(dl.load_labels(docs, CFG, root=tmp_path)) == 2 * len({doc_digest(d) for d in docs})
    rec = dl.read_record(tmp_path, "ac", dl.record_key(doc_digest(docs[0]), "ac", CFG))
    with pytest.raises(FileExistsError):
        dl.write_record(tmp_path, rec)


def test_a_tampered_record_is_refused(tmp_path):
    d = tiny_doc()
    dl.label_docs([d], CFG, root=tmp_path)
    key = dl.record_key(doc_digest(d), "ac", CFG)
    path = tmp_path / "ac" / key[:2] / f"{key}.json.gz"
    rec = json.loads(gzip.decompress(path.read_bytes()))
    rec["key"] = "0" * 64
    path.write_bytes(gzip.compress(json.dumps(rec).encode()))
    with pytest.raises(ValueError, match="content address"):
        dl.read_record(tmp_path, "ac", key)
