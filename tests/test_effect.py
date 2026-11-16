"""P3 effect module: pool, packets, annotation checks, kappa, merge (D28 Rounds 1-2; OQ-P3-1 label sets)."""

import pytest

from ahg import effect as ef

CFG = {"pool_stages": ["remote_candidate", "local_candidate"], "role_ambiguous_stages": ["role_not_server", "local_candidate"],
       "os_sample": {"family": "os", "stage": "local_candidate", "per_group": 2, "seed": 1},
       "pilot": {"seed": 1, "remote": 1, "role_ambiguous": 1, "local_service": 1, "local_os_per_os": 1}}


def unit(stage, family="http", os="linux", role="server", cohort="core", product="p"):
    return {"stage": stage, "family": family, "os": os, "role": role, "cohort": cohort, "product": product}


def row(cid, av, units, desc="Attacker can execute arbitrary code.", msrc=None):
    return {"cve_id": cid, "description": desc, "msrc": msrc, "cwe": [["CWE-78", "n", "Primary"]], "references": [],
            "vector_resolution": {"candidates": {"AV": [av]}, "values": {"AV": av, "PR": "N"}}, "units": units}


def item(ls="remote", roles=()):
    return {"cve_id": "CVE-1", "label_set": ls, "texts": {"nvd_description": "Allows root access\n to the host."},
            "units": [{"product": p, "role_question": True} for p in roles] or [{"product": "x", "role_question": False}]}


def ann(label="RCE_CONFIRMED", tp=None, ev=None, roles=None, **kw):
    return {"cve_id": "CVE-1", "label": label, "target_privilege": tp, "effects": ["code_execution"],
            "evidence": ev if ev is not None else [{"source": "nvd_description", "span": "root access to the host"}],
            "roles": roles or {}, "text_product_mismatch": False, **kw}


def test_pool_includes_candidates_and_ambiguous_roles_and_excludes_os_and_cohortless():
    rows = [row("A", "N", [unit("remote_candidate")]),
            row("B", "N", [unit("role_not_server", role="ambiguous")]),
            row("C", "N", [unit("role_not_server", role="client")]),
            row("D", "L", [unit("local_candidate", family="os", role="os_component")]),
            row("E", "N", [unit("remote_candidate", cohort=None)])]
    full, os_local = ef.pool_units(rows, CFG)
    assert sorted(c for c, _ in full) == ["A", "B"]
    assert [c for c, _ in os_local] == ["D"]


def test_os_sample_is_seeded_per_group_and_small_groups_taken_whole():
    os_local = [(f"C{i}", unit("local_candidate", "os", "linux", "os_component")) for i in range(5)]
    os_local += [("W", unit("local_candidate", "os", "windows", "os_component"))]
    s1, s2 = ef.sample_os(os_local, CFG), ef.sample_os(os_local, CFG)
    assert s1 == s2
    assert sum(u["os"] == "linux" for _, u in s1) == 2 and [c for c, _ in s1 if c == "W"] == ["W"]


def test_label_set_follows_single_av_only():
    assert ef.label_set_of(row("A", "N", [])) == "remote" and ef.label_set_of(row("A", "L", [])) == "local"
    r = row("A", "N", [])
    r["vector_resolution"]["candidates"]["AV"] = ["N", "L"]
    assert ef.label_set_of(r) is None


def test_packet_has_msrc_ghsa_and_truncated_faq_and_marks_role_question():
    msrc = {"impacts": ["Elevation of Privilege"], "title": "T", "faq": "x" * 50, "cna_note": []}
    r = row("A", "L", [], msrc=msrc)
    r["references"] = [{"url": "https://github.com/o/r/security/advisories/GHSA-aaaa-bbbb-cccc", "tags": []}]
    p = ef.packet(r, [unit("local_candidate", role="ambiguous")], {"GHSA-aaaa-bbbb-cccc": "text"}, 10)
    assert p["texts"]["msrc_impact"] == "Elevation of Privilege" and p["texts"]["msrc_faq"] == "x" * 10
    assert p["texts"]["ghsa:GHSA-aaaa-bbbb-cccc"] == "text"[:10] and p["units"][0]["role_question"]
    assert "cwe" in p["supporting"] and "cwe" not in p["texts"]


def test_valid_remote_and_local_annotations_pass():
    assert ef.check_annotation(ann(), item()) == []
    assert ef.check_annotation(ann("ADMIN_CONFIRMED", "root"), item("local")) == []
    assert ef.check_annotation(ann("UNKNOWN", "unspecified", ev=[]), item("local")) == []


@pytest.mark.parametrize("bad,ls,frag", [
    (ann("ADMIN_CONFIRMED", "root"), "remote", "not in the remote set"),
    (ann("RCE_CONFIRMED", "root"), "remote", "must be null"),
    (ann("ADMIN_CONFIRMED", "other"), "local", "needs target_privilege"),
    (ann("NON_ADMIN", "root"), "local", "cannot carry admin"),
    (ann("RCE_CONFIRMED", ev=[]), "remote", "needs at least one evidence span"),
    (ann(ev=[{"source": "nvd_description", "span": "made up words"}]), "remote", "not verbatim"),
    (ann(ev=[{"source": "cwe", "span": "CWE-78"}]), "remote", "not a quotable text"),
    (ann(ev=[{"source": "nvd_description", "span": "r" * 301}]), "remote", "longer than"),
])
def test_bad_annotations_are_rejected(bad, ls, frag):
    assert any(frag in e for e in ef.check_annotation(bad, item(ls)))


def test_span_match_ignores_line_wrapping():
    assert ef.check_annotation(ann(ev=[{"source": "nvd_description", "span": "Allows root access to the host."}]), item()) == []


def test_roles_must_cover_exactly_the_open_products_with_evidence():
    it = item(roles=("srv",))
    ok = ann(roles={"srv": {"role": "server", "source": "nvd_description", "span": "root access"}})
    assert ef.check_annotation(ok, it) == []
    assert any("roles must cover" in e for e in ef.check_annotation(ann(), it))
    unclear = ann(roles={"srv": {"role": "unclear", "source": None, "span": None}})
    assert ef.check_annotation(unclear, it) == []
    nospan = ann(roles={"srv": {"role": "client", "source": "nvd_description", "span": "nope"}})
    assert any("role srv" in e for e in ef.check_annotation(nospan, it))


def test_kappa_known_values():
    assert ef.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert ef.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert ef.cohen_kappa(["a", "a"], ["a", "a"]) is None  # chance agreement 1: undefined
    assert ef.cohen_kappa([], []) is None
    assert ef.cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "a"]) == pytest.approx(0.5)


def test_agreement_per_field_and_label_set():
    a1, a2 = ann("RCE_CONFIRMED"), ann("NON_RCE")
    r = ef.agreement([(a1, a1, "remote"), (a1, a2, "remote")])
    assert r["label"]["n"] == 2 and r["label"]["agree"] == 0.5 and r["label/remote"]["n"] == 2


def test_merge_takes_agreed_then_adjudicated_never_a_third_value():
    a, b = ann("RCE_CONFIRMED"), ann("NON_RCE")
    assert ef.disputes(a, b, "remote") == ["label"]
    assert ef.merge(a, b, "remote", {"label": {"pick": "B"}})["fields"]["label"] == "NON_RCE"
    assert ef.merge(a, b, "remote", {"label": {"pick": "REVIEW_REQUIRED"}})["fields"]["label"] == "REVIEW_REQUIRED"
    m = ef.merge(a, b, "remote", {})
    assert m["fields"]["label"] == "REVIEW_REQUIRED" and m["how"]["label"] == "unadjudicated"
    assert ef.merge(a, a, "remote", {})["how"]["label"] == "agreed"


def test_ruling_check():
    it = item()
    assert ef.check_ruling({"pick": "A", "source": "nvd_description", "span": "root access"}, it) == []
    assert ef.check_ruling({"pick": "REVIEW_REQUIRED"}, it) == []
    assert ef.check_ruling({"pick": "C"}, it)
    assert ef.check_ruling({"pick": "A", "source": "nvd_description", "span": "invented"}, it)


def test_pilot_select_is_seeded_stratified_and_unique():
    items = [{"cve_id": f"R{i}", "label_set": "remote", "units": [unit_view(False, "http", "linux")]} for i in range(5)]
    items += [{"cve_id": f"S{i}", "label_set": "local", "units": [unit_view(False, "http", "linux")]} for i in range(3)]
    items += [{"cve_id": f"M{i}", "label_set": "remote", "units": [unit_view(True, "dns", "linux")]} for i in range(3)]
    items += [{"cve_id": "OL", "label_set": "local", "units": [unit_view(False, "os", "linux")]},
              {"cve_id": "OW", "label_set": "local", "units": [unit_view(False, "os", "windows")]}]
    p1, p2 = ef.pilot_select(items, CFG), ef.pilot_select(items, CFG)
    assert p1 == p2 and len({i["cve_id"] for i in p1}) == len(p1) == 5
    assert {"OL", "OW"} <= {i["cve_id"] for i in p1}


def unit_view(amb, family, os):
    return {"product": "p", "family": family, "os": os, "role_question": amb}


def test_text_product_mismatch_must_be_boolean():
    bad = ann()
    bad["text_product_mismatch"] = "yes"
    assert any("text_product_mismatch" in e for e in ef.check_annotation(bad, item()))


def test_sample_cells_quota_redistribution_and_seed():
    cells = {("a",): [f"a{i}" for i in range(50)], ("b",): [f"b{i}" for i in range(50)], ("c",): ["c0", "c1"]}
    s = ef.sample_cells(cells, 20, 1)
    assert len(s) == 20 and sum(v == ("c",) for v in s.values()) == 2  # small cell taken whole
    assert sum(v == ("a",) for v in s.values()) == 9 and sum(v == ("b",) for v in s.values()) == 9
    assert s == ef.sample_cells(cells, 20, 1) and s != ef.sample_cells(cells, 20, 2)
    assert len(ef.sample_cells({("a",): ["x"]}, 20, 1)) == 1  # fewer CVEs than n: take all


def test_class_metrics():
    m = ef.class_metrics(["A", "A", "B", "B"], ["A", "B", "B", "B"], ("A", "B", "C"))
    assert m["classes"]["A"]["precision"] == 0.5 and m["classes"]["A"]["recall"] == 1.0
    assert m["classes"]["B"]["precision"] == 1.0 and m["classes"]["B"]["recall"] == pytest.approx(2 / 3)
    assert m["classes"]["C"]["precision"] is None and m["classes"]["C"]["recall"] is None and m["accuracy"] == 0.75


def test_eligible_applies_role_rule_to_remote_and_ac_filter_to_all():
    r = {"ac_status": "explicit", "label_set": "remote", "label": "RCE_CONFIRMED", "role": "server"}
    assert ef.eligible(r)
    assert not ef.eligible({**r, "role": "client"}) and not ef.eligible({**r, "role": "unclear"})
    assert not ef.eligible({**r, "ac_status": "conflict"}) and not ef.eligible({**r, "label": "NON_RCE"})
    loc = {"ac_status": "explicit", "label_set": "local", "label": "ADMIN_CONFIRMED", "role": "os_component"}
    assert ef.eligible(loc) and not ef.eligible({**loc, "label": "UNKNOWN"}) and not ef.eligible({**loc, "ac_status": "absent"})


def test_human_check_row_has_text_only_and_empty_answers():
    it = {"cve_id": "C", "label_set": "local", "texts": {"nvd_description": "d", "ghsa:GHSA-x": "g"},
          "units": [{"product": "p", "role_question": True}]}
    r = ef.human_check_row(it)
    assert list(r) == ef.HUMAN_COLUMNS and r["answer_effect"] == "" and r["role_question_products"] == "p" and "GHSA" in r["other_texts"]
    assert not any("label" in c and c != "label_set" for c in ef.HUMAN_COLUMNS)
