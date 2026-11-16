"""P3 stage-1 phrase rules: claim a label only on explicit wording (decided 15 Nov 2026)."""

import pytest

from ahg import effect as ef
from ahg import effect_rules as er


def item(ls, desc, **texts):
    return {"cve_id": "CVE-1", "label_set": ls, "texts": {"nvd_description": desc, **texts}, "units": []}


@pytest.mark.parametrize("desc,label", [
    ("allows remote attackers to execute arbitrary code via a crafted packet.", "RCE_CONFIRMED"),
    ("Remote code execution in the parser.", "RCE_CONFIRMED"),
    ("An attacker can run arbitrary commands on the server.", "RCE_CONFIRMED"),
    ("allows remote attackers to cause a denial of service (daemon crash) via a malformed request.", "unresolved"),  # "crash" blocks NON_
    ("allows remote attackers to cause a denial of service via a long query.", "NON_RCE"),
    ("allows remote attackers to obtain sensitive information via a crafted URL.", "NON_RCE"),
    ("A buffer overflow in the parser.", "unresolved"),  # flaw type alone: strict rule
    ("An out-of-bounds read in foo.", "unresolved"),
    ("This does not allow code execution.", "unresolved"),  # negated
    ("Potentially remote code execution may be possible.", "unresolved"),  # hedged
    ("denial of service or possibly execute arbitrary code", "unresolved"),  # both effects: block word keeps NON_ off, hedge keeps RCE off
])
def test_remote_rules(desc, label):
    assert er.apply_rules(item("remote", desc))["label"] == label


def test_remote_msrc_impact_is_authoritative():
    assert er.apply_rules(item("remote", "Windows X Vulnerability", msrc_impact="Remote Code Execution"))["rule"] == "remote.msrc_impact_rce"
    # MSRC RCE blocks any NON_ label, so it cannot conflict; the positive claim is verified in stage 2 anyway.
    v = er.apply_rules(item("remote", "Information disclosure bug.", msrc_impact="Remote Code Execution"))
    assert v["label"] == "RCE_CONFIRMED"
    v = er.apply_rules(item("remote", "Windows X Vulnerability", msrc_impact="Denial of Service"))
    assert (v["label"], v["rule"]) == ("NON_RCE", "remote.non_msrc_impact")
    assert v["span"] == "Denial of Service"  # verbatim, not lowercased


@pytest.mark.parametrize("desc,label", [
    ("allows local users to gain root privileges via a crafted ioctl.", "ADMIN_CONFIRMED"),
    ("allows local users to execute arbitrary code as root.", "ADMIN_CONFIRMED"),
    ("allows local users to gain privileges.", "unresolved"),  # no stated target: never inferred
    ("A local attacker with administrator privileges can read the file.", "unresolved"),  # precondition, not a gain
    ("does not allow an attacker to gain root privileges.", "unresolved"),
    ("local users can cause a denial of service (memory consumption).", "NON_ADMIN"),
])
def test_local_rules(desc, label):
    assert er.apply_rules(item("local", desc))["label"] == label


def test_local_msrc_eop_without_target_is_unresolved_and_faq_target_is_admin():
    base = {"msrc_impact": "Elevation of Privilege"}
    assert er.apply_rules(item("local", "Windows X Elevation of Privilege Vulnerability", **base))["label"] == "unresolved"
    faq = "An attacker who successfully exploited this vulnerability could gain SYSTEM privileges."
    v = er.apply_rules(item("local", "Windows X Elevation of Privilege Vulnerability", msrc_faq=faq, **base))
    assert v["label"] == "ADMIN_CONFIRMED" and v["source"] == "msrc_faq" and v["span"] in faq


def test_rule_spans_are_verbatim_in_their_source():
    it = item("remote", "allows remote attackers to execute arbitrary code via X.")
    v = er.apply_rules(it)
    assert ef.norm(v["span"]) in ef.norm(it["texts"][v["source"]]) and v["version"] == er.RULES_VERSION


CFG = {"os_sample": {"family": "os"}, "audit": {"per_pool": 2, "seed": 1}}


def pooled(cid, fam="http", amb=False):
    return {"cve_id": cid, "label_set": "remote", "units": [{"family": fam, "role_question": amb}]}


def test_stage2_selection_reasons_and_audit_exclusions():
    items = [pooled("P"), pooled("AMB", amb=True), pooled("U"), pooled("OSU", "os"), pooled("OSP", "os"),
             *[pooled(f"N{i}") for i in range(5)], *[pooled(f"ON{i}", "os") for i in range(5)], pooled("NAMB", amb=True)]
    v = {i["cve_id"]: {"label": "unresolved"} for i in items}
    v["P"]["label"] = v["OSP"]["label"] = "RCE_CONFIRMED"
    for i in items:
        if i["cve_id"][0] == "N" or i["cve_id"].startswith("ON"):
            v[i["cve_id"]]["label"] = "NON_RCE"
    sel = ef.stage2_select(items, v, CFG)
    assert sel["P"] == ["rule_positive"] and sel["AMB"] == ["ambiguous_role", "rule_unresolved_service"] and sel["U"] == ["rule_unresolved_service"]
    assert "OSU" not in sel  # OS unresolved is not annotated
    assert sel["OSP"] == ["rule_positive"] and sel["NAMB"] == ["ambiguous_role"]  # selected already: not in the audit pool
    assert sum(r[0] == "audit_svc" for r in sel.values()) == 2 and sum(r[0] == "audit_os" for r in sel.values()) == 2
    assert ef.stage2_select(items, v, CFG) == sel  # seeded


def test_os_sample_none_keeps_every_unit():
    units = [(f"C{i}", {"os": "linux", "cohort": "core", "product": "k"}) for i in range(5)]
    assert ef.sample_os(units, {"os_sample": {"per_group": None, "seed": 1}}) == units


def test_annotator_view_drops_duplicate_title_and_supporting():
    it = {"cve_id": "C", "label_set": "local", "supporting": {"cwe": ["CWE-1"]}, "texts": {"nvd_description": "A  B", "msrc_title": "A B"},
          "units": [{"product": "p", "role_question": True}, {"product": "q", "role_question": False}]}
    v = ef.annotator_view(it)
    assert "msrc_title" not in v["texts"] and "supporting" not in v and v["role_question"] == ["p"] and v["products"] == ["p", "q"]


def test_audit_unresolved_sample_is_seeded_os_only_and_excludes_used():
    items = [{"cve_id": f"O{i}", "label_set": "local", "units": [{"family": "os", "role_question": False}]} for i in range(10)]
    items += [{"cve_id": "S", "label_set": "local", "units": [{"family": "http", "role_question": False}]}]
    v = {i["cve_id"]: {"label": "unresolved"} for i in items}
    v["O0"]["label"] = "ADMIN_CONFIRMED"
    cfg = {"os_sample": {"family": "os"}, "audit_unresolved": {"n": 4, "seed": 3}}
    a = ef.audit_unresolved_sample(items, v, {"O1"}, cfg)
    assert a == ef.audit_unresolved_sample(items, v, {"O1"}, cfg) and len(a) == 4
    assert not ({"O0", "O1", "S"} & set(a))


def _it(desc, **more):
    return {"cve_id": "C", "label_set": "remote", "texts": {"nvd_description": desc, **more}}


def _ev(span, src="nvd_description"):
    return [{"source": src, "span": span}]


def test_r1_unchanged_by_r2():
    assert er.RULES_VERSION == "r1" and er.RULES_VERSION_R2 == "r2"


def test_r2_demotes_hedged_effect_only():
    d = "A flaw in the parser can crash it, or probably even a remote code execution."
    assert er.r2_demotes(_it(d), "RCE_CONFIRMED", _ev("probably even a remote code execution"))
    assert er.r2_demotes(_it("mod_x info disclosure and possible RCE"), "RCE_CONFIRMED", _ev("possible RCE"))
    assert er.r2_demotes(_it("OS command injection might occur via a user name"), "RCE_CONFIRMED", _ev("OS command injection might occur"))


def test_r2_keeps_could_allow_unhedged_wording_msrc_and_unplaceable_span():
    d = "This could allow an attacker to execute arbitrary code."
    assert not er.r2_demotes(_it(d), "RCE_CONFIRMED", _ev("execute arbitrary code"))  # "could allow" is not a hedge
    d2 = "Possibly a crash. A second bug lets an attacker execute arbitrary code on the server."
    assert not er.r2_demotes(_it(d2), "RCE_CONFIRMED", _ev("Possibly a crash"))  # explicit wording elsewhere
    d3 = "possible remote code execution"
    assert not er.r2_demotes(_it(d3, msrc_impact="Remote Code Execution"), "RCE_CONFIRMED", _ev(d3))  # MSRC Impact is authoritative
    assert not er.r2_demotes(_it(d3), "RCE_CONFIRMED", _ev("not in the text"))
    assert not er.r2_demotes(_it(d3), "NON_RCE", _ev(d3))  # only positives are demoted
    assert not er.r2_demotes(_it("A. Hedge: perhaps. A local user can gain root privileges."), "ADMIN_CONFIRMED", _ev("gain root privileges"))  # hedge in another sentence


def test_r2_admin_hedge():
    d = "A local user might possibly gain root privileges via a symlink."
    assert er.r2_demotes({**_it(d), "label_set": "local"}, "ADMIN_CONFIRMED", _ev("gain root privileges via a symlink"))


def test_r2_stock_allow_phrase_is_not_a_hedge():
    d = "pkg before 1.2 might allow remote attackers to execute arbitrary code by leveraging HTTP."
    assert not er.r2_demotes(_it(d), "RCE_CONFIRMED", _ev("might allow remote attackers to execute arbitrary code"))
    d2 = "pkg may allow local users to gain root privileges via a symlink."
    assert not er.r2_demotes({**_it(d2), "label_set": "local"}, "ADMIN_CONFIRMED", _ev("gain root privileges via a symlink"))
    assert er.r2_demotes(_it("OS command injection might occur via a user name"), "RCE_CONFIRMED", _ev("OS command injection might occur"))  # other "might" stays
    assert er.r2_demotes(_it("might allow a crash, probably remote code execution"), "RCE_CONFIRMED", _ev("probably remote code execution"))
