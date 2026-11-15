"""ahg.cvedata: CPE part (F6), platform conditions, CV1 vector resolution, cohorts, role rule (E5), funnel."""

import pytest

from ahg import cvedata as cd


def match(criteria, vulnerable=True, **rng):
    return {"criteria": criteria, "vulnerable": vulnerable, **rng}


CONFIGS = [
    {"nodes": [{"operator": "AND", "cpeMatch": [], "nodes": [
        {"operator": "OR", "cpeMatch": [match("cpe:2.3:a:samba:samba:*:*:*:*:*:*:*:*", versionEndExcluding="4.17.0")]},
        {"operator": "OR", "cpeMatch": [match("cpe:2.3:o:debian:debian_linux:11.0:*:*:*:*:*:*:*", False)]},
    ]}]},
    {"nodes": [{"operator": "OR", "negate": True, "cpeMatch": [match("cpe:2.3:a:samba:samba:4.16.5:*:*:*:*:*:*:*")]}]},
    {"nodes": [{"operator": "OR", "cpeMatch": [match("cpe:2.3:o:samba:samba:-:*:*:*:*:*:*:*")]}]},
]  # fmt: skip


def test_parse_cpe_fields_and_rejects_malformed():
    c = cd.parse_cpe("cpe:2.3:a:apache:http_server:2.4.1:*:*:*:*:*:*:*")
    assert (c["part"], c["vendor"], c["product"], c["version"]) == ("a", "apache", "http_server", "2.4.1")
    assert cd.parse_cpe(r"cpe:2.3:a:x:y\:z:1:*:*:*:*:*:*:*")["product"] == r"y\:z"
    with pytest.raises(ValueError):
        cd.parse_cpe("cpe:/a:apache:http_server")


def test_cpe_roles_keep_platform_conditions_and_ranges():
    ms = cd.cpe_matches(CONFIGS)
    roles = [(m["part"], m["product"], m["role"]) for m in ms]
    assert roles == [("a", "samba", "vulnerable"), ("o", "debian_linux", "platform"), ("a", "samba", "negated"),
                     ("o", "samba", "vulnerable")]  # fmt: skip
    assert ms[0]["versionEndExcluding"] == "4.17.0"
    assert cd.affected_products(ms) == ["a:samba:samba", "o:samba:samba"]  # platform never affected


def test_product_hits_requires_expected_part_f6():
    ms = cd.cpe_matches(CONFIGS)
    assert cd.product_hits(ms, "samba:samba", "a") == (True, ["a", "o"])
    only_o = cd.cpe_matches(CONFIGS[2:])
    assert cd.product_hits(only_o, "samba:samba", "a") == (False, ["o"])  # reported, never a hit
    assert cd.product_hits(ms, "debian:debian_linux", "o") == (False, [])  # platform only


def vec(vector, origin="nvd", source="nvd@nist.gov", type_="Primary", pids=()):
    return {"origin": origin, "source": source, "type": type_, "vector": vector,
            "version": cd.parse_vector(vector)["version"], "metrics": cd.parse_vector(vector), "product_ids": list(pids)}


def test_parse_vector_versions():
    assert cd.parse_vector("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H")["UI"] == "R"
    assert cd.parse_vector("AV:N/AC:L/Au:N/C:P/I:P/A:P")["version"] == "2.0"
    with pytest.raises(ValueError):
        cd.parse_vector("CVSS:3.1/AV:N/AV:L")


def test_resolve_vector_agreed_conflict_and_no_v3():
    a = vec("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    b = vec("CVSS:3.0/AV:N/AC:H/PR:N/UI:N/S:C/C:H/I:H/A:H", source="cna", type_="Secondary")
    r = cd.resolve_vector([a, b])
    assert r["status"] == "AGREED" and r["values"] == {"AV": "N", "PR": "N", "UI": "N"}
    assert r["cvss_scope_values"] == ["C", "U"]  # S differs: reported, not an AV/PR/UI conflict
    c = vec("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", source="cna", type_="Secondary")
    r = cd.resolve_vector([a, c])
    assert r["status"] == "REVIEW_REQUIRED" and r["conflict_fields"] == ["AV", "PR"] and r["values"] == {}
    assert cd.resolve_vector([vec("AV:N/AC:L/Au:N/C:P/I:P/A:P")])["status"] == "NO_V3"


def test_msrc_vectors_scoped_to_in_scope_products():
    e = {"vectors": [vec("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", "msrc", pids=["ws2019"]),
                     vec("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H", "msrc", pids=["win10"])]}  # fmt: skip
    assert [v["product_ids"] for v in cd.msrc_vectors_in_scope(e, {"ws2019"})] == [["ws2019"]]


def test_cohort_bounds_inclusive():
    co = {"core": ["2016-01-01", "2023-12-31"], "temporal": ["2024-01-01", "2026-10-21"]}
    assert cd.cohort("2023-12-31T23:59:00.000", co) == "core"
    assert cd.cohort("2024-01-01T00:00:00", co) == "temporal"
    assert cd.cohort("2026-10-21T10:00:00", co) == "temporal"
    assert cd.cohort("2026-10-22T00:00:00", co) is None and cd.cohort("2015-12-31", co) is None


# Wording copied from the two D28 E5 check advisories and typical descriptions.
FREERDP_GHSA = ("Summary Out-Of-Bounds Read in nsc_rle_decompress_data Affected FreeRDP based clients only. "
                "FreeRDP proxy not affected as image decoding is not done by proxy (data passthrough) Details")


@pytest.mark.parametrize("text,short,want", [
    (FREERDP_GHSA, False, {"client"}),
    ("Vulnerability in the MySQL Client product of Oracle MySQL (component: C API).", False, {"client"}),
    ("Vulnerability in the MySQL Server product of Oracle MySQL (component: Server: Optimizer).", False, {"server"}),
    ("A malicious RDP server can trigger an out-of-bounds write in the client.", False, {"client"}),
    ("A remote attacker can send a crafted request to the server to crash it.", False, {"server"}),
    ("A flaw in libssh allows a malicious client to bypass authentication.", False, {"server"}),
    ("An out-of-bounds read in libfoo allows clients to crash it.", False, set()),
    ("The server is not affected.", False, set()),
    ("Windows SMB Client Remote Code Execution Vulnerability", True, {"client"}),
    ("Windows DNS Server Remote Code Execution Vulnerability", True, {"server"}),
    ("Microsoft SQL Server Native Client OLE DB Provider Remote Code Execution Vulnerability", True, {"client"}),
    ("Microsoft Exchange Server Remote Code Execution Vulnerability", True, set()),
])  # fmt: skip
def test_role_signals(text, short, want):
    assert {s["role"] for s in cd.role_signals(text, "t", short)} == want


def test_decide_role_wording_then_prior():
    c, s = {"role": "client"}, {"role": "server"}
    assert cd.decide_role([c], "server")[0] == "client"
    assert cd.decide_role([s], "multi")[0] == "server"
    assert cd.decide_role([c, s], "server")[0] == "ambiguous"
    assert [cd.decide_role([], p)[0] for p in ("server", "library", "multi")] == ["server", "library", "ambiguous"]
    assert cd.decide_role([s], "os_component")[0] == "os_component"  # kernel / Windows OS: never a server role


def res(av, pr, ui):
    return {"status": "AGREED" if max(len(av), len(pr), len(ui)) == 1 else "REVIEW_REQUIRED",
            "candidates": {"AV": av, "PR": pr, "UI": ui}}


@pytest.mark.parametrize("r,role,want", [
    ({"status": "NO_V3"}, "server", "no_v3_vector"),
    (res(["A"], ["N"], ["N"]), "server", "AV:A"),
    (res(["P"], ["N"], ["N"]), "server", "AV:P"),
    (res(["L", "N"], ["N"], ["N"]), "server", "REVIEW_REQUIRED_AV"),
    (res(["N"], ["H", "L"], ["N"]), "server", "network_PR:L/H"),  # every provider's PR is excluded
    (res(["N"], ["L", "N"], ["R"]), "server", "UI:R"),  # agreed UI:R excludes before the PR conflict matters
    (res(["N"], ["N"], ["N"]), "client", "role_not_server"),
    (res(["N"], ["N"], ["N", "R"]), "server", "REVIEW_REQUIRED"),
    (res(["N"], ["N"], ["N"]), "server", "remote_candidate"),
    (res(["L"], ["H"], ["N"]), "os_component", "local_candidate"),  # local: PR and role do not gate here
    (res(["L"], ["L"], ["R"]), "server", "UI:R"),
    (res(["L"], ["L", "N"], ["N"]), "server", "REVIEW_REQUIRED"),
])  # fmt: skip
def test_funnel_stage(r, role, want):
    assert cd.funnel_stage(r, role) == want


def test_msrc_entries_parse_impact_tags_builds_vectors():
    doc = {
        "ProductTree": {"FullProductName": [{"ProductID": "1", "Value": "Windows Server 2019"},
                                            {"ProductID": "2", "Value": "Windows 10 Version 1809 for x64-based Systems"}]},
        "Vulnerability": [{
            "CVE": "CVE-2020-0001", "Title": {"Value": "Windows SMB Server Remote Code Execution Vulnerability"},
            "Notes": [{"Type": 7, "Value": "Windows SMB Server"}, {"Type": 4, "Value": "<p>An <b>unauthenticated</b> attacker</p>"}],
            "ProductStatuses": [{"Type": 3, "ProductID": ["1", "2"]}],
            "Threats": [{"Type": 0, "Description": {"Value": "Remote Code Execution"}, "ProductID": ["1", "2"]},
                        {"Type": 3, "Description": {"Value": "Critical"}, "ProductID": ["1"]}],
            "Remediations": [{"Type": 2, "Description": {"Value": "4551762"}, "FixedBuild": "10.0.17763.1098", "ProductID": ["1"]}],
            "CVSSScoreSets": [{"BaseScore": 10, "Vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", "ProductID": ["1"]}],
            "RevisionHistory": [{"Date": "2020-03-12T07:00:00"}], "ReleaseDate": "2020-03-12T07:00:00",
        }],
    }  # fmt: skip
    (e,) = cd.msrc_entries(doc, "2020-Mar")
    assert e["tags"] == ["Windows SMB Server"] and e["faq"] == "An unauthenticated attacker"
    assert e["products"]["1"]["impact"] == "Remote Code Execution" and e["products"]["1"]["severity"] == "Critical"
    assert e["products"]["1"]["fixes"][0]["fixed_build"] == "10.0.17763.1098" and "fixes" not in e["products"]["2"]
    assert e["vectors"][0]["metrics"]["AV"] == "N" and e["vectors"][0]["product_ids"] == ["1"]


def test_ghsa_text_extracts_description_block():
    page = ('<html><div class="Box-body"><div class="markdown-body comment-body p-0" >'
            '<h3>Summary</h3><p>Affected FreeRDP based clients only.</p></div></div>'
            '<div class="Box-header">Severity</div></html>')  # fmt: skip
    assert cd.ghsa_text(page) == "Summary Affected FreeRDP based clients only."
    assert cd.ghsa_text("<html></html>") == ""
