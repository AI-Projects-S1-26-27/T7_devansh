"""P5 version-fixture catalogue (D28 Round 4 catalogue rule; ahg.fixtures and scripts/build_fixtures.py).

Every fixture here is a hand-written synthetic page or record; no network access."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from ahg import fixtures as fx
from ahg.rawfetch import write_manifest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- versions


def test_version_order_numbers_pre_and_post_releases():
    k = fx.version_key
    assert k("2.4.9") < k("2.4.10") < k("2.5")  # numeric, not string, order
    assert k("11.0.0-M1") < k("11.0.0-M10") < k("11.0.0")  # milestones precede the release
    assert k("9.1.0rc1") < k("9.1.0")
    assert k("1.3.8") < k("1.3.8a") < k("1.3.8b") < k("1.3.9rc1") < k("1.3.9")  # ProFTPD letters are post-releases
    assert k("9.6") < k("9.6p1") < k("9.6p2") < k("9.7p1")
    assert k("9.11.3") < k("9.11.3-P1") < k("9.11.4")
    assert k("5.10") == k("5.10.0") and k("6.1") < k("6.1.1")


def test_prerelease_detection():
    assert fx.is_prerelease("11.0.0-M3") and fx.is_prerelease("9.19.0rc1") and fx.is_prerelease("1.1.0-beta1")
    assert not fx.is_prerelease("1.3.8b") and not fx.is_prerelease("9.6p1") and not fx.is_prerelease("9.11.3-P1")


# ---------------------------------------------------------------- ranges


def cfg(*matches, negate=False):
    return [{"nodes": [{"operator": "OR", "negate": negate, "cpeMatch": list(matches)}]}]


def m(criteria, vulnerable=True, **bounds):
    return {"criteria": criteria, "vulnerable": vulnerable, **bounds}


def test_range_bounds_follow_nvd_semantics():
    r = fx.Range("c", start_incl="9.0.0", end_excl="9.0.83")
    assert r.contains("9.0.0", "") and r.contains("9.0.82", "") and not r.contains("9.0.83", "")
    assert fx.Range("c", end_incl="9.5").contains("9.5", "p1")  # bounds compare the version field only
    assert not fx.Range("c", start_excl="2.4.49").contains("2.4.49", "")
    exact = fx.Range("c", exact="2.4.49", update="-")
    assert exact.contains("2.4.49", "") and not exact.contains("2.4.49", "p1") and not exact.contains("2.4.50", "")
    assert fx.Range("c", exact="9.5", update="p1").contains("9.5", "p1")
    assert fx.Range("c", start_incl="9.0.0", end_excl="9.0.83").label() == "[9.0.0, 9.0.83)"
    assert exact.label() == "=2.4.49"


def test_cve_ranges_keep_vulnerable_product_matches_only():
    conf = cfg(
        m("cpe:2.3:a:apache:tomcat:*:*:*:*:*:*:*:*", versionStartIncluding="9.0.0", versionEndExcluding="9.0.83"),
        m("cpe:2.3:a:apache:tomcat:*:*:*:*:*:*:*:*", versionStartIncluding="9.0.0", versionEndExcluding="9.0.83"),  # duplicate
        m("cpe:2.3:a:apache:tomcat:8.5.1:*:*:*:*:*:*:*"),
        m("cpe:2.3:o:debian:debian_linux:11:*:*:*:*:*:*:*", vulnerable=False),  # platform co-condition
        m("cpe:2.3:a:apache:http_server:*:*:*:*:*:*:*:*", versionEndExcluding="2.4.58"),  # other product
        m("cpe:2.3:o:apache:tomcat:7.0.1:*:*:*:*:*:*:*"),  # wrong CPE part (F6)
        m("cpe:2.3:a:apache:tomcat:-:*:*:*:*:*:*:*"),  # version not applicable: reported, not guessed
    )
    neg = cfg(m("cpe:2.3:a:apache:tomcat:10.0.0:*:*:*:*:*:*:*"), negate=True)
    ranges, notes = fx.cve_ranges(conf + neg, "apache:tomcat", "a")
    assert sorted(r.label() for r in ranges) == ["=8.5.1", "[9.0.0, 9.0.83)"]
    assert len(notes) == 1 and "version '-'" in notes[0]


def unit(product, stage="remote_candidate", cohort="core", ac="explicit", os_="linux", family="http"):
    return {"product": product, "stage": stage, "cohort": cohort, "ac_status": ac, "os": os_, "family": family}


def test_candidate_units_take_candidate_stages_with_cohort_and_explicit_ac():
    rows = [
        {"cve_id": "CVE-2020-1", "units": [unit("apache:tomcat"), unit("f5:nginx", stage="UI:R")]},
        {"cve_id": "CVE-2020-2", "units": [unit("apache:tomcat", stage="local_candidate")]},
        {"cve_id": "CVE-2020-3", "units": [unit("apache:tomcat", ac="conflict")]},
        {"cve_id": "CVE-2020-4", "units": [unit("apache:tomcat", cohort=None)]},
    ]
    got = fx.candidate_units(rows)
    assert set(got) == {"apache:tomcat"}
    assert sorted(got["apache:tomcat"]["cves"]) == ["CVE-2020-1", "CVE-2020-2"]


# ---------------------------------------------------------------- releases

APACHE_LISTING = """<pre><img src="/i.gif"> <a href="?C=N">Name</a>
<img src="/icons/c.gif"> <a href="httpd-2.4.57.tar.gz">httpd-2.4.57.tar.gz</a>   2023-04-06 13:00  9.4M
<img src="/icons/c.gif"> <a href="httpd-2.4.57.tar.bz2">httpd-2.4.57.tar.bz2</a>  2023-04-05 13:00  7.0M
<img src="/icons/c.gif"> <a href="httpd-2.4.58.tar.gz">httpd-2.4.58.tar.gz</a>   2023-10-19 09:09  9.4M
<img src="/icons/c.gif"> <a href="httpd-2.4.58.tar.gz.asc">httpd-2.4.58.tar.gz.asc</a>   2023-10-19 09:09  833
</pre>"""
NGINX_LISTING = '<a href="nginx-1.25.3.tar.gz">nginx-1.25.3.tar.gz</a>                30-Aug-2023 14:19    1M\n'
TABLE_LISTING = ('<tr><td><a href="samba-4.19.3.tar.gz">samba-4.19.3.tar.gz</a></td>'
                 '<td class="indexcollastmod">2023-11-27 12:10  </td><td> 40M</td></tr>')


def test_parse_listing_formats_and_dates():
    got = fx.parse_listing(APACHE_LISTING, r"httpd-(?P<name>2\.\d+\.\d+)\.tar\.(?:gz|bz2)")
    assert [(n, d) for n, d, _ in got] == [("2.4.57", "2023-04-06"), ("2.4.57", "2023-04-05"), ("2.4.58", "2023-10-19")]
    assert fx.parse_listing(NGINX_LISTING, r"nginx-(?P<name>\d+\.\d+\.\d+)\.tar\.gz")[0][:2] == ("1.25.3", "2023-08-30")
    assert fx.parse_listing(TABLE_LISTING, r"samba-(?P<name>\d+\.\d+\.\d+)\.tar\.gz")[0][:2] == ("4.19.3", "2023-11-27")


def test_parse_text_and_github_releases():
    page = "<html><body><p>Wing FTP Server v8.2.4 Released: 4/Sep/2026</p><p>Fixed a bug</p></body></html>"
    got = fx.parse_text_releases(page, r"^Wing FTP Server v(?P<name>\d+(?:\.\d+)+) Released: (?P<date>\S+)", "%d/%b/%Y")
    assert [(n, d) for n, d, _ in got] == [("8.2.4", "2026-09-04")]
    assert fx.parse_date("2024-8-21", "ymd_loose") == "2024-08-21"
    gh = json.dumps([
        {"tag_name": "3.5.1", "published_at": "2024-04-22T10:00:00Z", "draft": False, "prerelease": False},
        {"tag_name": "3.6.0-rc1", "published_at": "2024-05-01T10:00:00Z", "draft": False, "prerelease": True},
        {"tag_name": "3.4.0", "published_at": None, "draft": True, "prerelease": False},
    ])
    assert fx.parse_github_releases(gh, r"(?P<name>\d+\.\d+\.\d+)") == [("3.5.1", "2024-04-22", "3.5.1")]


def test_build_releases_drops_prereleases_keeps_earliest_date_and_splits_cpe():
    found = [("9.6p1", "2023-12-19", "e", "r"), ("9.6p1", "2023-12-18", "e2", "r"), ("9.7p1", None, "e", "r")]
    rels, pre = fx.build_releases(found, r"(?P<version>\d+\.\d+)(?P<update>p\d+)")
    assert [(r.name, r.cpe_version, r.cpe_update, r.date) for r in rels] == [
        ("9.6p1", "9.6", "p1", "2023-12-18"), ("9.7p1", "9.7", "p1", None)]
    rels, pre = fx.build_releases([("11.0.0-M1", None, "", "r"), ("11.0.0", None, "", "r")], None)
    assert [r.name for r in rels] == ["11.0.0"] and pre == ["11.0.0-M1"]
    with pytest.raises(ValueError):
        fx.cpe_split("9.6", r"(?P<version>\d+\.\d+)(?P<update>p\d+)")


# ---------------------------------------------------------------- evidence

TOMCAT_PAGE = """<html><body>
<h3 id="Fixed_in_Apache_Tomcat_9.0.83">Fixed in Apache Tomcat 9.0.83</h3>
<p>Important: Request smuggling</p><p>CVE-2023-46589</p>
<p>Low: Fix for CVE-2023-1111 did not address APR</p><p>CVE-2023-2222</p>
<h3>Fixed in Apache Tomcat 9.0.82</h3><p>CVE-2023-3333</p>
<h3>Fixed in Apache Tomcat 11.0.0-M11</h3><p>CVE-2023-4444</p>
</body></html>"""


def test_sections_assign_cves_to_the_header_version():
    docs = fx.ev_sections(TOMCAT_PAGE, r"^Fixed in Apache Tomcat (?P<v>\d+\.\d+\.\d+(?:-M\d+)?)", r"\d+\.\d+\.\d+(?:-M\d+)?",
                          "raw", cve_at_line_start=True)
    assert [(d.cve, d.version) for d in docs] == [
        ("CVE-2023-46589", "9.0.83"), ("CVE-2023-2222", "9.0.83"), ("CVE-2023-3333", "9.0.82"),
        ("CVE-2023-4444", "11.0.0-M11")]  # CVE-2023-1111 is named in prose: not a fix of that section
    loose = fx.ev_sections("version 2.90\n  Fix CVE-2023-50387 and CVE-2023-50868.\nversion 2.89\n", r"^version (?P<v>\d+\.\d+)$",
                           r"\d+\.\d+", "raw")
    assert [(d.cve, d.version) for d in loose] == [("CVE-2023-50387", "2.90"), ("CVE-2023-50868", "2.90")]


def test_sections_follow_dated_headers_after_a_table_of_contents():
    """The Tomcat pages list every "Fixed in" header in a contents block, then repeat each one with its
    release date in front. CVEs must go to the dated section they sit in, not to the last contents entry."""
    page = """<html><body><ul><li>Fixed in Apache Tomcat 9.0.83</li><li>Fixed in Apache Tomcat 9.0.82</li></ul>
    <h3><span>15 November 2023</span> Fixed in Apache Tomcat 9.0.83</h3><p>CVE-2023-46589</p>
    <h3><span>10 October 2023</span> Fixed in Apache Tomcat 9.0.82</h3><p>CVE-2023-42795</p></body></html>"""
    docs = fx.ev_sections(page, r"Fixed in Apache Tomcat (?P<v>\d+\.\d+\.\d+(?:[.-]M\d+)?)$", r"\d+\.\d+\.\d+", "raw", True)
    assert [(d.cve, d.version) for d in docs] == [("CVE-2023-46589", "9.0.83"), ("CVE-2023-42795", "9.0.82")]


def test_proftpd_version_regex_keeps_rc_suffixes():
    import json as _json
    cfgfile = Path(__file__).resolve().parents[1] / "configs" / "fixtures" / "p5.json"
    vr = _json.loads(cfgfile.read_text())["products"]["proftpd:proftpd"]["version_regex"]
    assert fx.versions_in("1.3.10rc6 and 1.3.8b, 1.3.9", vr) == ["1.3.10rc6", "1.3.8b", "1.3.9"]


def test_page_regex_nginx_and_per_cve_pages():
    page = "<html><body><p>Buffer overflow</p><p>CVE-2026-1001</p><p>Not vulnerable: 1.31.7+, 1.30.6+</p>" \
           "<p>Vulnerable: 1.0.0-1.31.6</p></body></html>"
    docs = fx.ev_page_regex(page, r"(?P<cves>(?:CVE-\d{4}-\d+\n)+)Not vulnerable: (?P<v>[^\n]*)", r"\d+\.\d+\.\d+", "raw")
    assert [(d.cve, d.version) for d in docs] == [("CVE-2026-1001", "1.31.7"), ("CVE-2026-1001", "1.30.6")]
    samba = "<html><body><p>Additionally, Samba 4.19.1, 4.18.8 and 4.17.12 have</p><p>been issued as security releases.</p></body></html>"
    pat = r"Samba (?:versions? )?(?P<v>\d+\.\d+\.\d+(?:(?:,| and|,? and) \d+\.\d+\.\d+)*) (?:has|have) been issued as (?:a )?security releases?"
    docs = fx.ev_page_regex(samba, pat, r"\d+\.\d+\.\d+", "raw", page_cve="CVE-2023-4091", flatten=True)
    assert [(d.cve, d.version) for d in docs] == [("CVE-2023-4091", v) for v in ("4.19.1", "4.18.8", "4.17.12")]


def test_table_reads_only_the_named_columns():
    page = """<table><tr><th>Reference</th><th>Affected</th><th>Fixed</th><th>Score</th></tr>
    <tr><td>CVE-2024-10979</td><td>17, 16</td><td>17.1, 16.5</td><td>8.8</td></tr></table>
    <table><tr><td>Date</td><td>CVE ID</td><td>9.20</td><td>9.18</td></tr>
    <tr><td>2024-07-23</td><td>CVE-2024-1737</td><td>9.20.0</td><td>-</td></tr></table>"""
    pg = fx.ev_table(page, "Fixed", r"\d+\.\d+(?:\.\d+)?", "raw")
    assert [(d.cve, d.version) for d in pg] == [("CVE-2024-10979", "17.1"), ("CVE-2024-10979", "16.5")]
    isc = fx.ev_table(page, r"\d+\.\d+", r"\d+\.\d+\.\d+", "raw")
    assert [(d.cve, d.version) for d in isc] == [("CVE-2024-1737", "9.20.0")]


def test_ghsa_repo_patched_versions():
    page = json.dumps([
        {"ghsa_id": "GHSA-1", "cve_id": "CVE-2024-1", "withdrawn_at": None,
         "vulnerabilities": [{"patched_versions": "0.9.24", "vulnerable_version_range": "<= 0.9.23"}]},
        {"ghsa_id": "GHSA-2", "cve_id": None, "vulnerabilities": [{"patched_versions": "1.0.0"}]},
        {"ghsa_id": "GHSA-3", "cve_id": "CVE-2024-3", "withdrawn_at": "2024-01-01", "vulnerabilities": [{"patched_versions": "2.0.0"}]},
    ])
    assert [(d.cve, d.version) for d in fx.ev_ghsa_repo(page, r"\d+\.\d+\.\d+", "raw")] == [("CVE-2024-1", "0.9.24")]


def json5(short, versions, product="Linux"):
    return {"cveMetadata": {"cveId": "CVE-2024-26600"},
            "containers": {"cna": {"providerMetadata": {"shortName": short},
                                   "affected": [{"product": product, "versions": versions}]}}}


def test_cve_json5_reads_unaffected_entries_from_the_vendor_cna_only():
    versions = [
        {"version": "6.5", "status": "affected"},
        {"version": "0", "lessThan": "6.5", "status": "unaffected", "versionType": "semver"},
        {"version": "6.6.4", "lessThanOrEqual": "6.6.*", "status": "unaffected", "versionType": "semver"},
        {"version": "6.7", "lessThanOrEqual": "*", "status": "unaffected", "versionType": "original_commit_for_fix"},
        {"version": "abc123", "lessThan": "def456", "status": "unaffected", "versionType": "git"},
        {"version": "5.10.0", "lessThan": "5.10.200", "status": "affected", "versionType": "semver"},  # a bound, never a fix
    ]
    docs, refusal = fx.ev_cve_json5(json5("Linux", versions), {"short_name": "Linux"}, "Linux", "raw")
    assert refusal is None and [d.version for d in docs] == ["0", "6.6.4", "6.7"]
    docs, refusal = fx.ev_cve_json5(json5("mitre", versions), {"short_name": "Linux"}, "Linux", "raw")
    assert docs == [] and "not the vendor CNA" in refusal
    assert fx.ev_cve_json5(json5("Linux", versions, product="Other"), {"short_name": "Linux"}, "Linux", "raw")[0] == []
    # Repository copies of kernel records carry only the CNA's orgId.
    rec = json5("Linux", versions)
    rec["containers"]["cna"]["providerMetadata"] = {"orgId": "f4215fc3"}
    assert len(fx.ev_cve_json5(rec, {"org_id": "f4215fc3"}, "Linux", "raw")[0]) == 3
    assert fx.ev_cve_json5(rec, {"org_id": "other"}, "Linux", "raw")[0] == []


# ---------------------------------------------------------------- catalogue rule


def rels(*names, split=None):
    return fx.build_releases([(n, f"2020-01-{i + 1:02d}", n, "listing") for i, n in enumerate(names)], split)[0]


def doc(cve, v):
    return fx.FixDoc(cve, v, "advisory", "sections", "x")


def test_release_index_match_and_inside():
    idx = fx.ReleaseIndex(rels("9.5p1", "9.6p1", "9.6p2", split=r"(?P<version>\d+\.\d+)(?P<update>p\d+)"))
    assert idx.match("9.6P1").name == "9.6p1"
    assert idx.match("9.5").name == "9.5p1"  # unique release with that CPE version
    assert idx.match("9.6") is None  # 9.6p1 and 9.6p2 share it: ambiguous, never guessed
    assert [idx.releases[i].name for i in idx.inside(fx.Range("c", end_incl="9.6"))] == ["9.5p1", "9.6p1", "9.6p2"]
    assert [idx.releases[i].name for i in idx.inside(fx.Range("c", exact="9.6", update="p2"))] == ["9.6p2"]


def test_catalogue_per_interval_fixtures_from_documented_fixes():
    idx = fx.ReleaseIndex(rels("8.5.95", "8.5.96", "9.0.81", "9.0.82", "9.0.83", "10.1.15", "10.1.16"))
    cve = "CVE-2023-46589"
    ranges = {cve: [fx.Range("a", start_incl="8.5.0", end_incl="8.5.95"), fx.Range("b", start_incl="9.0.0", end_excl="9.0.83"),
                    fx.Range("c", start_incl="10.1.0", end_incl="10.1.15")]}
    docs = [doc(cve, "8.5.96"), doc(cve, "9.0.83"), doc(cve, "10.1.16"), doc(cve, "11.0.0-M11")]
    cat = fx.catalogue_product(ranges, idx, docs)
    pairs = sorted((f["fixed_for"][0]["interval"], f["fixed_for"][0]["after"], v) for v, f in cat.fixtures.items() if f["fixed_for"])
    assert pairs == [("10.1.15..10.1.15", "10.1.15", "10.1.16"), ("8.5.95..8.5.95", "8.5.95", "8.5.96"),
                     ("9.0.81..9.0.82", "9.0.82", "9.0.83")]
    assert sorted(v for v, f in cat.fixtures.items() if f["vulnerable_for"]) == ["10.1.15", "8.5.95", "9.0.82"]
    assert cat.dropped == []
    assert [u["version"] for u in cat.unmatched_docs] == ["11.0.0-M11"]  # milestone: not a stable release


def test_enumerated_versions_collapse_into_maximal_runs():
    """NVD lists some CVEs version by version; each run of consecutive covered releases is one interval."""
    idx = fx.ReleaseIndex(rels("9.6.1", "9.6.2", "9.6.3", "10.0", "10.1", "10.2"))
    cve = "CVE-2017-1"
    ranges = {cve: [fx.Range(f"m{v}", exact=v) for v in ("9.6.1", "9.6.2", "10.0", "10.1")]}
    cat = fx.catalogue_product(ranges, idx, [doc(cve, "9.6.3"), doc(cve, "10.2")])
    assert sorted(v for v, f in cat.fixtures.items() if f["vulnerable_for"]) == ["10.1", "9.6.2"]
    assert sorted((f["fixed_for"][0]["interval"], v) for v, f in cat.fixtures.items() if f["fixed_for"]) == [
        ("10.0..10.1", "10.2"), ("9.6.1..9.6.2", "9.6.3")]


def test_catalogue_never_assumes_a_fix_and_logs_why():
    idx = fx.ReleaseIndex(rels("2.4.55", "2.4.56", "2.4.57", "2.4.58"))
    a, b, c = "CVE-2023-1", "CVE-2023-2", "CVE-2023-3"
    ranges = {
        a: [fx.Range("x", end_excl="2.4.56")],  # no documentation at all: no fixed fixture
        b: [fx.Range("x", end_excl="2.4.56")],  # documented fix 2.4.57, 2.4.56 undocumented but not affected
        c: [fx.Range("x", end_excl="2.4.57"), fx.Range("y", exact="2.0.1")],  # doc says 2.4.56: NVD lists it as affected
    }
    cat = fx.catalogue_product(ranges, idx, [doc(b, "2.4.57"), doc(c, "2.4.56")])
    assert cat.fixtures["2.4.57"]["fixed_for"][0]["cve"] == b
    reasons = {(d["cve"], d["reason"]) for d in cat.dropped}
    assert (a, "no_documented_fix_after_vulnerable_fixture") in reasons
    assert (c, "no_stable_release_in_cpe_match") in reasons  # 2.0.1 is not a release: logged, never a fixture
    conflict = [d for d in cat.dropped if d["cve"] == c and d["interval"] == "2.4.55..2.4.56"][0]
    assert conflict["conflicting_docs"] == ["2.4.56"]


def test_fix_on_a_later_affected_branch_is_not_this_intervals_fix():
    idx = fx.ReleaseIndex(rels("4.19.325", "5.0", "5.4.249", "5.4.250"))
    cve = "CVE-2024-9"
    ranges = {cve: [fx.Range("a", start_incl="4.19", end_excl="4.20"), fx.Range("b", start_incl="5.4", end_excl="5.4.250")]}
    cat = fx.catalogue_product(ranges, idx, [doc(cve, "5.4.250")])
    assert cat.fixtures["5.4.250"]["fixed_for"] == [{"cve": cve, "interval": "5.4.249..5.4.249", "after": "5.4.249", "evidence": [
        {"raw": "advisory", "kind": "sections", "excerpt": "x"}]}]
    dropped = [d for d in cat.dropped if d["interval"] == "4.19.325..4.19.325"]
    assert dropped[0]["reason"] == "no_documented_fix_after_vulnerable_fixture"  # 5.4.249 (affected) lies between


def test_unverified_fixtures_use_exact_and_inclusive_nvd_strings_only():
    exact = lambda v, u="*": fx.Range(f"cpe:2.3:a:v:p:{v}:{u}:*:*:*:*:*:*", v, u)  # noqa: E731
    star = "cpe:2.3:a:v:p:*:*:*:*:*:*:*:*"
    cves = {"CVE-2020-1": [exact("1.0"), exact("1.2"), exact("2.0-rc1"), fx.Range(star, end_excl="3.0")],
            "CVE-2020-2": [fx.Range(star, start_incl="4.0", end_incl="4.5"), fx.Range(star, start_incl="5.0", end_excl="5.3")],
            "CVE-2020-3": [exact("1.2", "hotfix1")],
            "CVE-2020-4": [exact("9.9")],
            "CVE-2020-5": [fx.Range(star, start_excl="6.0", end_excl="6.4")],
            "CVE-2020-6": [fx.Range(star, end_incl="7.0-rc1")]}  # fmt: skip
    raw = {c: {fx.nvd_match_key({"criteria": r.criteria, "versionStartIncluding": r.start_incl, "versionStartExcluding": r.start_excl,
                                 "versionEndIncluding": r.end_incl, "versionEndExcluding": r.end_excl}) for r in rs}
           for c, rs in cves.items() if c != "CVE-2020-4"}  # fmt: skip
    fixtures, dropped = fx.unverified_fixtures(cves, raw)
    got = [(f["version"], [(v["cve"], v["version_from"]) for v in f["vulnerable_for"]], f["fixed_for"], f["release_date"]) for f in fixtures]
    # CVE-2020-1: exact 1.2 (the 3.0 bound is exclusive). CVE-2020-2: the highest inclusive bound is 5.0, a start bound
    # (4.5 < 5.0; 5.3 is exclusive). The exclusive-only and pre-release-only CVEs get nothing.
    assert got == [("1.2", [("CVE-2020-1", "exact")], [], None), ("1.2:hotfix1", [("CVE-2020-3", "exact")], [], None),
                   ("5.0", [("CVE-2020-2", "versionStartIncluding")], [], None)]  # fmt: skip
    assert all(f["unverified_release"] for f in fixtures)
    # 1.2 also lies under CVE-2020-6's 7.0-rc1 bound; the hotfix CPE matches its own update only
    assert fixtures[0]["affected_by"] == ["CVE-2020-1", "CVE-2020-6"]
    assert fixtures[1]["affected_by"] == ["CVE-2020-1", "CVE-2020-3", "CVE-2020-6"]
    assert fixtures[2]["affected_by"] == ["CVE-2020-2", "CVE-2020-6"]
    assert {(d["cve"], d["reason"]) for d in dropped} == {
        ("CVE-2020-4", "cpe_match_not_in_raw_nvd_record"), ("CVE-2020-5", "no_exact_or_inclusive_version_in_cpe_matches"),
        ("CVE-2020-6", "only_prerelease_versions")}


def test_unverified_bound_must_match_the_raw_nvd_record():
    rg = fx.Range("cpe:2.3:a:v:p:*:*:*:*:*:*:*:*", end_incl="2.0")
    other = {fx.nvd_match_key({"criteria": rg.criteria, "versionEndIncluding": "1.9"})}  # raw record has a different bound
    fixtures, dropped = fx.unverified_fixtures({"CVE-2020-1": [rg]}, {"CVE-2020-1": other})
    assert fixtures == [] and dropped[0]["reason"] == "cpe_match_not_in_raw_nvd_record"


def test_build_refuses_unverified_fixtures_for_a_product_with_verified_ones():
    bf = load_script("build_fixtures")
    with pytest.raises(SystemExit):
        bf.add_unverified({"product": "a:b", "fixtures": [{"version": "1.0"}]}, {"cves": {}}, {})


def test_runs():
    assert fx.runs([1, 2, 3, 7, 8, 10]) == [(1, 3), (7, 8), (10, 10)] and fx.runs([]) == []


def test_affected_by_lists_every_matching_cve():
    idx = fx.ReleaseIndex(rels("1.0", "1.1", "1.2", "2.0"))
    ranges = {"CVE-2020-10": [fx.Range("a", end_excl="1.2")], "CVE-2020-9": [fx.Range("b", exact="1.1")],
              "CVE-2021-1": [fx.Range("c", start_incl="2.0")]}
    assert fx.affected_by(["1.1", "1.2", "2.0"], idx, ranges) == {
        "1.1": ["CVE-2020-9", "CVE-2020-10"], "1.2": [], "2.0": ["CVE-2021-1"]}  # numeric CVE order


# ---------------------------------------------------------------- Windows


def msrc_row(cve, products, date="2024-01-09T08:00:00"):
    return {"cve_id": cve, "msrc": {"docs": [date[:4] + "-Jan"], "initial_release": date, "products": {
        pid: {"name": name, "fixes": [{"fixed_build": b} for b in builds]} for pid, (name, builds) in products.items()}}}


def test_windows_prior_msrc_build():
    rows = [
        msrc_row("CVE-2023-1", {"1": ("Windows Server 2019", ["10.0.17763.5000"])}, "2023-11-14T08:00:00"),
        msrc_row("CVE-2024-2", {"1": ("Windows Server 2019", ["10.0.17763.5329"]),
                                "2": ("Windows Server 2019 (Server Core installation)", ["10.0.17763.5329"])}),
        msrc_row("CVE-2024-3", {"1": ("Windows Server 2019", ["10.0.17763.4000", None])}),
    ]
    builds = fx.windows_builds(rows)
    assert sorted(builds["Windows Server 2019"]) == ["10.0.17763.4000", "10.0.17763.5000", "10.0.17763.5329"]
    assert builds["Windows Server 2019"]["10.0.17763.5000"]["date"] == "2023-11-14"
    names = fx.msrc_product_names("windows_os", ["Windows Server 2019", "Windows Server 2019 (Server Core installation)"],
                                  {"default": {"names": ["Windows Server 2019"]}})
    assert names == ["Windows Server 2019"]
    pairs, dropped = fx.windows_fixtures(rows[1], names, builds)
    assert pairs == [{"product": "Windows Server 2019", "fixed": "10.0.17763.5329", "vulnerable": "10.0.17763.5000"}]
    pairs, dropped = fx.windows_fixtures(rows[2], names, builds)
    assert pairs == [] and dropped[0]["reason"] == "no_prior_msrc_build"
    assert fx.msrc_product_names("Exchange", ["Microsoft Exchange Server 2019 Cumulative Update 13", "Windows Server 2019"],
                                 {"default": {"names": []}, "Exchange": {"prefix": "Microsoft Exchange Server"}}) == [
        "Microsoft Exchange Server 2019 Cumulative Update 13"]


# ---------------------------------------------------------------- eligible view


def label(cve, product, label_set, lab, role="server", ac="explicit"):
    return {"cve_id": cve, "product": product, "label_set": label_set, "label": lab, "role": role, "ac_status": ac}


def test_eligible_units_follow_the_effect_rule():
    rows = [label("CVE-1", "a", "remote", "RCE_CONFIRMED"), label("CVE-2", "a", "remote", "RCE_CONFIRMED", role="client"),
            label("CVE-3", "a", "remote", "NON_RCE"), label("CVE-4", "k", "local", "ADMIN_CONFIRMED", role="client"),
            label("CVE-5", "k", "local", "ADMIN_CONFIRMED", ac="conflict"), label("CVE-6", "k", "local", "NON_ADMIN")]
    assert fx.eligible_units(rows) == {("CVE-1", "a"), ("CVE-4", "k")}


def test_eligible_view_keeps_only_eligible_roles():
    f1 = {"version": "1.0", "vulnerable_for": [{"cve": "CVE-1"}, {"cve": "CVE-2"}], "fixed_for": [], "affected_by": ["CVE-1", "CVE-2"]}
    f2 = {"version": "1.1", "vulnerable_for": [], "fixed_for": [{"cve": "CVE-2"}], "affected_by": []}
    win = {"version": "10.0.1", "vulnerable_for": [{"cve": "CVE-7", "unit": "windows_os"}, {"cve": "CVE-8", "unit": "IIS"}],
           "fixed_for": [], "affected_by": ["CVE-7", "CVE-8"]}
    cat = {"linux": {"a": {"status": "ok", "candidate_cves": {"CVE-1": {}, "CVE-2": {}}, "fixtures": [f1, f2],
                           "dropped": [{"cve": "CVE-2", "reason": "x"}], "unmatched_docs": []},
                     "b": {"status": "ok", "fixtures": [], "dropped": []}},
           "windows": {"WS2019": {"units": ["windows_os", "IIS"], "fixtures": [win], "dropped": [], "affected_unknown_no_fixed_build": []}}}
    view = fx.eligible_view(cat, {("CVE-1", "a"), ("CVE-7", "windows_os")})
    assert list(view["linux"]) == ["a"]
    assert view["linux"]["a"]["fixtures"] == [{**f1, "vulnerable_for": [{"cve": "CVE-1"}], "affected_by": ["CVE-1"]}]
    assert view["linux"]["a"]["dropped"] == [] and list(view["linux"]["a"]["candidate_cves"]) == ["CVE-1"]
    assert view["windows"]["WS2019"]["fixtures"][0]["vulnerable_for"] == [{"cve": "CVE-7", "unit": "windows_os"}]
    assert view["windows"]["WS2019"]["fixtures"][0]["affected_by"] == ["CVE-7"]


# ---------------------------------------------------------------- driver script, end to end on synthetic inputs


def write_raw(folder: Path, files: dict[str, str]) -> None:
    """A finished raw snapshot as rawfetch writes it: files, fetch log, manifest."""
    folder.mkdir(parents=True)
    with (folder / "fetch_log.jsonl").open("w") as log:
        for name, text in files.items():
            p = folder / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            log.write(json.dumps({"name": name, "url": f"https://example.org/{name}", "date_utc": "2026-11-16T00:00:00+00:00",
                                  "http": 200, "bytes": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "content_type": "text/html", "script": "t", "git_commit": "x", "git_dirty": False}) + "\n")
    write_manifest(folder)


def test_build_end_to_end(tmp_path):
    bf = load_script("build_fixtures")
    tomcat_cfg = cfg(m("cpe:2.3:a:apache:tomcat:*:*:*:*:*:*:*:*", versionStartIncluding="9.0.0", versionEndExcluding="9.0.83"))
    mysql_cfg = cfg(m("cpe:2.3:a:oracle:mysql:5.7.1:*:*:*:*:*:*:*"), m("cpe:2.3:a:oracle:mysql:5.7.2:*:*:*:*:*:*:*"),
                    m("cpe:2.3:a:oracle:mysql:*:*:*:*:*:*:*:*", versionEndIncluding="5.7.9"))
    rows = [
        {"cve_id": "CVE-2023-46589", "nvd": {"source_identifier": "x"}, "configurations": tomcat_cfg, "msrc": None,
         "units": [unit("apache:tomcat")]},
        {"cve_id": "CVE-2023-9", "nvd": None, "configurations": mysql_cfg, "msrc": None,
         "units": [unit("oracle:mysql", stage="local_candidate", family="sql")]},
        {"cve_id": "CVE-2024-2", "nvd": None, "configurations": [],
         **msrc_row("CVE-2024-2", {"1": ("Windows Server 2019", ["10.0.17763.5329"])}),
         "units": [unit("windows_os", stage="local_candidate", os_="windows", family="os")]},
        {"cve_id": "CVE-2023-1", "nvd": None, "configurations": [], "units": [],
         **msrc_row("CVE-2023-1", {"1": ("Windows Server 2019", ["10.0.17763.5000"])}, "2023-11-14T08:00:00")},
    ]
    master = tmp_path / "cves.jsonl.gz"
    master.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
    labels = tmp_path / "labels.jsonl.gz"
    labels.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in (
        label("CVE-2024-2", "windows_os", "local", "ADMIN_CONFIRMED"),
        label("CVE-2023-9", "oracle:mysql", "local", "ADMIN_CONFIRMED"))).encode()))
    nvd = tmp_path / "nvd"
    nvd.mkdir()
    page = gzip.compress(json.dumps({"url": "u", "fetched_utc": "t", "params": {},
                                     "response": {"vulnerabilities": [{"cve": {"id": "CVE-2023-9", "configurations": mysql_cfg}}]}}).encode())
    (nvd / "sql__oracle__mysql__00000.json.gz").write_bytes(page)
    (nvd / "manifest.json").write_text(json.dumps({"files": [{
        "product": "oracle:mysql", "path": "sql__oracle__mysql__00000.json.gz", "url": "https://services.nvd.nist.gov/x",
        "fetched_utc": "2026-10-21T08:00:00+00:00", "sha256": hashlib.sha256(page).hexdigest()}]}))
    msrc = tmp_path / "msrc"
    write_raw(msrc, {"cvrf_2024-Jan.json": "{}"})
    raw = tmp_path / "raw"
    listing = '<a href="v9.0.82/">v9.0.82/</a> 2023-10-10 10:00 -\n<a href="v9.0.83/">v9.0.83/</a> 2023-11-15 09:35 -\n'
    write_raw(raw, {"apache__tomcat/l.html": listing, "apache__tomcat/s.html": TOMCAT_PAGE})
    config = {
        "decisions": "test", "master_dataset": str(master), "master_sha256": hashlib.sha256(master.read_bytes()).hexdigest(),
        "eligible_labels": str(labels), "eligible_labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "msrc_fetch_logs": {str(msrc): "fetch_log.jsonl"}, "catalogue_root": str(tmp_path / "cat"),
        "results_root": str(tmp_path / "results"), "candidate_stages": ["remote_candidate", "local_candidate"],
        "cpe_part": {"default": "a"}, "windows_products": {"default": {"names": ["Windows Server 2019"]}},
        "not_covered": {"oracle:mysql": "no dated release list"},
        "unverified_release": {"products": ["oracle:mysql"], "nvd_raw_dir": str(nvd), "decision": "test"},
        "products": {"apache:tomcat": {
            "version_regex": r"\d+\.\d+\.\d+(?:-M\d+)?",
            "releases": [{"id": "l.html", "kind": "listing", "url": "u", "pattern": r"v(?P<name>\d+\.\d+\.\d+)/"}],
            "evidence": [{"id": "s.html", "kind": "sections", "url": "u", "header": r"^Fixed in Apache Tomcat (?P<v>\d+\.\d+\.\d+(?:-M\d+)?)",
                          "cve_at_line_start": True}]}},
    }  # fmt: skip
    bf.build(config, [raw])
    out = next((tmp_path / "cat").iterdir())
    cat = json.loads(gzip.decompress((out / "catalogue.json.gz").read_bytes()))
    tom = cat["linux"]["apache:tomcat"]
    assert [(f["version"], f["release_date"], bool(f["vulnerable_for"]), bool(f["fixed_for"])) for f in tom["fixtures"]] == [
        ("9.0.82", "2023-10-10", True, False), ("9.0.83", "2023-11-15", False, True)]
    assert tom["fixtures"][0]["affected_by"] == ["CVE-2023-46589"] and tom["fixtures"][1]["affected_by"] == []
    ev = tom["fixtures"][1]["fixed_for"][0]["evidence"][0]
    assert ev["url"] == "https://example.org/apache__tomcat/s.html" and len(ev["sha256"]) == 64 and ev["retrieved_utc"]
    assert tom["fixtures"][0]["release_source"]["raw"] == f"{tmp_path.name}/raw/apache__tomcat/l.html"  # <source>/<date>/<file>
    my = cat["linux"]["oracle:mysql"]
    assert my["status"] == "unverified_release" and my["verified_status"] == "dropped"
    assert [(f["version"], f["release_date"], f["unverified_release"], f["fixed_for"], f["affected_by"]) for f in my["fixtures"]] == [
        ("5.7.9", None, True, [], ["CVE-2023-9"])]  # highest exact or inclusive string: the 5.7.9 end bound (OQ-P5-11)
    assert my["fixtures"][0]["release_source"][0]["retrieved_utc"] == "2026-10-21T08:00:00+00:00"
    win = cat["windows"]["Windows Server 2019"]
    assert [(f["version"], f["affected_by"]) for f in win["fixtures"]] == [
        ("10.0.17763.5000", ["CVE-2024-2"]), ("10.0.17763.5329", [])]
    assert win["fixtures"][1]["release_source"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    run = next((tmp_path / "results" / "catalogue").iterdir())
    assert "apache:tomcat" in (run / "report.md").read_text()
    regained = json.loads(gzip.decompress((run / "raw.json.gz").read_bytes()))["unverified_regained"]
    assert regained == [{"product": "oracle:mysql", "status": "unverified_release", "unverified_fixtures": 1, "candidate_cves": 1,
                         "candidate_cves_with_fixture": 1, "eligible_remote": 0, "eligible_remote_regained": 0,
                         "eligible_local": 1, "eligible_local_regained": 1}]  # fmt: skip
    el = json.loads(gzip.decompress((out / "catalogue_eligible.json.gz").read_bytes()))
    assert list(el["linux"]) == ["oracle:mysql"]  # the Tomcat CVE has no eligible label
    assert el["linux"]["oracle:mysql"]["fixtures"][0]["unverified_release"] is True
    assert [f["version"] for f in el["windows"]["Windows Server 2019"]["fixtures"]] == ["10.0.17763.5000", "10.0.17763.5329"]


def test_raw_reads_a_base_and_its_supplement_and_refuses_duplicates(tmp_path):
    bf = load_script("build_fixtures")
    base, sup = tmp_path / "fixtures" / "2026-11-16", tmp_path / "fixtures_supplement" / "2026-11-16"
    write_raw(base, {"a/x.html": "x"})
    write_raw(sup, {"a/y.html": "y"})
    raw = bf.Raw([base, sup])
    assert raw.text("a/y.html") == "y" and raw.ref("a/x.html")["raw"] == "fixtures/2026-11-16/a/x.html"
    assert raw.ref("a/y.html")["raw"] == "fixtures_supplement/2026-11-16/a/y.html"
    dup = tmp_path / "other" / "2026-11-16"
    write_raw(dup, {"a/x.html": "other"})
    with pytest.raises(ValueError):
        bf.Raw([base, dup])


def test_fetch_never_shadows_its_base_snapshots():
    """Regression: a local variable named `base` in the index branch once replaced the base snapshot list."""
    import ast
    tree = ast.parse((SCRIPTS / "build_fixtures.py").read_text())
    fetch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fetch")
    assigned = {t.id for n in ast.walk(fetch) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert "base" not in assigned


def test_ghsa_pages_are_read_even_when_incomplete(tmp_path):
    bf = load_script("build_fixtures")
    asc = [{"ghsa_id": f"A{i}", "cve_id": None, "vulnerabilities": []} for i in range(100)]
    desc = [{"ghsa_id": f"D{i}", "cve_id": None, "vulnerabilities": []} for i in range(100)]
    write_raw(tmp_path / "fixtures" / "2026-11-16", {"p/adv_asc.json": json.dumps(asc), "p/adv_desc.json": json.dumps(desc)})
    names, note = bf.ghsa_union(bf.Raw([tmp_path / "fixtures" / "2026-11-16"]), "p", {"id": "adv"})
    assert names == ["p/adv_asc.json", "p/adv_desc.json"] and note.startswith("incomplete")


def test_candidate_units_include_eligible_units_outside_candidate_stages():
    rows = [{"cve_id": "CVE-2026-81642", "units": [unit("nlnetlabs:unbound", stage="role_not_server", family="dns")]}]
    assert fx.candidate_units(rows) == {}
    got = fx.candidate_units(rows, include={("CVE-2026-81642", "nlnetlabs:unbound")})
    assert list(got["nlnetlabs:unbound"]["cves"]) == ["CVE-2026-81642"]


def test_build_refuses_a_changed_master(tmp_path):
    bf = load_script("build_fixtures")
    master = tmp_path / "m.jsonl.gz"
    master.write_bytes(gzip.compress(b"{}\n"))
    with pytest.raises(SystemExit):
        bf.load_master({"master_dataset": str(master), "master_sha256": "0" * 64})


def test_p5_config_inputs_are_repo_local_and_labels_match_frozen_registry():
    """M3 (DECIDED-BY-AGENT): after the merge into d28-rebuild, P5 inputs resolve inside this checkout, not a
    sibling track worktree, and the eligible view reads the canonical frozen p3-labels-v2.1 (OQ-M2-1 registry)."""
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "fixtures" / "p5.json").read_text())
    paths = [cfg["master_dataset"], cfg["eligible_labels"], cfg["unverified_release"]["nvd_raw_dir"], *cfg["msrc_fetch_logs"]]
    assert all(not Path(p).is_absolute() and ".." not in Path(p).parts for p in paths)
    frozen = json.loads((root / "configs" / "d28_effect_labels.json").read_text())["versions"]["p3-labels-v2.1"]
    assert (cfg["eligible_labels"], cfg["eligible_labels_sha256"]) == (frozen["path"], frozen["sha256"])


def test_eligible_units_is_the_effect_rule_on_every_label_combination():
    """OQ-P5-7 (M3): one eligibility rule. Every (label set, label, role, AC status) combination is eligible in
    the fixtures view exactly when ahg.effect.eligible says so."""
    import itertools

    from ahg import effect

    combos = itertools.product(("remote", "local"), ("RCE_CONFIRMED", "ADMIN_CONFIRMED", "UNKNOWN", "NON_RCE"),
                               ("server", "client", "both", None), ("explicit", "missing"))
    rows = [{"cve_id": f"CVE-{i}", "product": "p", "label_set": s, "label": lab, "role": role, "ac_status": ac}
            for i, (s, lab, role, ac) in enumerate(combos)]
    assert fx.eligible_units(rows) == {(r["cve_id"], "p") for r in rows if effect.eligible(r)}
    assert 0 < len(fx.eligible_units(rows)) < len(rows)
