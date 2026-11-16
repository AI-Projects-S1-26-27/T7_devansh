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
    docs, refusal = fx.ev_cve_json5(json5("Linux", versions), "Linux", "Linux", "raw")
    assert refusal is None and [d.version for d in docs] == ["0", "6.6.4", "6.7"]
    docs, refusal = fx.ev_cve_json5(json5("mitre", versions), "Linux", "Linux", "raw")
    assert docs == [] and "not the vendor CNA" in refusal
    assert fx.ev_cve_json5(json5("Linux", versions, product="Other"), "Linux", "Linux", "raw")[0] == []


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


def test_catalogue_per_range_fixtures_from_documented_fixes():
    idx = fx.ReleaseIndex(rels("8.5.95", "8.5.96", "9.0.81", "9.0.82", "9.0.83", "10.1.15", "10.1.16"))
    cve = "CVE-2023-46589"
    ranges = {cve: [fx.Range("a", start_incl="8.5.0", end_incl="8.5.95"), fx.Range("b", start_incl="9.0.0", end_excl="9.0.83"),
                    fx.Range("c", start_incl="10.1.0", end_incl="10.1.15")]}
    docs = [doc(cve, "8.5.96"), doc(cve, "9.0.83"), doc(cve, "10.1.16"), doc(cve, "11.0.0-M11")]
    cat = fx.catalogue_product(ranges, idx, docs)
    pairs = sorted((f["fixed_for"][0]["after"], v) for v, f in cat.fixtures.items() if f["fixed_for"])
    assert pairs == [("10.1.15", "10.1.16"), ("8.5.95", "8.5.96"), ("9.0.82", "9.0.83")]
    assert sorted(v for v, f in cat.fixtures.items() if f["vulnerable_for"]) == ["10.1.15", "8.5.95", "9.0.82"]
    assert cat.dropped == []
    assert [u["version"] for u in cat.unmatched_docs] == ["11.0.0-M11"]  # milestone: not a stable release


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
    assert (c, "no_stable_release_in_range") in reasons  # 2.0.1 is not a release
    conflict = [d for d in cat.dropped if d["cve"] == c and d["range"] == "(*, 2.4.57)"][0]
    assert conflict["conflicting_docs"] == ["2.4.56"]


def test_fix_on_a_later_affected_branch_is_not_this_ranges_fix():
    idx = fx.ReleaseIndex(rels("4.19.325", "5.4.249", "5.4.250"))
    cve = "CVE-2024-9"
    ranges = {cve: [fx.Range("a", start_incl="4.19", end_excl="4.20"), fx.Range("b", start_incl="5.4", end_excl="5.4.250")]}
    cat = fx.catalogue_product(ranges, idx, [doc(cve, "5.4.250")])
    assert cat.fixtures["5.4.250"]["fixed_for"] == [{"cve": cve, "range": "[5.4, 5.4.250)", "after": "5.4.249", "evidence": [
        {"raw": "advisory", "kind": "sections", "excerpt": "x"}]}]
    dropped = [d for d in cat.dropped if d["range"] == "[4.19, 4.20)"]
    assert dropped[0]["reason"] == "no_documented_fix_after_vulnerable_fixture"  # 5.4.249 (affected) lies between


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
    rows = [
        {"cve_id": "CVE-2023-46589", "nvd": {"source_identifier": "x"}, "configurations": tomcat_cfg, "msrc": None,
         "units": [unit("apache:tomcat")]},
        {"cve_id": "CVE-2023-9", "nvd": None, "configurations": [], "msrc": None, "units": [unit("oracle:mysql", family="sql")]},
        {"cve_id": "CVE-2024-2", "nvd": None, "configurations": [],
         **msrc_row("CVE-2024-2", {"1": ("Windows Server 2019", ["10.0.17763.5329"])}),
         "units": [unit("windows_os", stage="local_candidate", os_="windows", family="os")]},
        {"cve_id": "CVE-2023-1", "nvd": None, "configurations": [], "units": [],
         **msrc_row("CVE-2023-1", {"1": ("Windows Server 2019", ["10.0.17763.5000"])}, "2023-11-14T08:00:00")},
    ]
    master = tmp_path / "cves.jsonl.gz"
    master.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
    msrc = tmp_path / "msrc"
    write_raw(msrc, {"cvrf_2024-Jan.json": "{}"})
    raw = tmp_path / "raw"
    listing = '<a href="v9.0.82/">v9.0.82/</a> 2023-10-10 10:00 -\n<a href="v9.0.83/">v9.0.83/</a> 2023-11-15 09:35 -\n'
    write_raw(raw, {"apache__tomcat/l.html": listing, "apache__tomcat/s.html": TOMCAT_PAGE})
    config = {
        "decisions": "test", "master_dataset": str(master), "master_sha256": hashlib.sha256(master.read_bytes()).hexdigest(),
        "msrc_fetch_logs": {str(msrc): "fetch_log.jsonl"}, "catalogue_root": str(tmp_path / "cat"),
        "results_root": str(tmp_path / "results"), "candidate_stages": ["remote_candidate", "local_candidate"],
        "cpe_part": {"default": "a"}, "windows_products": {"default": {"names": ["Windows Server 2019"]}},
        "not_covered": {"oracle:mysql": "no dated release list"},
        "products": {"apache:tomcat": {
            "version_regex": r"\d+\.\d+\.\d+(?:-M\d+)?",
            "releases": [{"id": "l.html", "kind": "listing", "url": "u", "pattern": r"v(?P<name>\d+\.\d+\.\d+)/"}],
            "evidence": [{"id": "s.html", "kind": "sections", "url": "u", "header": r"^Fixed in Apache Tomcat (?P<v>\d+\.\d+\.\d+(?:-M\d+)?)",
                          "cve_at_line_start": True}]}},
    }  # fmt: skip
    bf.build(config, raw)
    out = next((tmp_path / "cat").iterdir())
    cat = json.loads(gzip.decompress((out / "catalogue.json.gz").read_bytes()))
    tom = cat["linux"]["apache:tomcat"]
    assert [(f["version"], f["release_date"], bool(f["vulnerable_for"]), bool(f["fixed_for"])) for f in tom["fixtures"]] == [
        ("9.0.82", "2023-10-10", True, False), ("9.0.83", "2023-11-15", False, True)]
    assert tom["fixtures"][0]["affected_by"] == ["CVE-2023-46589"] and tom["fixtures"][1]["affected_by"] == []
    ev = tom["fixtures"][1]["fixed_for"][0]["evidence"][0]
    assert ev["url"] == "https://example.org/apache__tomcat/s.html" and len(ev["sha256"]) == 64 and ev["retrieved_utc"]
    assert tom["fixtures"][0]["release_source"]["raw"] == "apache__tomcat/l.html"
    assert cat["linux"]["oracle:mysql"]["status"] == "dropped"
    win = cat["windows"]["Windows Server 2019"]
    assert [(f["version"], f["affected_by"]) for f in win["fixtures"]] == [
        ("10.0.17763.5000", ["CVE-2024-2"]), ("10.0.17763.5329", [])]
    assert win["fixtures"][1]["release_source"][0]["sha256"] == hashlib.sha256(b"{}").hexdigest()
    run = next((tmp_path / "results" / "catalogue").iterdir())
    assert "apache:tomcat" in (run / "report.md").read_text()


def test_build_refuses_a_changed_master(tmp_path):
    bf = load_script("build_fixtures")
    master = tmp_path / "m.jsonl.gz"
    master.write_bytes(gzip.compress(b"{}\n"))
    with pytest.raises(SystemExit):
        bf.load_master({"master_dataset": str(master), "master_sha256": "0" * 64})
