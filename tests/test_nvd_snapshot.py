"""REP6 snapshot validation (D28 Round 3): offsets, per-page counts, duplicates, totals."""

from ahg.nvd_snapshot import validate


def page(q, start, ids, total, size=2, returned_start=None, rpp=None):
    vulns = [{"cve": {"id": i}} for i in ids]
    resp = {"startIndex": start if returned_start is None else returned_start, "totalResults": total,
            "resultsPerPage": len(ids) if rpp is None else rpp, "vulnerabilities": vulns}  # fmt: skip
    return {"query": q, "file": f"{q}-{start}", "requested_start": start, "page_size": size, "response": resp}


def good():
    return [page("a", 0, ["C1", "C2"], 3), page("a", 2, ["C3"], 3), page("b", 0, ["C3"], 1), page("e", 0, [], 0)]


def test_valid_snapshot_passes_and_counts_overlap():
    r = validate(good())
    assert r["ok"], r["errors"]
    assert (r["pages"], r["records"], r["unique_cves"], r["cves_in_several_queries"]) == (4, 4, 3, 1)


def test_offset_mismatch_fails():
    ps = good()
    ps[1] = page("a", 2, ["C3"], 3, returned_start=0)
    assert any("returned startIndex" in e for e in validate(ps)["errors"])


def test_short_page_and_wrong_results_per_page_fail():
    ps = good()
    ps[0] = page("a", 0, ["C1"], 3)
    errs = validate(ps)["errors"]
    assert any("1 records, expected 2" in e for e in errs) and any("!= totalResults" in e for e in errs)
    ps = good()
    ps[0] = page("a", 0, ["C1", "C2"], 3, rpp=5)
    assert any("resultsPerPage" in e for e in validate(ps)["errors"])


def test_missing_page_duplicate_and_total_drift_fail():
    assert any("requested starts" in e for e in validate([page("a", 0, ["C1", "C2"], 3)])["errors"])
    dup = [page("a", 0, ["C1", "C2"], 3), page("a", 2, ["C1"], 3)]
    assert any("duplicate CVE IDs" in e for e in validate(dup)["errors"])
    drift = [page("a", 0, ["C1", "C2"], 3), page("a", 2, ["C3"], 4)]
    assert any("totalResults differ" in e for e in validate(drift)["errors"])


def test_request_plan_check_against_recorded_commit(tmp_path):
    """Inventory #20: the snapshot is checked against the fetching code at its commit; dirty stays unresolved."""
    import gzip
    import json
    import subprocess
    import urllib.parse

    from ahg.nvd_snapshot import request_plan_check

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src" / "ahg").mkdir(parents=True)
    (repo / "scripts" / "fetch_nvd.py").write_text('API = "https://x/api"\nPAGE_SIZE = 2\nPACE = 6.5\n')
    (repo / "src" / "ahg" / "cve.py").write_text('FAMILY_PRODUCTS: dict = {"ssh": ("openbsd:openssh",)}\n')
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()  # noqa: E731
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    commit = git("rev-parse", "HEAD")

    def snapshot(dirty, start=0):
        snap = tmp_path / f"snap{dirty}{start}"
        snap.mkdir()
        params = {"virtualMatchString": "cpe:2.3:*:openbsd:openssh", "resultsPerPage": 2, "startIndex": start}
        url = "https://x/api?" + urllib.parse.urlencode(params, safe=":/*")
        (snap / "p.json.gz").write_bytes(gzip.compress(json.dumps({"url": url, "params": params}).encode()))
        files = [{"path": "p.json.gz", "family": "ssh", "product": "openbsd:openssh", "startIndex": 0, "url": url}]
        (snap / "manifest.json").write_text(json.dumps({"api": "https://x/api", "page_size": 2, "pace_seconds": 6.5,
                                                        "git_commit": commit, "git_dirty_src_or_scripts": dirty, "files": files}))
        return snap

    assert request_plan_check(snapshot(False), repo)["status"] == "CONSISTENT_CLEAN"
    assert request_plan_check(snapshot(True), repo)["status"] == "CONSISTENT_DIRTY_UNRESOLVABLE"
    bad = request_plan_check(snapshot(False, start=5), repo)
    assert bad["status"] == "INCONSISTENT" and "differs" in bad["errors"][0]
