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
