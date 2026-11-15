"""REP6: strict validation of a paged NVD CVE API 2.0 snapshot (D28 Round 3).

A snapshot is a list of pages. Each page is a dict:
  query: key of the paged query it belongs to (pages of one query share it);
  file: file name, for the report;
  requested_start: the startIndex the request asked for;
  page_size: the resultsPerPage the request asked for;
  response: the API's JSON response.

Checks, per query:
  - every page's returned startIndex equals the requested one (offset check);
  - every page reports the same totalResults;
  - the requested starts are exactly 0, size, 2*size, ... covering totalResults, with no gap or extra page;
  - every page holds exactly min(size, total - start) records, and its resultsPerPage says the same;
  - no CVE ID appears twice within the query.
Across queries, CVE IDs that occur in several queries are counted (expected when queries overlap, e.g. one
query per product) but are not errors. The report lists every failed check; ok is True only if none failed.
"""

from collections import Counter, defaultdict


def validate(pages: list[dict]) -> dict:
    by_query: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        by_query[p["query"]].append(p)
    errors, per_query, all_ids = [], {}, Counter()
    for q, ps in sorted(by_query.items()):
        ps = sorted(ps, key=lambda p: p["requested_start"])
        size = {p["page_size"] for p in ps}
        totals = {p["response"]["totalResults"] for p in ps}
        if len(size) != 1:
            errors.append(f"{q}: page sizes differ {sorted(size)}")
        if len(totals) != 1:
            errors.append(f"{q}: totalResults differ across pages {sorted(totals)}")
        size, total = min(size), max(totals)
        want_starts = list(range(0, total, size)) or [0]  # an empty result is still one page
        got_starts = [p["requested_start"] for p in ps]
        if got_starts != want_starts:
            errors.append(f"{q}: requested starts {got_starts} != expected {want_starts}")
        ids = Counter()
        for p in ps:
            r, start = p["response"], p["requested_start"]
            n = len(r["vulnerabilities"])
            expect = max(0, min(size, total - start))
            if r.get("startIndex") != start:
                errors.append(f"{p['file']}: returned startIndex {r.get('startIndex')} != requested {start}")
            if n != expect:
                errors.append(f"{p['file']}: {n} records, expected {expect}")
            if r.get("resultsPerPage") != n:
                errors.append(f"{p['file']}: resultsPerPage {r.get('resultsPerPage')} != records {n}")
            ids.update(v["cve"]["id"] for v in r["vulnerabilities"])
        dups = sorted(i for i, c in ids.items() if c > 1)
        if dups:
            errors.append(f"{q}: duplicate CVE IDs within the query: {dups[:20]}{' ...' if len(dups) > 20 else ''}")
        records = sum(ids.values())
        if records != total:
            errors.append(f"{q}: {records} records across pages != totalResults {total}")
        per_query[q] = {"pages": len(ps), "totalResults": total, "records": records, "unique": len(ids)}
        all_ids.update(ids.keys())
    return {
        "ok": not errors, "errors": errors, "queries": per_query, "pages": len(pages),
        "records": sum(v["records"] for v in per_query.values()), "unique_cves": len(all_ids),
        "cves_in_several_queries": sum(1 for c in all_ids.values() if c > 1),
    }  # fmt: skip
