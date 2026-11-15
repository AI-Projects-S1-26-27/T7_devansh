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

import ast
import gzip
import hashlib
import json
import subprocess
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path


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


def load_pages(snap_dir) -> list[dict]:
    """The pages of an NVD snapshot in the form validate() takes, after checking each file's SHA-256.

    Two manifest layouts exist:
      - scripts/fetch_nvd.py (2026-10-21): gzipped envelopes {url, params, fetched_utc, response}, one
        query per (family, product), listed under "files" with "path";
      - scripts/fetch_nvd_cpe.py (D28): response bodies saved as received, listed by ahg.rawfetch under
        "files" with "name" and "url"; the query is the URL without startIndex.
    """
    snap = Path(snap_dir)
    manifest = json.loads((snap / "manifest.json").read_text())
    pages = []
    for e in manifest["files"]:
        name = e.get("path") or e["name"]
        raw = (snap / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != e["sha256"]:
            raise ValueError(f"{snap / name}: SHA-256 differs from the manifest; snapshot was modified")
        if "path" in e:
            env = json.loads(gzip.decompress(raw))
            q, resp = f"{e['family']}/{e['product']}", env["response"]
            start, size = env["params"]["startIndex"], env["params"]["resultsPerPage"]
        else:
            params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(e["url"]).query))
            start, size, resp = int(params.pop("startIndex")), int(params["resultsPerPage"]), json.loads(raw)
            q = " ".join(f"{k}={v}" for k, v in sorted(params.items()))
        pages.append({"query": q, "file": name, "requested_start": start, "page_size": size, "response": resp})
    return pages


def request_plan_check(snap_dir, repo_root) -> dict:
    """Inventory #20: machine check of a scripts/fetch_nvd.py snapshot against the code at its recorded commit.

    Replaces the hand-written PROVENANCE_NOTES override. It reads scripts/fetch_nvd.py and src/ahg/cve.py at
    the manifest's git_commit (git show) and checks that the snapshot is exactly that code's request plan:
    API, page size and pace match its constants; the (family, product) queries are its FAMILY_PRODUCTS; and
    every page's URL and parameters are the ones it builds for that product and startIndex.

    What it cannot check: which files were on disk at fetch time. A git_dirty flag of true therefore stays
    unresolved (status CONSISTENT_DIRTY_UNRESOLVABLE), never overridden.
    """
    snap = Path(snap_dir)
    m = json.loads((snap / "manifest.json").read_text())
    commit = m["git_commit"]

    def show(path):
        r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo_root, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    fetch_src, cve_src = show("scripts/fetch_nvd.py"), show("src/ahg/cve.py")
    if fetch_src is None or cve_src is None:
        return {"status": "COMMIT_NOT_AVAILABLE", "commit": commit, "errors": []}

    def const(src, name):
        for node in ast.parse(src).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return ast.literal_eval(node.value)
        raise KeyError(name)

    api, size, pace = const(fetch_src, "API"), const(fetch_src, "PAGE_SIZE"), const(fetch_src, "PACE")
    products = {(f, p) for f, ps in const(cve_src, "FAMILY_PRODUCTS").items() for p in ps}
    errors = []
    if (m["api"], m["page_size"], m["pace_seconds"]) != (api, size, pace):
        errors.append(f"manifest api/page_size/pace {(m['api'], m['page_size'], m['pace_seconds'])} != code {(api, size, pace)}")
    got = {(e["family"], e["product"]) for e in m["files"]}
    if got != products:
        errors.append(f"queried products differ from FAMILY_PRODUCTS: extra {sorted(got - products)}, missing {sorted(products - got)}")
    for e in m["files"]:
        env = json.loads(gzip.decompress((snap / e["path"]).read_bytes()))
        params = {"virtualMatchString": f"cpe:2.3:*:{e['product']}", "resultsPerPage": size, "startIndex": e["startIndex"]}
        url = api + "?" + urllib.parse.urlencode(params, safe=":/*")
        if env["params"] != params or env["url"] != url or e["url"] != url:
            errors.append(f"{e['path']}: request differs from the code's plan")
    dirty = m.get("git_dirty_src_or_scripts")
    status = "INCONSISTENT" if errors else ("CONSISTENT_DIRTY_UNRESOLVABLE" if dirty else "CONSISTENT_CLEAN")
    return {"status": status, "commit": commit, "git_dirty_src_or_scripts": dirty, "pages_checked": len(m["files"]),
            "errors": errors}  # fmt: skip
