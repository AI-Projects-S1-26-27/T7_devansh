"""Fetch every NVD CVE record matching some CPE prefixes, published in a date range, into
data/raw/nvd/<date>/<label>/ (one folder and one manifest per invocation).

Usage: python scripts/fetch_nvd_cpe.py LABEL PUB_START PUB_END CPE [CPE ...] [--date YYYY-MM-DD]
  e.g. python scripts/fetch_nvd_cpe.py linux_kernel 2016-01-01 2026-10-21 cpe:2.3:o:linux:linux_kernel

The CVE API 2.0 allows at most 120 consecutive days per published-date query, so the range is split into
WINDOW_DAYS-day windows, each paged at PAGE_SIZE records (2000-record pages were cut off mid-transfer
during the 2026-10-21 snapshot). Requests are paced at PACE seconds: without an API key NVD allows
5 requests per 30 s. Each response body is saved exactly as received (BUILD_RULES rule 5).

The manifest is written only after the strict REP6 validation (ahg.nvd_snapshot) passes, and it holds the
full validation report under "rep6". A failed validation is printed and no manifest is written.
"""

import argparse
import json
import sys
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ahg.nvd_snapshot import validate
from ahg.rawfetch import Fetcher, write_manifest

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE_SIZE, PACE, WINDOW_DAYS = 500, 6.5, 100
ROOT = Path(__file__).resolve().parents[1]


def windows(first: date, last: date, days: int = WINDOW_DAYS) -> list[tuple[date, date]]:
    """Consecutive [start, end] day ranges of at most `days` days covering first..last exactly."""
    out, d = [], first
    while d <= last:
        e = min(d + timedelta(days=days - 1), last)
        out.append((d, e))
        d = e + timedelta(days=1)
    return out


def page_url(cpe: str, w: tuple[date, date], start: int) -> str:
    params = {
        "virtualMatchString": cpe, "pubStartDate": f"{w[0]}T00:00:00.000", "pubEndDate": f"{w[1]}T23:59:59.999",
        "resultsPerPage": PAGE_SIZE, "startIndex": start,
    }  # fmt: skip
    return API + "?" + urllib.parse.urlencode(params, safe=":")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("pub_start", type=date.fromisoformat)
    ap.add_argument("pub_end", type=date.fromisoformat)
    ap.add_argument("cpes", nargs="+")
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    a = ap.parse_args()
    date.fromisoformat(a.date)
    if not a.label.replace("_", "").isalnum():
        sys.exit("LABEL: letters, digits and _ only")
    f = Fetcher(ROOT / "data" / "raw" / "nvd" / a.date / a.label, "scripts/fetch_nvd_cpe.py", PACE)
    pages = []  # (query key, requested start, page size, response) for REP6
    for cpe, w in ((c, w) for c in a.cpes for w in windows(a.pub_start, a.pub_end)):
        slug = cpe.split(":", 3)[3].replace(":", "__")
        start, total = 0, None
        while total is None or start < total:
            name = f"{slug}__{w[0]}__{start:06d}.json"
            rec = f.fetch(name, page_url(cpe, w, start))
            if rec.get("http", 200) != 200:
                sys.exit(f"failed {name}: {rec}; rerun to resume")
            resp = json.loads((f.out / name).read_bytes())
            total = resp["totalResults"]
            pages.append({"query": f"{cpe} {w[0]}..{w[1]}", "file": name, "requested_start": start,
                          "page_size": PAGE_SIZE, "response": resp})  # fmt: skip
            print(f"{slug} {w[0]}..{w[1]} start={start:5d} got={len(resp['vulnerabilities']):4d} total={total}", flush=True)
            start += PAGE_SIZE
    report = validate(pages)
    print(json.dumps({k: report[k] for k in ("ok", "pages", "records", "unique_cves", "errors")}, indent=1))
    if not report["ok"]:
        sys.exit("REP6 validation failed; manifest not written")
    # The full REP6 report is the manifest's "rep6" entry: the logged validation record of this snapshot.
    m = write_manifest(f.out, {"api": API, "label": a.label, "cpes": a.cpes, "pub_range": [str(a.pub_start), str(a.pub_end)],
                               "page_size": PAGE_SIZE, "window_days": WINDOW_DAYS, "pace_seconds": PACE,
                               "rep6": report})  # fmt: skip
    print(f"manifest: {len(m['files'])} files")


if __name__ == "__main__":
    main()
