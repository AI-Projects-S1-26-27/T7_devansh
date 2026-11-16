"""Fetch MSRC CVRF monthly documents (API v3.0) into data/raw/msrc/<date>/ with a manifest.

Usage: python scripts/fetch_msrc.py FROM TO [--ids ID ...] [--date YYYY-MM-DD]
  FROM, TO: first and last month, YYYY-MM (inclusive), e.g. 2024-01 2026-10.
  --ids: extra update IDs to fetch as well (e.g. months missing from an older snapshot).
  --date: snapshot folder name, default today (UTC).

The update index (cvrf/v3.0/updates) is fetched first and saved; every update whose ID falls in the
month range (including out-of-band IDs such as 2017-May-B) is then fetched as JSON. The manifest is
written only if every requested document was fetched with HTTP 200.
"""

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from ahg.rawfetch import Fetcher, write_manifest

API = "https://api.msrc.microsoft.com/cvrf/v3.0"
PACE = 2.0  # MSRC documents no rate limit; 2 s keeps the load small
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
ROOT = Path(__file__).resolve().parents[1]
JSON = {"Accept": "application/json"}


def update_month(update_id: str) -> tuple[int, int] | None:
    """(year, month) of an MSRC update ID such as '2024-Jan' or '2017-May-B'; None if it has another form."""
    m = re.fullmatch(r"(\d{4})-([A-Za-z]{3})(-[A-Za-z0-9]+)?", update_id)
    if not m or m.group(2).capitalize() not in MONTHS:
        return None
    return int(m.group(1)), MONTHS.index(m.group(2).capitalize()) + 1


def ids_in_range(index: dict, first: tuple[int, int], last: tuple[int, int]) -> list[str]:
    """Update IDs from the index whose month lies in [first, last], sorted by (month, ID)."""
    picked = [(update_month(u["ID"]), u["ID"]) for u in index["value"]]
    return [i for ym, i in sorted(p for p in picked if p[0] is not None and first <= p[0] <= last)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("last")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    a = ap.parse_args()
    first, last = (tuple(int(x) for x in s.split("-")) for s in (a.first, a.last))
    date.fromisoformat(a.date)  # reject a malformed folder name
    f = Fetcher(ROOT / "data" / "raw" / "msrc" / a.date, "scripts/fetch_msrc.py", PACE)

    rec = f.fetch("updates.json", f"{API}/updates", JSON)
    if rec.get("http", 200) != 200:
        sys.exit(f"update index not fetched: {rec}")
    index = json.loads((f.out / "updates.json").read_bytes())
    wanted = ids_in_range(index, first, last)
    known = {u["ID"] for u in index["value"]}
    for extra in a.ids:
        if extra not in known:
            sys.exit(f"--ids {extra}: not in the MSRC update index")
    wanted += [i for i in a.ids if i not in wanted]
    failed = []
    for uid in wanted:
        r = f.fetch(f"cvrf_{uid}.json", f"{API}/cvrf/{uid}", JSON)
        print(r.get("http", r.get("skipped")), uid, r.get("bytes", ""), flush=True)
        if r.get("http", 200) != 200:
            failed.append(uid)
    if failed:
        sys.exit(f"failed: {failed}; rerun to resume (fetched files are kept), manifest not written")
    m = write_manifest(f.out, {"api": API, "range": [a.first, a.last], "extra_ids": a.ids, "update_ids": wanted})
    print(f"manifest: {len(m['files'])} files")


if __name__ == "__main__":
    main()
