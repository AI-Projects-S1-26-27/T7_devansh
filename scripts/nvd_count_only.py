"""nvd_count_only.py FILE.tsv [--date YYYY-MM-DD]: NVD CVE API 2.0 count-only queries.

FILE.tsv has one query per line: label<TAB>query-string (the part after "cves/2.0?"); lines starting with
"#" are skipped. Each query is sent with resultsPerPage=1 and only totalResults is kept: the response body
is parsed in memory and discarded, never written. Each query is appended (label, URL, UTC time, HTTP status,
totalResults, records in the discarded body, fetching script's git commit and dirty flag) to
data/raw/sources/<date>/nvd_count_queries.jsonl, where <date> defaults to today (UTC). Requests are paced
at 6.5 s (no API key: NVD allows 5 requests per 30 s).
"""

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from ahg.rawfetch import Fetcher, utc_now

BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0?"
PACE = 6.5


def count(f: Fetcher, log: Path, label: str, q: str) -> dict:
    url = BASE + q + "&resultsPerPage=1"
    status, body, _ = f._get(url, {})
    total = nrec = None
    if status == 200:
        d = json.loads(body)
        total, nrec = d.get("totalResults"), len(d.get("vulnerabilities", []))
    rec = {"label": label, "query": url, "time_utc": utc_now(), "http": status, "totalResults": total,
           "records_in_body_discarded": nrec, "script": "scripts/nvd_count_only.py", "git_commit": f.commit,
           "git_dirty": f.dirty}  # fmt: skip
    with log.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(status, total, label, flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    a = ap.parse_args()
    date.fromisoformat(a.date)
    f = Fetcher(ROOT / "data" / "raw" / "sources" / a.date, "scripts/nvd_count_only.py", PACE)
    rows = [line.split("\t") for line in a.tsv.read_text().splitlines() if line.strip() and not line.startswith("#")]
    for label, q in rows:
        count(f, f.out / "nvd_count_queries.jsonl", label, q)


if __name__ == "__main__":
    main()
