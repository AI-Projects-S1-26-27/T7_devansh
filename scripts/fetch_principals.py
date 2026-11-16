"""Fetch the candidate documentation pages for the runtime-principal table (D28 P6) into
data/raw/principals/<date>/ with a manifest (BUILD_RULES rule 5).

Usage: python scripts/fetch_principals.py [--date YYYY-MM-DD]

Pages come from configs/gen/principal_sources.json and are fetched one at a time, paced, by
ahg.rawfetch (generic client name, per-host pacing, append-only). The table itself
(configs/gen/principals.json) cites only what these pages state verbatim; anything else is UNKNOWN.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ahg.rawfetch import Fetcher, write_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "configs" / "gen" / "principal_sources.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    a = ap.parse_args()
    cfg = json.loads(SOURCES.read_text())
    f = Fetcher(ROOT / "data" / "raw" / "principals" / a.date, "scripts/fetch_principals.py", cfg["pace_seconds"])
    for page in cfg["pages"]:  # sequential: one request at a time, paced per host by the Fetcher
        rec = f.fetch(page["name"], page["url"])
        print(page["name"], rec.get("http", rec.get("skipped")))
    m = write_manifest(f.out, {"sources_config": "configs/gen/principal_sources.json"})
    print(f"{len(m['files'])} files, {len(m['failed_fetches'])} failed")


if __name__ == "__main__":
    main()
