"""Fetch the candidate documentation pages for the runtime-principal table (D28 P6) into
data/raw/principals/<date>/ with a manifest (BUILD_RULES rule 5). With --source, the same paced fetch writes
another cited source's pages into data/raw/<source>/<date>/ (e.g. the NASim firewall rule, OQ-P6-18).

Usage: python scripts/fetch_principals.py [--date YYYY-MM-DD] [--sources configs/gen/principal_sources.json]
                                          [--source principals]

Pages come from the sources config (one snapshot per config and date) and are fetched one at a time, paced, by
ahg.rawfetch (generic client name, per-host pacing, append-only). The table itself
(configs/gen/principals.json) cites only what these pages state verbatim; anything else is UNKNOWN.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ahg.rawfetch import Fetcher, write_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "configs" / "gen" / "principal_sources.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--sources", type=Path, default=SOURCES)
    ap.add_argument("--source", default="principals", help="raw source folder name under data/raw/")
    a = ap.parse_args()
    cfg = json.loads(a.sources.read_text())
    if not re.fullmatch(r"[a-z0-9_]+", a.source):
        raise SystemExit(f"--source must be a plain folder name, not {a.source!r}")
    f = Fetcher(ROOT / "data" / "raw" / a.source / a.date, "scripts/fetch_principals.py", cfg["pace_seconds"])
    for page in cfg["pages"]:  # sequential: one request at a time, paced per host by the Fetcher
        rec = f.fetch(page["name"], page["url"])
        print(page["name"], rec.get("http", rec.get("skipped")))
    m = write_manifest(f.out, {"sources_config": a.sources.resolve().relative_to(ROOT).as_posix()})
    print(f"{len(m['files'])} files, {len(m['failed_fetches'])} failed")


if __name__ == "__main__":
    main()
