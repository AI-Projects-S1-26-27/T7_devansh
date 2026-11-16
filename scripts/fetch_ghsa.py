"""Fetch the GitHub security advisory pages (GHSA) referenced by NVD records into data/raw/ghsa/<date>/.

Usage: python scripts/fetch_ghsa.py SNAPSHOT_DIR [SNAPSHOT_DIR ...] [--date YYYY-MM-DD] [--manifest]

Every reference URL of the form https://github.com/<owner>/<repo>/security/advisories/GHSA-xxxx-xxxx-xxxx
in a CVE published from 2016-01-01 on (the start of the D28 core cohort) is fetched once, as
<GHSA-id>__<owner>__<repo>.html; a global-database reference (https://github.com/advisories/GHSA-...) is
fetched as <GHSA-id>__global.html. These pages carry the maintainers' own statements of which component
is affected (e.g. "clients only", D28 Round 1 E5), which the CVE description often omits. The pages are
copyrighted: under D28 E8 only their manifest entries are archived; the raw copies stay private.
--manifest writes manifest.json after the batch, which makes the folder final.
"""

import argparse
import re
from datetime import date, datetime, timezone
from pathlib import Path

from ahg.nvd_snapshot import load_pages
from ahg.rawfetch import Fetcher, write_manifest

ROOT = Path(__file__).resolve().parents[1]
_ID = r"(GHSA(?:-[23456789cfghjmpqrvwx]{4}){3})"
GHSA_URL = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+)/security/advisories/" + _ID + r"/?$")
GHSA_GLOBAL = re.compile(r"^https://github\.com/advisories/" + _ID + r"/?$")
PACE = 3.0


def ghsa_refs(pages: list[dict], since: str = "2016-01-01") -> dict[str, str]:
    """{file name: canonical advisory URL} for every GHSA advisory referenced by a CVE published on/after `since`."""
    out = {}
    for p in pages:
        for v in p["response"]["vulnerabilities"]:
            cve = v["cve"]
            if cve["published"][:10] < since:
                continue
            for ref in cve.get("references", []):
                m = GHSA_URL.match(ref["url"])
                g = GHSA_GLOBAL.match(ref["url"])
                if m:
                    owner, repo, gid = m.groups()
                    out[f"{gid}__{owner}__{repo}.html"] = f"https://github.com/{owner}/{repo}/security/advisories/{gid}"
                elif g:
                    out[f"{g.group(1)}__global.html"] = f"https://github.com/advisories/{g.group(1)}"
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshots", nargs="+", type=Path)
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    date.fromisoformat(a.date)
    refs = {}
    for s in a.snapshots:
        refs.update(ghsa_refs(load_pages(s)))
    f = Fetcher(ROOT / "data" / "raw" / "ghsa" / a.date, "scripts/fetch_ghsa.py", PACE)
    for name, url in refs.items():
        r = f.fetch(name, url)
        print(r.get("http", r.get("skipped")), r.get("bytes", ""), name, flush=True)
    if a.manifest:
        m = write_manifest(f.out, {"snapshots": [str(s) for s in a.snapshots], "advisories": len(refs)})
        print(f"manifest: {len(m['files'])} files, {len(m['failed_fetches'])} failed fetches")


if __name__ == "__main__":
    main()
