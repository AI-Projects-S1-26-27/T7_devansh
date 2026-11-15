"""fetch_sources.py FILE.tsv [--date YYYY-MM-DD] [--txt-dir DIR] [--pace SECONDS]: fetch a batch of pages.

FILE.tsv has one page per line: name<TAB>url[<TAB>accept-header]; lines starting with "#" are skipped. Each
page is saved to data/raw/sources/<date>/<name> (<date> defaults to today, UTC) through ahg.rawfetch, which
appends {name, url, date_utc, http, bytes, sha256, content_type, script, git_commit, git_dirty} to that
folder's fetch_log.jsonl and never overwrites a file already on disk. With --txt-dir, a whitespace-normalised
text copy of each HTML page is written there too (a reading aid, not a raw input).
"""

import argparse
import html
import re
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from ahg.rawfetch import Fetcher


def html_text(data: bytes) -> str:
    t = data.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--txt-dir", type=Path)
    ap.add_argument("--pace", type=float, default=2.0)
    a = ap.parse_args()
    date.fromisoformat(a.date)
    f = Fetcher(ROOT / "data" / "raw" / "sources" / a.date, "scripts/fetch_sources.py", a.pace)
    for line in a.tsv.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, url, *accept = line.split("\t")
        rec = f.fetch(name, url, {"Accept": accept[0]} if accept else None)
        print(rec.get("http", rec.get("skipped")), rec.get("bytes", ""), name, rec.get("content_type", ""), flush=True)
        if a.txt_dir and rec.get("sha256") and "html" in rec.get("content_type", ""):
            a.txt_dir.mkdir(parents=True, exist_ok=True)
            (a.txt_dir / (Path(name).name + ".txt")).write_text(html_text((f.out / name).read_bytes()))


if __name__ == "__main__":
    main()
