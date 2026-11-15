"""Fetch raw inputs into data/raw/<source>/<YYYY-MM-DD>/ with a per-file manifest (BUILD_RULES rule 5).

Every fetch appends one record to <dir>/fetch_log.jsonl: name, URL, UTC time, HTTP status, SHA-256,
size, content type, and the fetching script's git commit and dirty flag. A file already on disk is never
overwritten (raw data is append-only, CLAUDE.md §1 rule 11), so an interrupted batch can be resumed.
manifest.json is written last, from the log and the bytes on disk, and fails if they disagree.

Requests are paced per host, because NVD (no API key: 5 requests / 30 s) and MSRC are shared services.
"""

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ahg import runlog

USER_AGENT = "ahg-raw-fetch/1.0"  # generic client name; no personal details (privacy rule)
LOG_NAME, MANIFEST_NAME = "fetch_log.jsonl", "manifest.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Fetcher:
    """Paced HTTP GETs into one snapshot directory, each logged with its provenance."""

    def __init__(self, out_dir: Path, script: str, pace: float, retry_wait: float = 35.0, timeout: float = 180.0):
        self.out, self.script, self.pace, self.retry_wait, self.timeout = Path(out_dir), script, pace, retry_wait, timeout
        if (self.out / MANIFEST_NAME).exists():
            raise FileExistsError(f"{self.out}/{MANIFEST_NAME} exists: snapshot is final, refusing to modify it")
        self.out.mkdir(parents=True, exist_ok=True)
        self.commit, self.dirty = runlog._git("rev-parse", "HEAD"), runlog.git_dirty()
        self._last: dict[str, float] = {}

    def _wait(self, host: str, seconds: float) -> None:
        wait = seconds - (time.time() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def _get(self, url: str, headers: dict[str, str]) -> tuple[int, bytes, str]:
        """One GET with a single delayed retry on network errors and 5xx/429. Returns (status, body, type)."""
        host = urllib.parse.urlsplit(url).netloc
        for attempt in (1, 2):
            self._wait(host, self.pace if attempt == 1 else self.retry_wait)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.status, r.read(), r.headers.get("Content-Type", "")
            except urllib.error.HTTPError as e:
                body, ctype = e.read(), e.headers.get("Content-Type", "")
                if attempt == 2 or not (e.code == 429 or e.code >= 500):
                    return e.code, body, ctype
            except OSError as e:  # connection reset, TLS failure, timeout, truncated read
                if attempt == 2:
                    return 0, repr(e).encode(), ""
        raise AssertionError("unreachable")

    def fetch(self, name: str, url: str, headers: dict[str, str] | None = None) -> dict:
        """Fetch url into <out>/<name> unless that file exists; return its log record.

        Only a 200 response is written under `name`. Any other outcome is logged (with bytes 0 and no
        hash) and nothing is written, so a failed fetch can never pass for a raw input.
        """
        dest = self.out / name
        if dest.exists():
            return {"name": name, "url": url, "skipped": "exists", "sha256": sha256(dest.read_bytes())}
        status, body, ctype = self._get(url, headers or {})
        ok = status == 200
        rec = {
            "name": name, "url": url, "date_utc": utc_now(), "http": status, "content_type": ctype,
            "bytes": len(body) if ok else 0, "sha256": sha256(body) if ok else None,
            "script": self.script, "git_commit": self.commit, "git_dirty": self.dirty,
        }  # fmt: skip
        if not ok:
            rec["error"] = body[:200].decode("utf-8", "replace")
        if ok:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(body)
            tmp.rename(dest)  # never leave a half-written file under the final name
        with (self.out / LOG_NAME).open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec


def write_manifest(out_dir: Path, extra: dict | None = None) -> dict:
    """Write manifest.json for a finished snapshot: every file on disk with its last successful log record.

    Fails if a file on disk has no successful log record, or its bytes differ from the logged SHA-256.
    """
    out = Path(out_dir)
    log = [json.loads(line) for line in (out / LOG_NAME).read_text().splitlines() if line.strip()]
    last_ok = {r["name"]: r for r in log if r.get("sha256")}
    files = []
    for p in sorted(q for q in out.rglob("*") if q.is_file() and q.name not in (LOG_NAME, MANIFEST_NAME)):
        name = p.relative_to(out).as_posix()
        rec = last_ok.get(name)
        if rec is None:
            raise ValueError(f"{name}: on disk but no successful fetch record")
        if sha256(p.read_bytes()) != rec["sha256"]:
            raise ValueError(f"{name}: bytes differ from the logged SHA-256")
        files.append({k: rec[k] for k in ("name", "url", "date_utc", "http", "bytes", "sha256", "content_type",
                                          "script", "git_commit", "git_dirty")})  # fmt: skip
    manifest = {"written_utc": utc_now(), "files": files, "failed_fetches": [r for r in log if not r.get("sha256")],
                "fetch_log_sha256": sha256((out / LOG_NAME).read_bytes()), **(extra or {})}  # fmt: skip
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest
