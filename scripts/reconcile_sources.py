"""Reconcile a data/raw/sources/<date>/ folder with its fetch_log.jsonl (BUILD_RULES rule 5).

Usage: python scripts/reconcile_sources.py CONFIG.json [--dry-run]

Checks every logged file's SHA-256 (last successful record), lists logged files missing on disk with the
folder they were re-fetched into (whose own log must hold them with HTTP 200), and lists every unlogged file
with its SHA-256, size and modification time together with the origin given in the config. The run fails if
an unlogged file has no origin, a hash differs, or a missing file was not re-fetched. Nothing in the raw
folder is changed; the result is a runlog run under results/d28_data/sources_reconciliation/.
"""

import argparse
import fnmatch
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ahg import runlog

ROOT = Path(__file__).resolve().parents[1]


def reconcile(folder: Path, explanations: dict[str, str], refetch: dict[str, str]) -> dict:
    log = [json.loads(line) for line in (folder / "fetch_log.jsonl").read_text().splitlines() if line.strip()]
    last = {}
    for r in log:
        last[r["name"]] = r  # the last record of a name describes the file on disk
    errors, verified, missing, unlogged = [], 0, [], []
    for name, r in sorted(last.items()):
        p = folder / name
        if not p.exists():
            dest = refetch.get(name)
            ok = False
            if dest:
                rl = [json.loads(x) for x in (ROOT / dest / "fetch_log.jsonl").read_text().splitlines() if x.strip()]
                hit = [x for x in rl if x["name"] == name and x.get("http") == 200 and x.get("sha256")]
                ok = bool(hit) and hashlib.sha256((ROOT / dest / name).read_bytes()).hexdigest() == hit[-1]["sha256"]
                missing.append({"name": name, "logged": r, "refetched_into": dest, "refetch_record": hit[-1] if hit else None})
            if not ok:
                errors.append(f"{name}: logged but missing, and not re-fetched")
            continue
        if not r.get("sha256") or hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            errors.append(f"{name}: SHA-256 differs from its last log record")
        else:
            verified += 1
    for p in sorted(q for q in folder.rglob("*") if q.is_file() and q.name != "fetch_log.jsonl"):
        name = p.relative_to(folder).as_posix()
        if name in last:
            continue
        why = next((v for k, v in explanations.items() if fnmatch.fnmatch(name, k)), None)
        if why is None:
            errors.append(f"{name}: unlogged and unexplained")
        st = p.stat()
        unlogged.append({"name": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": st.st_size,
                         "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                         "origin": why})  # fmt: skip
    return {"folder": str(folder.relative_to(ROOT)), "log_records": len(log), "logged_names": len(last),
            "verified": verified, "missing": missing, "unlogged": unlogged, "errors": errors}  # fmt: skip


def render(r: dict) -> str:
    lines = [f"# Reconciliation of {r['folder']}", "",
             f"Log records: {r['log_records']}; distinct names: {r['logged_names']}; hashes verified: {r['verified']}.", "",
             "## Logged but missing on disk", "", "| file | re-fetched into | HTTP | SHA-256 |", "|---|---|---|---|"]  # fmt: skip
    lines += [f"| {m['name']} | {m['refetched_into']} | {(m['refetch_record'] or {}).get('http')} | "
              f"{(m['refetch_record'] or {}).get('sha256')} |" for m in r["missing"]]  # fmt: skip
    lines += ["", "## On disk without a log record", "", "| file | bytes | mtime (UTC) | SHA-256 | origin |", "|---|---|---|---|---|"]
    lines += [f"| {u['name']} | {u['bytes']} | {u['mtime_utc']} | `{u['sha256']}` | {u['origin']} |" for u in r["unlogged"]]
    lines += ["", f"Errors: {r['errors'] or 'none'}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    r = reconcile(ROOT / cfg["folder"], cfg["explanations"], cfg.get("missing_refetch", {}))
    if a.dry_run or r["errors"]:
        print(render(r))
        sys.exit(1 if r["errors"] else 0)
    print(runlog.write_run("sources_reconciliation", r, render(r), [], cfg, root=ROOT / "results" / "d28_data"))


if __name__ == "__main__":
    main()
