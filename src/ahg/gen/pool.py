"""The D28 action pool: which (CVE, product) units may become actions (P6, decided 16 Nov 2026).

Built as a filter over the frozen effect labels `p3-labels-v2.1` (read through the SHA-256-checked
`ahg.effect.load_frozen_labels`; the frozen file is never rewritten) joined with the master dataset (also
SHA-256-checked). A unit enters the pool only if every rule holds:

- AC explicit and not conflicting (Round 5: never imputed; a conflict is REVIEW_REQUIRED for AC);
- the CVE's resolved vector is not REVIEW_REQUIRED (Round 3 CV1);
- remote (AV:N): PR:N, UI:N, RCE_CONFIRMED, role server (Rounds 1-2, E3, E4, E5);
- local (AV:L): PR:N or PR:L, UI:N, ADMIN_CONFIRMED (Round 2 "Local PR:L"; E4 holds for every course CVE;
  local units are not role-filtered, OQ-P4-1).

Every excluded unit is counted under the first rule it fails (the funnel). The pool keeps each CVE's full
NVD configurations, because matching is CPE-only (ahg.gen.cpe).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ahg import effect

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Unit:
    """One pool unit: a CVE on one product, with what the generator needs to match and cost it."""

    cve_id: str
    product: str  # master/label product: "vendor:product" (Linux) or the MSRC component name (Windows)
    family: str  # service family, or "os" for the kernel / Windows OS level
    os: str  # "linux" | "windows"
    kind: str  # "remote" | "local"
    ac: str  # "L" | "H"
    cohort: str  # "core" | "temporal"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_master(path: Path, sha256: str) -> dict[str, dict]:
    """Master rows by CVE ID; refuses a file whose SHA-256 is not the pinned one."""
    got = sha256_file(path)
    if got != sha256:
        raise ValueError(f"{path}: SHA-256 {got} does not match the pinned {sha256}")
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    return {r["cve_id"]: r for r in rows}


def exclusion(row: dict, master_row: dict) -> str | None:
    """The first pool rule a label row fails, as a reason string; None if it enters the pool."""
    if row["ac_status"] != "explicit":
        return f"ac_{row['ac_status']}"
    vr = master_row["vector_resolution"]
    if vr["status"] == "REVIEW_REQUIRED":
        return "vector_review_required"
    v = vr["values"]
    if row["label_set"] == "remote":
        if v.get("AV") != "N":
            return f"remote_av_{v.get('AV')}"
        if v.get("PR") != "N":
            return f"remote_pr_{v.get('PR')}"
        if v.get("UI") != "N":
            return f"remote_ui_{v.get('UI')}"
        if row["label"] != "RCE_CONFIRMED":
            return f"remote_label_{row['label']}"
        if row["role"] != "server":
            return f"remote_role_{row['role']}"
        return None
    if row["label_set"] == "local":
        if v.get("AV") != "L":
            return f"local_av_{v.get('AV')}"
        if v.get("PR") not in ("N", "L"):
            return f"local_pr_{v.get('PR')}"
        if v.get("UI") != "N":
            return f"local_ui_{v.get('UI')}"
        if row["label"] != "ADMIN_CONFIRMED":
            return f"local_label_{row['label']}"
        return None
    return f"label_set_{row['label_set']}"


def build_pool(labels: list[dict], master: dict[str, dict]) -> tuple[list[Unit], dict]:
    """(pool units sorted by CVE then product, funnel). The funnel counts every label row once."""
    units, funnel = [], Counter()
    for row in labels:
        why = exclusion(row, master[row["cve_id"]])
        if why is not None:
            funnel[f"{row['label_set']}:{why}" if not why.startswith(row["label_set"]) else why] += 1
            continue
        units.append(Unit(row["cve_id"], row["product"], row["family"], row["os"], row["label_set"], row["ac"],
                          row["cohort"]))
    units.sort(key=lambda u: (cve_key(u.cve_id), u.product))
    if len({(u.cve_id, u.product) for u in units}) != len(units):
        raise ValueError("a (CVE, product) unit appears twice in the frozen labels")
    kinds = Counter(u.kind for u in units)
    return units, {"label_rows": len(labels), "pool": len(units), "remote": kinds["remote"], "local": kinds["local"],
                   "excluded": dict(sorted(funnel.items()))}


def load_pool(cfg: dict, root: Path = ROOT) -> tuple[list[Unit], dict, dict[str, dict]]:
    """(pool, funnel, master) from the generator config's pinned inputs."""
    master = load_master(root / cfg["master"]["path"], cfg["master"]["sha256"])
    labels = effect.load_frozen_labels(cfg["labels"], root=root)
    units, funnel = build_pool(labels, master)
    return units, funnel, master


def cve_key(cve_id: str) -> tuple[int, int]:
    """Numeric CVE order (Round 5, OQ-CA2-1): year, then sequence number, as integers."""
    _, year, seq = cve_id.split("-")
    return int(year), int(seq)


def touches(master_row: dict, products: set[str]) -> bool:
    """Does a CVE touch any of `products` (held-out product split, Round 10)? Any master unit of the
    product, or any CPE vendor:product NVD lists as affected, counts."""
    if any(u["product"] in products for u in master_row["units"]):
        return True
    return any(p.split(":", 1)[1] in products for p in master_row["affected_products"])


def check_cve_ids(cves: frozenset[str], master: dict[str, dict]) -> None:
    """Excluded CVEs must be canonical master IDs (CV9: one master row per canonical CVE ID), so a typo or an
    alias cannot silently exclude nothing."""
    bad = sorted(c for c in cves if c not in master)
    if bad:
        raise ValueError(f"excluded CVEs not in the master dataset (canonical IDs only): {bad[:5]}")


def restrict(units: list[Unit], master: dict[str, dict], cohort: str, excluded: frozenset[str],
             excluded_cves: frozenset[str] = frozenset()) -> list[Unit]:
    """The units a generator run may use: its cohort only, no CVE touching an excluded product, and no
    excluded CVE (the unseen-CVE split, Round 10; §18 17 Nov 2026). Units are keyed by canonical CVE ID, so
    excluding a CVE removes every product unit of it."""
    if cohort not in ("core", "temporal"):
        raise ValueError(f"cohort must be 'core' or 'temporal', not {cohort!r}")
    check_cve_ids(excluded_cves, master)
    return [u for u in units if u.cohort == cohort and u.cve_id not in excluded_cves
            and not (excluded and touches(master[u.cve_id], set(excluded)))]


def cve_set_sha256(cves) -> str:
    """SHA-256 of a CVE set: its IDs in numeric order, one per line, each line ending in a newline."""
    return hashlib.sha256("".join(f"{c}\n" for c in sorted(cves, key=cve_key)).encode()).hexdigest()


def cve_holdout(units: list[Unit], share: float, seed: str) -> tuple[list[str], dict]:
    """The unseen-CVE holdout (Round 10; decisions.md §18 item 6, 17 Nov 2026): every CVE with an AC:H pool
    unit, plus AC:L-only CVEs drawn at random until the holdout is `share` of the pool's CVE IDs.

    Grouped by canonical CVE ID: a CVE is one item whatever its number of product units, and a CVE with any
    AC:H unit counts as AC:H. The target size is round_half_up(share * IDs) (halves up, as OQ-P6-1). The
    AC:L draw is random.Random(seed).sample over the AC:L-only IDs in numeric order, so it depends on the
    seed and the pool only. Raises if the AC:H CVEs alone exceed the target (the rule cannot hold).
    Returns (held-out IDs in numeric order, counts)."""
    import math
    import random

    acs: dict[str, set[str]] = {}
    for u in units:
        acs.setdefault(u.cve_id, set()).add(u.ac)
    ids = sorted(acs, key=cve_key)
    high = [c for c in ids if "H" in acs[c]]
    low = [c for c in ids if acs[c] == {"L"}]
    target = math.floor(share * len(ids) + 0.5)
    if len(high) > target:
        raise ValueError(f"{len(high)} AC:H CVEs exceed the {share:.0%} target of {target} of {len(ids)} CVE IDs")
    drawn = random.Random(seed).sample(low, target - len(high))
    held = sorted(set(high) | set(drawn), key=cve_key)
    return held, {"pool_cve_ids": len(ids), "target": target, "ac_h_cves": len(high),
                  "ac_h_cves_with_an_ac_l_unit": sum(acs[c] == {"H", "L"} for c in high),
                  "ac_l_only_cves": len(low), "ac_l_drawn": len(drawn)}
