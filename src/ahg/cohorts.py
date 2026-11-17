"""D28 cohort registry, cohort builds and the D28 test-access gate (P7; decisions Rounds 9, 10, 11, 18).

A cohort is a seeded set of generated networks (one task each) with one parameter set: archetype, sizes,
generator cohort (core / temporal), OS mix, p and product exclusions. configs/d28_cohorts.json lists them.

Seeds. Cohort c's network k at size n has the generator seed "<namespace>/n<n>/<k>". Namespaces are unique
and contain no "/", so two cohorts never share a seed string, hence never share an rng stream (the
generator seeds "archetype:seed"). They are new strings, so they cannot collide with the integer seeds of the
old study (Round 10).

Selection.
- training and validation ("solvable_quota", Round 9; validation decided 17 Nov 2026, OQ-P7-1): candidates
  k = 0, 1, ... per size, until the size's quota of solvable networks is reached or the generation limit is
  hit; every candidate is logged with its seed, archetype, size, OS mix, p, solvability and reason. A
  shortfall is reported, never reallocated to another size.
- every test cohort ("fixed", Round 9): exactly per_size networks k = 0 .. per_size-1 per size, with
  no solvability filtering; unsolvable ones stay in and are counted with their reason (Round 7). A cell the
  inventory cannot realise (generator InfeasibleCell) stays in as a failed member, also counted.

Test cohorts are built and read only with a token from record_test_read, which appends to
results/test_access_log.jsonl (Round 10). A second evaluation read of a test cohort is refused unless a named
D-entry permits it. Nothing in this module opens a test cohort without such a token.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .splits import EVALUATION, LABEL_GENERATION, TEST_ACCESS_LOG, TEST_PURPOSES

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "d28_cohorts.json"
PREREG = ROOT / "configs" / "d28_prereg.json"
TASK_STORE = ROOT / "data" / "d28" / "tasks"

ROLES = ("training", "validation", "test")
ANALYSES = (None, "primary", "exploratory", "sensitivity", "proposed")
SELECTIONS = ("solvable_quota", "fixed")
EXCLUDES = ("heldout_products", "none")
REQUIREMENTS = ("n4_pass",)
GEN_COHORTS = ("core", "temporal")
TRAINING_MAX_N = 20  # Round 9: training networks have n <= 20
SMOKE_PREFIX = "p7-smoke"
TEST_LOG_SCHEMA = "d28-test-access/1"
D_ENTRY = re.compile(r"^D\d+\b")  # a named D-entry, e.g. "D29 §2"


class BlockedCohort(RuntimeError):
    """The cohort cannot be built yet: a decision it depends on is open (open_questions.md)."""


class TestAccessError(PermissionError):
    """A test cohort was touched without a valid test-access token, or a second read was refused."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class Cohort:
    name: str
    role: str
    analysis: str | None
    hypothesis: str | None
    namespace: str
    archetype: str
    sizes: tuple[int, ...]
    gen_cohort: str
    os_linux: float
    p: float
    exclude: str
    selection: str
    per_size: int | None  # null: not set yet (test cohorts until the P6 pilot rates, decided 17 Nov 2026)
    max_candidates_per_size: int | None
    requires: tuple[str, ...]
    blocked_by: tuple[str, ...]
    smoke: bool = False

    def seed(self, n: int, k: int) -> str:
        if n not in self.sizes:
            raise ValueError(f"{self.name} has no size {n}")
        return f"{self.namespace}/n{n}/{k}"

    @property
    def os_mix(self) -> dict[str, float]:
        return {"linux": self.os_linux, "windows": 1 - self.os_linux}

    @property
    def is_test(self) -> bool:
        return self.role == "test"

    def factors(self) -> dict:
        """The parameters a one-factor-at-a-time comparison looks at (Round 10)."""
        return {"archetype": self.archetype, "sizes": self.sizes, "gen_cohort": self.gen_cohort,
                "os_linux": self.os_linux, "p": self.p, "exclude": self.exclude}


@dataclass(frozen=True)
class Registry:
    cohorts: dict[str, Cohort]
    frozen: bool
    heldout_products: tuple[str, ...] | None
    training_total_solvable: int
    sha256: str
    heldout_cves: tuple[str, ...] | None = None
    heldout_cve_roles: tuple[str, ...] = ()

    def __getitem__(self, name: str) -> Cohort:
        return self.cohorts[name]

    @property
    def training(self) -> Cohort:
        return next(c for c in self.cohorts.values() if c.role == "training" and not c.smoke)

    def excluded(self, c: Cohort) -> frozenset[str]:
        """Products whose CVEs the generator must leave out for cohort c (Rounds 10, 11)."""
        if c.exclude == "none":
            return frozenset()
        if self.heldout_products is None:
            raise BlockedCohort(f"{c.name}: the held-out product set is not chosen (OQ-P6-8)")
        return frozenset(self.heldout_products)

    def problems(self, c: Cohort, prereg: Path = PREREG) -> list[str]:
        """Why cohort c cannot be built now; [] if it can. Smoke cohorts need no freeze."""
        out = [f"blocked by {b}" for b in c.blocked_by]
        if c.smoke:
            return out
        if not self.frozen:
            out.append("the cohort registry is not frozen (configs/d28_cohorts.json 'frozen')")
        if c.per_size is None:
            out.append("the cohort size is not set (OQ-P7-1: after the P6 pilot solvability rates)")
        if c.selection == "solvable_quota" and c.max_candidates_per_size is None:
            out.append("the generation limit is not set (OQ-P7-2: after the P6 pilot solvability rates)")
        if c.role in self.heldout_cve_roles:
            # OQ-P7-4 (decided 17 Nov 2026): training and validation exclude the held-out CVEs, which reach the
            # generator through its "excluded CVEs" parameter; that parameter is being added on d28-gen.
            out.append("the held-out CVEs cannot be excluded yet: the generator's excluded-CVEs parameter "
                       "(d28-gen) is not on this branch")
        if c.exclude == "heldout_products" and self.heldout_products is None:
            out.append("the held-out product set is not chosen (OQ-P6-8)")
        if not prereg_frozen(prereg):
            out.append("hypotheses are not frozen: configs/d28_prereg.json is not committed (D28 §13.1)")
        return out


def _cohort(d: dict, smoke: bool) -> Cohort:
    keys = {"name", "role", "analysis", "hypothesis", "namespace", "archetype", "sizes", "gen_cohort", "os_linux",
            "p", "exclude", "selection", "per_size", "max_candidates_per_size", "requires", "blocked_by"}
    if set(d) != keys:
        raise ValueError(f"cohort {d.get('name')}: keys {sorted(set(d) ^ keys)} missing or unknown")
    return Cohort(**{**d, "sizes": tuple(d["sizes"]), "requires": tuple(d["requires"]),
                     "blocked_by": tuple(d["blocked_by"])}, smoke=smoke)


def validate(cohorts: list[Cohort], total: int, p_values: list[float]) -> None:
    """Structural rules of the registry; raises ValueError on the first violation."""
    names = [c.name for c in cohorts]
    if len(set(names)) != len(names):
        raise ValueError("cohort names must be unique")
    spaces = [c.namespace for c in cohorts]
    if len(set(spaces)) != len(spaces):
        raise ValueError("seed namespaces must be unique (disjoint seeds)")
    for c in cohorts:
        if "/" in c.namespace or not c.namespace:
            raise ValueError(f"{c.name}: a namespace may not contain '/'")
        if c.smoke != c.namespace.startswith(SMOKE_PREFIX):
            raise ValueError(f"{c.name}: smoke cohorts, and only they, live under the {SMOKE_PREFIX!r} namespace")
        if not c.smoke and not c.name.startswith("d28_"):
            raise ValueError(f"{c.name}: D28 cohort names start with 'd28_' (OQ-P0-11)")
        if c.role not in ROLES or c.analysis not in ANALYSES or c.selection not in SELECTIONS \
                or c.exclude not in EXCLUDES or c.gen_cohort not in GEN_COHORTS:
            raise ValueError(f"{c.name}: unknown role, analysis, selection, exclude or generator cohort")
        if (c.role == "test") != (c.analysis is not None):
            raise ValueError(f"{c.name}: test cohorts, and only they, have an analysis class")
        if set(c.requires) - set(REQUIREMENTS):
            raise ValueError(f"{c.name}: unknown requirement {sorted(set(c.requires) - set(REQUIREMENTS))}")
        if c.p not in p_values or not 0 <= c.os_linux <= 1:
            raise ValueError(f"{c.name}: p {c.p} or OS mix {c.os_linux} outside the generator's values")
        if not c.sizes or len(set(c.sizes)) != len(c.sizes):
            raise ValueError(f"{c.name}: sizes are required")
        # A test cohort's size may be null until set (OQ-P7-1); problems() then refuses to build it.
        if not (c.per_size is None and c.role == "test") and (not isinstance(c.per_size, int) or c.per_size < 1):
            raise ValueError(f"{c.name}: a positive per_size is required")
        if c.role in ("training", "validation"):
            # Round 9 (training) and OQ-P7-1 (validation, decided 17 Nov 2026): a quota of solvable networks per
            # size, n <= 20. The generation limit may be null until set (OQ-P7-2); problems() then refuses.
            if c.selection != "solvable_quota" or max(c.sizes) > TRAINING_MAX_N:
                raise ValueError(f"{c.name}: {c.role} selects solvable networks with n <= {TRAINING_MAX_N}")
            lim = c.max_candidates_per_size
            if lim is not None and (not isinstance(lim, int) or lim < c.per_size):
                raise ValueError(f"{c.name}: {c.role} needs a generation limit >= its quota")
            if c.role == "training" and not c.smoke and c.per_size * len(c.sizes) != total:
                raise ValueError(f"{c.name}: {len(c.sizes)} equal quotas of {c.per_size} != {total} networks")
        elif c.selection != "fixed" or c.max_candidates_per_size is not None:
            raise ValueError(f"{c.name}: test cohorts are fixed-size, with no solvability filtering")
    real = [c for c in cohorts if not c.smoke]
    for role in ("training", "validation"):
        if sum(c.role == role for c in real) != 1:
            raise ValueError(f"exactly one {role} cohort is required")


def load_registry(path: Path = REGISTRY, gen_config: Path | None = None) -> Registry:
    raw = path.read_bytes()
    cfg = json.loads(raw)
    from .gen.generate import CONFIG as GEN_CONFIG  # the generator's own p values (read-only use of its config)
    p_values = json.loads((gen_config or GEN_CONFIG).read_text())["p_values"]
    cohorts = [_cohort(d, False) for d in cfg["cohorts"]] + [_cohort(d, True) for d in cfg.get("smoke", [])]
    validate(cohorts, cfg["training_total_solvable"], p_values)
    held = cfg["heldout_products"]
    cves = cfg["heldout_cves"]
    if set(cves["excluded_from_roles"]) - set(ROLES):
        raise ValueError(f"heldout_cves: unknown roles {cves['excluded_from_roles']}")
    return Registry({c.name: c for c in cohorts}, bool(cfg["frozen"]), None if held is None else tuple(held),
                    cfg["training_total_solvable"], hashlib.sha256(raw).hexdigest(),
                    None if cves["ids"] is None else tuple(cves["ids"]), tuple(cves["excluded_from_roles"]))


def prereg_frozen(path: Path = PREREG) -> bool:
    """True once configs/d28_prereg.json is committed (D28 §13.1: the freeze is that commit)."""
    if not path.exists():
        return False
    r = subprocess.run(["git", "ls-files", "--error-unmatch", path.name], cwd=path.parent, capture_output=True)
    return r.returncode == 0


# ------------------------------------------------------------------ the content-addressed task store

def store_doc(doc: dict, store: Path = TASK_STORE) -> str:
    """Write a task document under its SHA-256 (canonical JSON) and return the digest. Idempotent."""
    from .gen.generate import doc_digest, dumps
    digest = doc_digest(doc)
    path = store / digest[:2] / f"{digest}.json.gz"
    if path.exists():
        load_doc(digest, store)  # verifies the stored bytes
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(gzip.compress(dumps(doc).encode(), mtime=0))
    tmp.replace(path)  # atomic: a crash never leaves a half-written task
    return digest


def load_doc(digest: str, store: Path = TASK_STORE) -> dict:
    from .gen.generate import doc_digest
    doc = json.loads(gzip.decompress((store / digest[:2] / f"{digest}.json.gz").read_bytes()))
    if doc_digest(doc) != digest:
        raise ValueError(f"stored task {digest} does not hash to its name")
    return doc


# ------------------------------------------------------------------ the D28 test-access gate

def _log_entries(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def record_test_read(registry: Registry, split: str, purpose: str, commit: str, *, dataset_sha256: str | None,
                     eval_config_sha256: str | None = None, checkpoint_sha256: dict | None = None,
                     permitted_by: str | None = None, log: Path = TEST_ACCESS_LOG) -> dict:
    """Append one access to a D28 test cohort and return it as the token the cohort functions require.

    purpose "label generation": building the cohort and labelling it; reads no model output, may repeat
    (sharded, resumable runs), each run logged. purpose "evaluation": the single frozen read (Round 10); it
    must name the dataset, the evaluation config and every checkpoint, and a second one is refused unless
    permitted_by names the D-entry that allows it."""
    c = registry.cohorts.get(split)
    if c is None or not c.is_test:
        raise TestAccessError(f"{split!r} is not a test cohort of the registry")
    if purpose not in TEST_PURPOSES:
        raise TestAccessError(f"purpose must be one of {TEST_PURPOSES}, not {purpose!r}")
    if permitted_by is not None and not D_ENTRY.match(permitted_by):
        raise TestAccessError(f"permitted_by must name a D-entry (e.g. 'D29'), not {permitted_by!r}")
    if purpose == EVALUATION:
        if not (dataset_sha256 and eval_config_sha256 and checkpoint_sha256):
            raise TestAccessError("an evaluation read records the dataset, evaluation config and checkpoint hashes")
        past = [e for e in _log_entries(log) if e.get("schema") == TEST_LOG_SCHEMA and e["split"] == split
                and e["purpose"] == EVALUATION]
        if past and permitted_by is None:
            raise TestAccessError(f"test cohort {split} was already read on {past[0]['date_utc']}; "
                                  "a second read needs a named D-entry (permitted_by)")
    entry = {"schema": TEST_LOG_SCHEMA, "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "split": split, "purpose": purpose, "dataset_sha256": dataset_sha256,
             "eval_config_sha256": eval_config_sha256, "checkpoint_sha256": checkpoint_sha256,
             "registry_sha256": registry.sha256, "permitted_by": permitted_by, "git_commit": commit,
             "argv": sys.argv}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def check_access(c: Cohort, access: dict | None, purposes: tuple[str, ...], log: Path = TEST_ACCESS_LOG) -> None:
    """Refuse to touch test cohort c unless `access` is a logged token for it with an allowed purpose."""
    if not c.is_test:
        return
    if access is None:
        raise TestAccessError(f"{c.name} is a test cohort: it is opened only through record_test_read")
    if access.get("split") != c.name or access.get("purpose") not in purposes:
        raise TestAccessError(f"the token is for {access.get('split')}/{access.get('purpose')}, "
                              f"not {c.name}/{purposes}")
    if access not in _log_entries(log):
        raise TestAccessError("the token is not in the test-access log")


# ------------------------------------------------------------------ building a cohort

def _candidate(ctx, c: Cohort, n: int, k: int, excluded: frozenset[str], store: Path) -> dict:
    from .gen import generate as gen
    from .gen.hosts import InfeasibleCell
    row = {"seed": c.seed(n, k), "index": k, "archetype": c.archetype, "n": n, "gen_cohort": c.gen_cohort,
           "os_linux": c.os_linux, "p": c.p}
    try:
        doc = gen.generate(ctx, c.archetype, row["seed"], n, cohort=c.gen_cohort, os_mix=c.os_mix, p=c.p,
                           excluded=excluded)
    except InfeasibleCell as e:  # kept and counted with its reason, never silently dropped (Round 7)
        return row | {"digest": None, "solvable": False, "reason": f"infeasible_cell: {e}"}
    return row | {"digest": store_doc(doc, store), "solvable": doc["solvable"], "reason": doc["unsolvable_reason"],
                  "unverified_release": doc["flags"]["unverified_release"]}


def build(ctx, registry: Registry, name: str, *, store: Path = TASK_STORE, access: dict | None = None,
          n4_pass: set[int] | None = None, log: Path = TEST_ACCESS_LOG, prereg: Path = PREREG) -> dict:
    """Build cohort `name`; return its record: every candidate, the members (task digests) and counts.

    access: a token from record_test_read (test cohorts only). n4_pass: the sizes at which N4 passed, for
    cohorts that require it (the structure split runs only there, Round 6)."""
    c = registry[name]
    why = registry.problems(c, prereg)
    if why:
        raise BlockedCohort(f"{name}: " + "; ".join(why))
    check_access(c, access, (LABEL_GENERATION, EVALUATION), log)
    sizes = list(c.sizes)
    not_run = []
    if "n4_pass" in c.requires:
        if n4_pass is None:
            raise BlockedCohort(f"{name} runs only at sizes where N4 passed; pass n4_pass")
        not_run = [n for n in sizes if n not in n4_pass]
        sizes = [n for n in sizes if n in n4_pass]
    excluded = registry.excluded(c)
    candidates, members, shortfall = [], [], {}
    for n in sizes:
        if c.selection == "fixed":
            rows = [_candidate(ctx, c, n, k, excluded, store) for k in range(c.per_size)]
            candidates += rows
            members += rows
            continue
        got = 0
        for k in range(c.max_candidates_per_size):
            row = _candidate(ctx, c, n, k, excluded, store)
            candidates.append(row)
            if row["solvable"]:
                members.append(row)
                got += 1
                if got == c.per_size:
                    break
        shortfall[n] = c.per_size - got  # reported, never reallocated (Round 9)
    digests = [m["digest"] for m in members]
    by_size = {}
    for n in sizes:
        rows = [r for r in candidates if r["n"] == n]
        mem = [r for r in members if r["n"] == n]
        reasons: dict[str, int] = {}
        for r in mem:
            if not r["solvable"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        by_size[str(n)] = {"candidates": len(rows), "solvable_candidates": sum(r["solvable"] for r in rows),
                           "members": len(mem), "solvable_members": sum(r["solvable"] for r in mem),
                           "unsolvable_members_by_reason": dict(sorted(reasons.items()))}
    return {"cohort": name, "spec": {**c.__dict__, "sizes": list(c.sizes), "requires": list(c.requires),
                                     "blocked_by": list(c.blocked_by)},
            "registry_sha256": registry.sha256, "inputs": ctx.input_hashes, "excluded": sorted(excluded),
            "sizes_run": sizes, "sizes_not_run_n4": not_run, "candidates": candidates,
            "members": [{"digest": m["digest"], "seed": m["seed"], "n": m["n"], "solvable": m["solvable"],
                         "reason": m["reason"]} for m in members],
            "dataset_sha256": hashlib.sha256(json.dumps(digests).encode()).hexdigest(),
            "shortfall": shortfall, "by_size": by_size}


def open_members(record: dict, registry: Registry, *, store: Path = TASK_STORE, access: dict | None = None,
                 purposes: tuple[str, ...] = (LABEL_GENERATION, EVALUATION),
                 log: Path = TEST_ACCESS_LOG) -> list[tuple[dict, dict | None]]:
    """(member row, task document or None for a failed cell) for every member of a built cohort."""
    c = registry[record["cohort"]]
    check_access(c, access, purposes, log)
    if access is not None and access.get("dataset_sha256") not in (None, record["dataset_sha256"]):
        raise TestAccessError("the token names a different dataset than this cohort record")
    return [(m, None if m["digest"] is None else load_doc(m["digest"], store)) for m in record["members"]]
