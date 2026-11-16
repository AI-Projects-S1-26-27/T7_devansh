"""D28 exact reference labels for generated tasks (P7; decisions Round 9, Round 5, Round 8 C2).

One label record per (task document, cost regime). The labeller is labels.label_instance: the five offline
sources, A*+LM-cut per state under the per-state expansion budget, every label checked h_max, LM-cut <= h*,
against exhaustive enumeration when it fits, and against the independent oracle; a mismatch raises
HardFailure and aborts the run (CLAUDE.md rule 6), it is never recorded as an outcome.

Outcomes, recorded separately (Round 9):
- LABELLED: h*(s0) is certified; per-state budget hits are listed as BUDGET_EXHAUSTED states, with no label.
- BUDGET_EXHAUSTED: A*+LM-cut from s0 hit the budget; the task has no label, and its optimum is uncertified.
- DEAD_END: the task is unsolvable (exact reachability fixpoint, h_max(s0) = infinity); s0 is a proven dead
  end, excluded from regression labels. In a solvable delete-free task no reachable state is a dead end
  (every state reached from s0 contains s0's facts), so dead ends occur only at s0 of unsolvable tasks.
- SOLVER_ERROR: the labeller raised something other than HardFailure (e.g. MemoryError); type and message
  are kept, the task has no label.

The per-state budget is one frozen number (configs/labels/d28.json "state_budget", set from the training-only
budget pilot); s0 gets the same budget as every other state (OQ-P7-7). Storage is content-addressed: a
record's key is the SHA-256 of (labeller version, task digest, regime, label config), its file
<root>/<regime>/<key[:2]>/<key>.json.gz. A run resumes by skipping keys already on disk (after checking them).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from .cost import check_regime
from .heuristics import INF, hmax
from .labels import CONFIG_KEYS as D19_KEYS, SOURCES, config_sha, label_instance
from .measure import HardFailure

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "labels" / "d28.json"
LABEL_ROOT = ROOT / "data" / "labels" / "d28"
LABELLER = "ahg.d28labels/1"
LABELLED, BUDGET_EXHAUSTED, DEAD_END, SOLVER_ERROR = "LABELLED", "BUDGET_EXHAUSTED", "DEAD_END", "SOLVER_ERROR"
OUTCOMES = (LABELLED, BUDGET_EXHAUSTED, DEAD_END, SOLVER_ERROR)
CONFIG_KEYS = (D19_KEYS - {"init_budget"}) | {"regimes"}


def load_config(path: Path = CONFIG) -> dict:
    cfg = json.loads(path.read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}  # "_note" keys document, never configure
    check_config(cfg)
    return cfg


def check_config(cfg: dict, max_n: int | None = None) -> None:
    if set(cfg) != CONFIG_KEYS:
        raise ValueError(f"label config keys {sorted(set(cfg) ^ CONFIG_KEYS)} missing or unknown")
    for r in cfg["regimes"]:
        check_regime(r)
    if max_n is not None and cfg["oracle_all_max_n"] < max_n:
        # Round 9: every label is checked against the oracle, not a sample (D19 sampled above n = 12).
        raise ValueError(f"oracle_all_max_n {cfg['oracle_all_max_n']} < n = {max_n}: labels would be sampled")


def with_budget(cfg: dict, state_budget: int | None) -> dict:
    """The config with its per-state budget set. A null budget (not frozen yet) needs an explicit value."""
    budget = cfg["state_budget"] if state_budget is None else state_budget
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("the per-state budget is not frozen yet (state_budget null); pass one explicitly")
    return {**cfg, "state_budget": budget}


def record_key(digest: str, regime: str, cfg: dict) -> str:
    return hashlib.sha256(json.dumps([LABELLER, digest, regime, config_sha(cfg)]).encode()).hexdigest()


def label_doc(doc: dict, regime: str, cfg: dict) -> dict:
    """Label one task document under one regime; returns the record (see the module docstring)."""
    from .gen.generate import doc_digest, to_task
    check_regime(regime)
    task = to_task(doc, regime)
    digest = doc_digest(doc)
    prm = doc["params"]
    base = {"labeller": LABELLER, "key": record_key(digest, regime, cfg), "doc_digest": digest, "regime": regime,
            "config_sha256": config_sha(cfg), "seed": prm["seed"], "n": prm["n"], "archetype": prm["archetype"],
            "rows": [], "budget_exhausted": [], "dead_ends": [], "error": None, "stats": None,
            "hstar_init": None, "certified": False}
    reachable = hmax(task, task.init) < INF  # the common exact pre-check (Round 7)
    if reachable != doc["solvable"]:
        raise HardFailure(f"{digest}: generator solvable={doc['solvable']} but h_max(s0) finite={reachable}")
    if not reachable:
        return base | {"outcome": DEAD_END, "dead_ends": [hex(task.init)]}
    lcfg = {k: v for k, v in cfg.items() if k != "regimes"} | {"init_budget": cfg["state_budget"]}
    key = f"{prm['archetype']}:{prm['n']}:{prm['seed']}"  # the rng streams of the five sources
    t0 = time.perf_counter()
    try:
        out = label_instance(task, key, prm["n"], lcfg)
    except HardFailure:
        raise  # a cross-check mismatch aborts the whole run (rule 6)
    except Exception as e:  # noqa: BLE001 - recorded as SOLVER_ERROR with its type, never as a label
        return base | {"outcome": SOLVER_ERROR, "error": {"type": type(e).__name__, "message": str(e)[:2000]},
                       "stats": {"seconds": round(time.perf_counter() - t0, 4)}}
    if out["stats"] is None:
        return base | {"outcome": BUDGET_EXHAUSTED, "budget_exhausted": [hex(task.init)],
                       "stats": {"init_expanded": out["failures"][0]["expanded"]}}
    return base | {"outcome": LABELLED, "certified": True, "hstar_init": out["stats"]["hstar_init"],
                   "rows": [[hex(s), h, src] for s, h, src in out["rows"]],
                   "budget_exhausted": [{"state": f["state"], "sources": f["sources"]} for f in out["failures"]],
                   "stats": out["stats"]}


def _path(root: Path, regime: str, key: str) -> Path:
    return root / regime / key[:2] / f"{key}.json.gz"


def read_record(root: Path, regime: str, key: str) -> dict | None:
    path = _path(root, regime, key)
    if not path.exists():
        return None
    rec = json.loads(gzip.decompress(path.read_bytes()))
    if rec.get("key") != key or rec.get("regime") != regime or rec.get("labeller") != LABELLER:
        raise ValueError(f"{path}: the stored record does not match its content address")
    return rec


def write_record(root: Path, rec: dict) -> str:
    """Write atomically; return the file's SHA-256. Never overwrites an existing record (append-only)."""
    path = _path(root, rec["regime"], rec["key"])
    if path.exists():
        raise FileExistsError(f"{path} exists; label records are append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(json.dumps(rec, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def label_docs(docs: list[dict], cfg: dict, root: Path = LABEL_ROOT, shard: tuple[int, int] | None = None,
               commit: str | None = None) -> dict:
    """Label every (doc, regime) of this shard that is not on disk yet; return the run summary.

    shard (k, K): docs with index i % K == k. Resumable: existing records are verified and counted as
    resumed. Counts are per regime, per outcome and per source (Round 9 "Recording")."""
    if shard is not None and not 0 <= shard[0] < shard[1]:
        raise ValueError(f"bad shard {shard}")
    from .gen.generate import doc_digest
    files, counts = [], Counter()
    by_source = {r: Counter() for r in cfg["regimes"]}
    for i, doc in enumerate(docs):
        if shard is not None and i % shard[1] != shard[0]:
            continue
        for regime in cfg["regimes"]:
            key = record_key(doc_digest(doc), regime, cfg)
            rec = read_record(root, regime, key)
            if rec is None:
                rec = label_doc(doc, regime, cfg) | {"git_commit": commit}
                sha = write_record(root, rec)
                counts[f"{regime}|written"] += 1
            else:
                sha = hashlib.sha256(_path(root, regime, key).read_bytes()).hexdigest()
                counts[f"{regime}|resumed"] += 1
            counts[f"{regime}|{rec['outcome']}"] += 1
            counts[f"{regime}|budget_exhausted_states"] += len(rec["budget_exhausted"]) if rec["outcome"] == LABELLED else 0
            for _, _, src in rec["rows"]:
                for s in src:
                    by_source[regime][s] += 1
            files.append({"doc_digest": rec["doc_digest"], "regime": regime, "key": key, "outcome": rec["outcome"],
                          "file_sha256": sha})
    return {"labeller": LABELLER, "config": cfg, "config_sha256": config_sha(cfg), "shard": shard,
            "counts": dict(sorted(counts.items())),
            "labels_by_source": {r: {s: by_source[r][s] for s in SOURCES} for r in cfg["regimes"]},
            "files": files}


def load_labels(docs: list[dict], cfg: dict, root: Path = LABEL_ROOT) -> dict[tuple[str, str], dict]:
    """(task digest, regime) -> label record, for every pair on disk."""
    from .gen.generate import doc_digest
    out = {}
    for doc in docs:
        d = doc_digest(doc)
        for regime in cfg["regimes"]:
            rec = read_record(root, regime, record_key(d, regime, cfg))
            if rec is not None:
                out[(d, regime)] = rec
    return out

