"""Offline training and validation batches from D28 cohorts and their label records (Round 9, P10 bridge).

The aggregator grid and the on-policy runs take in-memory LabelBatch objects; the D19 batches on disk
(dataset.open_batch) cannot hold D28 tasks. cohort_batch builds one batch per (cohort, regime): the cohort
record's members (training or validation only; test cohorts are refused here, BUILD_RULES rule 7), each
encoded by d28encode.instance under the regime, and every certified row of its label record as an Example.

Only LABELLED records give examples. A member without a record, or with BUDGET_EXHAUSTED / DEAD_END /
SOLVER_ERROR, keeps its instance slot (so indices stay aligned with the cohort) and is listed in failures
with its outcome, never as a label (Round 9: predictions and missing labels are never labels).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import cohorts as co
from . import d28labels
from .cost import check_regime
from .dataset import Example, LabelBatch
from .learning import d28encode
from .onpolicy import make_batch

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "data" / "d28" / "cohorts"
SPLIT_OF_ROLE = {"training": "train", "validation": "val"}
PREFIX = "cohort:"  # a train_batches / val_batches entry naming a D28 cohort instead of a D19 batch path


def is_cohort_ref(entry: str) -> bool:
    return str(entry).startswith(PREFIX)


def cohort_batch(name: str, regime: str, *, reg=None, label_cfg: dict | None = None,
                 label_root: Path | None = None, store: Path | None = None,
                 records: Path | None = None) -> LabelBatch:
    """The offline LabelBatch of one built training or validation cohort under one regime."""
    check_regime(regime)
    reg = reg or co.load_registry()
    c = reg[name]
    if c.is_test or c.role not in SPLIT_OF_ROLE:
        # Test cohorts are read only by the evaluation task (BUILD_RULES rule 7), never for training.
        raise PermissionError(f"cohort {name!r} (role {c.role!r}) cannot become a training or validation batch")
    path = (records or RECORDS) / f"{c.name}.json"  # the record scripts/d28_pipe.py build writes
    if not path.exists():
        raise FileNotFoundError(f"{path}: build the cohort first")
    rec = json.loads(path.read_text())
    cfg = label_cfg or d28labels.load_config()
    root = label_root or d28labels.LABEL_ROOT
    docs = [d for _, d in co.open_members(rec, reg, store=store or co.TASK_STORE)]
    instances, examples, failures = [], [], []
    from .gen.generate import doc_digest
    for i, doc in enumerate(docs):
        if doc is None:
            instances.append(None)
            continue
        digest = doc_digest(doc)
        lab = d28labels.read_record(root, regime, d28labels.record_key(digest, regime, cfg))
        instances.append(d28encode.instance(doc, regime))
        if lab is None or lab["outcome"] != d28labels.LABELLED:
            failures.append({"instance": i, "doc_digest": digest,
                             "outcome": "MISSING" if lab is None else lab["outcome"]})
            continue
        for s, h, src in lab["rows"]:
            examples.append(Example(i, int(s, 16), int(h), list(src) if isinstance(src, list) else [src]))
    return make_batch(f"{name}-{regime}", SPLIT_OF_ROLE[c.role], regime, instances, examples, failures)


def load_cohort_batches(entries: list[str], regime: str) -> list[LabelBatch]:
    return [cohort_batch(e[len(PREFIX):], regime) for e in entries]
