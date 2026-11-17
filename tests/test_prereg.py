"""The D28 hypotheses freeze (decisions.md §13.1; Devansh 17 Nov 2026, OQ-P8-1 option (a)).

configs/d28_prereg.json records the frozen hypotheses' SHA-256, the frozen test cohort sizes, the generation caps
and the budget rule; the registry must carry exactly those values. When the local inputs are present (the
hypotheses file and the two pilot runs), the file must also equal what scripts/freeze_prereg.py derives from them."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ahg import cohorts as co

ROOT = Path(__file__).resolve().parents[1]
PREREG = json.loads((ROOT / "configs" / "d28_prereg.json").read_text())
REG = co.load_registry()
LABELS = json.loads((ROOT / "configs" / "labels" / "d28.json").read_text())


def load_script():
    spec = importlib.util.spec_from_file_location("freeze_prereg", ROOT / "scripts" / "freeze_prereg.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["freeze_prereg"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_prereg_records_the_freeze():
    assert PREREG["schema"] == "d28-prereg/1" and PREREG["hypotheses"]["status"] == "FROZEN"
    assert len(PREREG["hypotheses"]["sha256"]) == 64 and PREREG["freeze_date"] == "2026-11-17"
    assert PREREG["state_budget"]["value"] is None  # filled mechanically into configs/labels/d28.json, not here
    sb = PREREG["state_budget"]
    assert (sb["candidates"], sb["target"], sb["cap"]) == (REG_BUDGET["candidates"], REG_BUDGET["target"], REG_BUDGET["cap"])


REG_BUDGET = json.loads(co.REGISTRY.read_text())["budget_pilot"]


def test_registry_is_frozen_with_exactly_the_prereg_values():
    assert REG.frozen
    sizes = PREREG["test_cohort_sizes"]["per_size"]
    tests = {c.name: c for c in REG.cohorts.values() if c.is_test and not c.smoke}
    assert set(sizes) == set(tests)
    for name, c in tests.items():
        assert {str(n): c.size_at(n) for n in c.sizes} == sizes[name], name
    caps = PREREG["generation_caps"]
    for role, c in (("training", REG.training), ("validation", REG["d28_val"])):
        assert {str(n): c.cap_at(n) for n in c.sizes} == caps[role]["max_candidates_per_size"]
        assert caps[role]["quota_per_size"] == c.per_size
        assert caps[role]["total"] == sum(c.cap_at(n) for n in c.sizes)
    # Devansh's freeze message: the all-13-sizes guarantee gives 13,065 (training) and 2,432 (validation)
    assert (caps["training"]["total"], caps["validation"]["total"]) == (13065, 2432)
    assert sizes["d28_test_id"] == {"12": 650, "20": 674}  # the §18 P6d frozen proposal, as reported in P6.md


def test_nothing_but_the_freeze_commit_blocks_the_real_cohorts():
    for c in REG.cohorts.values():
        if not c.smoke:
            assert all("hypotheses are not frozen" in w for w in REG.problems(c)), (c.name, REG.problems(c))


def test_state_budget_is_unset_or_a_candidate_of_the_rule():
    v = LABELS["state_budget"]
    assert v is None or v in PREREG["state_budget"]["candidates"]


@pytest.mark.skipif(not (ROOT / "docs" / "d28" / "hypotheses.md").exists(), reason="local hypotheses file absent")
def test_the_local_hypotheses_file_is_the_frozen_one():
    import hashlib
    text = (ROOT / "docs" / "d28" / "hypotheses.md").read_bytes()
    assert hashlib.sha256(text).hexdigest() == PREREG["hypotheses"]["sha256"]  # never edited after the freeze
    assert b"FROZEN" in text.splitlines()[0]


def test_the_prereg_is_derived_from_the_runs():
    fz = load_script()
    if not all((ROOT / r / "raw.json.gz").exists() for r in (fz.TEST_SIZES_RUN, fz.CAPS_RUN)) or \
            not (ROOT / "docs" / "d28" / "hypotheses.md").exists():
        pytest.skip("pilot runs or the local hypotheses file not in this checkout")
    built, reg = fz.build()
    strip = lambda d: {k: v for k, v in d.items() if k not in ("freeze_date", "commit_time_utc",
                                                                "labels_config_state_budget_at_freeze")}
    assert strip(built) == strip(PREREG)
    cur = json.loads(co.REGISTRY.read_text())
    assert [c["per_size"] for c in reg["cohorts"]] == [c["per_size"] for c in cur["cohorts"]]
    assert [c["max_candidates_per_size"] for c in reg["cohorts"]] == [c["max_candidates_per_size"] for c in cur["cohorts"]]
