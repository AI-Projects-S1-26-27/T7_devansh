"""AL1 gap gate: the verdict is computed mechanically from pilot metrics and the frozen config (decisions.md §21)."""

import json
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "configs" / "al1_gate.json"
RULES = {  # metric -> criterion key; every criterion is a lower bound
    "gap_state_share": "gap_state_share_min",
    "gap_instance_share": "gap_instance_share_min",
    "median_best_classical_ratio": "median_best_classical_ratio_min",
    "solvable_share": "solvable_share_min",
    "recovery_or_revisit_plan_share": "recovery_or_revisit_plan_share_min",
    "route_change_instance_share": "route_change_instance_share_min",
}


def load_gate(path: Path = GATE) -> dict:
    g = json.loads(path.read_text())
    if g.get("schema") != "al1-gate/1" or set(g["criteria"]) != set(RULES.values()):
        raise ValueError(f"{path}: not an al1-gate/1 config with exactly the six criteria")
    return g


def verdict(metrics: dict, gate: dict) -> dict:
    """{"pass": bool, "checks": {metric: {"value", "min", "ok"}}}. A missing or None metric fails its check."""
    checks = {}
    for m, k in RULES.items():
        v, lo = metrics.get(m), gate["criteria"][k]
        checks[m] = {"value": v, "min": lo, "ok": v is not None and v >= lo}
    return {"pass": all(c["ok"] for c in checks.values()), "checks": checks}
