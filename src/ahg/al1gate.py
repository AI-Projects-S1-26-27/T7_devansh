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


def decision(summary: dict, rule: dict) -> dict:
    """AL2 go/stop, mechanical (configs/gen/al2_diag.json "decision_rule", fixed before any run).

    summary: a gap run's {config key: aggregate}. The rule reads summary[rule["config"]] and says "go" only if
    at least one "any_of" lower bound holds and every "all_of" lower bound holds. Bounds are inclusive; a
    missing or None metric fails its check, so an empty cell can never say go.
    """
    agg = summary.get(rule["config"], {})

    def check(group: str) -> dict:
        out = {}
        for k, lo in rule[group].items():
            m = k.removesuffix("_min")
            v = agg.get(m)
            out[m] = {"value": v, "min": lo, "ok": v is not None and v >= lo}
        return out

    any_of, all_of = check("any_of"), check("all_of")
    go = any(c["ok"] for c in any_of.values()) and all(c["ok"] for c in all_of.values())
    return {"config": rule["config"], "go": go, "any_of": any_of, "all_of": all_of}
