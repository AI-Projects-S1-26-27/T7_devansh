"""Feature-leakage audit (D28 Round 11, LRN6): run per split before training, fail the run on any violation.

A FeaturePolicy is an allowlist of exact feature names and a denylist of
case-insensitive regular expressions (CVE IDs, vendor/product tokens, versions,
dates, CPE URIs, seeds, ...). A feature passes only if it is on the allowlist and
matches no deny pattern; the denylist wins over the allowlist, and a policy whose
allowlist itself matches a deny pattern is refused when it is built. The concrete
lists are a separate task (P9); this module only enforces them.

audit() checks the feature names a dataset actually exposes and returns a report;
run_audit() writes that report (it is kept, per LRN6) and raises LeakageError if
the report has any violation, so a training run cannot start on a leaking dataset.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeaturePolicy:
    allow: frozenset[str]
    deny: tuple[str, ...]

    def __post_init__(self):
        for p in self.deny:
            re.compile(p)  # an invalid pattern fails here, not mid-audit
        clash = sorted(f for f in self.allow if self.denied_by(f))
        if clash:
            raise ValueError(f"policy allows features its own denylist forbids: {clash}")

    def denied_by(self, name: str) -> str | None:
        return next((p for p in self.deny if re.search(p, name, flags=re.IGNORECASE)), None)

    @classmethod
    def from_dict(cls, d: dict) -> "FeaturePolicy":
        if set(d) != {"allow", "deny"}:
            raise ValueError("a feature policy has exactly the keys 'allow' and 'deny'")
        return cls(frozenset(d["allow"]), tuple(d["deny"]))

    @classmethod
    def from_json(cls, path: str | Path) -> "FeaturePolicy":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {"allow": sorted(self.allow), "deny": list(self.deny)}


def audit(feature_names: list[str], policy: FeaturePolicy, split: str) -> dict:
    """The audit report for one split's dataset: every feature name and every violation with its reason."""
    if not feature_names:
        raise LeakageError(f"split {split!r}: no feature names to audit (an empty audit proves nothing)")
    violations, seen = [], set()
    for name in feature_names:
        if name in seen:
            violations.append({"feature": name, "reason": "duplicate feature name"})
            continue
        seen.add(name)
        pattern = policy.denied_by(name)
        if pattern is not None:
            violations.append({"feature": name, "reason": f"denied: matches {pattern!r}"})
        elif name not in policy.allow:
            violations.append({"feature": name, "reason": "not on the allowlist"})
    return {"split": split, "features": list(feature_names), "policy": policy.to_dict(),
            "violations": violations, "ok": not violations}


def run_audit(feature_names: list[str], policy: FeaturePolicy, split: str, report_path: str | Path) -> dict:
    """Audit, write the report to report_path, and raise LeakageError on any violation."""
    report = audit(feature_names, policy, split)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n")
    if not report["ok"]:
        raise LeakageError(f"split {split!r}: {len(report['violations'])} feature violation(s), "
                           f"report {path}: {report['violations'][:5]}")
    return report
