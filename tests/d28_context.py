"""Test stand-in for the generator's planner-visible OS and principal classes (P7c, OQ-P7c-1).

Decided 17 Nov 2026 (Devansh): the generator exposes each host's OS family (network.hosts[].os, null on the
internet host) and each principal index's identity class (network.principal_classes: "attacker" for p0,
"admin" for an admin principal by citation, "UNKNOWN", else "service"), so the D28 encoder never reads the
metadata block. That change belongs to d28-gen and is not on this branch yet; until it is, these tests derive
the two fields from the metadata exactly as specified. If a document already carries them (the generator
change merged), add_planner_context checks they equal the specification instead of overwriting them.
"""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMIN = frozenset(json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())["admin_principals"])


def principal_class(name: str, index: int) -> str:
    if index == 0:
        return "attacker"
    if name in ADMIN:
        return "admin"
    return "UNKNOWN" if name == "UNKNOWN" else "service"


def add_planner_context(doc: dict) -> dict:
    """A copy of a generated document with network.hosts[].os and network.principal_classes as specified."""
    out = copy.deepcopy(doc)
    os_of = {h["id"]: h["os"] for h in doc["metadata"]["hosts"]}
    hosts = [{**h, "os": os_of.get(h["id"])} for h in doc["network"]["hosts"]]
    classes = [principal_class(p, i) for i, p in enumerate(doc["metadata"]["principals"])]
    net = doc["network"]
    if "principal_classes" in net:
        assert net["principal_classes"] == classes and net["hosts"] == hosts, "generator context != specification"
        return out
    out["network"] = {**net, "hosts": hosts, "principal_classes": classes}
    return out


def generating_with_context(generate):
    """generate() wrapped so every document carries the planner context (monkeypatch target)."""
    def wrapped(*a, **k):
        return add_planner_context(generate(*a, **k))
    return wrapped
