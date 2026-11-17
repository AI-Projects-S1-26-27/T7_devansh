"""Independent check of the generator's planner-visible OS and principal classes (P7c, OQ-P7c-1; §18 P6e).

Decided 17 Nov 2026 (Devansh): the generator exposes each host's OS (network.hosts[].os = {family, version}, null
on the internet host; the version is the drawn Windows Server version, null on Linux, OQ-P6-28) and each principal
index's identity class (network.principal_classes: "attacker" for p0, "admin" for an admin principal by citation,
"UNKNOWN", else "service"). The generator writes both since d28-gen a1ee057 (merged in M4). This module derives
the two fields from the metadata block exactly as specified: for a document that carries them,
add_planner_context checks that they equal the specification; a hand-built document without them gets them added.
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


def os_of(meta_host: dict) -> dict:
    """§18 P6e from the metadata host: the family, and on Windows the drawn Server version (the OS component's
    product); Linux has no OS version (OQ-P6-28)."""
    version = None
    if meta_host["os"] == "windows":
        version = next(c["product"] for c in meta_host["components"] if c["family"] == "os")
    return {"family": meta_host["os"], "version": version}


def add_planner_context(doc: dict) -> dict:
    """A copy of a generated document with network.hosts[].os and network.principal_classes as specified."""
    out = copy.deepcopy(doc)
    meta = {h["id"]: h for h in doc["metadata"]["hosts"]}
    hosts = [{**h, "os": os_of(meta[h["id"]]) if h["id"] in meta else None} for h in doc["network"]["hosts"]]
    classes = [principal_class(p, i) for i, p in enumerate(doc["metadata"]["principals"])]
    net = doc["network"]
    if "principal_classes" in net:
        assert net["principal_classes"] == classes and net["hosts"] == hosts, "generator context != specification"
        return out
    out["network"] = {**net, "hosts": hosts, "principal_classes": classes}
    return out


def generating_with_context(generate):
    """generate() wrapped so every document is checked against (or given) the planner context."""
    def wrapped(*a, **k):
        return add_planner_context(generate(*a, **k))
    return wrapped
