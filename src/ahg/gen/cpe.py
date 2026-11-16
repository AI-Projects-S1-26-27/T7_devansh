"""Primary CPE applicability evaluator (P6 decision "Matching is CPE-only", decided 16 Nov 2026).

A CVE applies to an installed component only through the CVE's own NVD configurations; MSRC tags,
catalogue `affected_by` lists and CVE text never decide a match. The CVE is never sampled.

Semantics (NVD applicability statements; CPE 2.3 name matching, NISTIR 7696):

- A CVE has a list of configurations; it applies to a host if ANY configuration is true for the host.
- A configuration combines its nodes with its operator (AND/OR; OR if absent). A node combines its
  cpeMatch entries and child nodes with its own operator; `negate: true` inverts a node.
- A cpeMatch is true for a host if any of the host's CPE names matches it. `vulnerable: false` matches are
  co-conditions on the host (platform, Round 3): they make a configuration true but never make the matched
  component vulnerable.
- A component is vulnerable to the CVE if some configuration is true for the host AND contains a
  non-negated `vulnerable: true` match that matches the component's own CPE name. That is the match record.

Attribute matching (strict CPE 2.3 name matching): a criteria value `*` (ANY) matches every host value;
any other criteria value, including `-` (NA), matches only the equal host value (case-insensitive, CPE
escapes removed). So a criteria naming `x64` does not match a host whose target_hw is `*`, and a criteria
version `-` does not match a host whose version is a build number. Version bounds (`versionStart*`,
`versionEnd*`) apply when the criteria version is `*`; they compare the version field only, and only a
host version that starts with a digit can be inside a range (a version such as `r2` is not on a numeric
scale, so it is never inside one).

Version order is `ahg.fixtures.version_key`, the order the P5 catalogue intervals were built with. The
second evaluator (`ahg.gen.cpe_check`) implements all of this independently.
"""

from __future__ import annotations

from functools import lru_cache

from ahg.cvedata import parse_cpe
from ahg.fixtures import version_key

_parse = lru_cache(maxsize=None)(parse_cpe)  # pure; the same criteria strings recur across hosts
_vkey = lru_cache(maxsize=None)(version_key)

BOUND_KEYS = ("versionStartIncluding", "versionStartExcluding", "versionEndIncluding", "versionEndExcluding")
ATTRS = ("part", "vendor", "product", "version", "update", "edition", "language", "sw_edition", "target_sw",
         "target_hw", "other")


def _unescape(v: str) -> str:
    """CPE 2.3 formatted-string escapes (`\\.`, `\\:`) removed, lower-cased, for comparison."""
    return v.replace("\\", "").lower()


def _attr_ok(crit: str, host: str) -> bool:
    """Strict CPE 2.3 attribute match: criteria ANY matches everything, any other value only itself."""
    return crit == "*" or _unescape(crit) == _unescape(host)


def _in_range(match: dict, version: str) -> bool:
    """Is a host version inside the match's bounds? Only digit-led versions can be (see module doc)."""
    if not version[:1].isdigit():
        return False
    k = _vkey(version)
    lo_i, lo_x = match.get("versionStartIncluding"), match.get("versionStartExcluding")
    hi_i, hi_x = match.get("versionEndIncluding"), match.get("versionEndExcluding")
    return ((lo_i is None or k >= _vkey(lo_i)) and (lo_x is None or k > _vkey(lo_x))
            and (hi_i is None or k <= _vkey(hi_i)) and (hi_x is None or k < _vkey(hi_x)))


def name_matches(match: dict, host_cpe: str) -> bool:
    """Does one host CPE name satisfy one cpeMatch (attributes plus version bounds)?"""
    c, h = _parse(match["criteria"]), _parse(host_cpe)
    for a in ATTRS:
        if a == "version" and c["version"] == "*" and any(k in match for k in BOUND_KEYS):
            if not _in_range(match, h["version"]):
                return False
        elif not _attr_ok(c[a], h[a]):
            return False
    return True


def _node_true(node: dict, host: list[str]) -> bool:
    terms = [any(name_matches(m, n) for n in host) for m in node.get("cpeMatch", [])]
    terms += [_node_true(ch, host) for ch in [*node.get("nodes", []), *node.get("children", [])]]
    if not terms:  # an empty node states nothing, so it is never true (not even under AND)
        return bool(node.get("negate"))
    val = all(terms) if node.get("operator", "OR").upper() == "AND" else any(terms)
    return val != bool(node.get("negate"))


def config_true(config: dict, host: list[str]) -> bool:
    """Is one NVD configuration true for a host given as the list of its CPE names?"""
    vals = [_node_true(n, host) for n in config.get("nodes", [])]
    if not vals:
        return bool(config.get("negate"))
    val = all(vals) if config.get("operator", "OR").upper() == "AND" else any(vals)
    return val != bool(config.get("negate"))


def vulnerable_matches(configurations: list[dict]) -> list[dict]:
    """Every non-negated `vulnerable: true` cpeMatch of a CVE, at any depth, in configuration order."""
    return [m for cfg in configurations for node in cfg.get("nodes", [])
            for m in _vulnerable_matches(node, bool(cfg.get("negate")))]


def _vulnerable_matches(node: dict, negated: bool):
    negated ^= bool(node.get("negate"))
    for m in node.get("cpeMatch", []):
        if m.get("vulnerable") and not negated:
            yield m
    for ch in [*node.get("nodes", []), *node.get("children", [])]:
        yield from _vulnerable_matches(ch, negated)


def component_matches(configurations: list[dict], host: list[str], component: list[str]) -> list[dict]:
    """The evidence that a component (given by its CPE names, all among the host's) is vulnerable: one entry
    per true configuration, with the vulnerable criteria that matched it. Empty list: the CVE does not apply."""
    out = []
    for i, cfg in enumerate(configurations):
        if not config_true(cfg, host):
            continue
        hits = sorted({m["criteria"] + "".join(f"|{k}={m[k]}" for k in BOUND_KEYS if k in m)
                       for node in cfg.get("nodes", []) for m in _vulnerable_matches(node, bool(cfg.get("negate")))
                       if any(name_matches(m, n) for n in component)})
        if hits:
            out.append({"configuration": i, "criteria": hits})
    return out
