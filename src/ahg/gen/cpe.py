"""Primary applicability evaluator: NVD CPE configurations plus MSRC records (P6, P6b; decided 16 Nov 2026).

A CVE applies to an installed component only through recorded applicability data: the CVE's NVD
configurations, and for Microsoft CVEs its MSRC affected-product records (P6b, OQ-P6-10). MSRC tags,
catalogue `affected_by` lists and CVE text never decide a match. The CVE is never sampled.

NVD semantics (applicability statements; CPE 2.3 name matching, NISTIR 7696, with the P6b reading of `-`):

- A CVE has a list of configurations; one that is true for the host can make a component vulnerable.
- A configuration combines its nodes with its operator (AND/OR; OR if absent). A node combines its
  cpeMatch entries and child nodes with its own operator; `negate: true` inverts a node.
- A cpeMatch is true for a host if any of the host's CPE names matches it. `vulnerable: false` matches are
  co-conditions on the host (platform, Round 3): they make a configuration true but never make the matched
  component vulnerable.
- A component is vulnerable to the CVE if some configuration is true for the host AND contains a
  non-negated `vulnerable: true` match that matches the component's own CPE name, AND, when that match
  names a Microsoft product without version bounds, the MSRC build check passes (below).

Attribute matching: a criteria value `*` (ANY) matches every host value. A criteria version `-` means the
whole product (P6b, OQ-P6-6): it matches every host version. Any other criteria value matches only the
equal host value (case-insensitive, CPE escapes removed), so a criteria naming `x64` does not match a host
whose target_hw is `*`. Version bounds (`versionStart*`, `versionEnd*`) apply when the criteria version is
`*`; they compare the version field only, and only a host version that starts with a digit can be inside
a range (a version such as `r2` is not on a numeric scale).

MSRC (Microsoft products, P6b): the MSRC affected-product records of a CVE are those of its pool unit's
MSRC product IDs with status type 3 (Known Affected). The host's component names one MSRC product (the
Windows Server version, or the Exchange / SQL Server release line) and carries a build. The build check
passes if that product has an affected record with at least one fixed build and the host build is earlier
than the lowest of them (the lowest-FixedBuild rule of OQ-P5-7), unless the host lists one of the record's
fixing KBs among the standalone KBs installed on top of its cumulative build (P6b addendum, 17 Nov 2026).
Without a fixed build the check fails: a KB number alone does not place a build (OQ-P6-14). An affected record whose build check passes is itself
an applicability record ("msrc" evidence), stored and traced like an NVD match.

Version order is `ahg.fixtures.version_key`, the order the P5 catalogue was built with. The second
evaluator (`ahg.gen.cpe_check`) implements all of this independently.
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
MSRC_AFFECTED = 3  # CVRF ProductStatus type "Known Affected"


def _unescape(v: str) -> str:
    """CPE 2.3 formatted-string escapes (`\\.`, `\\:`) removed, lower-cased, for comparison."""
    return v.replace("\\", "").lower()


def _attr_ok(crit: str, host: str) -> bool:
    """Criteria ANY matches everything, any other value only itself."""
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


def bounded(match: dict) -> bool:
    return any(k in match for k in BOUND_KEYS)


def name_matches(match: dict, host_cpe: str) -> bool:
    """Does one host CPE name satisfy one cpeMatch (attributes plus version bounds)?"""
    c, h = _parse(match["criteria"]), _parse(host_cpe)
    for a in ATTRS:
        if a == "version":
            if c["version"] == "*" and bounded(match):
                if not _in_range(match, h["version"]):
                    return False
                continue
            if c["version"] == "-":  # P6b: '-' means the whole product
                continue
        if not _attr_ok(c[a], h[a]):
            return False
    return True


def needs_build_check(match: dict) -> bool:
    """A Microsoft criteria without version bounds cannot place a build: MSRC data decides (P6b)."""
    return _parse(match["criteria"])["vendor"].replace("\\", "").lower() == "microsoft" and not bounded(match)


def msrc_fixed_build(msrc_products: dict, unit_ids: list[str], name: str) -> str | None:
    """Lowest fixed build of the CVE's affected record for MSRC product `name` (among the unit's IDs)."""
    builds = [f["fixed_build"] for pid in unit_ids if pid in msrc_products
              for p in [msrc_products[pid]] if p["name"] == name and p.get("status_type") == MSRC_AFFECTED
              for f in p.get("fixes", []) if f.get("fixed_build")]
    return min(builds, key=_vkey) if builds else None


def kb_id(kb) -> str:
    """A KB number without its "KB" prefix ("KB5034127" and "5034127" are the same update)."""
    return str(kb).strip().upper().removeprefix("KB")


def msrc_fix_kbs(msrc_products: dict, unit_ids: list[str], name: str) -> set[str]:
    """Every KB that MSRC lists as fixing the CVE for MSRC product `name` (among the unit's IDs)."""
    return {kb_id(f["kb"]) for pid in unit_ids if pid in msrc_products
            for p in [msrc_products[pid]] if p["name"] == name and p.get("status_type") == MSRC_AFFECTED
            for f in p.get("fixes", []) if f.get("kb")}


def msrc_affected(msrc: dict | None) -> str | None:
    """The fixed build the host build is earlier than, or None if the MSRC build check fails.

    msrc: {"products": CVE's MSRC products, "unit_ids": the unit's MSRC product IDs, "name": the host's MSRC
    product name, "build": the host build, "kbs": standalone KBs installed on top of that build (optional)},
    or None for a component with no MSRC product. A host is unaffected if its build is at or above the
    fixed build, or if one of the CVE's fixing KBs is in its list.
    """
    if not msrc or not msrc.get("name"):
        return None
    if {kb_id(k) for k in msrc.get("kbs") or ()} & msrc_fix_kbs(msrc["products"], msrc["unit_ids"], msrc["name"]):
        return None
    fb = msrc_fixed_build(msrc["products"], msrc["unit_ids"], msrc["name"])
    return fb if fb is not None and _vkey(msrc["build"]) < _vkey(fb) else None


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


def marks(match: dict, component: list[str], build_ok: bool) -> bool:
    """Does a vulnerable match make the component vulnerable (name match, plus the build check if needed)?"""
    return any(name_matches(match, n) for n in component) and (build_ok or not needs_build_check(match))


def component_matches(configurations: list[dict], host: list[str], component: list[str],
                      msrc: dict | None = None) -> list[dict]:
    """The evidence that a component (given by its CPE names, all among the host's) is vulnerable.

    One entry per true NVD configuration with the vulnerable criteria that matched it, then one "msrc" entry
    if the MSRC affected record applies. Empty list: the CVE does not apply.
    """
    fb = msrc_affected(msrc)
    out = []
    for i, cfg in enumerate(configurations):
        if not config_true(cfg, host):
            continue
        hits = sorted({m["criteria"] + "".join(f"|{k}={m[k]}" for k in BOUND_KEYS if k in m)
                       for node in cfg.get("nodes", []) for m in _vulnerable_matches(node, bool(cfg.get("negate")))
                       if marks(m, component, fb is not None)})
        if hits:
            out.append({"configuration": i, "criteria": hits})
    if fb is not None:
        out.append({"configuration": "msrc", "product": msrc["name"], "build": msrc["build"], "fixed_build": fb,
                    "kbs": sorted(msrc.get("kbs") or ())})
    return out
