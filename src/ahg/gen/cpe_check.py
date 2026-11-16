"""Second, independently written CPE applicability evaluator (P6: "a second, independently written
evaluator must agree on every match").

It imports nothing from ahg, so it cannot share a bug with ahg.gen.cpe, ahg.cvedata or ahg.fixtures. It
implements the same declared semantics by a different route:

- CPE strings are split by a hand-written scanner (not a regex), escapes dropped while scanning;
- versions are compared pairwise by `compare_versions` (a three-way comparison over alternating number /
  word segments), not mapped to sort keys;
- the configuration tree is evaluated with an explicit stack in post-order, not by recursion;
- vulnerable components are collected in the same pass over each true configuration.

Declared semantics (identical to ahg.gen.cpe): configurations OR'ed; nodes combined by their operator (OR
by default), `negate` inverts; a cpeMatch holds if any host CPE name satisfies it; criteria `*` matches any
value, a criteria version `-` means the whole product (P6b), any other criteria value matches only the
equal value (case-insensitive); bounds apply to a `*` version, compare the version field only, and only to
a digit-led host version; a component is vulnerable if a true configuration has a non-negated
`vulnerable: true` cpeMatch satisfied by one of the component's names and, for a Microsoft criteria with no
bounds, the MSRC build check passes. MSRC (P6b): among the unit's MSRC product IDs, the records with status
type 3 named as the host's MSRC product; the check passes if one has a fixed build and the host build is
below the lowest, unless one of the record's fixing KBs is among the host's standalone KBs; such a record
is also an applicability record of its own ("msrc").
Version order: numbers numerically; the words dev < alpha < beta < m/milestone < pre/preview < rc sort
below the bare release; any other word sorts above it; a run of trailing zero numbers is ignored
(5.10 == 5.10.0).
"""

FIELDS = 11  # part, vendor, product, version, update, edition, language, sw_edition, target_sw, target_hw, other
BOUNDS = ("versionStartIncluding", "versionStartExcluding", "versionEndIncluding", "versionEndExcluding")
PRE = {"dev": 0, "alpha": 1, "beta": 2, "m": 3, "milestone": 3, "pre": 4, "preview": 4, "rc": 5}


def split_cpe(s: str) -> list[str]:
    """The 11 attribute values of a `cpe:2.3:...` string, lower-cased, escapes removed."""
    out, cur, i = [], [], 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            cur.append(s[i + 1])
            i += 2
            continue
        if ch == ":":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    if len(out) != 13 or out[0] != "cpe" or out[1] != "2.3":
        raise ValueError(f"bad CPE string {s!r}")
    return [v.lower() for v in out[2:]]


def _segments(v: str) -> list:
    """Alternating runs: ints for digit runs, str for letter runs; separators dropped."""
    segs, i, v = [], 0, v.lower()
    while i < len(v):
        if v[i].isdigit():
            j = i
            while j < len(v) and v[j].isdigit():
                j += 1
            segs.append(int(v[i:j]))
            i = j
        elif v[i].isalpha():
            j = i
            while j < len(v) and v[j].isalpha():
                j += 1
            segs.append(v[i:j])
            i = j
        else:
            i += 1
    # drop zeros that end a run of numbers (followed by a word, or by the end), keeping the run's first number
    out = []
    for k, s in enumerate(segs):
        out.append(s)
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        if not isinstance(nxt, int):
            while len(out) >= 2 and out[-1] == 0 and isinstance(out[-2], int):
                out.pop()
    return out


def _rank(seg) -> tuple[int, object]:
    """Class of a segment at one position: pre-release word 0, end-of-version 1, other word 2, number 3."""
    if seg is None:
        return (1, 0)
    if isinstance(seg, int):
        return (3, seg)
    if seg in PRE:
        return (0, PRE[seg])
    return (2, seg)


def compare_versions(a: str, b: str) -> int:
    """-1, 0 or 1."""
    x, y = _segments(a), _segments(b)
    for i in range(max(len(x), len(y)) + 1):
        ra = _rank(x[i] if i < len(x) else None)
        rb = _rank(y[i] if i < len(y) else None)
        if ra != rb:
            return -1 if ra < rb else 1
        if ra == (1, 0):
            return 0
    return 0


def satisfies(match: dict, name: str) -> bool:
    crit, host = split_cpe(match["criteria"]), split_cpe(name)
    ranged = crit[3] == "*" and any(b in match for b in BOUNDS)
    for idx in range(FIELDS):
        if idx == 3 and crit[3] == "-":
            continue  # the whole product
        if idx == 3 and ranged:
            v = host[3]
            if not v or not v[0].isdigit():
                return False
            if "versionStartIncluding" in match and compare_versions(v, match["versionStartIncluding"]) < 0:
                return False
            if "versionStartExcluding" in match and compare_versions(v, match["versionStartExcluding"]) <= 0:
                return False
            if "versionEndIncluding" in match and compare_versions(v, match["versionEndIncluding"]) > 0:
                return False
            if "versionEndExcluding" in match and compare_versions(v, match["versionEndExcluding"]) >= 0:
                return False
            continue
        if crit[idx] != "*" and crit[idx] != host[idx]:
            return False
    return True


def _evaluate(cfg: dict, host: list[str]) -> bool:
    """Post-order evaluation of one configuration with an explicit stack."""
    top = {"operator": cfg.get("operator", "OR"), "negate": cfg.get("negate", False), "nodes": cfg.get("nodes", [])}
    stack, values = [(top, False)], {}
    while stack:
        node, expanded = stack.pop()
        kids = list(node.get("nodes", [])) + list(node.get("children", []))
        if not expanded:
            stack.append((node, True))
            for k in kids:
                stack.append((k, False))
            continue
        terms = [any(satisfies(m, h) for h in host) for m in node.get("cpeMatch", [])]
        terms += [values[id(k)] for k in kids]
        if not terms:
            val = False
        elif str(node.get("operator", "OR")).upper() == "AND":
            val = all(terms)
        else:
            val = any(terms)
        values[id(node)] = (not val) if node.get("negate") else val
    return values[id(top)]


def _positive_vulnerable(cfg: dict) -> list[dict]:
    out, stack = [], [(n, bool(cfg.get("negate"))) for n in cfg.get("nodes", [])]
    while stack:
        node, neg = stack.pop()
        neg = neg != bool(node.get("negate"))
        out += [m for m in node.get("cpeMatch", []) if m.get("vulnerable") is True and not neg]
        stack += [(k, neg) for k in list(node.get("nodes", [])) + list(node.get("children", []))]
    return out


def msrc_check(msrc: dict | None) -> bool:
    """True if the host build is below the lowest fixed build of an affected MSRC record for its product."""
    if msrc is None or not msrc.get("name"):
        return False
    installed = set()
    for k in msrc.get("kbs") or []:
        k = str(k).strip().upper()
        installed.add(k[2:] if k.startswith("KB") else k)
    lowest = None
    for pid in msrc["unit_ids"]:
        rec = msrc["products"].get(pid)
        if rec is None or rec["name"] != msrc["name"] or rec.get("status_type") != 3:
            continue
        for fix in rec.get("fixes") or []:
            kb = str(fix.get("kb") or "").strip().upper()
            if kb and (kb[2:] if kb.startswith("KB") else kb) in installed:
                return False  # the fixing standalone KB is installed
            b = fix.get("fixed_build")
            if b and (lowest is None or compare_versions(b, lowest) < 0):
                lowest = b
    return lowest is not None and compare_versions(msrc["build"], lowest) < 0


def _microsoft_unbounded(match: dict) -> bool:
    return split_cpe(match["criteria"])[1] == "microsoft" and not any(b in match for b in BOUNDS)


def applies(configurations: list[dict], host: list[str], component: list[str], msrc: dict | None = None) -> list:
    """Indices of the configurations that make `component` vulnerable on this host, then "msrc" if the MSRC
    affected record applies (empty: no match)."""
    build_ok = msrc_check(msrc)
    hits: list = []
    for i, cfg in enumerate(configurations):
        if not _evaluate(cfg, host):
            continue
        for m in _positive_vulnerable(cfg):
            if any(satisfies(m, n) for n in component) and (build_ok or not _microsoft_unbounded(m)):
                hits.append(i)
                break
    if build_ok:
        hits.append("msrc")
    return hits
