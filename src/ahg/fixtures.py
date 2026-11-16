"""Version-fixture catalogue (D28 Round 4 "Fixtures and assignment", catalogue rule; task P5).

For each candidate (product, CVE):
1. The affected intervals come from the CVE's full NVD CPE applicability criteria: the vulnerable,
   non-negated matches of the product cover a set of upstream releases, and each maximal run of
   consecutive covered releases is one interval (P5 decisions 16 Nov 2026: "per range", then "maximal
   affected runs" for versions NVD enumerates one by one).
2. The vulnerable fixture of an interval is the latest stable upstream release inside it (P5 decision:
   stable releases only; rc/beta/milestone builds are never fixtures).
3. The fixed fixture of an interval is the first release after that vulnerable fixture which the vendor
   explicitly documents as fixed: a vendor advisory or release-notes page, or, where the CVE's CNA is the
   vendor itself, a CVE JSON 5 version entry with status "unaffected" (P5 decisions: vendor pages with a
   CNA fallback; a "lessThan" bound alone is never a fix).
4. Fixtures are deduplicated per product.
5. Each fixture records its exact version, release date, source URL, retrieval time and the SHA-256 of the
   raw file that proves it (P5 decision: SHA-256 of the evidence files, no tarball downloads).

Catalogue constraints (Round 4): a CPE bound is never taken as a release. Only versions found in a fetched
upstream release list become fixtures; a documented fix that is not such a release is dropped and logged.
A release after a range is never assumed fixed: it needs its own documented evidence.

Windows (P5 decision "prior MSRC build"): builds come from MSRC data only. The fixed fixture is the
FixedBuild MSRC gives for the CVE and product; the vulnerable fixture is the highest FixedBuild MSRC lists
for the same product below it (the previous cumulative update in the snapshot).
"""

from __future__ import annotations

import html
import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime

from ahg.cvedata import parse_cpe, walk_matches

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}")

# ---------------------------------------------------------------- versions

# Words that mark a pre-release. They sort *before* the release they precede (11.0.0-M1 < 11.0.0), unlike
# other letters, which upstream projects use for post-releases (ProFTPD 1.3.8b, OpenSSH 9.6p1, BIND 9.11.3-P1).
PRE_WORDS = {"dev": 0, "alpha": 1, "beta": 2, "m": 3, "milestone": 3, "pre": 4, "preview": 4, "rc": 5}
_END = (1,)  # end-of-version marker: above any pre-release token, below any number or post-release letter


def version_key(v: str) -> tuple:
    """Total order on version strings, close to NVD's: numbers numerically, pre-release words below the
    bare release, other letters above it."""
    toks: list[tuple] = []
    for t in re.findall(r"\d+|[a-z]+", v.lower()):
        if t.isdigit():
            toks.append((3, int(t)))
        elif t in PRE_WORDS:
            toks.append((0, PRE_WORDS[t]))
        else:
            toks.append((2, t))
    # Trailing zeros of each run of numbers carry no order ("5.10" == "5.10.0"; "9.1.0rc1" < "9.1.0").
    out: list[tuple] = []
    for t in toks + [_END]:
        if t[0] != 3:
            while len(out) > 1 and out[-1] == (3, 0) and out[-2][0] == 3:
                out.pop()
        out.append(t)
    return tuple(out)


def is_prerelease(v: str) -> bool:
    """True if the version carries a pre-release word (rc, beta, milestone, ...)."""
    return any(t in PRE_WORDS for t in re.findall(r"[a-z]+", v.lower()))


# ---------------------------------------------------------------- CVE side: affected intervals


@dataclass(frozen=True)
class Range:
    """One vulnerable CPE match of a product: an exact version or a version interval (NVD semantics:
    bounds compare the CPE version field only; the update field must match unless it is '*')."""

    criteria: str
    exact: str | None = None  # a specific CPE version; None for a '*' match with bounds
    update: str = "*"
    start_incl: str | None = None
    start_excl: str | None = None
    end_incl: str | None = None
    end_excl: str | None = None

    def contains(self, version: str, update: str) -> bool:
        """Does a release with CPE (version, update) satisfy this match?"""
        if self.update == "-" and update:
            return False
        if self.update not in ("*", "-", "") and update.lower() != self.update.lower():
            return False
        k = version_key(version)
        if self.exact is not None:
            return k == version_key(self.exact)
        if self.start_incl is not None and k < version_key(self.start_incl):
            return False
        if self.start_excl is not None and k <= version_key(self.start_excl):
            return False
        if self.end_incl is not None and k > version_key(self.end_incl):
            return False
        if self.end_excl is not None and k >= version_key(self.end_excl):
            return False
        return True

    def label(self) -> str:
        """Readable form, e.g. '[9.0.0, 9.0.83)' or '=2.4.49'."""
        if self.exact is not None:
            return f"={self.exact}" + (f":{self.update}" if self.update not in ("*", "-") else "")
        lo = f"[{self.start_incl}" if self.start_incl else (f"({self.start_excl}" if self.start_excl else "(*")
        hi = f"{self.end_incl}]" if self.end_incl else (f"{self.end_excl})" if self.end_excl else "*)")
        return f"{lo}, {hi}"


def cve_ranges(configurations: list[dict], product: str, part: str) -> tuple[list[Range], list[str]]:
    """The product's affected intervals from one CVE's full CPE applicability criteria.

    Only vulnerable, non-negated matches of vendor:product with the expected CPE part count (F6; platform
    matches are co-conditions and never create a vulnerability, Round 3). Identical matches across
    configurations are merged. Returns (ranges, notes); a match with version '-' (not applicable) or a
    '*' match without bounds cannot give an interval and is reported in notes, never guessed.
    """
    vendor, prod = product.split(":")
    out: dict[tuple, Range] = {}
    notes: list[str] = []
    for _i, m, negated in walk_matches(configurations):
        if negated or not m.get("vulnerable"):
            continue
        c = parse_cpe(m["criteria"])
        if (c["part"], c["vendor"], c["product"]) != (part, vendor, prod):
            continue
        bounds = {k: m.get(k) for k in ("versionStartIncluding", "versionStartExcluding", "versionEndIncluding",
                                        "versionEndExcluding")}  # fmt: skip
        if c["version"] == "-":
            notes.append(f"version '-' match skipped: {m['criteria']}")
            continue
        if c["version"] == "*" and not any(bounds.values()):
            notes.append(f"unbounded '*' match skipped: {m['criteria']}")
            continue
        exact = None if c["version"] == "*" else c["version"].replace("\\", "")
        r = Range(m["criteria"], exact, c["update"], bounds["versionStartIncluding"], bounds["versionStartExcluding"],
                  bounds["versionEndIncluding"], bounds["versionEndExcluding"])  # fmt: skip
        out.setdefault((r.exact, r.update, r.start_incl, r.start_excl, r.end_incl, r.end_excl), r)
    return list(out.values()), notes


def candidate_units(rows, stages: tuple[str, ...] = ("remote_candidate", "local_candidate"),
                    include: set[tuple[str, str]] | frozenset = frozenset()) -> dict[str, dict]:
    """Candidate units per product: units at a pre-P3 candidate stage, in a cohort, with an explicit v3.x
    AC, plus every (CVE, product) unit in `include`.

    P5 decisions (16 Nov 2026): the catalogue covers the pre-P3 superset and every entry keeps its CVE, so
    the eligible catalogue is a filter of it. `include` adds the eligible units whose master stage is not a
    candidate stage (P3 re-decided their role), so the eligible view misses none.
    Returns {product: {"os", "families", "cves": {cve: {"cohort", "stage", "family", "row": row}}}}.
    """
    out: dict[str, dict] = {}
    for r in rows:
        for u in r["units"]:
            forced = (r["cve_id"], u["product"]) in include
            if not forced and (u["stage"] not in stages or not u["cohort"] or u.get("ac_status") != "explicit"):
                continue
            p = out.setdefault(u["product"], {"os": u["os"], "families": set(), "cves": {}})
            p["families"].add(u["family"])
            p["cves"][r["cve_id"]] = {"cohort": u["cohort"], "stage": u["stage"], "family": u["family"], "row": r}
    return out


# ---------------------------------------------------------------- releases


@dataclass(frozen=True)
class Release:
    """One upstream release: the published version text, its CPE (version, update) and where it was seen."""

    name: str
    cpe_version: str
    cpe_update: str
    date: str | None
    raw_name: str  # raw file that lists it
    entry: str  # the listing entry or line that names it

    @property
    def key(self) -> tuple:
        return version_key(self.name)


def html_text_lines(text: str) -> list[str]:
    """Visible text of an HTML (or plain-text) page, one block element per line, whitespace collapsed."""
    if re.search(r"<(?:html|body|div|p|table|pre)\b", text[:20000], re.I):
        text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
        text = re.sub(r"(?i)<br\s*/?>|</(?:p|div|li|tr|h\d|dt|dd|table|td|th|pre|section|article)>|<(?:p|div|li|tr|h\d|dt|dd|td|th)\b[^>]*>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return [line for line in lines if line]


_DATE_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%d/%b/%Y", "%Y-%m-%d %H:%M", "%d-%b-%Y %H:%M", "%d %B %Y", "%B %d, %Y")
_LISTING_DATE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{2}-[A-Z][a-z]{2}-\d{4})[ T]+\d{2}:\d{2}")


def parse_date(text: str, fmt: str | None = None) -> str:
    """ISO date (YYYY-MM-DD) from a page's date text; raises ValueError if no known format fits."""
    text = text.strip()
    if fmt == "ymd_loose":  # 2024-8-21 (Xlight history page)
        y, m, d = (int(x) for x in text.split("-"))
        return datetime(y, m, d).date().isoformat()
    for f in ([fmt] if fmt else _DATE_FORMATS):
        try:
            return datetime.strptime(text, f).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unknown date format: {text!r}")


def cpe_split(name: str, split: str | None) -> tuple[str, str]:
    """(CPE version, CPE update) of a published version, e.g. '9.6p1' -> ('9.6', 'p1') for OpenSSH."""
    if not split:
        return name, ""
    m = re.fullmatch(split, name)
    if not m:
        raise ValueError(f"release {name!r} does not fit cpe_split {split!r}")
    return m["version"], (m.groupdict().get("update") or "").lower()


def parse_listing(page: str, pattern: str) -> list[tuple[str, str | None, str]]:
    """Files or folders of a web server directory listing: (version, date, entry) for every link whose
    target fits `pattern` (named group `name`). The date is the first listing timestamp after the link,
    before the next link (Apache, nginx and lighttpd index formats all put it there)."""
    rx = re.compile(pattern)
    links = list(re.finditer(r'<a\s+href="([^"]+)"', page, re.I))
    out = []
    for i, a in enumerate(links):
        target = html.unescape(a.group(1))
        m = rx.fullmatch(target)
        if not m:
            continue
        seg = page[a.end(): links[i + 1].start() if i + 1 < len(links) else len(page)]
        d = _LISTING_DATE.search(seg)
        out.append((m["name"], parse_date(d.group(1)) if d else None, target))
    return out


def parse_text_releases(page: str, pattern: str, date_format: str | None) -> list[tuple[str, str | None, str]]:
    """Releases named on a vendor's version-history page: `pattern` (groups `name`, optional `date`) is
    matched against each visible text line."""
    rx = re.compile(pattern)
    out = []
    for line in html_text_lines(page):
        m = rx.search(line)
        if m:
            d = m.groupdict().get("date")
            out.append((m["name"], parse_date(d, date_format) if d else None, line[:200]))
    return out


def parse_github_releases(page: str, pattern: str) -> list[tuple[str, str | None, str]]:
    """Published, non-draft, non-prerelease GitHub releases whose tag fits `pattern` (group `name`)."""
    rx = re.compile(pattern)
    out = []
    for r in json.loads(page):
        m = rx.fullmatch(r.get("tag_name") or "")
        if m and not r.get("draft") and not r.get("prerelease"):
            out.append((m["name"], (r.get("published_at") or "")[:10] or None, r["tag_name"]))
    return out


def build_releases(found: list[tuple[str, str | None, str, str]], split: str | None) -> tuple[list[Release], list[str]]:
    """Stable releases (one per version, earliest date kept) sorted by version, plus the pre-release names
    excluded (P5 decision: stable releases only). `found` holds (name, date, entry, raw_name)."""
    by_name: dict[str, Release] = {}
    pre: set[str] = set()
    for name, date, entry, raw in found:
        if is_prerelease(name):
            pre.add(name)
            continue
        v, u = cpe_split(name, split)
        old = by_name.get(name)
        if old is None or (date and (old.date is None or date < old.date)):
            by_name[name] = Release(name, v, u, date, raw, entry)
    return sorted(by_name.values(), key=lambda r: r.key), sorted(pre)


# ---------------------------------------------------------------- evidence of fixed releases


@dataclass(frozen=True)
class FixDoc:
    """A vendor statement that `version` fixes (or is not affected by) `cve`."""

    cve: str
    version: str
    raw_name: str
    kind: str
    excerpt: str


def versions_in(text: str, version_regex: str) -> list[str]:
    """Every version string in a piece of evidence text, in order, without duplicates."""
    seen: dict[str, None] = {}
    for m in re.finditer(version_regex, text):
        seen.setdefault(m.group(0), None)
    return list(seen)


def ev_sections(page: str, header: str, version_regex: str, raw_name: str, cve_at_line_start: bool = False) -> list[FixDoc]:
    """Release notes or advisory pages grouped under version headers ("Fixed in Apache Tomcat 9.0.83",
    "version 2.90", "OpenSSH 9.6/9.6p1 (2023-12-18)"): a CVE ID inside a section is documented as fixed in
    the header's version(s). With cve_at_line_start, only CVE IDs that open a line count (pages whose
    entries start with the CVE, where IDs inside prose refer to other issues)."""
    rx = re.compile(header)
    out: list[FixDoc] = []
    current: list[str] = []
    head = ""
    for line in html_text_lines(page):
        hm = rx.search(line)
        if hm:
            # The header's version text: group `v` when the pattern has one, else the whole match.
            current, head = versions_in(hm.groupdict().get("v") or hm.group(0), version_regex), line
            continue
        if not current:
            continue
        cves = [line[: m.end()] for m in [CVE_RE.match(line)] if m] if cve_at_line_start else CVE_RE.findall(line)
        for cve in dict.fromkeys(cves):
            for v in current:
                out.append(FixDoc(cve, v, raw_name, "sections", f"{head[:120]} | {line[:160]}"))
    return out


def ev_page_regex(page: str, pattern: str, version_regex: str, raw_name: str, page_cve: str | None = None,
                  flatten: bool = False) -> list[FixDoc]:
    """Advisory text matched by `pattern` (groups `v` and optionally `cves`) over the page's visible text,
    lines joined by newlines (or by spaces with flatten). The CVEs are those in group `cves`, else the CVE
    the page is about (per-CVE pages: ISC KB, Samba)."""
    text = ("\n" if not flatten else " ").join(html_text_lines(page)) + "\n"
    out = []
    for m in re.finditer(pattern, text):
        cves = CVE_RE.findall(m.groupdict().get("cves") or "") or ([page_cve] if page_cve else [])
        for cve in dict.fromkeys(cves):
            for v in versions_in(m["v"], version_regex):
                out.append(FixDoc(cve, v, raw_name, "page_regex", re.sub(r"\s+", " ", m.group(0))[:240]))
    return out


def html_tables(page: str) -> list[list[list[str]]]:
    """Every <table> as rows of cell texts."""
    tables = []
    for t in re.findall(r"(?is)<table\b.*?</table>", page):
        rows = []
        for tr in re.findall(r"(?is)<tr\b.*?</tr>", t):
            cells = re.findall(r"(?is)<t[hd]\b[^>]*>(.*?)</t[hd]>", tr)
            rows.append([re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip() for c in cells])
        tables.append(rows)
    return tables


def ev_table(page: str, column: str, version_regex: str, raw_name: str) -> list[FixDoc]:
    """Tables whose header names fixed versions ("Fixed" on the PostgreSQL page; the branch columns of
    ISC's BIND matrix, whose cells list the fixing release): a row's CVEs are fixed in the versions of the
    columns whose header fits `column`. Header rows are the rows with no CVE ID that precede data rows."""
    rx = re.compile(column)
    out = []
    for rows in html_tables(page):
        header: list[str] = []
        for cells in rows:
            cves = CVE_RE.findall(" ".join(cells))
            if not cves:
                if cells and any(rx.fullmatch(c) for c in cells):
                    header = cells
                continue
            for i, h in enumerate(header):
                if i < len(cells) and rx.fullmatch(h):
                    for cve in dict.fromkeys(cves):
                        for v in versions_in(cells[i], version_regex):
                            out.append(FixDoc(cve, v, raw_name, "table", f"{h}: {cells[i][:120]}"))
    return out


def ev_ghsa_repo(page: str, version_regex: str, raw_name: str) -> list[FixDoc]:
    """A project's own GitHub repository security advisories (authored by the maintainers): every
    `patched_versions` entry is a documented fixed release for the advisory's CVE."""
    out = []
    for adv in json.loads(page):
        cve = adv.get("cve_id")
        if not cve or adv.get("withdrawn_at"):
            continue
        for vuln in adv.get("vulnerabilities") or []:
            for v in versions_in(vuln.get("patched_versions") or "", version_regex):
                out.append(FixDoc(cve, v, raw_name, "ghsa_repo", f"{adv.get('ghsa_id')} patched_versions={vuln.get('patched_versions')}"))
    return out


def ev_cve_json5(record: dict, vendor: dict, product_regex: str, raw_name: str) -> tuple[list[FixDoc], str | None]:
    """Version entries with status "unaffected" in a CVE JSON 5 record whose CNA is the vendor.

    Returns (docs, refusal). The record counts only if its CNA container's provider matches the configured
    vendor CNA: `vendor` names a "short_name" (records from the CVE Services API) or an "org_id" (the
    kernel CNA's own repository copies carry only the orgId). A "lessThan"/"lessThanOrEqual" bound on an
    affected entry is never read as a fix (P5 decision). Only affected[] entries whose product fits
    `product_regex` are read.
    """
    cna = record.get("containers", {}).get("cna", {})
    meta = cna.get("providerMetadata", {})
    cve = record.get("cveMetadata", {}).get("cveId")
    field_ = "short_name" if "short_name" in vendor else "org_id"
    got = meta.get("shortName" if field_ == "short_name" else "orgId")
    if got != vendor[field_]:
        return [], f"CNA {field_} {got!r} is not the vendor CNA {vendor[field_]!r}"
    out = []
    for aff in cna.get("affected", []):
        if not re.fullmatch(product_regex, aff.get("product") or ""):
            continue
        for v in aff.get("versions", []):
            if v.get("status") == "unaffected" and v.get("versionType") != "git" and v.get("version"):
                out.append(FixDoc(cve, v["version"], raw_name, "cve_json5",
                                  json.dumps({k: v[k] for k in sorted(v)}, sort_keys=True)[:200]))
    return out, None


# ---------------------------------------------------------------- catalogue (per product)


class ReleaseIndex:
    """Releases sorted by version, with fast range lookup.

    A CPE interval is contiguous in version order, so the releases inside it are found by bisection over
    the CPE version keys (the kernel has thousands of releases and ~11k CVEs; testing every pair is too
    slow). Releases are sorted by published name; their CPE version keys must then be non-decreasing,
    which the constructor checks.
    """

    def __init__(self, releases: list[Release]):
        self.releases = sorted(releases, key=lambda r: r.key)
        self.ckeys = [version_key(r.cpe_version) for r in self.releases]
        if any(a > b for a, b in zip(self.ckeys, self.ckeys[1:])):
            raise ValueError("release names and CPE versions are ordered differently; check cpe_split")
        self.pos = {r.name: i for i, r in enumerate(self.releases)}
        self._by_name = {r.name.lower(): r for r in self.releases}
        self._by_cpe: dict[tuple, list[Release]] = {}
        for r, k in zip(self.releases, self.ckeys):
            self._by_cpe.setdefault(k, []).append(r)

    def match(self, text: str) -> Release | None:
        """The release a documented version names: exact published name (case-insensitive), else the
        single release whose CPE version equals it (OpenSSH '9.6' -> 9.6p1). None if absent or ambiguous."""
        hit = self._by_name.get(text.lower())
        if hit is not None:
            return hit
        same = self._by_cpe.get(version_key(text), [])
        return same[0] if len(same) == 1 else None

    def inside(self, rg: Range) -> list[int]:
        """Indices of the releases that satisfy the match, ascending."""
        k = self.ckeys
        if rg.exact is not None:
            lo, hi = bisect_left(k, version_key(rg.exact)), bisect_right(k, version_key(rg.exact))
        else:
            lo, hi = 0, len(k)
            if rg.start_incl is not None:
                lo = max(lo, bisect_left(k, version_key(rg.start_incl)))
            if rg.start_excl is not None:
                lo = max(lo, bisect_right(k, version_key(rg.start_excl)))
            if rg.end_incl is not None:
                hi = min(hi, bisect_right(k, version_key(rg.end_incl)))
            if rg.end_excl is not None:
                hi = min(hi, bisect_left(k, version_key(rg.end_excl)))
        # contains() re-checks each hit: the update field is not part of the bisection.
        return [i for i in range(lo, hi) if rg.contains(self.releases[i].cpe_version, self.releases[i].cpe_update)]


@dataclass
class ProductCatalogue:
    """Fixtures of one product, keyed by release name, plus every dropped (CVE, range) with its reason."""

    fixtures: dict[str, dict] = field(default_factory=dict)
    dropped: list[dict] = field(default_factory=list)
    unmatched_docs: list[dict] = field(default_factory=list)


def _fixture(cat: ProductCatalogue, rel: Release) -> dict:
    return cat.fixtures.setdefault(rel.name, {
        "version": rel.name, "cpe_version": rel.cpe_version, "cpe_update": rel.cpe_update,
        "release_date": rel.date, "release_raw": rel.raw_name, "release_entry": rel.entry,
        "vulnerable_for": [], "fixed_for": [],
    })  # fmt: skip


def runs(indices: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive integers in a sorted list: [1, 2, 3, 7, 8] -> [(1, 3), (7, 8)]."""
    out: list[tuple[int, int]] = []
    for i in indices:
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def catalogue_product(cves: dict[str, list[Range]], index: ReleaseIndex, docs: list[FixDoc]) -> ProductCatalogue:
    """Apply the catalogue rule per (CVE, affected interval) for one product.

    Affected intervals (P5 decision 16 Nov 2026, "maximal affected runs"): the maximal runs of consecutive
    stable releases, in version order, that the CVE's CPE matches cover. Branch ranges stay separate,
    because unaffected releases lie between them; versions that NVD enumerates one by one collapse into
    their run. A CPE match that covers no stable release is dropped and logged.

    Vulnerable fixture V: the last release of the run. Fixed fixture F: the first release after V that is
    in none of the CVE's affected ranges and that the vendor documents as fixed for this CVE, provided no
    affected release lies between V and F, so F continues V's line of releases rather than following a
    later affected run ("that range's documented fix", P5 decision). A documented release that NVD lists
    as affected is a vendor/NVD conflict: logged, never used.
    """
    cat = ProductCatalogue()
    rels = index.releases
    by_cve: dict[str, list[FixDoc]] = {}
    for d in docs:
        by_cve.setdefault(d.cve, []).append(d)
    for cve in sorted(cves, key=cve_sort_key):
        ranges = cves[cve]
        if not ranges:
            cat.dropped.append({"cve": cve, "interval": None, "reason": "no_vulnerable_cpe_range"})
            continue
        hits = [index.inside(rg) for rg in ranges]
        for rg, h in zip(ranges, hits):
            if not h:
                cat.dropped.append({"cve": cve, "interval": None, "cpe_match": rg.label(), "reason": "no_stable_release_in_cpe_match"})
        affected = sorted({i for h in hits for i in h})
        affected_set = set(affected)
        fixed: dict[int, list[FixDoc]] = {}
        for d in by_cve.get(cve, []):
            rel = index.match(d.version)
            if rel is None:
                cat.unmatched_docs.append({"cve": cve, "version": d.version, "raw": d.raw_name, "kind": d.kind,
                                           "reason": "documented_fix_not_a_stable_release"})  # fmt: skip
            else:
                fixed.setdefault(index.pos[rel.name], []).append(d)
        conflicts = sorted(rels[j].name for j in fixed if j in affected_set)
        clean_fixed = sorted(j for j in fixed if j not in affected_set)
        for a, b in runs(affected):
            interval = f"{rels[a].name}..{rels[b].name}"
            v = rels[b]
            _fixture(cat, v)["vulnerable_for"].append({"cve": cve, "interval": interval})
            k = bisect_right(clean_fixed, b)
            fj = clean_fixed[k] if k < len(clean_fixed) else None
            # No affected release strictly between V and F (affected is sorted: count by bisection).
            if fj is not None and bisect_left(affected, fj) - bisect_right(affected, b) > 0:
                fj = None
            if fj is None:
                cat.dropped.append({"cve": cve, "interval": interval, "vulnerable": v.name,
                                    "reason": "no_documented_fix_after_vulnerable_fixture",
                                    **({"conflicting_docs": conflicts} if conflicts else {})})  # fmt: skip
                continue
            ev = [{"raw": d.raw_name, "kind": d.kind, "excerpt": d.excerpt} for d in fixed[fj]]
            _fixture(cat, rels[fj])["fixed_for"].append({"cve": cve, "interval": interval, "after": v.name, "evidence": ev})
    return cat


def affected_by(fixture_names: list[str], index: ReleaseIndex, cves: dict[str, list[Range]]) -> dict[str, list[str]]:
    """For each fixture, every candidate CVE of the product whose affected ranges contain it.

    Status per cohort (Round 4) is derived from this later.
    """
    wanted = {index.pos[n] for n in fixture_names}
    out: dict[str, set[str]] = {n: set() for n in fixture_names}
    for cve, ranges in cves.items():
        for rg in ranges:
            for i in index.inside(rg):
                if i in wanted:
                    out[index.releases[i].name].add(cve)
    return {n: sorted(s, key=cve_sort_key) for n, s in out.items()}


def nvd_match_key(m: dict) -> tuple:
    """Identity of one raw NVD CPE match: its criteria string and its four bounds."""
    return (m["criteria"], m.get("versionStartIncluding"), m.get("versionStartExcluding"),
            m.get("versionEndIncluding"), m.get("versionEndExcluding"))  # fmt: skip


def _range_key(rg: Range) -> tuple:
    return (rg.criteria, rg.start_incl, rg.start_excl, rg.end_incl, rg.end_excl)


def unverified_fixtures(cves: dict[str, list[Range]], nvd_matches: dict[str, set[tuple]]) -> tuple[list[dict], list[dict]]:
    """Vulnerable fixtures from version strings NVD itself lists as affected, for products with no verified
    fixture.

    Decisions 16 Nov 2026 (Devansh, after P5; rule 13): such fixtures carry "unverified_release": no release
    date, no upstream validation, vulnerable role only, never a fixed fixture; usable in the primary (p = 1)
    benchmark, not in p < 1 cohorts. Allowed strings (never an invented version):
    - the version (and update) field of a vulnerable CPE match (first decision);
    - its inclusive bounds, versionStartIncluding and versionEndIncluding (OQ-P5-11). Exclusive bounds are
      never used: NVD states that version is not affected.
    The match, bounds included, must also appear in the CVE's raw NVD record (`nvd_matches`, keys from
    nvd_match_key), so every string is one NVD lists. Per CVE, the fixture is the highest stable such
    string (OQ-P5-12; Round 4 takes the latest affected release). Returns (fixtures, dropped).
    """
    fixtures: dict[str, dict] = {}
    dropped: list[dict] = []
    for cve in sorted(cves, key=cve_sort_key):
        def options(rgs: list[Range]) -> list[tuple[str, str, str, Range]]:
            out = []
            for rg in rgs:
                upd = _update(rg.update)
                for v, kind in ((rg.exact, "exact"), (rg.start_incl, "versionStartIncluding"), (rg.end_incl, "versionEndIncluding")):
                    if v is not None:
                        out.append((v.replace("\\", ""), upd, kind, rg))
            return out

        allowed = options(cves[cve])
        listed = options([rg for rg in cves[cve] if _range_key(rg) in nvd_matches.get(cve, set())])
        stable = [o for o in listed if not is_prerelease(o[0]) and not is_prerelease(o[1])]
        if not stable:
            reason = ("no_exact_or_inclusive_version_in_cpe_matches" if not allowed else
                      "cpe_match_not_in_raw_nvd_record" if not listed else "only_prerelease_versions")
            dropped.append({"cve": cve, "interval": None, "reason": reason,
                            "cpe_matches": [rg.label() for rg in cves[cve]]})  # fmt: skip
            continue
        version, upd, kind, rg = max(stable, key=lambda o: version_key(f"{o[0]} {o[1]}"))
        name = f"{version}:{upd}" if upd else version
        f = fixtures.setdefault(name, {"version": name, "cpe_version": version, "cpe_update": upd, "release_date": None,
                                       "unverified_release": True, "vulnerable_for": [], "fixed_for": []})  # fmt: skip
        f["vulnerable_for"].append({"cve": cve, "interval": rg.label(), "cpe_criteria": rg.criteria, "version_from": kind})
    for f in fixtures.values():
        f["affected_by"] = sorted((c for c, rgs in cves.items() if any(rg.contains(f["cpe_version"], f["cpe_update"]) for rg in rgs)),
                                  key=cve_sort_key)  # fmt: skip
    return sorted(fixtures.values(), key=lambda f: version_key(f["version"])), dropped


def _update(u: str) -> str:
    """CPE update field as a release qualifier: '*', '-' and '' carry none."""
    return "" if u in ("*", "-", "") else u.replace("\\", "")


def cve_sort_key(cve: str) -> tuple[int, int]:
    """Numeric CVE order (D28 Round 5: year, then sequence number, as integers)."""
    _, y, n = cve.split("-")
    return int(y), int(n)


# ---------------------------------------------------------------- Windows (MSRC builds)


def msrc_product_names(unit_product: str, names: list[str], rules: dict) -> list[str]:
    """MSRC product names that host a Windows unit: the configured Windows Server versions for OS-level
    units, or names with a configured prefix (Exchange, SQL Server)."""
    rule = rules.get(unit_product, rules["default"])
    if "names" in rule:
        return [n for n in names if n in rule["names"]]
    return [n for n in names if n.startswith(rule["prefix"])]


def windows_builds(rows) -> dict[str, dict[str, dict]]:
    """Every FixedBuild MSRC lists, per product name: {name: {build: {"date", "cves", "docs"}}}.

    The date of a build is the earliest initial release of a CVE fixed in it (when MSRC published it).
    """
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        m = r.get("msrc")
        if not m:
            continue
        for p in m["products"].values():
            for fx in p.get("fixes", []):
                b = fx.get("fixed_build")
                if not b or not re.fullmatch(r"\d+(?:\.\d+)+", b):
                    continue
                e = out.setdefault(p["name"], {}).setdefault(b, {"date": None, "cves": set(), "docs": set()})
                e["cves"].add(r["cve_id"])
                e["docs"].update(m.get("docs") or [m.get("doc")])
                d = (m.get("initial_release") or "")[:10] or None
                if d and (e["date"] is None or d < e["date"]):
                    e["date"] = d
    return out


def windows_fixtures(row: dict, product_names: list[str], builds: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    """Vulnerable and fixed builds of one CVE on each hosting MSRC product (P5 decision "prior MSRC build").

    Returns (pairs, dropped); a pair is {"product", "fixed", "vulnerable"}. Several FixedBuilds for one
    product (different packages) use the lowest, the first build MSRC documents as fixed.
    """
    pairs, dropped = [], []
    for p in row["msrc"]["products"].values():
        if p["name"] not in product_names:
            continue
        fixed = sorted({fx["fixed_build"] for fx in p.get("fixes", []) if fx.get("fixed_build")
                        and re.fullmatch(r"\d+(?:\.\d+)+", fx["fixed_build"])}, key=version_key)  # fmt: skip
        if not fixed:
            dropped.append({"cve": row["cve_id"], "product": p["name"], "reason": "no_msrc_fixed_build"})
            continue
        f = fixed[0]
        prior = [b for b in builds.get(p["name"], {}) if version_key(b) < version_key(f)]
        if not prior:
            dropped.append({"cve": row["cve_id"], "product": p["name"], "fixed": f, "reason": "no_prior_msrc_build"})
            continue
        pairs.append({"product": p["name"], "fixed": f, "vulnerable": max(prior, key=version_key)})
    return pairs, dropped


# ---------------------------------------------------------------- eligible view (P5 decision, 16 Nov 2026)


def eligible_units(label_rows) -> set[tuple[str, str]]:
    """(CVE, product) units eligible after effect classification (frozen P3 labels, e.g. p3-labels-v2.1).

    Same rule as ahg.effect.eligible on branch d28-effect (D28 E5 and OQ-P4-1), restated here because that
    module is not on this track: AC explicit; remote units RCE_CONFIRMED with a server role; local units
    ADMIN_CONFIRMED with no role filter.
    """
    out = set()
    for r in label_rows:
        if r["ac_status"] != "explicit":
            continue
        ok = (r["label"] == "RCE_CONFIRMED" and r["role"] == "server") if r["label_set"] == "remote" \
            else r["label"] == "ADMIN_CONFIRMED"
        if ok:
            out.add((r["cve_id"], r["product"]))
    return out


def _keep(items: list[dict], ok) -> list[dict]:
    return [i for i in items if ok(i)]


def eligible_view(catalogue: dict, eligible: set[tuple[str, str]]) -> dict:
    """The catalogue restricted to eligible (CVE, product) units: roles, affected-by lists and drops of other
    CVEs are removed; a fixture with no remaining role is removed; a product with no eligible CVE is left out.

    Linux entries are keyed by the unit's product. Windows entries are keyed by MSRC product and each role
    names its unit; affected-by lists keep a CVE if it is eligible for any unit hosted on that product.
    """
    out: dict = {"linux": {}, "windows": {}}
    for product, e in catalogue["linux"].items():
        cves = {c for c, p in eligible if p == product}
        if not cves:
            continue
        fixtures = []
        for f in e["fixtures"]:
            vf = _keep(f["vulnerable_for"], lambda i: i["cve"] in cves)
            ff = _keep(f["fixed_for"], lambda i: i["cve"] in cves)
            if vf or ff:
                fixtures.append({**f, "vulnerable_for": vf, "fixed_for": ff, "affected_by": [c for c in f["affected_by"] if c in cves]})
        out["linux"][product] = {**e, "candidate_cves": {c: v for c, v in (e.get("candidate_cves") or {}).items() if c in cves},
                                 "fixtures": fixtures, "status": e["status"] if fixtures else "dropped",
                                 "dropped": _keep(e["dropped"], lambda i: i["cve"] in cves),
                                 "unmatched_docs": _keep(e.get("unmatched_docs", []), lambda i: i["cve"] in cves)}  # fmt: skip
    for name, e in catalogue["windows"].items():
        units = set(e["units"])
        cves = {c for c, p in eligible if p in units}
        ok = lambda i: (i["cve"], i["unit"]) in eligible  # noqa: E731
        fixtures = []
        for f in e["fixtures"]:
            vf, ff = _keep(f["vulnerable_for"], ok), _keep(f["fixed_for"], ok)
            if vf or ff:
                fixtures.append({**f, "vulnerable_for": vf, "fixed_for": ff, "affected_by": [c for c in f["affected_by"] if c in cves]})
        dropped = _keep(e["dropped"], ok)
        if fixtures or dropped:
            out["windows"][name] = {**e, "fixtures": fixtures, "dropped": dropped, "status": "ok" if fixtures else "dropped",
                                    "affected_unknown_no_fixed_build": [c for c in e["affected_unknown_no_fixed_build"] if c in cves]}  # fmt: skip
    return out
