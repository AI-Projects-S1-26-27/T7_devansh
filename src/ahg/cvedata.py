"""D28 master CVE dataset: pure functions over NVD, MSRC and GHSA records (D28 Rounds 1-4, data side).

One row per canonical CVE ID (CV9). The row keeps every affected product, every provider's CVSS vector with
its provenance (CV1), CWE, the full CPE configurations with the part of every match (F6) and the
vulnerable:false platform conditions, dates, cohort (CV2/CV10), and for Microsoft CVEs the MSRC impact,
component tag and affected builds. Effect classification is not done here (task P3).

The functions here never fetch anything and never read files; scripts/build_cve_master.py does the I/O.
"""

import html
import re
from collections import Counter
from datetime import date

V3_KEYS = ("cvssMetricV31", "cvssMetricV30")
OTHER_KEYS = ("cvssMetricV40", "cvssMetricV2")
ROLE_FIELDS = ("AV", "PR", "UI")  # the fields CV1 resolves across providers
CPE_FIELDS = ("part", "vendor", "product", "version", "update", "edition", "language", "sw_edition", "target_sw",
              "target_hw", "other")  # fmt: skip


# ---------------------------------------------------------------- CPE (F6, platform conditions)


def parse_cpe(criteria: str) -> dict[str, str]:
    """Split a CPE 2.3 formatted string into its eleven named fields (escaped colons are kept)."""
    parts = re.split(r"(?<!\\):", criteria)
    if len(parts) != 13 or parts[:2] != ["cpe", "2.3"]:
        raise ValueError(f"not a CPE 2.3 formatted string: {criteria!r}")
    return dict(zip(CPE_FIELDS, parts[2:]))


def walk_matches(configurations: list[dict]):
    """Yield (configuration index, cpeMatch dict, negated) for every match at any depth.

    A node with negate: true inverts its contents; a match under an odd number of negations states where
    the CVE does *not* apply (AUDIT_1 D1). Nested "nodes"/"children" are visited (D4).
    """
    def walk(node, negated):
        negated ^= bool(node.get("negate"))
        for m in node.get("cpeMatch", []):
            yield m, negated
        for child in [*node.get("nodes", []), *node.get("children", [])]:
            yield from walk(child, negated)

    for i, cfg in enumerate(configurations):
        for node in cfg.get("nodes", []):
            for m, neg in walk(node, bool(cfg.get("negate"))):
                yield i, m, neg


def cpe_matches(configurations: list[dict]) -> list[dict]:
    """Every cpeMatch flattened, with its parsed part/vendor/product and its role in the statement.

    role: "vulnerable" (vulnerable: true, not negated), "platform" (vulnerable: false, not negated: an
    applicability co-condition that never makes the platform vulnerable, D28 Round 3) or "negated".
    """
    out = []
    for i, m, neg in walk_matches(configurations):
        c = parse_cpe(m["criteria"])
        role = "negated" if neg else ("vulnerable" if m.get("vulnerable") else "platform")
        rng = {k: m[k] for k in ("versionStartIncluding", "versionStartExcluding", "versionEndIncluding",
                                 "versionEndExcluding") if k in m}  # fmt: skip
        out.append({"configuration": i, "role": role, "criteria": m["criteria"], "part": c["part"],
                    "vendor": c["vendor"], "product": c["product"], "version": c["version"], **rng})  # fmt: skip
    return out


def affected_products(matches: list[dict]) -> list[str]:
    """Sorted distinct part:vendor:product of every vulnerable match (CV9: all affected products)."""
    return sorted({f"{m['part']}:{m['vendor']}:{m['product']}" for m in matches if m["role"] == "vulnerable"})


def product_hits(matches: list[dict], product: str, part: str) -> tuple[bool, list[str]]:
    """(vulnerable with the expected CPE part, parts seen on vulnerable matches of vendor:product).

    F6: a match counts only if its part (a/o/h) is the expected one; other parts are reported, never used.
    """
    vendor, prod = product.split(":")
    parts = sorted({m["part"] for m in matches if m["role"] == "vulnerable" and (m["vendor"], m["product"]) == (vendor, prod)})
    return part in parts, parts


# ---------------------------------------------------------------- CVSS vectors (CV1)


def parse_vector(vector: str) -> dict[str, str]:
    """Metrics of a CVSS vector string, e.g. 'CVSS:3.1/AV:N/AC:L/...' -> {'version': '3.1', 'AV': 'N', ...}.

    v2 vectors have no prefix ('AV:N/AC:L/Au:N/...'); version is then '2.0'.
    """
    fields = vector.split("/")
    out = {}
    if fields[0].startswith("CVSS:"):
        out["version"] = fields.pop(0)[5:]
    else:
        out["version"] = "2.0"
    for f in fields:
        k, sep, v = f.partition(":")
        if not sep or k in out:
            raise ValueError(f"malformed CVSS vector {vector!r}")
        out[k] = v
    return out


def nvd_vectors(cve: dict) -> list[dict]:
    """Every CVSS metric NVD carries for the CVE (all versions, all providers), with provenance."""
    out = []
    for key in (*V3_KEYS, *OTHER_KEYS):
        for m in cve.get("metrics", {}).get(key, []):
            d = m["cvssData"]
            out.append({"origin": "nvd", "source": m.get("source"), "type": m.get("type"), "version": d["version"],
                        "vector": d["vectorString"], "base_score": d.get("baseScore"),
                        "metrics": parse_vector(d["vectorString"])})  # fmt: skip
    return out


def resolve_vector(vectors: list[dict]) -> dict:
    """CV1 resolution of AV/PR/UI over every provider's v3.x vector.

    status:
      AGREED: every v3 vector gives the same AV, PR and UI; values holds them.
      REVIEW_REQUIRED: some field has several values; candidates lists the values per field, and
        conflict_fields names the fields that differ. No provider is preferred (D28 E2: no precedence).
      NO_V3: no v3.x vector from any provider (v2/v4 are stored but not resolved here; see OQ-P2-1).
    Vectors given here must already be restricted to the products in scope (the scope comparison):
    see msrc_vectors_in_scope.
    """
    v3 = [v for v in vectors if v["version"].startswith("3.")]
    if not v3:
        return {"status": "NO_V3", "values": {}, "candidates": {}, "conflict_fields": [], "n_v3": 0}
    cand = {f: sorted({v["metrics"][f] for v in v3}) for f in ROLE_FIELDS}
    conflict = [f for f in ROLE_FIELDS if len(cand[f]) > 1]
    scope = sorted({v["metrics"].get("S", "") for v in v3})
    return {"status": "REVIEW_REQUIRED" if conflict else "AGREED",
            "values": {} if conflict else {f: cand[f][0] for f in ROLE_FIELDS},
            "candidates": cand, "conflict_fields": conflict, "n_v3": len(v3),
            "providers": sorted({f"{v['origin']}:{v['source']}" for v in v3}),
            "cvss_scope_values": scope}  # fmt: skip


# ---------------------------------------------------------------- MSRC CVRF


def note_text(value: str | None) -> str:
    """Plain text of an MSRC note (HTML fragment)."""
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()


def msrc_entries(doc: dict, doc_id: str) -> list[dict]:
    """One dict per CVE in a CVRF document: title, tags, FAQ text, per-product impact, severity, vectors, builds."""
    names = {p["ProductID"]: p["Value"] for p in doc.get("ProductTree", {}).get("FullProductName", [])}
    out = []
    for v in doc.get("Vulnerability") or []:
        products: dict[str, dict] = {}
        for s in v.get("ProductStatuses", []):
            for pid in s.get("ProductID", []):
                products.setdefault(pid, {"name": names.get(pid, ""), "status_type": s.get("Type")})
        for t in v.get("Threats", []):
            key = {0: "impact", 3: "severity"}.get(t.get("Type"))
            for pid in t.get("ProductID", []) if key else []:
                products.setdefault(pid, {"name": names.get(pid, "")})[key] = (t.get("Description") or {}).get("Value")
        for r in v.get("Remediations", []):
            if r.get("Type") == 2:  # vendor fix
                for pid in r.get("ProductID", []):
                    p = products.setdefault(pid, {"name": names.get(pid, "")})
                    p.setdefault("fixes", []).append({"fixed_build": r.get("FixedBuild"), "kb": (r.get("Description") or {}).get("Value"),
                                                      "subtype": r.get("SubType"), "url": r.get("URL")})  # fmt: skip
        vectors = [{"origin": "msrc", "source": "msrc-cvrf", "type": None, "vector": c["Vector"],
                    "version": parse_vector(c["Vector"])["version"], "base_score": c.get("BaseScore"),
                    "metrics": parse_vector(c["Vector"]), "product_ids": sorted(c.get("ProductID", []))}
                   for c in v.get("CVSSScoreSets", []) if c.get("Vector")]  # fmt: skip
        notes = v.get("Notes", [])
        revs = [r.get("Date") for r in v.get("RevisionHistory", []) if r.get("Date")]
        first = [r["Date"] for r in v.get("RevisionHistory", []) if r.get("Number") in ("1.0", "1") and r.get("Date")]
        out.append({
            "cve_id": v.get("CVE"), "doc": doc_id, "title": (v.get("Title") or {}).get("Value"),
            "tags": sorted({n.get("Value") for n in notes if n.get("Type") == 7 and n.get("Value")}),
            "faq": " ".join(note_text(n.get("Value")) for n in notes if n.get("Type") == 4),
            "description": " ".join(note_text(n.get("Value")) for n in notes if n.get("Type") == 2),
            "cna_note": sorted({n.get("Value") for n in notes if n.get("Type") == 8 and n.get("Value")}),
            "cwe": sorted({c.get("ID") for c in v.get("CWE", []) or [] if c.get("ID")}),
            # ReleaseDate is often unset ("0001-01-01", ReleaseDateSpecified false); revision 1.0 is the
            # first publication, else the earliest revision.
            "release_date": v.get("ReleaseDate") if v.get("ReleaseDateSpecified") else None,
            "initial_release": min(first or revs or [None], key=lambda d: d or ""),
            "revision_dates": sorted(revs),
            "products": dict(sorted(products.items())), "vectors": vectors,
        })  # fmt: skip
    return out


def msrc_vectors_in_scope(entry: dict, product_ids: set[str]) -> list[dict]:
    """The MSRC vectors that score at least one in-scope product (scope comparison, CV1).

    MSRC scores per product; a vector given only for products outside the scope (e.g. a client SKU) says
    nothing about the in-scope component and must not create a conflict.
    """
    return [v for v in entry["vectors"] if set(v["product_ids"]) & product_ids]


# ---------------------------------------------------------------- cohorts


def cohort(published: str, cohorts: dict[str, list[str]]) -> str | None:
    """Name of the cohort whose closed date range holds the publication date, else None."""
    d = date.fromisoformat(published[:10])
    for name, (lo, hi) in cohorts.items():
        if date.fromisoformat(lo) <= d <= date.fromisoformat(hi):
            return name
    return None


# ---------------------------------------------------------------- role rule (E5 / CV5 / CV11)

_NEG = re.compile(r"[^.;]*\b(?:not|never|un)\s*(?:be\s+)?(?:affected|vulnerable|impacted|exploitable)\b[^.;]*", re.I)
_CLIENT = [
    ("clients only", r"\bclients?\s+only\b"),
    ("client-side", r"\bclient[- ]side\b"),
    ("in the client", r"\b(?:in|affects?|affecting)\s+(?:the\s+)?(?:(?!allows?\b|permits?\b|lets?\b)[\w.-]+\s+){0,2}clients?\b(?!\s+(?:to|can|could|may|might|that|who|with)\b)"),
    ("client product", r"\bvulnerability in the [\w ]*client product\b"),
    ("attacker-controlled server", r"\b(?:malicious|rogue|compromised|attacker[- ]controlled|attacker's|fake)\s+(?:[\w.]+\s+){0,1}servers?\b"),
    ("connects to a malicious host", r"\bconnect(?:s|ing|ed)?\s+to\s+(?:a|an)\s+(?:malicious|rogue|compromised|attacker|specially crafted|crafted)"),
    ("server sends crafted data", r"\b(?:a|the)\s+(?:malicious\s+)?(?:server|peer)\s+(?:could|can|may|might)\s+(?:send|return|respond|reply)"),
]
_SERVER = [
    ("server-side", r"\bserver[- ]side\b"),
    ("server product", r"\bvulnerability in the [\w ]*server product\b"),
    ("attacker-controlled client", r"\b(?:malicious|rogue|compromised|attacker[- ]controlled|unauthenticated|remote)\s+clients?\b"),
    ("crafted input sent to a server", r"\b(?:send|sending|sends|submit|submitting|submits)\s+(?:a\s+|an\s+)?(?:specially[- ]crafted|crafted|malicious|malformed)\s+(?:[\w./-]+\s+){0,3}(?:requests?|packets?|quer(?:y|ies)|messages?|commands?|headers?|urls?|uris?|payloads?|data|traffic|connections?)\s+to\s+(?:a|an|the|any|affected|vulnerable|targeted|target)\s+(?:[\w./-]+\s+){0,3}(?:servers?|services?|daemons?|hosts?|systems?|instances?|endpoints?)\b"),
]
# Short component names (MSRC tags and titles): a plain word is the component's own role. Product names
# that contain "server" are removed first (they name a product, not a role).
_PRODUCT_NAMES = re.compile(r"\b(?:sql server|exchange server|windows server|sharepoint server|server core)\b", re.I)


def strip_negated(text: str) -> str:
    """Remove clauses that say something is not affected ("FreeRDP proxy are not affected")."""
    return _NEG.sub(" ", text)


def role_signals(text: str, source: str, short_name: bool = False) -> list[dict]:
    """Client/server evidence in one text, each with the pattern and the matched words."""
    t = strip_negated(text or "")
    found = []
    pats = [("client", n, p) for n, p in _CLIENT] + [("server", n, p) for n, p in _SERVER]
    if short_name:
        t = _PRODUCT_NAMES.sub(" ", t)
        pats += [("client", "component name says client", r"\bclient\b"), ("server", "component name says server", r"\bserver\b|\bservices?\b")]
    for role, name, pat in pats:
        m = re.search(pat, t, re.I)
        if m:
            found.append({"role": role, "source": source, "pattern": name, "match": m.group(0)[:120]})
    return found


def vocab_signals(text: str, vocab: dict[str, list[str]], source: str) -> list[dict]:
    """Product-specific role wording (daemon and client-program names), after removing negated clauses."""
    t = strip_negated(text or "")
    found = []
    for role in ("client", "server"):
        for pat in vocab.get(role, []):
            m = re.search(pat, t)
            if m:
                found.append({"role": role, "source": source, "pattern": f"vocabulary: {pat}", "match": m.group(0)[:120]})
                break  # one hit per role and text is enough evidence
    return found


def decide_role(signals: list[dict], prior: str) -> tuple[str, str]:
    """(role, reason) from the collected signals and the product's role prior.

    Explicit wording wins: only client signals -> client; only server signals -> server; both -> ambiguous.
    With no wording, the prior decides: "server" (single-role server product) -> server, "library" ->
    library, "os_component" -> os_component, "multi" (ships client and server parts) -> ambiguous.
    Ambiguous cases are excluded from the course pool and counted (D28 E5).
    """
    if prior == "os_component":
        return "os_component", "kernel and Windows OS-level CVEs are OS components, not server roles (D28 E5)"
    roles = {s["role"] for s in signals}
    if roles == {"client"}:
        return "client", "explicit client wording only"
    if roles == {"server"}:
        return "server", "explicit server wording only"
    if roles == {"client", "server"}:
        return "ambiguous", "both client and server wording"
    return {"server": ("server", "no role wording; single-role server product"),
            "library": ("library", "no role wording; library product"),
            "multi": ("ambiguous", "no role wording; product ships client and server parts")}[prior]


def ghsa_text(page_html: str) -> str:
    """Plain text of the description block of a GitHub security advisory page."""
    i = page_html.find('class="markdown-body')
    if i < 0:
        return ""
    j = page_html.find('<div class="Box-header', i)
    body = page_html[page_html.find(">", i) + 1 : j if j > 0 else i + 60000]
    return note_text(body)


# ---------------------------------------------------------------- funnel (D28 Round 1-2 exclusions)

FUNNEL_STAGES = ("in_cohort", "no_v3_vector", "AV:A", "AV:P", "REVIEW_REQUIRED_AV", "network_PR:L/H", "UI:R",
                 "role_not_server", "REVIEW_REQUIRED", "remote_candidate", "local_candidate")  # fmt: skip


def funnel_stage(resolution: dict, role: str) -> str:
    """The stage at which one (CVE, product) unit leaves the funnel, or its candidate type.

    Each filter applies to a field only when every provider's value is excluded by it, so a CVE that an
    agreed field already excludes needs no conflict review; conflicts that still matter at the end are
    REVIEW_REQUIRED. Order: AV:A, AV:P, then remote (AV:N): network PR:L/H, UI:R, role != server;
    local (AV:L): UI:R. PR does not gate local CVEs here (D28 Round 2: PR:N/L via
    local_unprivileged_execution, PR:H via admin).
    """
    if resolution["status"] == "NO_V3":
        return "no_v3_vector"
    c = resolution["candidates"]
    av = set(c["AV"])
    if av == {"A"}:
        return "AV:A"
    if av == {"P"}:
        return "AV:P"
    if av <= {"A", "P"}:
        return "AV:A"  # both excluded values; counted with AV:A (rare; the vector conflict is kept in the row)
    if len(av) > 1:
        return "REVIEW_REQUIRED_AV"  # AV:N vs AV:L decides remote vs local; nothing to apply first
    pending = False
    if av == {"N"}:
        if set(c["PR"]) <= {"L", "H"}:
            return "network_PR:L/H"
        pending |= len(c["PR"]) > 1
        if set(c["UI"]) == {"R"}:
            return "UI:R"
        pending |= len(c["UI"]) > 1
        if role != "server":
            return "role_not_server"
        return "REVIEW_REQUIRED" if pending else "remote_candidate"
    # AV:L
    if set(c["UI"]) == {"R"}:
        return "UI:R"
    pending |= len(c["UI"]) > 1 or len(c["PR"]) > 1
    return "REVIEW_REQUIRED" if pending else "local_candidate"


def funnel_counts(units: list[dict]) -> dict:
    """Counts of units per (family, os, product, cohort, stage); units carry those five keys."""
    return Counter((u["family"], u["os"], u["product"], u["cohort"], u["stage"]) for u in units)
