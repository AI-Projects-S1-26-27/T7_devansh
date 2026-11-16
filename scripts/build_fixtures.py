"""Version-fixture catalogue (task P5; D28 Round 4 catalogue rule; see ahg.fixtures).

Usage:
  python scripts/build_fixtures.py fetch configs/fixtures/p5.json [--date YYYY-MM-DD]
  python scripts/build_fixtures.py fetch configs/fixtures/p5.json --source fixtures_supplement --base data/raw/fixtures/<date>
  python scripts/build_fixtures.py build configs/fixtures/p5.json data/raw/fixtures/<date> [data/raw/fixtures_supplement/<date>]

fetch: every upstream page the catalogue reads (release listings, vendor advisories and release notes,
vendor CNA records) into data/raw/fixtures/<date>/ through ahg.rawfetch: one request at a time, paced
per host, each logged with URL, UTC time, HTTP status and SHA-256 (BUILD_RULES rule 5). The manifest is
written last. Rerunning resumes: files already on disk are kept, never refetched. A supplement (--base)
fetches only the names its finished base snapshots lack, e.g. pages of units added after the base.

build: reads the master CVE dataset (SHA-256 checked) and the finished raw snapshot (manifest required),
applies the catalogue rule, and writes the versioned catalogue to data/d28/fixtures/<stamp>-<commit>/
plus a runlog run (summary, per-product report) under results/d28_fixtures/catalogue/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
import tarfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ahg import fixtures as fx
from ahg import runlog
from ahg.cvedata import walk_matches
from ahg.rawfetch import Fetcher, write_manifest

ROOT = Path(__file__).resolve().parents[1]
GITHUB_API = "https://api.github.com/repos"
CVEAWG = "https://cveawg.mitre.org/api/cve/"
GITHUB_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def rel(p: Path) -> str:
    """Path relative to the repository root when inside it (as recorded in run metadata), else absolute."""
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def slug(product: str) -> str:
    return product.replace(":", "__")


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_pinned_jsonl(path: str, sha256: str) -> list[dict]:
    """Rows of a gzipped JSON-lines input; refuses a file whose SHA-256 differs from the configured one."""
    data = (ROOT / path).resolve().read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        sys.exit(f"{path}: SHA-256 {got} != configured {sha256}")
    return [json.loads(line) for line in gzip.decompress(data).decode().splitlines() if line.strip()]


def load_master(cfg: dict) -> list[dict]:
    """The master CVE rows (SHA-256 checked)."""
    return load_pinned_jsonl(cfg["master_dataset"], cfg["master_sha256"])


def candidates(cfg: dict, rows: list[dict]) -> tuple[dict, set[tuple[str, str]], dict[str, set[tuple[str, str]]]]:
    """(candidate units per product, eligible units, eligible units per label set 'remote'/'local'): the
    pre-P3 superset plus every eligible unit."""
    labels = load_pinned_jsonl(cfg["eligible_labels"], cfg["eligible_labels_sha256"])
    eligible = fx.eligible_units(labels)
    by_set = {s: fx.eligible_units(r for r in labels if r["label_set"] == s) for s in ("remote", "local")}
    return fx.candidate_units(rows, tuple(cfg["candidate_stages"]), eligible), eligible, by_set


def patterns(src: dict) -> list[str]:
    return src.get("patterns") or [src["pattern"]]


def decode_raw(data: bytes) -> str:
    """Text of a raw file. Wayback `id_` captures return the original server bytes, which may be
    gzip-encoded; they are stored as fetched (raw files are never edited) and decompressed only here."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- fetch


def github_pages(get, name: str, url: str, max_pages: int) -> None:
    """GitHub releases, 100 per page, until a short page (the API pages with ?page=N)."""
    for n in range(1, max_pages + 1):
        rec = get(f"{name}/page{n:02d}.json", f"{url}?per_page=100&page={n}", GITHUB_HEADERS)
        if rec.get("http", 200) != 200 or len(json.loads(Path(rec["path"]).read_bytes())) < 100:
            return
    sys.exit(f"{url}: more than {max_pages} pages; raise github_pages_max")


def fetch(cfg: dict, date: str, base: list[Path], source: str) -> None:
    """Fetch every page the catalogue reads into data/raw/<source>/<date>/ (BUILD_RULES rule 5). Names
    already in a finished `base` snapshot are not fetched again: a supplement (source
    "fixtures_supplement") holds only what the base lacks, and build reads both."""
    rows = load_master(cfg)
    cands = candidates(cfg, rows)[0]
    have = Raw(base) if base else None
    f = Fetcher(ROOT / "data" / "raw" / source / date, "scripts/build_fixtures.py", cfg["pace_seconds"])
    failed = []

    def get(name: str, url: str, headers: dict | None = None) -> dict:
        if have is not None and have.has(name):
            return {"name": name, "skipped": "in base snapshot", "path": str(have.path(name))}
        rec = f.fetch(name, url, headers)
        if rec.get("http", 200) != 200:
            failed.append(rec)
        rec["path"] = str(f.out / name)
        print(f"{rec.get('http', 'kept')} {name}", flush=True)
        return rec

    cna_cves: dict[str, str] = {}
    for product in sorted(cfg["products"]):
        pc, s = cfg["products"][product], slug(product)
        cves = sorted(cands.get(product, {}).get("cves", {}), key=fx.cve_sort_key)
        for src in pc["releases"]:
            if src["kind"] == "github_releases":
                github_pages(get, f"{s}/{src['id']}", f"{GITHUB_API}/{src['repo']}/releases", cfg["github_pages_max"])
            else:
                get(f"{s}/{src['id']}", src["url"])
        if "git_tags" in pc:
            t = pc["git_tags"]
            github_pages(get, f"{s}/{t['id']}", f"{GITHUB_API}/{t['repo']}/tags", cfg["github_pages_max"])
        for src in pc["evidence"]:
            kind = src["kind"]
            if kind in ("sections", "page_regex", "table"):
                get(f"{s}/{src['id']}", src["url"])
            elif kind == "per_cve":
                # A template names every CVE's page; a `urls` map names only the CVEs a vendor page is
                # about (each one the NVD record itself links as a vendor advisory, cited in the config).
                for cve in cves:
                    url = src["urls"].get(cve) if "urls" in src else src["url_template"].format(cve=cve, cve_lower=cve.lower())
                    if url:
                        get(f"{s}/{src['id']}/{cve}.html", url)
            elif kind == "index":
                rec = get(f"{s}/{src['id']}/index.html", src["url"])
                if rec.get("http", 200) == 200:
                    page = decode_raw(Path(rec["path"]).read_bytes())
                    url_dir = src["url"].rsplit("/", 1)[0] + "/"
                    for link in sorted(set(re.findall(r'href="(' + src["link_pattern"] + r')"', page))):
                        get(f"{s}/{src['id']}/{link}", url_dir + link)
            elif kind == "ghsa_repo":
                # The endpoint pages with cursors in a Link header, which rawfetch does not keep. Two
                # requests sorted oldest-first and newest-first cover every advisory when they overlap;
                # build checks that they do.
                for direction in ("asc", "desc"):
                    get(f"{s}/{src['id']}_{direction}.json",
                        f"{GITHUB_API}/{src['repo']}/security-advisories?per_page=100&sort=published&direction={direction}",
                        GITHUB_HEADERS)  # fmt: skip
            else:
                sys.exit(f"{product}: unknown evidence kind {kind!r}")
        if "cna" in pc:
            for cve in cves:
                if cands[product]["cves"][cve]["row"]["nvd"] and \
                        cands[product]["cves"][cve]["row"]["nvd"]["source_identifier"] in pc["cna"]["sources"]:
                    cna_cves[cve] = product
        if "cna_tarball" in pc:
            get(pc["cna_tarball"]["id"], pc["cna_tarball"]["url"])
    for name, hc in sorted(cfg.get("windows_update_history", {}).items()):
        get(f"windows_update_history/{hc['id']}", hc["url"])  # P5w: one Microsoft update-history page per version
    for cve in sorted(cna_cves, key=fx.cve_sort_key):
        get(f"cveawg/{cve}.json", CVEAWG + cve)
    m = write_manifest(f.out, {"task": "P5", "config": cfg, "candidate_products": sorted(cands),
                               "base_snapshots": [rel(b) for b in base]})  # fmt: skip
    print(f"manifest: {len(m['files'])} files, {len(m['failed_fetches'])} failed fetches")
    for r in failed:
        print(f"FAILED {r.get('http')} {r['name']} {r['url']}")


# ---------------------------------------------------------------- build


class Raw:
    """Finished raw snapshots (a base and its supplements): file bytes and their manifest records."""

    def __init__(self, folders: list[Path]):
        self.dirs = folders
        self.files: dict[str, dict] = {}
        self.where: dict[str, Path] = {}
        self.failed: list[dict] = []
        for d in folders:
            manifest = json.loads((d / "manifest.json").read_text())  # an unfinished snapshot is refused
            for r in manifest["files"]:
                if r["name"] in self.files:
                    raise ValueError(f"{r['name']} is in two snapshots: {self.where[r['name']]} and {d}")
                self.files[r["name"]], self.where[r["name"]] = r, d
            self.failed += [{**r, "snapshot": d.name} for r in manifest["failed_fetches"]]

    def has(self, name: str) -> bool:
        return name in self.files

    def path(self, name: str) -> Path:
        return self.where[name] / name

    def text(self, name: str) -> str:
        return decode_raw(self.path(name).read_bytes())

    def ref(self, name: str) -> dict:
        r = self.files[name]
        d = self.where[name]  # <source>/<date>/<name>, as under data/raw/
        return {"raw": f"{d.parent.name}/{d.name}/{name}", "url": r["url"], "retrieved_utc": r["date_utc"], "sha256": r["sha256"]}

    def under(self, prefix: str) -> list[str]:
        return sorted(n for n in self.files if n.startswith(prefix))


def product_releases(product: str, pc: dict, raw: Raw) -> tuple[list[fx.Release], list[str], list[str]]:
    """(stable releases, pre-releases excluded, raw names read) for one product."""
    s, found, used = slug(product), [], []
    for src in pc["releases"]:
        names = raw.under(f"{s}/{src['id']}/") if src["kind"] == "github_releases" else [f"{s}/{src['id']}"]
        for name in names:
            if not raw.has(name):
                continue
            used.append(name)
            page = raw.text(name)
            if src["kind"] == "listing":
                got = fx.parse_listing(page, src["pattern"])
            elif src["kind"] == "text":
                got = fx.parse_text_releases(page, src["pattern"], src.get("date_format"))
            elif src["kind"] == "github_releases":
                got = fx.parse_github_releases(page, src["pattern"])
            elif src["kind"] == "page":
                got = fx.parse_release_page(page, src["pattern"], src["date_pattern"], src.get("date_format"))
            else:
                raise ValueError(f"{product}: unknown release kind {src['kind']!r}")
            if src.get("undated"):
                # An official vendor version list without dates still validates the release (OQ-P5v-2);
                # the date is recorded as "undated", never guessed.
                got = [(n, d or fx.UNDATED, e) for n, d, e in got]
            found += [(n, d, e, name) for n, d, e in got]
    rels, pre = fx.build_releases(found, pc.get("cpe_split"))
    return rels, pre, list(dict.fromkeys(used))  # one page may serve several patterns (Cerberus date formats)


def product_tags(product: str, pc: dict, raw: Raw) -> dict[str, dict] | None:
    """The vendor's official Git tags per release name ({name: {tag, commit, url, source}}), or None if the
    product configures none. Tags corroborate the release list; they never add a release."""
    if "git_tags" not in pc:
        return None
    t = pc["git_tags"]
    out: dict[str, dict] = {}
    for name in raw.under(f"{slug(product)}/{t['id']}/"):
        for rel_name, info in fx.parse_github_tags(raw.text(name), t["pattern"]).items():
            out.setdefault(rel_name, {**info, "url": f"https://github.com/{t['repo']}/tree/{info['tag']}", "source": raw.ref(name)})
    return out


def ghsa_union(raw: Raw, s: str, src: dict) -> tuple[list[str], str | None]:
    """The two GitHub advisory pages, plus a note if together they may miss advisories.

    Every advisory on the pages is a valid vendor statement, so the pages are always read; the note only
    says that a CVE without a document may have an advisory that was not fetched.
    """
    names = [f"{s}/{src['id']}_{d}.json" for d in ("asc", "desc") if raw.has(f"{s}/{src['id']}_{d}.json")]
    if len(names) < 2:
        return names, "advisory pages missing"
    ids = [{a["ghsa_id"] for a in json.loads(raw.text(n))} for n in names]
    if all(len(i) == 100 for i in ids) and not (ids[0] & ids[1]):
        return names, "incomplete: more than 200 advisories (the oldest and newest 100 do not overlap); the middle ones were not fetched"
    return names, None


def product_docs(product: str, pc: dict, cves: list[str], raw: Raw, tar_records: dict) -> tuple[list[fx.FixDoc], list[str], list[str]]:
    """(fix documents, raw names read, notes) for one product."""
    s, vr = slug(product), pc["version_regex"]
    docs: list[fx.FixDoc] = []
    used: list[str] = []
    notes: list[str] = []
    for src in pc["evidence"]:
        kind = src["kind"]
        if kind in ("sections", "page_regex", "table"):
            pages = [(f"{s}/{src['id']}", None)]
        elif kind == "per_cve":
            pages = [(f"{s}/{src['id']}/{c}.html", c) for c in cves]
        elif kind == "index":
            pages = [(n, None) for n in raw.under(f"{s}/{src['id']}/") if not n.endswith("/index.html")]
        elif kind == "ghsa_repo":
            names, refusal = ghsa_union(raw, s, src)
            if refusal:
                notes.append(f"{src['id']}: {refusal}")
            pages = [(n, None) for n in names]
        else:
            raise ValueError(f"{product}: unknown evidence kind {kind!r}")
        for name, page_cve in pages:
            if not raw.has(name):
                continue
            used.append(name)
            page = raw.text(name)
            if kind == "sections":
                docs += fx.ev_sections(page, src["header"], vr, name, src.get("cve_at_line_start", False))
            elif kind == "table":
                docs += fx.ev_table(page, src["column"], vr, name)
            elif kind == "ghsa_repo":
                docs += fx.ev_ghsa_repo(page, vr, name)
            else:
                for pat in patterns(src):
                    docs += fx.ev_page_regex(page, pat, vr, name, page_cve, src.get("flatten", False))
    if "cna" in pc:
        for cve in cves:
            name = f"cveawg/{cve}.json"
            if raw.has(name):
                got, refusal = fx.ev_cve_json5(json.loads(raw.text(name)), {"short_name": pc["cna"]["short_name"]}, pc["cna"]["product_regex"], name)
                used.append(name)
                docs += got
                if refusal:
                    notes.append(f"{cve}: {refusal}")
    if "cna_tarball" in pc:
        t = pc["cna_tarball"]
        for cve in cves:
            if cve in tar_records:
                member, record = tar_records[cve]
                got, refusal = fx.ev_cve_json5(record, {"org_id": t["org_id"]}, t["product_regex"], f"{t['id']}#{member}")
                docs += got
                if refusal:
                    notes.append(f"{cve}: {refusal}")
        used.append(t["id"])
    # One spelling per release: vendors write the same hotfix as "15.5.4 HF1", "15.4.2 HF 2" or
    # "15.5.4 Hotfix 1"; `version_aliases` rewrites documented versions to the release-list spelling.
    for pattern, repl in pc.get("version_aliases", []):
        docs = [replace(d, version=re.sub(pattern, repl, d.version)) for d in docs]
    # Keep only documents about this product's candidate CVEs (pages also name other issues).
    keep = set(cves)
    return [d for d in docs if d.cve in keep], sorted(set(used)), notes


def advisory_fix_releases(product: str, pc: dict, docs: list[fx.FixDoc], rels: list[fx.Release]) -> list[fx.Release]:
    """Hotfixes that an official vendor advisory names as the fix (OQ-P5v-3): for an evidence source with
    `fix_releases` (a regex on the documented version), each documented version that fits it and is not in
    the release list becomes a release, dated "undated", whose source is that advisory."""
    known = {r.name.lower() for r in rels}
    out: dict[str, fx.Release] = {}
    for src in pc["evidence"]:
        if "fix_releases" not in src:
            continue
        prefix = f"{slug(product)}/{src['id']}"
        for d in docs:
            if d.raw_name.startswith(prefix) and re.fullmatch(src["fix_releases"], d.version) and d.version.lower() not in known:
                v, u = fx.cpe_split(d.version, pc.get("cpe_split"))
                out.setdefault(d.version, fx.Release(d.version, v, u, fx.UNDATED, d.raw_name,
                                                     f"fixed release named by vendor advisory: {d.excerpt[:150]}"))
    return sorted(out.values(), key=lambda r: r.key)


def kernel_records(raw: Raw, t: dict) -> dict[str, tuple[str, dict]]:
    """CVE JSON 5 records from the kernel CNA's vulns repository snapshot: {cve: (member, record)}.

    The repository is the kernel CNA's own publication repository, so every published record in it must
    carry the configured orgId; a snapshot with any other orgId is refused rather than trusted.
    """
    out = {}
    rx = re.compile(t["member_regex"])
    with tarfile.open(raw.path(t["id"]), "r:gz") as tar:
        for m in tar:
            hit = rx.fullmatch(m.name)
            if hit and m.isfile():
                out[hit["cve"]] = (m.name, json.load(tar.extractfile(m)))
    orgs = Counter(r["containers"]["cna"]["providerMetadata"].get("orgId") for _, r in out.values())
    if set(orgs) != {t["org_id"]}:
        sys.exit(f"{t['id']}: orgIds {dict(orgs)} are not all the configured kernel CNA {t['org_id']}")
    return out


def nvd_raw_criteria(cfg: dict, product: str) -> tuple[dict[str, set[tuple]], dict[str, dict]]:
    """Vulnerable CPE matches (criteria and bounds, fx.nvd_match_key) per CVE in the raw NVD pages of one product, and each CVE's page provenance.

    The pages are the NVD snapshot the master dataset was built from (BUILD_RULES rule 5 manifest); a file
    whose SHA-256 differs from its manifest record is refused. Unverified fixtures may only use matches
    found here, so every version string is one NVD itself lists.
    """
    d = (ROOT / cfg["unverified_release"]["nvd_raw_dir"]).resolve()
    manifest = json.loads((d / "manifest.json").read_text())
    criteria: dict[str, set[str]] = {}
    refs: dict[str, dict] = {}
    for r in manifest["files"]:
        if r["product"] != product:
            continue
        data = (d / r["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            sys.exit(f"{d / r['path']}: SHA-256 differs from its manifest record")
        for v in json.loads(gzip.decompress(data))["response"]["vulnerabilities"]:  # page = {url, fetched_utc, params, response}
            cve = v["cve"]["id"]
            refs[cve] = {"raw": f"{cfg['unverified_release']['nvd_raw_dir']}/{r['path']}", "url": r["url"], "retrieved_utc": r["fetched_utc"], "sha256": r["sha256"],
                         "nvd_detail": f"https://nvd.nist.gov/vuln/detail/{cve}"}  # fmt: skip
            criteria.setdefault(cve, set()).update(
                fx.nvd_match_key(m) for _, m, neg in walk_matches(v["cve"].get("configurations", [])) if m.get("vulnerable") and not neg)
    return criteria, refs


def add_unverified(entry: dict, info: dict, cfg: dict) -> dict:
    """Unverified-release vulnerable fixtures (decision 16 Nov 2026) for the listed product's CVEs that have
    no verified vulnerable fixture.

    P5v: "a fixture with evidence becomes a normal fixture; one without stays unverified_release". A CVE
    that the catalogue rule gives a verified vulnerable fixture never also gets an unverified one, and an
    unverified version that is also a verified release name is refused (two fixtures, one version).
    """
    product = entry["product"]
    part = cfg["cpe_part"].get(product, cfg["cpe_part"]["default"])
    ranges = {c: fx.cve_ranges(info["cves"][c]["row"]["configurations"], product, part)[0] for c in info["cves"]}
    verified = entry["fixtures"]
    covered = {v["cve"] for f in verified for v in f["vulnerable_for"]}
    criteria, refs = nvd_raw_criteria(cfg, product)
    fixtures, dropped = fx.unverified_fixtures(ranges, criteria, only=set(ranges) - covered)
    clash = {f["version"] for f in fixtures} & {f["version"] for f in verified}
    if clash:
        sys.exit(f"{product}: unverified fixtures {sorted(clash)} share a version with verified fixtures")
    for f in fixtures:
        f["release_source"] = [refs[v["cve"]] for v in f["vulnerable_for"]]
    status = entry["status"] if verified else ("unverified_release" if fixtures else "dropped")
    out = {**entry, "verified_status": entry["status"], "verified_drop_reason": entry.get("drop_reason"),
           "unverified_release_decision": cfg["unverified_release"]["decision"],
           "fixtures": sorted(verified + fixtures, key=lambda f: fx.version_key(f["version"])),
           "dropped": entry["dropped"] + [{**d, "stage": "unverified_release"} for d in dropped],
           "status": status}  # fmt: skip
    if status != "dropped":
        out.pop("drop_reason", None)
    return out


def ref_of(raw: Raw, name: str) -> dict:
    """Provenance of a raw name, including members of the kernel tarball ('<tar>#<member>')."""
    base, _, member = name.partition("#")
    r = raw.ref(base)
    return {**r, "member": member} if member else r


def build_linux(product: str, info: dict, cfg: dict, raw: Raw, tar_records: dict) -> dict:
    pc = cfg["products"][product]
    part = cfg["cpe_part"].get(product, cfg["cpe_part"]["default"])
    cve_ids = sorted(info["cves"], key=fx.cve_sort_key)
    ranges, range_notes = {}, {}
    for cve in cve_ids:
        ranges[cve], notes = fx.cve_ranges(info["cves"][cve]["row"]["configurations"], product, part)
        if notes:
            range_notes[cve] = notes
    rels, pre, rel_used = product_releases(product, pc, raw)
    docs, ev_used, ev_notes = product_docs(product, pc, cve_ids, raw, tar_records)
    extra = advisory_fix_releases(product, pc, docs, rels)
    rels = sorted(rels + extra, key=lambda r: r.key)
    rel_used = list(dict.fromkeys(rel_used + [r.raw_name for r in extra]))
    base = {"product": product, "os": info["os"], "families": sorted(info["families"]),
            "candidate_cves": {c: {k: info["cves"][c][k] for k in ("cohort", "stage", "family")} for c in cve_ids},
            "release_sources": [raw.ref(n) for n in rel_used], "evidence_sources": [ref_of(raw, n) for n in ev_used],
            "n_stable_releases": len(rels), "prereleases_excluded": pre, "range_notes": range_notes,
            "evidence_notes": ev_notes + ([pc["evidence_note"]] if "evidence_note" in pc else [])}  # fmt: skip
    if not rels:
        return {**base, "status": "dropped", "drop_reason": "no stable upstream release parsed from the release sources",
                "fixtures": [], "dropped": [], "unmatched_docs": []}  # fmt: skip
    tags = product_tags(product, pc, raw)
    if tags is not None:
        base["git_tag_check"] = {"repo": pc["git_tags"]["repo"], "releases": len(rels),
                                 "releases_with_tag": sum(r.name in tags for r in rels),
                                 "releases_without_tag": [r.name for r in rels if r.name not in tags]}  # fmt: skip
    index = fx.ReleaseIndex(rels)
    cat = fx.catalogue_product(ranges, index, docs)
    aff = fx.affected_by(list(cat.fixtures), index, ranges)
    fixtures = []
    for f in sorted(cat.fixtures.values(), key=lambda e: fx.version_key(e["version"])):
        for ff in f["fixed_for"]:
            for ev in ff["evidence"]:
                ev.update(ref_of(raw, ev.pop("raw")))
        fixtures.append({**{k: v for k, v in f.items() if k != "release_raw"},
                         "release_source": raw.ref(f["release_raw"]), "affected_by": aff[f["version"]],
                         **({"git_tag": tags.get(f["version"])} if tags is not None else {})})  # fmt: skip
    for u in cat.unmatched_docs:
        u.update(ref_of(raw, u.pop("raw")))
    status = "ok" if fixtures else "dropped"
    return {**base, "status": status, **({"drop_reason": "no fixture found for any candidate CVE"} if not fixtures else {}),
            "fixtures": fixtures, "dropped": cat.dropped, "unmatched_docs": cat.unmatched_docs}  # fmt: skip


def msrc_refs(cfg: dict) -> dict[str, list[dict]]:
    """Raw evidence of each MSRC CVRF document ('2016-Jun'): file, URL, retrieval time, SHA-256 on disk."""
    out: dict[str, list[dict]] = {}
    for folder, log in cfg["msrc_fetch_logs"].items():
        d = (ROOT / folder).resolve()
        for line in (d / log).read_text().splitlines():
            r = json.loads(line)
            m = re.search(r"cvrf_(\d{4}-[A-Z][a-z]{2})\.json$", r.get("name", ""))
            if not m or str(r.get("http")) != "200" or not (d / r["name"]).exists():
                continue
            on_disk = hashlib.sha256((d / r["name"]).read_bytes()).hexdigest()
            out.setdefault(m.group(1), []).append({
                "raw": f"{folder}/{r['name']}", "url": r["url"], "retrieved_utc": r["date_utc"], "sha256": on_disk,
                "logged_sha256_matches": on_disk == r.get("sha256")})  # fmt: skip
    return out


def build_windows(cands: dict, rows: list[dict], cfg: dict, raw: Raw | None = None) -> dict[str, dict]:
    """Windows catalogue, one entry per MSRC product name (Windows Server version, Exchange CU, SQL build).

    Versions with a configured update-history page (`windows_update_history`, P5w: Server 2012 R2, 2016,
    2019, 2022) take their builds from that page: per CVE, the fixing KB MSRC lists for the product is
    mapped to its release; the fixed fixture is that release and the vulnerable fixture the release before
    it in the history (ahg.fixtures.history_pairs). Every other product keeps the P5 rule (prior MSRC
    FixedBuild, OQ-P5-4).
    """
    builds = fx.windows_builds(rows)
    refs = msrc_refs(cfg)
    rules = cfg["windows_products"]
    history_cfg = cfg.get("windows_update_history", {})
    out: dict[str, dict] = {}
    fixed_of: dict[str, dict[str, str]] = {}  # msrc product -> cve -> fixed build (for affected_by, P5 rule)
    hosted: dict[str, set[str]] = {}
    kbs_of: dict[str, dict[str, dict]] = {}  # history product -> cve -> {"kbs", "units", "row"}
    for unit_product, info in sorted(cands.items()):
        if info["os"] != "windows":
            continue
        for cve in sorted(info["cves"], key=fx.cve_sort_key):
            row = info["cves"][cve]["row"]
            names = fx.msrc_product_names(unit_product, [p["name"] for p in row["msrc"]["products"].values()], rules)
            for name in names:
                hosted.setdefault(name, set()).add(cve)
            for p in row["msrc"]["products"].values():
                if p["name"] in names and p["name"] in history_cfg:
                    e = kbs_of.setdefault(p["name"], {}).setdefault(cve, {"kbs": set(), "fixed_builds": set(), "units": {}, "row": row})
                    e["kbs"].update(f["kb"] for f in p.get("fixes", []) if f.get("kb"))
                    e["fixed_builds"].update(f["fixed_build"] for f in p.get("fixes", []) if f.get("fixed_build"))
                    e["units"][unit_product] = info["cves"][cve]["cohort"]
            p5_names = [n for n in names if n not in history_cfg]  # products that keep the P5 rule
            pairs, dropped = fx.windows_fixtures(row, p5_names, builds)
            if not names:
                dropped = [{"cve": cve, "product": None, "reason": "no_configured_msrc_product_listed"}]
            for d in dropped:
                e = out.setdefault(d["product"] or f"(none: {unit_product})", {"fixtures": {}, "dropped": [], "units": set()})
                e["dropped"].append({**d, "unit": unit_product})
                e["units"].add(unit_product)
            for p in pairs:
                e = out.setdefault(p["product"], {"fixtures": {}, "dropped": [], "units": set()})
                e["units"].add(unit_product)
                fixed_of.setdefault(p["product"], {})[cve] = p["fixed"]
                for role, b in (("vulnerable_for", p["vulnerable"]), ("fixed_for", p["fixed"])):
                    meta = builds[p["product"]][b]
                    fxe = e["fixtures"].setdefault(b, {
                        "version": b, "release_date": meta["date"],
                        "release_source": [r for doc in sorted(meta["docs"]) for r in refs.get(doc, [])],
                        "msrc_docs": sorted(meta["docs"]), "vulnerable_for": [], "fixed_for": []})  # fmt: skip
                    fxe[role].append({"cve": cve, "unit": unit_product, **({"after": p["vulnerable"]} if role == "fixed_for" else {})})
    result = {}
    for name, e in sorted(out.items()):
        fixed = fixed_of.get(name, {})
        unknown = sorted(hosted.get(name, set()) - set(fixed), key=fx.cve_sort_key)
        fixtures = []
        for b, f in sorted(e["fixtures"].items(), key=lambda kv: fx.version_key(kv[0])):
            aff = sorted((c for c, fb in fixed.items() if fx.version_key(b) < fx.version_key(fb)), key=fx.cve_sort_key)
            fixtures.append({**f, "affected_by": aff})
        result[name] = {"product": name, "os": "windows", "units": sorted(e["units"]), "status": "ok" if fixtures else "dropped",
                        "fixtures": fixtures, "dropped": e["dropped"], "affected_unknown_no_fixed_build": unknown}  # fmt: skip
    for name, hc in sorted(history_cfg.items()):
        if name in kbs_of:
            result[name] = history_entry(name, hc, kbs_of[name], hosted.get(name, set()), refs, raw)
    return result


def history_entry(name: str, hc: dict, cves: dict[str, dict], hosted: set[str], refs: dict, raw: Raw | None) -> dict:
    """One Windows version's catalogue entry from its update-history page (P5w)."""
    page = f"windows_update_history/{hc['id']}"
    if raw is None or not raw.has(page):
        sys.exit(f"{name}: update-history page {page} is not in the raw snapshots")
    ordered = fx.history_order(fx.parse_update_history(raw.text(page), hc.get("build_line")), bool(hc.get("build_line")))
    if not ordered:
        sys.exit(f"{name}: no release parsed from {page}")
    src = raw.ref(page)
    pairs, dropped_by_cve = fx.history_pairs({c: sorted(v["kbs"]) for c, v in cves.items()}, ordered)
    fixtures: dict[int, dict] = {}

    def fixture(i: int) -> dict:
        r = ordered[i]
        return fixtures.setdefault(i, {"version": r["version"], "kb": f"KB{r['kb']}", "builds": r["builds"], "release_date": r["date"],
                                       "release_entry": r["entry"], "release_source": [src], "vulnerable_for": [], "fixed_for": []})  # fmt: skip

    mismatch = []
    for cve in sorted(pairs, key=fx.cve_sort_key):
        p, info = pairs[cve], cves[cve]
        fixed_r = ordered[p["fixed"]]
        docs = sorted(info["row"]["msrc"].get("docs") or [info["row"]["msrc"].get("doc")])
        evidence = {"fix_kbs": sorted(info["kbs"]), "mapped_kb": f"KB{p['kb']}", "msrc_docs": docs,
                    "msrc_sources": [r for d in docs for r in refs.get(d, [])],
                    "msrc_fixed_builds": sorted(info["fixed_builds"])}  # fmt: skip
        # MSRC's own FixedBuild (2021+ documents), where given, must name the mapped release's build.
        if info["fixed_builds"] and fixed_r["builds"] and not set(info["fixed_builds"]) & set(fixed_r["builds"]):
            mismatch.append({"cve": cve, "mapped_kb": f"KB{p['kb']}", "history_builds": fixed_r["builds"],
                             "msrc_fixed_builds": sorted(info["fixed_builds"])})  # fmt: skip
        for unit, cohort in sorted(info["units"].items()):
            fixture(p["vulnerable"])["vulnerable_for"].append({"cve": cve, "unit": unit, "cohort": cohort})
            fixture(p["fixed"])["fixed_for"].append({"cve": cve, "unit": unit, "cohort": cohort,
                                                     "after": ordered[p["vulnerable"]]["version"], "evidence": evidence})  # fmt: skip
    for i, f in fixtures.items():
        # A release is affected by every hosted CVE whose fix comes later in the servicing order.
        f["affected_by"] = sorted((c for c, p in pairs.items() if i < p["fixed"]), key=fx.cve_sort_key)
    dropped = [{"cve": c, "product": name, "unit": u, "cohort": co, **d}
               for c, d in sorted(dropped_by_cve.items(), key=lambda kv: fx.cve_sort_key(kv[0])) for u, co in sorted(cves[c]["units"].items())]
    units = sorted({u for v in cves.values() for u in v["units"]})
    return {"product": name, "os": "windows", "units": units, "status": "ok" if fixtures else "dropped",
            "rule": "P5w update history", "release_source": src, "n_releases": len(ordered),
            "fixtures": [fixtures[i] for i in sorted(fixtures)], "dropped": dropped,
            "msrc_fixed_build_mismatch": mismatch,
            "affected_unknown_no_fixed_build": sorted(set(dropped_by_cve) | (hosted - set(cves)), key=fx.cve_sort_key)}  # fmt: skip


def summarize(entry: dict) -> dict:
    fixtures = entry["fixtures"]
    n_aff = [len(f["affected_by"]) for f in fixtures]
    cands = entry.get("candidate_cves")
    return {
        "status": entry["status"], "drop_reason": entry.get("drop_reason"),
        "candidate_cves": len(cands) if cands is not None else None,
        "fixtures": len(fixtures),
        "vulnerable_fixtures": sum(bool(f["vulnerable_for"]) for f in fixtures),
        "fixed_fixtures": sum(bool(f["fixed_for"]) for f in fixtures),
        "unverified_release_fixtures": sum(bool(f.get("unverified_release")) for f in fixtures),
        "cves_with_vulnerable_fixture": len({v["cve"] for f in fixtures for v in f["vulnerable_for"]}),
        "cves_with_fixed_fixture": len({v["cve"] for f in fixtures for v in f["fixed_for"]}),
        "affected_by_min": min(n_aff) if n_aff else None, "affected_by_median": statistics.median(n_aff) if n_aff else None,
        "affected_by_max": max(n_aff) if n_aff else None,
        "dropped_by_reason": dict(Counter(d["reason"] for d in entry["dropped"])),
        "unmatched_docs": len(entry.get("unmatched_docs", [])),
        "n_stable_releases": entry.get("n_stable_releases"),
    }  # fmt: skip


def summary_table(summary: dict) -> list[str]:
    lines = ["| OS | product | status | candidate CVEs | fixtures | vulnerable | fixed | CVEs w/ vuln fixture | CVEs w/ fixed fixture | affected-by min/median/max | dropped (CVE, range) by reason | unmatched fix docs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for (os_, product), s in sorted(summary.items()):
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(s["dropped_by_reason"].items())) or "-"
        aff = f"{s['affected_by_min']}/{s['affected_by_median']}/{s['affected_by_max']}" if s["fixtures"] else "-"
        status = s["status"] + (f" ({s['drop_reason']})" if s.get("drop_reason") else "")
        lines.append(f"| {os_} | {product} | {status} | {s['candidate_cves'] if s['candidate_cves'] is not None else '-'} | {s['fixtures']} | "
                     f"{s['vulnerable_fixtures']} | {s['fixed_fixtures']} | {s['cves_with_vulnerable_fixture']} | "
                     f"{s['cves_with_fixed_fixture']} | {aff} | {reasons} | {s['unmatched_docs']} |")  # fmt: skip
    return lines


def fixture_lines(catalogue: dict) -> list[str]:
    lines = []
    for key in ("linux", "windows"):
        for product, e in sorted(catalogue[key].items()):
            if not e["fixtures"]:
                continue
            lines += ["", f"### {key} / {product}", ""]
            for f in e["fixtures"]:
                roles = []
                if f["vulnerable_for"]:
                    roles.append(f"vulnerable for {len(f['vulnerable_for'])}")
                if f["fixed_for"]:
                    roles.append(f"fixed for {len(f['fixed_for'])}")
                ids = f["affected_by"]
                when = "unverified release, no date" if f.get("unverified_release") else f["release_date"]
                lines.append(f"- {f['version']} ({when}): {', '.join(roles)}; affected by {len(ids)}: "
                             + ", ".join(ids[:12]) + (" ..." if len(ids) > 12 else ""))  # fmt: skip
    return lines


def unverified_regained(catalogue: dict, eligible_by_set: dict[str, set[tuple[str, str]]]) -> list[dict]:
    """Per product in the unverified_release list: fixtures by kind (verified from release evidence,
    unverified_release), candidate CVEs with a vulnerable fixture of any kind and with a verified one, and
    the eligible remote and local units each covers (`regained`: any kind; `verified`: verified only)."""
    out = []
    for product, e in sorted(catalogue["linux"].items()):
        if "unverified_release_decision" not in e:
            continue
        covered = {v["cve"] for f in e["fixtures"] for v in f["vulnerable_for"]}
        covered_verified = {v["cve"] for f in e["fixtures"] if not f.get("unverified_release") for v in f["vulnerable_for"]}
        n_unverified = sum(bool(f.get("unverified_release")) for f in e["fixtures"])
        row = {"product": product, "status": e["status"], "verified_fixtures": len(e["fixtures"]) - n_unverified,
               "unverified_fixtures": n_unverified, "candidate_cves": len(e["candidate_cves"]),
               "candidate_cves_with_fixture": len(covered), "candidate_cves_with_verified_fixture": len(covered_verified)}  # fmt: skip
        for s, units in eligible_by_set.items():
            el = {c for c, p in units if p == product}
            row[f"eligible_{s}"], row[f"eligible_{s}_regained"] = len(el), len(el & covered)
            row[f"eligible_{s}_verified"] = len(el & covered_verified)
        out.append(row)
    return out


def regained_table(rows: list[dict]) -> list[str]:
    lines = ["| product | status | verified fixtures | unverified fixtures | candidate CVEs | with fixture | with verified fixture | "
             "eligible remote | covered | verified | eligible local | covered | verified |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        lines.append(f"| {r['product']} | {r['status']} | {r['verified_fixtures']} | {r['unverified_fixtures']} | {r['candidate_cves']} | "
                     f"{r['candidate_cves_with_fixture']} | {r['candidate_cves_with_verified_fixture']} | "
                     f"{r['eligible_remote']} | {r['eligible_remote_regained']} | {r['eligible_remote_verified']} | "
                     f"{r['eligible_local']} | {r['eligible_local_regained']} | {r['eligible_local_verified']} |")  # fmt: skip
    return lines


def windows_history_rows(catalogue: dict) -> list[dict]:
    """Per Windows version built from its update history (P5w) and cohort: CVEs, fixtures and drops."""
    out = []
    for name, e in sorted(catalogue["windows"].items()):
        if e.get("rule") != "P5w update history":
            continue
        cohorts = sorted({i["cohort"] for f in e["fixtures"] for i in f["vulnerable_for"] + f["fixed_for"]} | {d["cohort"] for d in e["dropped"]})
        for co in cohorts:
            vul = {i["cve"] for f in e["fixtures"] for i in f["vulnerable_for"] if i["cohort"] == co}
            fix = {i["cve"] for f in e["fixtures"] for i in f["fixed_for"] if i["cohort"] == co}
            drops = {(d["cve"], d["reason"]) for d in e["dropped"] if d["cohort"] == co}
            out.append({"product": name, "cohort": co, "n_releases": e["n_releases"], "cves": len(vul | fix | {c for c, _ in drops}),
                        "cves_with_vulnerable_fixture": len(vul), "cves_with_fixed_fixture": len(fix),
                        "vulnerable_fixtures": sum(any(i["cohort"] == co for i in f["vulnerable_for"]) for f in e["fixtures"]),
                        "fixed_fixtures": sum(any(i["cohort"] == co for i in f["fixed_for"]) for f in e["fixtures"]),
                        "dropped_by_reason": dict(Counter(r for _, r in drops)),
                        "msrc_fixed_build_mismatch": sum(1 for m in e["msrc_fixed_build_mismatch"] if m["cve"] in vul | fix)})  # fmt: skip
    return out


def windows_history_table(rows: list[dict]) -> list[str]:
    lines = ["| version | cohort | releases in history | CVEs | with vulnerable fixture | with fixed fixture | vulnerable fixtures | fixed fixtures | dropped CVEs by reason | MSRC FixedBuild mismatch |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(r["dropped_by_reason"].items())) or "-"
        lines.append(f"| {r['product']} | {r['cohort']} | {r['n_releases']} | {r['cves']} | {r['cves_with_vulnerable_fixture']} | "
                     f"{r['cves_with_fixed_fixture']} | {r['vulnerable_fixtures']} | {r['fixed_fixtures']} | {reasons} | {r['msrc_fixed_build_mismatch']} |")  # fmt: skip
    return lines


def windows_history_fixture_lines(catalogue: dict) -> list[str]:
    """Every update-history fixture with the CVEs it is the vulnerable or fixed fixture for, per cohort."""
    lines = []
    for name, e in sorted(catalogue["windows"].items()):
        if e.get("rule") != "P5w update history":
            continue
        lines += ["", f"### {name}", ""]
        for f in e["fixtures"]:
            parts = []
            for role, label in (("vulnerable_for", "vulnerable for"), ("fixed_for", "fixed for")):
                for co in ("core", "temporal"):
                    ids = sorted({i["cve"] for i in f[role] if i["cohort"] == co}, key=fx.cve_sort_key)
                    if ids:
                        parts.append(f"{label} ({co}) {', '.join(ids)}")
            lines.append(f"- {f['version']} ({f['kb']}, {f['release_date']}): " + "; ".join(parts))
    return lines


def report(summary: dict, catalogue: dict, meta: dict, el_summary: dict, el_catalogue: dict, regained: list[dict]) -> str:
    lines = ["# P5 fixture catalogue", "", f"Catalogue: `{meta['catalogue_path']}` (SHA-256 `{meta['catalogue_sha256']}`)",
             f"Eligible view: `{meta['eligible_path']}` (SHA-256 `{meta['eligible_sha256']}`), labels SHA-256 `{meta['eligible_labels_sha256']}`, "
             f"{meta['eligible_units']} eligible (CVE, product) units",
             f"Raw snapshots: {', '.join(f'`{d}` (manifest SHA-256 `{h}`)' for d, h in zip(meta['raw_dirs'], meta['raw_manifest_sha256']))}",
             f"Master dataset SHA-256: `{meta['master_sha256']}`", "",
             "Candidates: pre-P3 superset plus every eligible unit (P5 decisions). Fixture counts are distinct releases/builds per product.", "",
             "## Products of the unverified_release list (decision 16 Nov 2026; P5v)", "",
             "Verified fixtures come from vendor release evidence by the catalogue rule. Unverified fixtures (vulnerable role only, "
             "p = 1 only) remain for the CVEs that have no verified vulnerable fixture.", ""]  # fmt: skip
    lines += regained_table(regained)
    lines += ["", "## Windows versions from Microsoft update history (P5w)", "", "Superset:", ""] + windows_history_table(windows_history_rows(catalogue))
    lines += ["", "Eligible view:", ""] + windows_history_table(windows_history_rows(el_catalogue))
    lines += ["", "## Eligible view (p3-labels-v2.1)", ""]
    lines += summary_table(el_summary)
    lines += ["", "## Superset (pre-P3 candidates)", ""] + summary_table(summary)
    lines += ["", "## Eligible view: fixtures and the eligible CVEs each is affected by", "",
              "One line per fixture: version (release date): roles; affected-by count; CVE IDs (first 12)."]
    lines += fixture_lines(el_catalogue)
    lines += ["", "## Superset: fixtures and the CVEs each is affected by", ""] + fixture_lines(catalogue)
    lines += ["", "## P5w: update-history fixtures and the CVEs each is the vulnerable or fixed fixture for (superset)"] + windows_history_fixture_lines(catalogue)
    return "\n".join(lines) + "\n"


def build(cfg: dict, raw_dirs: list[Path]) -> None:
    rows = load_master(cfg)
    cands, eligible, eligible_by_set = candidates(cfg, rows)
    raw = Raw(raw_dirs)
    linux: dict[str, dict] = {}
    for product, info in sorted(cands.items()):
        if info["os"] != "linux":
            continue
        if product in cfg["not_covered"]:
            linux[product] = {"product": product, "os": "linux", "families": sorted(info["families"]),
                              "candidate_cves": {c: {k: info["cves"][c][k] for k in ("cohort", "stage", "family")} for c in info["cves"]},
                              "status": "dropped", "drop_reason": cfg["not_covered"][product],
                              "fixtures": [], "dropped": [], "unmatched_docs": []}  # fmt: skip
            if product in cfg["unverified_release"]["products"]:
                linux[product] = add_unverified(linux[product], info, cfg)
            continue
        if product not in cfg["products"]:
            sys.exit(f"{product}: candidate product with no configuration and no not_covered reason")
        tar_records = {}
        if "cna_tarball" in cfg["products"][product] and raw.has(cfg["products"][product]["cna_tarball"]["id"]):
            tar_records = kernel_records(raw, cfg["products"][product]["cna_tarball"])
        linux[product] = build_linux(product, info, cfg, raw, tar_records)
        if product in cfg["unverified_release"]["products"]:
            linux[product] = add_unverified(linux[product], info, cfg)
        print(f"{product}: {summarize(linux[product])}", flush=True)
    windows = build_windows(cands, rows, cfg, raw)
    catalogue = {"schema": "ahg.fixtures/1", "task": "P5", "decisions": cfg["decisions"], "linux": linux, "windows": windows,
                 "raw_failed_fetches": raw.failed}  # fmt: skip
    summary = {("linux", p): summarize(e) for p, e in linux.items()} | {("windows", p): summarize(e) for p, e in windows.items()}
    prov = runlog.provenance([], cfg)
    stamp = prov["date_utc"].replace(":", "").replace("+0000", "Z")
    out = ROOT / cfg["catalogue_root"] / f"{stamp}-{prov['git_commit'][:7]}"
    out.mkdir(parents=True, exist_ok=False)  # a catalogue version is never overwritten (rule 11)
    catalogue["provenance"] = prov
    blob = gzip.compress(json.dumps(catalogue, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    (out / "catalogue.json.gz").write_bytes(blob)
    el_cat = {"schema": "ahg.fixtures/1", "task": "P5", "view": "eligible (p3-labels-v2.1)",
              "eligible_labels_sha256": cfg["eligible_labels_sha256"], **fx.eligible_view(catalogue, eligible)}  # fmt: skip
    el_blob = gzip.compress(json.dumps(el_cat, sort_keys=True, separators=(",", ":")).encode(), mtime=0)
    (out / "catalogue_eligible.json.gz").write_bytes(el_blob)
    el_summary = {("linux", p): summarize(e) for p, e in el_cat["linux"].items()} | \
        {("windows", p): summarize(e) for p, e in el_cat["windows"].items()}
    meta = {"catalogue_path": rel(out / "catalogue.json.gz"), "catalogue_sha256": hashlib.sha256(blob).hexdigest(),
            "eligible_path": rel(out / "catalogue_eligible.json.gz"), "eligible_sha256": hashlib.sha256(el_blob).hexdigest(),
            "eligible_labels_sha256": cfg["eligible_labels_sha256"], "eligible_units": len(eligible),
            "raw_dirs": [rel(d) for d in raw_dirs],
            "raw_manifest_sha256": [hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest() for d in raw_dirs],
            "master_sha256": cfg["master_sha256"]}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    regained = unverified_regained(catalogue, eligible_by_set)
    run_raw = {"meta": meta, "unverified_regained": regained, "windows_history": windows_history_rows(catalogue),
               "windows_history_eligible": windows_history_rows(el_cat), "summary": [{"os": k[0], "product": k[1], **v} for k, v in sorted(summary.items())],
               "eligible_summary": [{"os": k[0], "product": k[1], **v} for k, v in sorted(el_summary.items())]}  # fmt: skip
    run = runlog.write_run("catalogue", run_raw, report(summary, catalogue, meta, el_summary, el_cat, regained), [], cfg,
                           root=ROOT / cfg["results_root"])  # fmt: skip
    print(f"catalogue: {meta['catalogue_path']}\nrun: {rel(run)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_f = sub.add_parser("fetch")
    a_f.add_argument("config", type=Path)
    a_f.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    a_f.add_argument("--source", default="fixtures", choices=["fixtures", "fixtures_supplement", "fixtures_p5v", "fixtures_p5v2", "fixtures_p5w"],
                     help="data/raw/<source>/<date>/; any source but 'fixtures' is a supplement and needs --base")
    a_f.add_argument("--base", type=Path, nargs="*", default=[], help="finished snapshots this one supplements")
    a_b = sub.add_parser("build")
    a_b.add_argument("config", type=Path)
    a_b.add_argument("raw_dirs", type=Path, nargs="+")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "fetch":
        date_ = datetime.strptime(a.date, "%Y-%m-%d").date().isoformat()  # rejects anything but YYYY-MM-DD
        if (a.source != "fixtures") != bool(a.base):
            sys.exit("a supplement snapshot (fixtures_supplement, fixtures_p5v, fixtures_p5v2, fixtures_p5w) needs --base, and only a supplement takes one")
        fetch(cfg, date_, [b.resolve() for b in a.base], a.source)
    else:
        build(cfg, [d.resolve() for d in a.raw_dirs])


if __name__ == "__main__":
    main()
