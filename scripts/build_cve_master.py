"""Build the D28 master CVE dataset and the P2 exclusion funnel (D28 Rounds 1-4, data side).

Usage: python scripts/build_cve_master.py configs/data/cve_master_p2.json

Reads only pinned raw snapshots (every file's SHA-256 is checked against its manifest or fetch log first),
never the network. Writes:
  - data/d28/master/<run-id>/cves.jsonl.gz: one row per canonical CVE ID (untracked; Zenodo, D28 §0);
  - results/d28_data/cve_master/<run-id>/ through ahg.runlog: REP6 reports, funnel, eligible counts,
    product counts, provider mix, conflicts, role evidence summary, and report.md.
The build fails if a role check named in D28 E5 does not come out as stated.
"""

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ahg import cvedata as cd
from ahg import runlog
from ahg.cve import FAMILY_PRODUCTS
from ahg.nvd_snapshot import load_pages, validate

ROOT = Path(__file__).resolve().parents[1]
ROLE_CHECKS = {"CVE-2023-39354": "client", "CVE-2023-40569": "client"}  # D28 E5 known checks (FreeRDP GHSA)
KEYWORDS = re.compile(r"SMB|Remote Desktop|RDP|DNS|IIS|Internet Information|SQL|Exchange|SSH|FTP", re.I)


def verified_files(folder: Path) -> dict[str, Path]:
    """{name: path} of a raw folder, after checking every file's SHA-256 against its manifest or fetch log.

    The record may sit in the folder itself or in its parent (data/raw/sources/2026-11-08/fetch_log.jsonl
    logs msrc/... names); names are returned relative to `folder`. Files without a record are not returned.
    """
    for base, prefix in ((folder, ""), (folder.parent, folder.name + "/")):
        m, log = base / "manifest.json", base / "fetch_log.jsonl"
        if m.exists():
            recs = {r["name"]: r["sha256"] for r in json.loads(m.read_text())["files"]}
            break
        if log.exists():
            recs = {}
            for line in log.read_text().splitlines():
                r = json.loads(line)
                if r.get("sha256"):
                    recs[r["name"]] = r["sha256"]  # the last successful record wins
            break
    else:
        sys.exit(f"{folder}: no manifest.json or fetch_log.jsonl here or in the parent")
    out = {}
    for name, sha in recs.items():
        if not name.startswith(prefix):
            continue
        p = base / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            sys.exit(f"{p}: SHA-256 differs from its record; raw data was modified")
        out[name[len(prefix):]] = p
    return out


def load_msrc(folders: list[Path]) -> tuple[dict[str, list[dict]], list[str]]:
    """{CVE ID: entries from every CVRF document that lists it}, and the documents read."""
    by_cve, docs = defaultdict(list), []
    for folder in folders:
        for name, p in sorted(verified_files(folder).items()):
            m = re.fullmatch(r"cvrf_(\d{4}-[A-Za-z]{3}(?:-[A-Za-z0-9]+)?)\.json", name)
            if not m:
                continue  # index, docs, swagger, the 2017-Mar XML duplicate
            docs.append(f"{folder.name}/{name}")
            for e in cd.msrc_entries(json.loads(p.read_bytes()), m.group(1)):
                by_cve[e["cve_id"]].append(e)
    return by_cve, docs


def latest(entries: list[dict]) -> dict:
    """The entry with the latest revision date (ties: the later document ID in sort order)."""
    return max(entries, key=lambda e: (max(e["revision_dates"] or [""]), e["doc"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text())
    cohorts = cfg["cohorts"]

    # ---- NVD: verify, REP6-validate, merge by CVE ID (latest lastModified wins; all sources kept)
    rep6, nvd, nvd_from = {}, {}, defaultdict(set)
    for label, rel in cfg["nvd_snapshots"].items():
        pages = load_pages(ROOT / rel)
        r = validate(pages)
        rep6[label] = {k: r[k] for k in ("ok", "errors", "pages", "records", "unique_cves", "cves_in_several_queries")}
        if not r["ok"]:
            sys.exit(f"REP6 failed for {rel}: {r['errors'][:5]}")
        for p in pages:
            for v in p["response"]["vulnerabilities"]:
                c = v["cve"]
                nvd_from[c["id"]].add(label)
                if c["id"] not in nvd or c["lastModified"] > nvd[c["id"]]["lastModified"]:
                    nvd[c["id"]] = c

    msrc, msrc_docs = load_msrc([ROOT / f for f in cfg["msrc_folders"]])
    ghsa = {}
    gdir = ROOT / cfg["ghsa_folder"]
    for name, p in verified_files(gdir).items():
        ghsa.setdefault(name.split("__")[0], []).append((name, cd.ghsa_text(p.read_text(errors="replace"))))

    linux_products = {p: fam for fam, ps in FAMILY_PRODUCTS.items() for p in ps}
    prior = cfg["product_role_prior"]
    win_versions, comps = set(cfg["windows_versions"]), cfg["windows_components"]
    app_re = {k: re.compile(v) for k, v in cfg["windows_app_products"].items()}
    build = Counter()
    unmapped_tags = Counter()
    rows = []

    for cid in sorted(set(nvd) | set(msrc)):
        if not re.fullmatch(r"CVE-\d{4}-\d{4,}", cid or ""):
            build["msrc_non_cve_id"] += 1
            continue
        cve = nvd.get(cid)
        matches = cd.cpe_matches(cve.get("configurations", [])) if cve else []
        units, part_mismatch = [], []
        for prod, fam in linux_products.items():
            hit, parts = cd.product_hits(matches, prod, cfg["linux_product_part"])
            if hit:
                units.append({"family": fam, "os": "linux", "product": prod, "prior": prior[prod]})
            elif parts:
                part_mismatch.append({"product": prod, "parts": parts})
        kp = cfg["linux_kernel"]
        hit, parts = cd.product_hits(matches, kp["product"], kp["part"])
        if hit:
            units.append({"family": "os", "os": "linux", "product": kp["product"], "prior": "os_component"})
        elif parts:
            part_mismatch.append({"product": kp["product"], "parts": parts})

        ms = latest(msrc[cid]) if cid in msrc else None
        scoped_pids: set[str] = set()
        if ms:
            win_pids = {pid for pid, p in ms["products"].items() if p["name"] in win_versions}
            mapped = [comps[t] for t in ms["tags"] if t in comps]
            for t in ms["tags"]:
                if t not in comps and KEYWORDS.search(t) and win_pids:
                    unmapped_tags[t] += 1
            seen = set()
            for fam, comp in mapped:
                if comp in app_re:
                    pids = {pid for pid, p in ms["products"].items() if app_re[comp].search(p["name"])}
                else:
                    pids = win_pids
                    if comp == "IIS" and re.search(r"\bFTP\b", ms["title"] or ""):
                        fam = "ftp"
                if pids and (fam, comp) not in seen:
                    seen.add((fam, comp))
                    scoped_pids |= pids
                    units.append({"family": fam, "os": "windows", "product": comp,
                                  "prior": cfg["windows_component_role_prior"][comp], "msrc_products": sorted(pids)})  # fmt: skip
            if not seen and win_pids:
                scoped_pids |= win_pids
                units.append({"family": "os", "os": "windows", "product": "windows_os", "prior": "os_component",
                              "msrc_products": sorted(win_pids)})  # fmt: skip
        if not units:
            build["no_in_scope_product" if cve else "msrc_not_in_scope"] += 1
            continue

        # ---- dates and cohort: NVD publication date where NVD has the record, else MSRC release date
        if cve:
            published, published_source = cve["published"], "nvd"
        else:
            published, published_source = ms["release_date"] or min(ms["revision_dates"]), "msrc"
            build["msrc_only_rows"] += 1
        coh = cd.cohort(published, cohorts)

        # ---- vectors (CV1): every NVD provider, plus MSRC vectors scoring an in-scope product
        vectors = cd.nvd_vectors(cve) if cve else []
        if ms:
            vectors += cd.msrc_vectors_in_scope(ms, scoped_pids)
        res = cd.resolve_vector(vectors)

        # ---- role (E5): wording from NVD description, GHSA advisories, MSRC title/tags/FAQ; CPE fields
        texts = []
        if cve:
            texts.append(("nvd_description", " ".join(d["value"] for d in cve["descriptions"] if d["lang"] == "en"), False))
            for gid in sorted({m.group(0) for r in cve.get("references", []) for m in [re.search(r"GHSA(?:-[\w]{4}){3}", r["url"])] if m}):
                texts += [(f"ghsa:{n}", t, False) for n, t in ghsa.get(gid, [])]
        if ms:
            texts += [("msrc_title", ms["title"] or "", True), ("msrc_faq", ms["faq"], False)]
            texts += [(f"msrc_tag:{t}", t, True) for t in ms["tags"]]
        signals = [s for src, t, short in texts for s in cd.role_signals(t, src, short)]
        for u in units:
            vendor_prod = u["product"].split(":")
            cpe_sig = [{"role": r, "source": "cpe", "pattern": f"CPE field says {r}", "match": m["criteria"]}
                       for m in matches if m["role"] == "vulnerable" and [m["vendor"], m["product"]] == vendor_prod
                       for r in ("client", "server") if re.search(rf"\b{r}\b", m["criteria"].split(":", 5)[5].replace("_", " "))]  # fmt: skip
            u["role"], u["role_reason"] = cd.decide_role(signals + cpe_sig, u["prior"])
            u["role_evidence"] = signals + cpe_sig
            u["cohort"] = coh
            u["stage"] = cd.funnel_stage(res, u["role"]) if coh else None

        rows.append({
            "cve_id": cid, "published": published, "published_source": published_source,
            "last_modified": cve["lastModified"] if cve else None, "cohort": coh,
            "nvd": {"vuln_status": cve["vulnStatus"], "source_identifier": cve.get("sourceIdentifier"),
                    "snapshots": sorted(nvd_from[cid]), "cna_only": not any(v["type"] == "Primary" for v in vectors if v["origin"] == "nvd")}
                   if cve else None,
            "description": next((d["value"] for d in cve["descriptions"] if d["lang"] == "en"), None) if cve else ms["description"],
            "cwe": sorted({(d["value"], w.get("source"), w.get("type")) for w in (cve or {}).get("weaknesses", []) for d in w["description"]}),
            "vectors": vectors, "vector_resolution": res,
            "cpe_matches": matches, "affected_products": cd.affected_products(matches),
            "platform_conditions": sorted({m["criteria"] for m in matches if m["role"] == "platform"}),
            "cpe_part_mismatch": part_mismatch, "configurations": cve.get("configurations", []) if cve else [],
            "references": [{"url": r["url"], "tags": r.get("tags", [])} for r in (cve or {}).get("references", [])],
            "msrc": {k: ms[k] for k in ("doc", "title", "tags", "faq", "cwe", "release_date", "revision_dates", "products", "cna_note")}
                    | {"docs": sorted(e["doc"] for e in msrc[cid]),
                       "impacts": sorted({p.get("impact") for p in ms["products"].values() if p.get("impact")})} if ms else None,
            "units": units,
        })  # fmt: skip

    # ---- D28 E5 known checks
    by_id = {r["cve_id"]: r for r in rows}
    checks = {}
    for cid, want in ROLE_CHECKS.items():
        got = sorted({u["role"] for u in by_id[cid]["units"]}) if cid in by_id else ["MISSING"]
        checks[cid] = {"want": want, "got": got, "ok": got == [want]}
    if not all(c["ok"] for c in checks.values()):
        sys.exit(f"D28 E5 role check failed: {checks}")

    # ---- aggregates (counts come only from the rows above)
    units = [u | {"cve_id": r["cve_id"]} for r in rows for u in r["units"] if u["cohort"]]
    funnel = Counter((u["family"], u["os"], u["product"], u["cohort"], u["stage"]) for u in units)
    funnel_rows = sorted({k[:4] for k in funnel})
    eligible = defaultdict(set)
    for u in units:
        if u["stage"] in ("remote_candidate", "local_candidate"):
            eligible[(u["family"], u["os"], u["cohort"], u["stage"])].add(u["cve_id"])
    products_listed = defaultdict(set)
    for p, fam in linux_products.items():
        products_listed[(fam, "linux")].add(p)
    products_listed[("os", "linux")].add(cfg["linux_kernel"]["product"])
    for tag, (fam, comp) in ((k, v) for k, v in comps.items() if not k.startswith("_")):
        products_listed[(fam, "windows")].add(comp)
    products_listed[("ftp", "windows")].add("IIS")
    products_listed[("os", "windows")].add("windows_os")
    prod_with_eligible = defaultdict(set)
    for u in units:
        if u["stage"] in ("remote_candidate", "local_candidate"):
            prod_with_eligible[(u["family"], u["os"], u["cohort"])].add(u["product"])
    mix = Counter()
    for r in rows:
        if r["cohort"]:
            prov = sorted({("NVD-Primary" if v["origin"] == "nvd" and v["type"] == "Primary" else
                            "MSRC" if v["origin"] == "msrc" else "CNA/Secondary") + "-v" + v["version"][0]
                           for v in r["vectors"]})  # fmt: skip
            mix[(r["cohort"], "+".join(prov) or "none")] += 1
    conflicts = Counter((r["cohort"], ",".join(r["vector_resolution"]["conflict_fields"])) for r in rows
                        if r["vector_resolution"]["status"] == "REVIEW_REQUIRED")  # fmt: skip
    scope_diff = Counter(r["cohort"] for r in rows if len(r["vector_resolution"].get("cvss_scope_values", [])) > 1)
    roles = Counter((u["family"], u["os"], u["cohort"], u["role"], u["role_reason"]) for u in units)

    # ---- write the dataset (untracked) and the run (tracked)
    raw = {
        "rep6": rep6, "msrc_docs": msrc_docs, "ghsa_pages": sum(len(v) for v in ghsa.values()),
        "build_counts": dict(build), "rows": len(rows),
        "rows_per_cohort": dict(Counter(str(r["cohort"]) for r in rows)),
        "cpe_part_mismatch_rows": sum(1 for r in rows if r["cpe_part_mismatch"]),
        "funnel": [[*k, n] for k, n in sorted(funnel.items())],
        "eligible": [[*k, len(v)] for k, v in sorted(eligible.items())],
        "products_listed": {f"{f}|{o}": sorted(v) for (f, o), v in sorted(products_listed.items())},
        "products_with_eligible": [[*k, sorted(v)] for k, v in sorted(prod_with_eligible.items())],
        "provider_mix": [[*k, n] for k, n in sorted(mix.items())],
        "vector_conflicts": [[*k, n] for k, n in sorted(conflicts.items(), key=str)],
        "cvss_scope_disagreement_rows": dict(scope_diff),
        "roles": [[*k, n] for k, n in sorted(roles.items(), key=str)],
        "unmapped_component_tags": dict(unmapped_tags.most_common()),
        "role_checks": checks,
        "stages": list(cd.FUNNEL_STAGES),
    }
    data = "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode() + b"\n"
    raw["dataset_sha256"] = hashlib.sha256(data).hexdigest()
    if a.dry_run:
        print(render(raw))
        return
    run = runlog.write_run("cve_master", raw, render(raw), [], cfg, root=ROOT / "results" / "d28_data")
    out = ROOT / "data" / "d28" / "master" / run.name
    out.mkdir(parents=True, exist_ok=False)
    (out / "cves.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
    print(run, out, raw["rows"], raw["dataset_sha256"])


def md_table(header: list[str], body: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in body])  # fmt: skip


def render(raw: dict) -> str:
    """report.md for the run, rendered only from raw (BUILD_RULES rule 6)."""
    stages = raw["stages"][1:]
    f = defaultdict(Counter)
    for fam, os_, prod, coh, stage, n in raw["funnel"]:
        f[(fam, os_, prod, coh)][stage] += n
    body = [[*k, sum(c.values()), *[c[s] for s in stages]] for k, c in sorted(f.items())]
    tot = defaultdict(Counter)
    for fam, os_, prod, coh, stage, n in raw["funnel"]:
        tot[coh][stage] += n
    body += [["**all**", "", "", coh, sum(c.values()), *[c[s] for s in stages]] for coh, c in sorted(tot.items())]
    el = defaultdict(Counter)
    for fam, os_, coh, stage, n in raw["eligible"]:
        el[(fam, os_, coh)][stage] = n
    prod = {(fam, os_, coh): v for fam, os_, coh, v in raw["products_with_eligible"]}
    pl = raw["products_listed"]
    out = [
        "# D28 master CVE dataset and exclusion funnel (P2)", "",
        f"Rows (canonical CVE IDs): {raw['rows']}; per cohort: {raw['rows_per_cohort']}; dataset SHA-256 `{raw['dataset_sha256']}`.", "",
        "## REP6 snapshot validation", "",
        md_table(["snapshot", "ok", "pages", "records", "unique CVEs", "CVEs in several queries"],
                 [[k, v["ok"], v["pages"], v["records"], v["unique_cves"], v["cves_in_several_queries"]] for k, v in raw["rep6"].items()]),
        "", f"MSRC CVRF documents read: {len(raw['msrc_docs'])}. GHSA advisory pages read: {raw['ghsa_pages']}.",
        f"Build counts: {raw['build_counts']}. Rows with a CPE part mismatch on a listed product (F6, not counted as hits): {raw['cpe_part_mismatch_rows']}.", "",
        "## Exclusion funnel per family × OS × product × cohort", "",
        "Units are (CVE, product) pairs. Each unit is counted once, at the first filter that excludes it, in column order.", "",
        md_table(["family", "OS", "product", "cohort", "in cohort", *stages], body), "",
        "## Eligible before effect classification, unique CVEs per family × OS × cohort", "",
        md_table(["family", "OS", "cohort", "remote (AV:N)", "local (AV:L)", "products with ≥1 eligible CVE", "products listed"],
                 [[*k, c["remote_candidate"], c["local_candidate"], len(prod.get(k, [])), len(pl.get(f"{k[0]}|{k[1]}", []))]
                  for k, c in sorted(el.items())]), "",
        "## Products per family per OS", "",
        md_table(["family", "OS", "products listed", "names"], [[*k.split("|"), len(v), ", ".join(v)] for k, v in pl.items()]), "",
        "## Provider mix per cohort (rows)", "",
        md_table(["cohort", "providers (v = CVSS major version)", "rows"], raw["provider_mix"]), "",
        "## AV/PR/UI conflicts between providers (REVIEW_REQUIRED rows)", "",
        md_table(["cohort", "fields that differ", "rows"], raw["vector_conflicts"]), "",
        f"Rows whose v3 vectors disagree on CVSS Scope (S): {raw['cvss_scope_disagreement_rows']}.", "",
        "## Role outcomes (units in a cohort)", "",
        md_table(["family", "OS", "cohort", "role", "reason", "units"], raw["roles"]), "",
        "## Role checks (D28 E5)", "", md_table(["CVE", "want", "got", "ok"], [[k, v["want"], v["got"], v["ok"]] for k, v in raw["role_checks"].items()]), "",
        "## MSRC tags with a family keyword not mapped to a family (on Windows Server CVEs)", "",
        md_table(["tag", "CVEs"], list(raw["unmapped_component_tags"].items())), "",
    ]  # fmt: skip
    return "\n".join(out)


if __name__ == "__main__":
    main()
