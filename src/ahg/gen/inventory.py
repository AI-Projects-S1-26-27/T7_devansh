"""What a generated host can run: products per (OS, family), their fixtures, CPE names and status (P6).

Sources: the P5 eligible-view catalogue (SHA-256-pinned) for fixtures and the action pool (ahg.gen.pool)
for CVEs. Every number the generator can draw comes from these two files.

Components and their CPE names (matching is CPE-only, ahg.gen.cpe):
- Linux service: `cpe:2.3:a:<vendor>:<product>:<version>:<update or ->:...`; the kernel is part `o`.
- Windows OS: the version's CPE name from the config, with the MSRC build as its version where NVD encodes
  builds (2016/2019/2022). NVD encodes Server 2012 R2 as `windows_server_2012:r2`, so a 2012 R2 build is
  not visible to CPE matching; the build is still recorded.
- Windows built-in services (SMB, RDS, DNS, IIS for http and ftp, OpenSSH for Windows) are components of
  the OS: they carry the OS CPE name and the OS build. A built-in is OS-compatible with a Windows version
  only if MSRC lists it for that version (the catalogue's `units`).
- Exchange, SQL Server: own builds from MSRC, CPE names by the config's name patterns.

Status of a fixture for a cohort (Round 4: from that cohort's pool CVEs only): `vulnerable` if some cohort
unit of the component has a non-negated vulnerable cpeMatch that matches one of the fixture's CPE names,
else `unaffected`. Co-conditions are ignored here, so an unaffected fixture is unaffected on every host;
a vulnerable one is vulnerable on its own (the final match on a real host still evaluates every
co-condition). A Windows OS build's status counts the units of the OS and of every built-in service,
because they all ride on that build. `unverified_release` fixtures are vulnerable-only and used at p = 1
only; an unaffected fixture is any other catalogue fixture with no cohort match (OQ-P6-3).
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ahg.gen import cpe as cpe_eval
from ahg.gen.pool import ROOT, Unit, sha256_file


@dataclass(frozen=True)
class Fixture:
    """One installable version of a product (or one Windows build), with its CPE names."""

    product: str  # generator product name (label product, or a Windows Server version)
    version: str
    cpe_names: tuple[str, ...]
    unverified: bool = False
    release_line: str = ""  # MSRC product name for Windows builds; "" for Linux
    builtins: tuple[str, ...] = ()  # Windows OS builds: the built-in products MSRC lists for this version


@dataclass
class Product:
    os: str
    family: str
    name: str  # label product, e.g. "apache:http_server", "Exchange", "Windows DNS Server", "Windows Server 2019"
    fixtures: list[Fixture] = field(default_factory=list)
    builtin: bool = False  # Windows built-in: no fixtures of its own, rides on the OS build


def load_catalogue(cfg: dict, root: Path = ROOT) -> dict:
    path = root / cfg["catalogue"]["path"]
    got = sha256_file(path)
    if got != cfg["catalogue"]["sha256"]:
        raise ValueError(f"{path}: SHA-256 {got} does not match the pinned {cfg['catalogue']['sha256']}")
    with gzip.open(path, "rt") as f:
        return json.load(f)


def linux_cpe(product: str, version: str, update: str, cfg: dict) -> str:
    part = cfg["cpe_part"].get(product, cfg["cpe_part"]["default"])
    vendor, prod = product.split(":")
    return f"cpe:2.3:{part}:{vendor}:{prod}:{version}:{update or '-'}:*:*:*:*:*:*"


def msrc_cpe_names(product: str, msrc_name: str, build: str, cfg: dict) -> tuple[str, ...] | None:
    """CPE names of an Exchange / SQL Server build from its MSRC product name; None if no pattern fits."""
    for pat in cfg["windows_products"][product]["cpe"]:
        m = re.match(pat["re"], msrc_name)
        if m:
            subst = lambda t: re.sub(r"\{(\d)\}", lambda g: m.group(int(g.group(1))), t).replace("{build}", build)
            return tuple(subst(t) for t in pat["names"])
    return None


def products(cfg: dict, catalogue: dict) -> tuple[list[Product], list[dict]]:
    """Every product the catalogue gives fixtures for, plus the Windows built-ins; (products, notes)."""
    out, notes = [], []
    never = set(cfg["never_products"])
    for name, row in sorted(catalogue["linux"].items()):
        if name in never:
            notes.append({"product": name, "dropped": cfg["never_products"][name]})
            continue
        fams = row["families"]
        if len(fams) != 1:
            raise ValueError(f"{name}: expected one family, got {fams}")
        fx = [Fixture(name, f["version"], (linux_cpe(name, f["cpe_version"], f["cpe_update"], cfg),),
                      bool(f.get("unverified_release"))) for f in row["fixtures"]]
        out.append(Product("linux", fams[0], name, fx))
    win = catalogue["windows"]
    for version, template in cfg["windows_versions"].items():
        row = win[version]
        builtins = tuple(u for u in row["units"] if u != "windows_os")
        fx = [Fixture(version, f["version"], (template.replace("{build}", f["version"]),), False, version, builtins)
              for f in row["fixtures"]]
        out.append(Product("windows", "os", version, fx))
    for prod, spec in cfg["windows_products"].items():
        fx = []
        for msrc_name, row in sorted(win.items()):
            if not msrc_name.startswith(spec["prefix"]) or not row["fixtures"]:
                continue
            for f in row["fixtures"]:
                names = msrc_cpe_names(prod, msrc_name, f["version"], cfg)
                if names is None:
                    notes.append({"product": prod, "release_line": msrc_name, "dropped": "no CPE name pattern"})
                    break
                fx.append(Fixture(prod, f["version"], names, False, msrc_name))
        out.append(Product("windows", spec["family"], prod, fx))
    for fam, prod in cfg["windows_builtin"].items():
        out.append(Product("windows", fam, prod, [], builtin=True))
    return out, notes


def potential_cves(units: list[Unit], vuln: dict[str, list[dict]], names: tuple[str, ...]) -> list[str]:
    """CVE IDs of `units` with a non-negated vulnerable cpeMatch matching one of `names` (status rule).

    vuln: CVE ID -> its vulnerable matches (cpe.vulnerable_matches), computed once per inventory.
    """
    keys = {tuple(n.split(":")[2:5]) for n in names}  # (part, vendor, product): a cheap pre-filter only
    hits = []
    for u in units:
        if any(tuple(m["criteria"].split(":")[2:5]) in keys and cpe_eval.name_matches(m, n)
               for m in vuln[u.cve_id] for n in names):
            hits.append(u.cve_id)
    return sorted(set(hits))



@dataclass
class Inventory:
    """Products a run may draw, with fixture statuses for one cohort, exclusion set and p regime."""

    cohort: str
    p: float
    products: dict[tuple[str, str], list[Product]]  # (os, family) -> available products, sorted by name
    windows_versions: list[Product]  # available Windows Server versions
    kernel: Product | None
    status: dict[tuple[str, str], str]  # (product, version) -> "vulnerable" | "unaffected"
    report: dict


def build_inventory(cfg: dict, catalogue: dict, units: list[Unit], master: dict[str, dict], cohort: str,
                    p: float, excluded: frozenset[str] = frozenset()) -> Inventory:
    """Availability per (cohort, p, exclusions). `units` must already be restricted (pool.restrict).

    A product is available if it is not excluded, has a cohort pool unit, and has a usable fixture of every
    status the p regime can draw: p = 1 needs a vulnerable fixture (unverified ones allowed); p < 1 needs a
    verified vulnerable one and an unaffected one. Built-ins need a Windows build they can match and ride on
    the OS draw. Windows versions and the kernel are treated as the OS family's products.
    """
    vuln = {u.cve_id: cpe_eval.vulnerable_matches(master[u.cve_id]["configurations"]) for u in units}
    by_prod: dict[tuple[str, str], list[Unit]] = defaultdict(list)
    for u in units:
        by_prod[(u.product, u.family)].append(u)
    prods, notes = products(cfg, catalogue)
    os_units = [u for u in units if u.os == "windows" and (u.family == "os" or u.product in cfg["windows_builtin"].values())]
    status, report = {}, {"cohort": cohort, "p": p, "excluded": sorted(excluded), "notes": notes, "products": []}
    avail: dict[tuple[str, str], list[Product]] = defaultdict(list)
    versions, kernel = [], None

    def classify(prod: Product, relevant: list[Unit]) -> dict:
        n = {"vulnerable": 0, "vulnerable_verified": 0, "unaffected": 0}
        for f in prod.fixtures:
            hit = potential_cves(relevant, vuln, f.cpe_names)
            st = "vulnerable" if hit else "unaffected"
            if st == "unaffected" and f.unverified:
                st = "unusable"  # unverified releases are vulnerable-only (Round 4)
            status[(prod.name, f.version)] = st
            if st == "vulnerable":
                n["vulnerable"] += 1
                n["vulnerable_verified"] += not f.unverified
            elif st == "unaffected":
                n["unaffected"] += 1
        return n

    for prod in prods:
        row = {"os": prod.os, "family": prod.family, "product": prod.name, "fixtures": len(prod.fixtures)}
        if prod.name in excluded:
            row["available"], row["reason"] = False, "excluded"
            report["products"].append(row)
            continue
        if prod.family == "os" and prod.os == "windows":
            relevant = [u for u in os_units if u.product == "windows_os" or u.product in
                        (prod.fixtures[0].builtins if prod.fixtures else ())]
        elif prod.builtin:
            relevant = by_prod.get((prod.name, prod.family), [])
        else:
            relevant = by_prod.get((prod.name, prod.family), [])
        row["cohort_units"] = len(relevant)
        if prod.builtin:
            ok = bool(relevant)
            row["reason"] = None if ok else "no_cohort_unit"
        else:
            n = classify(prod, relevant)
            row.update(n)
            if not relevant:
                ok, row["reason"] = False, "no_cohort_unit"
            elif p == 1.0:
                ok = n["vulnerable"] > 0
                row["reason"] = None if ok else "no_vulnerable_fixture"
            else:
                ok = n["vulnerable_verified"] > 0 and n["unaffected"] > 0
                row["reason"] = None if ok else ("no_unaffected_fixture" if n["unaffected"] == 0
                                                 else "no_verified_vulnerable_fixture")
        row["available"] = ok
        report["products"].append(row)
        if not ok:
            continue
        if prod.family == "os":
            if prod.os == "windows":
                versions.append(prod)
            else:
                kernel = prod
        else:
            avail[(prod.os, prod.family)].append(prod)
    # a built-in needs a Windows build it can match: checked against the available versions' fixtures
    for key in list(avail):
        keep = []
        for prod in avail[key]:
            if prod.builtin:
                rel = by_prod.get((prod.name, prod.family), [])
                fx = [f for v in versions if prod.name in v.fixtures[0].builtins for f in v.fixtures]
                if not any(potential_cves(rel, vuln, f.cpe_names) for f in fx):
                    next(r for r in report["products"] if r["product"] == prod.name and r["family"] == prod.family
                         ).update(available=False, reason="no_matching_windows_build")
                    continue
            keep.append(prod)
        avail[key] = sorted(keep, key=lambda q: q.name)
    report["windows_versions"] = [v.name for v in versions]
    report["kernel"] = kernel is not None
    return Inventory(cohort, p, {k: v for k, v in sorted(avail.items()) if v}, sorted(versions, key=lambda v: v.name),
                     kernel, status, report)
