"""scripts/build_cve_master.py helpers: hash-verified raw inputs and the MSRC entry choice."""

import hashlib
import importlib.util
import gzip
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_cve_master.py"
spec = importlib.util.spec_from_file_location("build_cve_master", SCRIPT)
bcm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcm)


def test_verified_files_reads_parent_fetch_log_and_detects_edits(tmp_path):
    (tmp_path / "msrc").mkdir()
    (tmp_path / "msrc" / "cvrf_2020-Jan.json").write_bytes(b"{}")
    (tmp_path / "other.html").write_bytes(b"x")
    log = [{"name": "msrc/cvrf_2020-Jan.json", "sha256": hashlib.sha256(b"{}").hexdigest()},
           {"name": "other.html", "sha256": hashlib.sha256(b"x").hexdigest()},
           {"name": "msrc/failed.json", "sha256": None}]  # fmt: skip
    (tmp_path / "fetch_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    assert list(bcm.verified_files(tmp_path / "msrc")) == ["cvrf_2020-Jan.json"]
    (tmp_path / "msrc" / "cvrf_2020-Jan.json").write_bytes(b"{ }")
    with pytest.raises(SystemExit, match="modified"):
        bcm.verified_files(tmp_path / "msrc")


def test_latest_msrc_entry_by_revision_date():
    a = {"doc": "2020-Jan", "revision_dates": ["2020-01-14T08:00:00"]}
    b = {"doc": "2019-Dec", "revision_dates": ["2019-12-10T08:00:00", "2020-02-11T08:00:00"]}
    assert bcm.latest([a, b]) is b


def _write_rawfetch_snapshot(folder: Path, pages: list[tuple[str, dict]]):
    folder.mkdir(parents=True)
    files = []
    for name, resp in pages:
        b = json.dumps(resp).encode()
        (folder / name).write_bytes(b)
        url = f"https://x/api?resultsPerPage=500&startIndex=0&virtualMatchString={name}"
        files.append({"name": name, "url": url, "sha256": hashlib.sha256(b).hexdigest()})
    (folder / "manifest.json").write_text(json.dumps({"files": files}))


def _cve(cid, pub, product, vector, desc, part="a", extra_vectors=()):
    metrics = [{"source": "nvd@nist.gov", "type": "Primary", "cvssData": {"version": "3.1", "vectorString": vector}}]
    metrics += [{"source": "cna@x", "type": "Secondary", "cvssData": {"version": "3.1", "vectorString": v}} for v in extra_vectors]
    return {"cve": {"id": cid, "published": pub, "lastModified": pub, "vulnStatus": "Analyzed", "sourceIdentifier": "x",
                    "descriptions": [{"lang": "en", "value": desc}], "metrics": {"cvssMetricV31": metrics},
                    "weaknesses": [], "references": [],
                    "configurations": [{"nodes": [{"operator": "OR", "cpeMatch": [
                        {"criteria": f"cpe:2.3:{part}:{product}:*:*:*:*:*:*:*:*", "vulnerable": True}]}]}]}}  # fmt: skip


def test_build_end_to_end_on_synthetic_snapshots(tmp_path, monkeypatch):
    """One server CVE becomes a remote candidate; client, AV:A, conflicting and wrong-part CVEs do not."""
    N = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    vulns = [
        _cve("CVE-2020-0001", "2020-01-01T00:00:00.000", "thekelleys:dnsmasq", N, "dnsmasq crash"),
        _cve("CVE-2020-0002", "2020-01-01T00:00:00.000", "freerdp:freerdp", N, "A malicious RDP server can crash the client."),
        _cve("CVE-2020-0003", "2020-01-01T00:00:00.000", "thekelleys:dnsmasq", N.replace("AV:N", "AV:A"), "x"),
        _cve("CVE-2025-0004", "2025-01-01T00:00:00.000", "thekelleys:dnsmasq", N, "x", extra_vectors=[N.replace("UI:N", "UI:R")]),
        _cve("CVE-2020-0005", "2020-01-01T00:00:00.000", "thekelleys:dnsmasq", N, "x", part="o"),
        _cve("CVE-2015-0006", "2015-01-01T00:00:00.000", "thekelleys:dnsmasq", N, "x"),
        _cve("CVE-2021-0007", "2021-01-01T00:00:00.000", "linux:linux_kernel", N.replace("AV:N", "AV:L"), "x", part="o"),
    ]  # fmt: skip
    resp = {"startIndex": 0, "totalResults": len(vulns), "resultsPerPage": len(vulns), "vulnerabilities": vulns}
    _write_rawfetch_snapshot(tmp_path / "nvd", [("p.json", resp)])
    (tmp_path / "msrc").mkdir()
    (tmp_path / "msrc" / "fetch_log.jsonl").write_text("")
    (tmp_path / "ghsa").mkdir()
    (tmp_path / "ghsa" / "fetch_log.jsonl").write_text("")
    cfg = json.loads((SCRIPT.parents[1] / "configs/data/cve_master_p2.json").read_text())
    cfg |= {"nvd_snapshots": {"t": "nvd"}, "msrc_folders": ["msrc"], "ghsa_folder": "ghsa"}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    captured = {}

    def fake_write_run(name, raw, md, seeds, config, root=None):
        captured["raw"], captured["md"] = raw, md
        return tmp_path / "run"

    monkeypatch.setattr(bcm, "ROOT", tmp_path)
    monkeypatch.setattr(bcm, "ROLE_CHECKS", {})
    monkeypatch.setattr(bcm.runlog, "write_run", fake_write_run)
    monkeypatch.setattr("sys.argv", ["build_cve_master.py", str(tmp_path / "cfg.json")])
    bcm.main()
    raw = captured["raw"]
    stages = {(f[2], f[3], f[4]): f[5] for f in raw["funnel"]}
    assert stages[("thekelleys:dnsmasq", "core", "remote_candidate")] == 1
    assert stages[("freerdp:freerdp", "core", "role_not_server")] == 1
    assert stages[("thekelleys:dnsmasq", "core", "AV:A")] == 1
    assert stages[("thekelleys:dnsmasq", "temporal", "REVIEW_REQUIRED")] == 1  # NVD UI:N vs CNA UI:R
    assert stages[("linux:linux_kernel", "core", "local_candidate")] == 1
    # the pre-2016 CVE stays in the dataset without a cohort; the part-o dnsmasq match is not a hit (F6)
    assert raw["rows"] == 6 and raw["rows_per_cohort"] == {"core": 4, "temporal": 1, "None": 1}
    assert raw["build_counts"] == {"cpe_part_mismatch_only": 1}
    rows = [json.loads(x) for x in gzip.decompress((tmp_path / "data/d28/master/run/cves.jsonl.gz").read_bytes()).splitlines()]
    assert {r["cve_id"] for r in rows} == {"CVE-2020-0001", "CVE-2020-0002", "CVE-2020-0003", "CVE-2025-0004",
                                           "CVE-2021-0007", "CVE-2015-0006"}  # fmt: skip
    r4 = next(r for r in rows if r["cve_id"] == "CVE-2025-0004")
    assert r4["vector_resolution"]["conflict_fields"] == ["UI"] and len(r4["vectors"]) == 2  # both providers kept
