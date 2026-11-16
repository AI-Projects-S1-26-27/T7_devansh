"""Fetch scripts: no machine-specific paths (inventory #6) and correct query planning."""

import importlib.util
import re
from datetime import date
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_hard_coded_user_or_scratchpad_paths():
    """Inventory #6: scripts resolve paths from the repo root, never from one machine's home or temp dirs."""
    bad = re.compile(r"/Users/|/home/|/private/tmp|scratchpad/")
    offenders = [p.name for p in SCRIPTS.glob("*.py") if bad.search(p.read_text())]
    assert offenders == []


def test_fetch_script_docstrings_name_the_script():
    """Inventory #6: the usage line names the file it is in, and the unimplemented "0 tried first" claim is gone."""
    for name in ("nvd_count_only", "fetch_sources"):
        doc = load(name).__doc__
        assert doc.lstrip().startswith(f"{name}.py"), doc[:40]
        assert "tried first" not in doc


def test_nvd_windows_cover_range_within_api_limit():
    w = load("fetch_nvd_cpe").windows(date(2016, 1, 1), date(2026, 10, 21))
    assert w[0][0] == date(2016, 1, 1) and w[-1][1] == date(2026, 10, 21)
    assert all((e - s).days + 1 <= 120 for s, e in w)  # NVD: at most 120 consecutive days per query
    assert all((b[0] - a[1]).days == 1 for a, b in zip(w, w[1:]))  # no gap, no overlap


def test_msrc_update_ids_in_month_range():
    m = load("fetch_msrc")
    index = {"value": [{"ID": i} for i in ("2023-Dec", "2024-Jan", "2024-Jan-B", "2018-FEB", "2026-Oct", "2026-Nov", "x")]}
    assert m.ids_in_range(index, (2024, 1), (2026, 10)) == ["2024-Jan", "2024-Jan-B", "2026-Oct"]
    assert m.update_month("2018-FEB") == (2018, 2) and m.update_month("x") is None


def test_ghsa_refs_picks_repository_advisories_in_window():
    m = load("fetch_ghsa")

    def cve(cid, pub, *urls):
        return {"cve": {"id": cid, "published": pub, "references": [{"url": u} for u in urls]}}

    base = "https://github.com/FreeRDP/FreeRDP/security/advisories/"
    pages = [{"response": {"vulnerabilities": [
        cve("CVE-2023-39354", "2023-08-31T17:15:09.000", base + "GHSA-c3r2-pxxp-f8r6", "https://example.org/x"),
        cve("CVE-2015-0001", "2015-01-01T00:00:00.000", base + "GHSA-hm8c-rcjg-c8qp"),
        cve("CVE-2023-1", "2023-01-01T00:00:00.000", "https://github.com/advisories/GHSA-hm8c-rcjg-c8qp"),
    ]}}]  # fmt: skip
    assert m.ghsa_refs(pages) == {
        "GHSA-c3r2-pxxp-f8r6__FreeRDP__FreeRDP.html": base + "GHSA-c3r2-pxxp-f8r6",
        "GHSA-hm8c-rcjg-c8qp__global.html": "https://github.com/advisories/GHSA-hm8c-rcjg-c8qp",
    }
