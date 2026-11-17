"""P6 runtime-principal table: its candidate sources, and every cited identity against the raw page."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "configs" / "gen" / "principal_sources.json").read_text())


def test_principal_sources_are_well_formed():
    names = [p["name"] for p in SOURCES["pages"]]
    assert len(names) == len(set(names))
    assert SOURCES["pace_seconds"] >= 2
    for p in SOURCES["pages"]:
        assert re.fullmatch(r"https?://[^\s]+", p["url"]) and "/" not in p["name"]
        assert p["for"] and all(re.fullmatch(r"[^|]+\|[a-z]+", k) for k in p["for"])


TABLE = json.loads((ROOT / "configs" / "gen" / "principals.json").read_text())


def page_text(raw: bytes, html_page: bool = True) -> str:
    """Whitespace collapsed (how quotes are compared); for an HTML page also tags and scripts removed and
    entities decoded. Plain-text pages keep their '<' and '>' (e.g. 'Exim < 4.92')."""
    import html

    t = raw.decode("utf-8", "replace")
    if html_page:
        t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return " ".join(t.split())


URLS = {p["url"] for f in sorted((ROOT / "configs" / "gen").glob("principal_sources*.json"))
        for p in json.loads(f.read_text())["pages"]}


def cited():
    """(key, statement) for every cited statement: product entries and CVE-specific ones."""
    for key, e in TABLE["entries"].items():
        if e["principal"] != "UNKNOWN":
            yield key, e
        for cve, c in e.get("cves", {}).items():
            yield f"{key}|{cve}", c


def test_every_entry_is_cited_or_unknown_with_a_reason():
    assert TABLE["entries"]
    for key, e in TABLE["entries"].items():
        assert re.fullmatch(r"[^|]+\|[a-z]+", key)
        if e["principal"] == "UNKNOWN":
            assert e["reason"] and "evidence" not in e
        assert all(re.fullmatch(r"CVE-\d{4}-\d+", c) for c in e.get("cves", {}))
    for key, e in cited():
        ev = e["evidence"]
        assert e["principal"] != "UNKNOWN" and e["process"]
        assert ev["url"] in URLS and re.fullmatch(r"[0-9a-f]{64}", ev["sha256"]) and ev["quotes"]
        assert ev["retrieved_utc"].startswith("20")


def test_root_or_system_only_where_the_quote_says_so():
    for key, e in cited():
        if e["principal"] in ("root", "SYSTEM"):
            assert any(e["principal"].lower() in q.lower() for q in e["evidence"]["quotes"]), key


def test_cited_quotes_are_verbatim_in_the_hashed_raw_page():
    import hashlib

    import pytest

    if not all((ROOT / m).exists() for m in TABLE["raw_manifests"]):
        pytest.skip("raw principal pages not in this checkout")
    man = {}
    for m, sha in TABLE["raw_manifests"].items():
        assert hashlib.sha256((ROOT / m).read_bytes()).hexdigest() == sha
        man |= {f"{Path(m).parent.as_posix()}/{f['name']}": f for f in json.loads((ROOT / m).read_text())["files"]}
    for key, e in cited():
        ev = e["evidence"]
        raw = (ROOT / ev["raw"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ev["sha256"] == man[ev["raw"]]["sha256"]
        assert man[ev["raw"]]["url"] == ev["url"]
        text = page_text(raw, html_page=ev["raw"].endswith(".html"))
        for q in ev["quotes"]:
            assert q in text, (key, q)


def test_the_table_covers_every_remote_pool_product():
    import pytest

    from ahg.gen import pool

    cfg = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())
    if not (ROOT / cfg["master"]["path"]).exists():
        pytest.skip("pinned inputs not in this checkout")
    units, _, _ = pool.load_pool(cfg)
    assert {f"{u.product}|{u.family}" for u in units if u.kind == "remote"} == set(TABLE["entries"])


def test_every_principal_sources_config_is_well_formed():
    for path in sorted((ROOT / "configs" / "gen").glob("principal_sources*.json")):
        cfg = json.loads(path.read_text())
        names = [p["name"] for p in cfg["pages"]]
        assert len(names) == len(set(names)) and cfg["pace_seconds"] >= 2, path
        for p in cfg["pages"]:
            assert re.fullmatch(r"https?://[^\s]+", p["url"]) and "/" not in p["name"]
            assert p["for"] and all(re.fullmatch(r"[^|]+\|[a-z]+", k) for k in p["for"])


def test_rollup_cumulative_citation_is_verbatim_in_its_hashed_page():
    """OQ-P6-16: Microsoft's own update-history page states that the newest update includes the previous ones."""
    import hashlib

    import pytest

    cite = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())["rollup_cumulative"]
    assert cite["url"].startswith("https://support.microsoft.com/") and re.fullmatch(r"[0-9a-f]{64}", cite["sha256"])
    raw_path = ROOT / cite["raw"]
    if not raw_path.exists():
        pytest.skip("P5w raw page not in this checkout")
    raw = raw_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == cite["sha256"]
    assert cite["quote"] in page_text(raw)


def test_citation_sources_configs_are_well_formed_and_the_fetch_script_takes_a_source():
    """Other cited sources (e.g. NASim's firewall rule, OQ-P6-18) use the same paced fetch into data/raw/<source>/."""
    paths = sorted((ROOT / "configs" / "gen").glob("citation_sources_*.json"))
    assert paths
    for path in paths:
        cfg = json.loads(path.read_text())
        names = [p["name"] for p in cfg["pages"]]
        assert len(names) == len(set(names)) and cfg["pace_seconds"] >= 2, path
        assert all(re.fullmatch(r"https://[^\s]+", p["url"]) and "/" not in p["name"] for p in cfg["pages"])
    script = (ROOT / "scripts" / "fetch_principals.py").read_text()
    assert '"--source"' in script and '"data" / "raw" / a.source / a.date' in script


def test_firewall_rule_citation_is_verbatim_in_its_hashed_nasim_pages():
    """OQ-P6-18 (§18, 17 Nov 2026): NASim's at-least-one firewall guarantee, and the NASim parts not taken."""
    import hashlib

    import pytest

    cite = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())["firewall_rule"]
    assert "2191a3a7f15a178bd00a6321151f3492ce118046" in cite["url"]
    pages = [(cite, False), (cite["not_taken"]["restrictiveness"], True)]
    for c, is_html in pages:
        assert re.fullmatch(r"[0-9a-f]{64}", c["sha256"])
        raw_path = ROOT / c["raw"]
        if not raw_path.exists():
            pytest.skip("NASim raw snapshot not in this checkout")
        raw = raw_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c["sha256"]
        assert c["quote"] in page_text(raw, html_page=is_html)
    src = page_text((ROOT / cite["not_taken"]["vulnerable_only"]["raw"]).read_bytes(), html_page=False)
    assert cite["not_taken"]["vulnerable_only"]["quote"] in src
