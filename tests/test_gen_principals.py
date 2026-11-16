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
