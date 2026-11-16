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
