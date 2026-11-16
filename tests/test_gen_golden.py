"""P6: the D28 generator reproduces its versioned golden digests (tests/golden_d28.py)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import golden_d28  # noqa: E402

GOLDEN = golden_d28.DATA / "golden_d28_gen_v1.json"
CFG = json.loads((golden_d28.ROOT / "configs" / "gen" / "p6.json").read_text())
HAVE = (golden_d28.ROOT / CFG["catalogue"]["path"]).exists() and (golden_d28.ROOT / CFG["master"]["path"]).exists()


@pytest.mark.skipif(not HAVE, reason="pinned P6 inputs not in this checkout")
def test_d28_generator_matches_golden_v1():
    from ahg.gen.generate import Context

    g = json.loads(GOLDEN.read_text())
    ctx = Context.load()
    data_inputs = {k: v for k, v in ctx.input_hashes.items() if k != "config"}
    assert data_inputs == {k: v for k, v in g["inputs"].items() if k != "config"}, "pinned inputs changed"
    assert set(g["digests"]) == {golden_d28.key(*c) for c in golden_d28.CASES}
    for c in golden_d28.CASES:
        assert golden_d28.case_digest(ctx, *c) == g["digests"][golden_d28.key(*c)], c


def test_golden_files_are_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(golden_d28, "DATA", tmp_path)
    (tmp_path / "golden_d28_gen_v1.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["golden_d28.py", "--version", "v1", "--change", "x"])
    with pytest.raises(SystemExit, match="never overwritten"):
        golden_d28.main()
