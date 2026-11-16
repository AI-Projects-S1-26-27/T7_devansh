"""D28 E6: MulVAL is cited for semantics only; no code claims an equivalence with it (P6 removes the
folded-MulVAL claim from ahg.attack_graph)."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_no_folded_mulval_equivalence_claim_in_code():
    for f in sorted(SRC.rglob("*.py")):
        text = re.sub(r"\s+", " ", f.read_text())
        assert "folded into the actions" not in text, f
        for hit in re.finditer(r"MulVAL", text):
            window = text[max(0, hit.start() - 200): hit.end() + 200]
            assert "no equivalence" in window, (f, window)
