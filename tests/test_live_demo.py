"""The live demo reproduces the expansion counts shown on the demo slides (hub_and_spoke seed 90129)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_live_demo_reproduces_slide_numbers():
    out = subprocess.run([sys.executable, "scripts/live_demo.py", "--family", "hub_and_spoke", "--seed", "90129"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    rows = {line[:22].strip(): line[22:].split() for line in out.splitlines() if line.startswith(("normal", "A* +"))}
    # Columns: expanded, cost, cost/opt, time. Expansions are deterministic; time is not checked.
    assert rows["normal A* (h0)"][:3] == ["12,987", "857", "1.000"]
    assert rows["A* + h_max"][0] == "146"
    assert rows["A* + LM-cut"][:2] == ["10", "857"]
    assert rows["A* + GNN Model A"][:3] == ["10", "857", "1.000"]
    assert rows["A* + GNN Model B"][0] == "11"
