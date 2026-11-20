# Learned Heuristics for A* Search over CVE-Grounded Attack Graphs

CS F407 Artificial Intelligence, BITS Pilani Hyderabad. Team 7 (Devansh Sharma, 2022A3PS1298H).

This branch (`final-demo`) is the frozen code of the study reported in the final report, plus the one-shot
test reads and a live demo script. The frozen GNN checkpoints are tracked under `results/training/d23t1/`
(see `results/training/d23t1/frozen.json`), and every test read is logged in `results/test_access_log.jsonl`.

## Setup (Windows, macOS or Linux)

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# install uv
#   macOS / Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   Windows (PowerShell):  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone -b final-demo https://github.com/AI-Projects-S1-26-27/T7_devansh.git
cd T7_devansh
uv sync --locked          # installs Python 3.12 if missing, and the pinned torch (CPU)
```

## Live demo

Builds a fresh network (seed >= 90000, outside every train/validation/test split), runs A* with every
heuristic on it and prints nodes expanded, plan cost, cost relative to the certified optimum, and time.

```bash
uv run python scripts/live_demo.py --family hub_and_spoke --seed 90129   # unseen topology (demo slides 10-15)
uv run python scripts/live_demo.py --family tiered --seed 90214          # training shape (demo slides 17-22)
uv run python scripts/live_demo.py --family flat --random                # a brand-new random network
```

Families: `flat`, `tiered`, `hub_and_spoke`. Options: `--n` hosts (default 20), `--D` credential tiers
(default 3). Costs are printed as exact integers C = 38 c (Sec. III of the report).

### Visual demo (recommended for presenting)

Runs the same searches and opens a click-through slideshow in the browser: the network with the optimal attack,
animated races (normal A* vs the GNN, h_max vs the GNN) on one shared expansion counter, and one animation per
heuristic showing the states it expands and the attack it returns. Arrow keys move between slides; `r` replays
the animation. Takes about 30 s to build. Needs internet once, for matplotlib.

```bash
uv run --with matplotlib python scripts/visual_demo.py --family hub_and_spoke --seed 90129
uv run --with matplotlib python scripts/visual_demo.py --family tiered --seed 90214
uv run --with matplotlib python scripts/visual_demo.py --family flat --random
```

Output is written to `demo_out/<family>-<seed>/slides.html` (open it by hand if no browser starts).

Small example of every heuristic and search algorithm on a toy network:

```bash
uv run python scripts/toy_demo.py
```

## Tests

```bash
uv run pytest
```

## Layout

- `src/ahg/` search (A*, tie-breaking), heuristics (h0, h_sp, h_max, h_add, h_FF, LM-cut), generator, labels,
  GNN (`learning/`), statistics
- `scripts/` pipeline entry points (generation, labels, training, evaluation, analysis) and the demos
- `configs/` every run's configuration
- `data/derived/cves.csv` the CVE table built from the pinned NVD snapshot
- `results/` training runs with checkpoints and provenance, evaluations, analysis, test access log
