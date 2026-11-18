# Learned Heuristics for A* Search over Attack Graphs

Research project. Author: Devansh Sharma.

An attack graph shows how an attacker can chain exploits through a network. Finding the cheapest attack is a
classical planning problem: states are the privileges the attacker holds, actions are CVE exploits, and costs come
from CVSS. This project asks whether a graph neural network (GNN), trained on optimal cost-to-go, can make A*
expand fewer nodes than classical heuristics (h_max, h_add, LM-cut) on attack graphs of unseen networks, and
whether that holds under shifts in network size, topology and vulnerabilities.

## What is in the repository

| Path | Contents |
|---|---|
| `src/ahg/` | the Python package: planning core, search, heuristics, oracle and reference solver, attack-graph generator, CVE data pipeline, labels, learning (`learning/`), evaluation and statistics |
| `scripts/` | command-line entry points (data fetching, generation, labelling, training, evaluation, analysis) |
| `configs/` | every experiment's configuration (JSON) |
| `tests/` | the test suite |
| `data/derived/` | the derived CVE tables (tracked); `data/raw/`, `data/labels/` and caches stay local |
| `third_party/` | Fast Downward 24.06.1, built by `scripts/setup_fast_downward.sh` (not tracked) |

## Two studies

- **Monotonic study** (frozen at the tag `monotonic-study-frozen`): synthetic flat, tiered and hub-and-spoke
  networks with CVEs from NVD, Active-Directory-style credential tiering and cross-host coupling; residual GNN
  heuristics trained on A*-evaluated states and evaluated once on held-out test splits.
- **D28 rebuild** (branch `d28-rebuild`, in progress): a revised benchmark with CVE effect classification,
  CPE-based product and OS fixtures, CVSS prerequisites, firewall zones, a frozen pre-registration and an
  exploratory access-loss model with delete effects.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run pytest -q                      # full test suite
scripts/setup_fast_downward.sh        # optional: the compiled baseline planner
```

## Running

```bash
uv run python scripts/toy_demo.py                         # smallest end-to-end example
uv run python scripts/d28_pipe.py --help                  # D28 pipeline: build, label, n4, landscape, budget, gap, evaluate, report
uv run python scripts/train.py --help                     # training
uv run python scripts/evaluate.py --help                  # A* evaluation harness
```

Every run writes a folder under `results/` with its configuration, git commit, seeds, library versions and
hardware. Test splits are read only through a logged, once-only access path.

## Data and acknowledgements

This product uses the NVD API but is not endorsed or certified by the NVD. CVE records are © MITRE; see `NOTICE`.
