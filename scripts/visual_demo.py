"""Visual live demo: build a fresh network, run A* with every heuristic, draw the searches, open a slideshow.

  uv run --with matplotlib python scripts/visual_demo.py --family hub_and_spoke --seed 90129
  uv run --with matplotlib python scripts/visual_demo.py --family tiered --seed 90214
  uv run --with matplotlib python scripts/visual_demo.py --family flat --random

Writes demo_out/<family>-<seed>/slides.html (one self-contained file, images embedded) and opens it in the browser.
Arrow keys or the buttons move between slides. Seeds >= 90000 are outside every train/validation/test split.
Network drawing code is draw_demo.py (same figures as presentation/demo.pdf).
"""

import argparse
import base64
import html
import json
import random
import sys
import time
import webbrowser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import draw_demo  # noqa: E402

from ahg.attack_graph import KnobInfeasible, UnreachableGoal, make_instance  # noqa: E402
from ahg.cve import load_table  # noqa: E402
from ahg.heuristics import HEURISTICS  # noqa: E402
from ahg.labels import CVES  # noqa: E402
from ahg.learning.heuristic import Learned  # noqa: E402
from ahg.learning.train import load  # noqa: E402
from ahg.search import best_first_search  # noqa: E402
from ahg.splits import COUPLINGS, cve_pool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"A": "results/training/d23t1/gnnres-64x3-lr0.001-rank-s0/2026-10-31T084318Z-0126a7d",
          "B": "results/training/d23t1/gnnres-64x3-lr0.001-mse-s0/2026-10-31T094018Z-0126a7d"}
BASE = ("h0", "hsp", "hmax", "hadd", "lmcut")
BUDGET = 200_000
CAPTION = {
    "h0": "No estimate: A* orders states by cost so far only, so it explores almost every partial attack before it "
          "can prove the cheapest one.",
    "hsp": "Counts remaining hops to the goal but ignores credential tiers and coupling, so it barely helps.",
    "hmax": "Prices only the single most expensive requirement of each AND step, so it under-counts coupled exploits "
            "and credentials.",
    "hadd": "Adds every requirement: close to the plan length, but it double-counts shared steps, can overestimate, "
            "and then loses the optimality guarantee.",
    "lmcut": "The strongest classical estimate: finds the bottleneck steps every attack must pay for. Few states, "
             "but each estimate is expensive to compute.",
    "A": "Our GNN (Model A, ranking loss): learned from exact costs, correction on top of h_max. Built for few "
         "expansions; plan cost may be slightly above optimal.",
    "B": "Our GNN (Model B, MSE): built for accurate values, so plans stay essentially optimal.",
}


def build(family, seed, n, D, pool):
    rho, m = COUPLINGS[seed % len(COUPLINGS)]
    if rho == 0.0:
        return None, rho, m
    try:
        return make_instance(family, seed, n, pool, cred_levels=D, coupling=rho, share_pool=m,
                             exposed_level0=True), rho, m
    except (UnreachableGoal, KnobInfeasible):
        return None, rho, m


def bits(s):
    return [i for i in range(s.bit_length()) if s >> i & 1]


def run_all(inst, learned):
    task = inst.task
    runs = {}
    for name in BASE + tuple(learned):
        h = HEURISTICS[name] if name in HEURISTICS else learned[name].bind(inst)
        trace = []
        t0 = time.perf_counter()
        r = best_first_search(task, h, max_expansions=BUDGET, trace=trace)
        dt = time.perf_counter() - t0
        runs[name] = {"expanded": r.expanded, "cost": r.cost if r.solved else None,
                      "plan": [task.actions[i].name for i in r.plan] if r.solved else None,
                      "plan_actions": r.plan, "h0": h(task, task.init), "ms": dt * 1000,
                      "trace": [bits(s) for s in trace]}
    return runs


def img(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="hub_and_spoke", choices=["flat", "tiered", "hub_and_spoke"])
    ap.add_argument("--seed", type=int, default=90129)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--D", type=int, default=3)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    pool = cve_pool(load_table(CVES), "train")
    learned = {k: Learned(*load(ROOT / p / "model.pt"), k) for k, p in MODELS.items()}
    seed = random.randint(90000, 99999) if args.random else args.seed
    inst, rho, m = build(args.family, seed, args.n, args.D, pool)
    while inst is None:  # skip seeds with no coupling or no reachable goal
        seed += 1
        inst, rho, m = build(args.family, seed, args.n, args.D, pool)
    task, net = inst.task, inst.network
    print(f"network {args.family} seed {seed}: {len(task.actions)} actions; running 7 searches ...", flush=True)
    runs = run_all(inst, learned)
    rec = {"family": args.family, "seed": seed, "n": args.n, "D": args.D, "rho": rho, "m": m,
           "hosts": [{"id": x.id, "name": x.name, "zone": x.zone} for x in net.hosts],
           "services": [{"id": s.id, "host": s.host, "family": s.family, "cve": inst.cves[s.id].cve_id,
                         "E": inst.cves[s.id].exploitability_score, "pr": inst.cves[s.id].privileges_required}
                        for s in net.services],
           "reach": [[r.src, r.service] for r in net.reach], "trust": [[t.src, t.dst] for t in net.trust],
           "target": inst.target, "levels": inst.levels, "coupled": inst.coupled, "facts": list(task.facts),
           "actions": [{"name": a.name, "pre": list(a.pre), "add": list(a.add), "cost": a.cost}
                       for a in task.actions],
           "runs": runs}
    out = ROOT / "demo_out" / f"{args.family}-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "instance.json").write_text(json.dumps(rec))
    sys.argv = ["draw_demo", str(out / "instance.json"), str(out)]
    draw_demo.main()

    opt = runs["lmcut"]["cost"]
    order = [k for k in draw_demo.ORDER if k in runs]
    names = {"h0": "normal A* (h0)", "hsp": "h_sp", "hmax": "h_max", "hadd": "h_add (inadmissible)",
             "lmcut": "LM-cut", "A": "GNN Model A", "B": "GNN Model B"}
    slides = []
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in runs["lmcut"]["plan"])
    slides.append(
        f"<h1>A fresh network: {args.family.replace('_', '-')}, seed {seed}</h1>"
        f"<div class=row><img src='{img(out / 'overview.png')}'><div>"
        f"<p>{len(net.hosts) - 1} hosts, {len(task.actions)} possible actions, {len(inst.coupled or {})} coupled "
        f"exploits. Seed {seed} is outside every train / validation / test split.</p>"
        f"<p><b>Cheapest attack ({len(runs['lmcut']['plan'])} steps, certified optimal by A*+LM-cut):</b></p>"
        f"<ol class=steps>{steps}</ol><p class=small>Blue: exploit steps. Purple dashed: the exploit also needs "
        f"control of a third host (AND). Orange: credentials harvested for the next tier.</p></div></div>")
    for k in order:
        r = runs[k]
        tag = "optimal" if r["cost"] == opt else f"{r['cost'] / opt:.2f}x optimal"
        slides.append(
            f"<h1>A* with {names[k]}: {r['expanded']:,} states</h1><img class=wide src='{img(out / f'search-{k}.png')}'>"
            f"<p>{CAPTION[k]}</p><p class=small>Plan {tag}. Time {r['ms']:.1f} ms. Blue: the attack this search "
            f"returned. Red halo: how often a host appears in the states it expanded.</p>")
    rows = "".join(
        f"<tr class='{'ours' if k in 'AB' and len(k) == 1 else ''}'><td>{names[k]}</td><td>{runs[k]['expanded']:,}</td>"
        f"<td>{runs[k]['cost'] / opt:.3f}</td><td>{runs[k]['ms']:.1f}</td></tr>" for k in order)
    slides.append(f"<h1>Summary on this network</h1><table><tr><th>search</th><th>states expanded</th>"
                  f"<th>plan cost / optimum</th><th>time (ms)</th></tr>{rows}</table>"
                  f"<p class=small>Over all test networks (report, Table I): Model A expands 0.59x the nodes of "
                  f"A*+h_max and ties LM-cut; it is slower than h_max in wall-clock time because each GNN call costs "
                  f"more than one h_max pass.</p>")
    page = ("<!doctype html><meta charset=utf-8><title>Live demo</title><style>"
            "body{font-family:Helvetica,Arial,sans-serif;margin:0;background:#fff;color:#111}"
            ".s{display:none;padding:18px 40px}.s.on{display:block}h1{color:#0a5aa0;font-size:28px;margin:0 0 10px}"
            "img{max-width:100%}img.wide{max-height:62vh;display:block;margin:auto}.row{display:flex;gap:24px}"
            ".row img{max-width:48%;max-height:75vh}.steps{font-size:14px;padding-left:22px}.small{font-size:13px;"
            "color:#555}p{font-size:18px}table{border-collapse:collapse;font-size:20px;margin:20px 0}"
            "td,th{border-bottom:1px solid #ccc;padding:6px 22px;text-align:right}td:first-child,th:first-child"
            "{text-align:left}tr.ours{background:#e8f1fb;font-weight:bold}.nav{position:fixed;bottom:8px;right:16px;"
            "font-size:14px;color:#555}button{font-size:16px}</style>"
            + "".join(f"<div class=s>{s}</div>" for s in slides)
            + "<div class=nav><button onclick=go(-1)>&larr;</button> <span id=c></span> "
              "<button onclick=go(1)>&rarr;</button></div><script>"
              "var S=document.querySelectorAll('.s'),i=0;function show(){S.forEach((s,j)=>s.classList.toggle('on',j==i));"
              "document.getElementById('c').textContent=(i+1)+' / '+S.length}"
              "function go(d){i=Math.max(0,Math.min(S.length-1,i+d));show()}"
              "document.onkeydown=e=>{if(e.key=='ArrowRight'||e.key==' ')go(1);if(e.key=='ArrowLeft')go(-1)};show()"
              "</script>")
    (out / "slides.html").write_text(page)
    print(f"wrote {out / 'slides.html'}")
    if not args.no_open:
        webbrowser.open((out / "slides.html").as_uri())


if __name__ == "__main__":
    main()
