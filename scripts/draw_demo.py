"""Stage 2 of the demo figures: draw one traced instance (output of trace_demo.py).

  uv run --with matplotlib python scripts/visual_demo.py (imported by) <instance.json> <out_dir>

Figures: overview.pdf (the network, tiers, coupling and the optimal attack) and one search-<h>.pdf per heuristic
(the plan that search returned, how often each host appears in the states it expanded, and the expansion counts
of every heuristic on this network). Writes generated/trace_numbers.tex with the counts used on the slides.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

LEVEL_COL = ["#d6eaff", "#fde3b0", "#f8b878", "#e8735f"]
NAMES = {"h0": r"normal A* ($h_0$)", "hsp": r"$h_{\mathrm{sp}}$", "hmax": r"$h_{\max}$", "hadd": r"$h_{\mathrm{add}}$",
         "lmcut": "LM-cut", "A": "GNN (Model A)", "B": "GNN (Model B)"}
ORDER = ["h0", "hsp", "hmax", "hadd", "lmcut", "A", "B"]
STEP = re.compile(r"(exploit (?P<cve>CVE-\d+-\d+) (?P<fam>\w+)@(?P<dst>[\w-]+) from (?P<src>[\w-]+)"
                  r"(?: needs@(?P<need>[\w-]+))?)|(harvest cred\((?P<k>\d)\)@(?P<at>[\w-]+))|"
                  r"(login (?P<lfam>\w+)@(?P<ldst>[\w-]+) from (?P<lsrc>[\w-]+) creds@(?P<lc>[\w-]+))")


def layout(d):
    zones = [h["zone"] for h in d["hosts"]]
    if "hub" not in zones:
        # layered families (tiered, flat): one row per zone, internet on top
        rows = [z for z in ("dmz", "tier1", "tier2", "tier3", "tier4", "internal") if z in zones]
        rows += sorted({z for z in zones if z not in rows and z != "internet"})
        pos = {0: (-0.2, 1.3)}
        for r, z in enumerate(rows):
            hs = [h["id"] for h in d["hosts"] if h["zone"] == z]
            y = 1.3 - 2.5 * (r + 1) / len(rows)
            for i, h in enumerate(hs):
                pos[h] = (-1.55 + 2.8 * (i + 0.5) / len(hs), y + (0.12 if i % 2 else 0.0))
        return pos
    pos = {0: (-1.55, 1.25)}
    hub = [h["id"] for h in d["hosts"] if h["zone"] == "hub"]
    spokes = [h["id"] for h in d["hosts"] if h["zone"] == "spoke"]
    for i, h in enumerate(hub):
        pos[h] = (0.0, 0.0) if len(hub) == 1 else (0.35 * math.cos(2 * math.pi * i / len(hub)),
                                                    0.35 * math.sin(2 * math.pi * i / len(hub)))
    for i, h in enumerate(spokes):
        a = math.pi / 2 - 2 * math.pi * i / len(spokes)
        pos[h] = (math.cos(a), math.sin(a))
    return pos


def parse(plan):
    out = []
    for i, s in enumerate(plan, 1):
        m = STEP.fullmatch(s)
        out.append((i, s, m.groupdict() if m else {}))
    return out


def draw_network(ax, d, pos, plan, heat=None):
    by_name = {h["name"]: h["id"] for h in d["hosts"]}
    levels = {int(k): v for k, v in (d["levels"] or {}).items()}
    fams = {}
    for s in d["services"]:
        fams.setdefault(s["host"], []).append(s["family"])
    # every reach edge, faint: who can talk to whom
    for src, sid in d["reach"]:
        dst = d["services"][sid]["host"]
        (x1, y1), (x2, y2) = pos[src], pos[dst]
        ax.plot([x1, x2], [y1, y2], color="#cccccc", lw=0.5, zorder=1)
    if heat:
        total = heat.pop("total")
        for h, c in heat.items():
            if h in pos and h != 0 and c:  # the internet fact is in every state
                ax.scatter(*pos[h], s=2400 * (c / total) ** 0.5 + 200, color="#e03030", alpha=0.18, zorder=2, lw=0)
    for h in d["hosts"]:
        x, y = pos[h["id"]]
        if h["id"] == 0:
            ax.scatter(x, y, s=700, marker="s", color="#dddddd", edgecolor="k", zorder=3)
            ax.text(x, y, "internet", ha="center", va="center", fontsize=6, zorder=4)
            continue
        lv = levels.get(h["id"], 0)
        goal = h["id"] == d["target"]
        ax.scatter(x, y, s=520, color=LEVEL_COL[min(lv, 3)], edgecolor="k", lw=2.2 if goal else 0.8, zorder=3,
                   marker="*" if goal else "o")
        ax.text(x, y, h["name"].split("-")[1] if "-" in h["name"] else h["name"], ha="center", va="center",
                fontsize=6, zorder=4, weight="bold")
        r = 1.17 if h["zone"] == "spoke" else -0.22
        tx, ty = (x * r, y * r) if h["zone"] == "spoke" else (x, y + r)
        ax.text(tx, ty, "/".join(sorted(fams.get(h["id"], []))), ha="center", va="center", fontsize=4.5,
                color="#555555", zorder=4)
    for i, _, g in parse(plan):
        if g.get("cve"):
            a, b = by_name[g["src"]], by_name[g["dst"]]
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=9, color="#0a5aa0",
                                         lw=1.6, zorder=5, shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=0.12"))
            mx_, my_ = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
            ax.text(mx_, my_, str(i), fontsize=5.5, color="white", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.15", fc="#0a5aa0", ec="none"))
            if g.get("need"):
                c = by_name[g["need"]]
                ax.add_patch(FancyArrowPatch(pos[c], pos[b], arrowstyle="-|>", mutation_scale=7, color="#8a2be2",
                                             lw=1.0, ls="--", zorder=5, shrinkA=9, shrinkB=9,
                                             connectionstyle="arc3,rad=-0.2"))
        elif g.get("k"):
            h = by_name[g["at"]]
            x, y = pos[h]
            ax.text(x + 0.09, y + 0.09, f"{i}: cred({g['k']})", fontsize=5, color="#b35900", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#fff4e0", ec="#b35900", lw=0.5))
    ax.set_xlim(-1.75, 1.35)
    ax.set_ylim(-1.3, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")


def legend(ax, upper=False):
    from matplotlib.lines import Line2D
    items = [Line2D([], [], marker="o", ls="", color=LEVEL_COL[k], markeredgecolor="k", label=f"tier {k}")
             for k in range(4)]
    items += [Line2D([], [], marker="*", ls="", color=LEVEL_COL[3], markeredgecolor="k", markersize=10, label="goal"),
              Line2D([], [], color="#0a5aa0", lw=1.6, label="exploit step"),
              Line2D([], [], color="#8a2be2", lw=1, ls="--", label="also needs (coupling)")]
    if upper:  # layered drawings: the bottom row is full, the top corners are free
        ax.legend(handles=items, loc="upper left", fontsize=5, frameon=False, ncol=2, bbox_to_anchor=(-0.02, 1.02))
    else:
        ax.legend(handles=items, loc="lower left", fontsize=5, frameon=False, ncol=2, bbox_to_anchor=(-0.02, -0.04))


def main():
    d = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pos = layout(d)
    runs = d["runs"]
    opt = runs["lmcut"]["cost"]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    draw_network(ax, d, pos, runs["lmcut"]["plan"])
    legend(ax, upper=not any(h["zone"] == "hub" for h in d["hosts"]))
    fig.savefig(out / "overview.png", bbox_inches="tight", dpi=170)
    plt.close(fig)

    names = [k for k in ORDER if k in runs]
    for k in names:
        r = runs[k]
        heat = Counter(h for s in r["trace"] for h in s)  # fact index h = host h for execCode facts
        heat["total"] = max(len(r["trace"]), 1)
        fig = plt.figure(figsize=(7.2, 3.6))
        ax = fig.add_axes([0.0, 0.0, 0.58, 1.0])
        draw_network(ax, d, pos, r["plan"], heat)
        bx = fig.add_axes([0.66, 0.18, 0.32, 0.7])
        vals = [runs[x]["expanded"] for x in names]
        cols = ["#e03030" if x == k else "#bbbbbb" for x in names]
        bx.barh(range(len(names)), vals, color=cols)
        bx.set_xscale("log")
        bx.set_yticks(range(len(names)), [NAMES[x] for x in names], fontsize=7)
        bx.invert_yaxis()
        for i, v in enumerate(vals):
            bx.text(v * 1.15, i, f"{v:,}", va="center", fontsize=6.5)
        bx.set_xlim(1, max(vals) * 12)
        bx.set_xlabel("states expanded (log scale)", fontsize=7)
        bx.tick_params(axis="x", labelsize=6)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        tag = "optimal" if r["cost"] == opt else f"{r['cost'] / opt:.2f}x optimal"
        bx.set_title(f"{NAMES[k]}: {r['expanded']:,} states, plan {tag}", fontsize=7.5)
        fig.savefig(out / f"search-{k}.png", bbox_inches="tight", dpi=170)
        plt.close(fig)

    tex = ["% generated by draw_demo.py from " + Path(sys.argv[1]).name + "; do not edit"]
    macro = {"h0": "Hzero", "hsp": "Hsp", "hmax": "Hmax", "hadd": "Hadd", "lmcut": "Lmcut", "A": "A", "B": "B"}
    for k in names:
        tex.append(f"\\newcommand{{\\T{macro[k]}Exp}}{{{runs[k]['expanded']:,}}}".replace(",", "{,}"))
        tex.append(f"\\newcommand{{\\T{macro[k]}Opt}}{{{'optimal' if runs[k]['cost'] == opt else 'suboptimal'}}}")
    tex.append(f"\\newcommand{{\\TSeed}}{{{d['seed']}}}")
    tex.append(f"\\newcommand{{\\THosts}}{{{len(d['hosts']) - 1}}}")
    tex.append(f"\\newcommand{{\\TActions}}{{{len(d['actions'])}}}")
    tex.append(f"\\newcommand{{\\TSteps}}{{{len(runs['lmcut']['plan'])}}}")
    tex.append(f"\\newcommand{{\\TCoupled}}{{{len(d['coupled'] or {})}}}")
    steps = []
    for i, s, g in parse(runs["lmcut"]["plan"]):
        if g.get("cve"):
            extra = f", also needs {g['need']}" if g.get("need") else ""
            steps.append(f"\\item {g['src']} $\\to$ {g['dst']}: {g['cve']} ({g['fam']}){extra}")
        elif g.get("k"):
            steps.append(f"\\item harvest cred({g['k']}) on {g['at']}")
        else:
            steps.append(f"\\item {s}")
    (out / "steps.tex").write_text("\n".join(steps) + "\n")
    prefix = sys.argv[3] if len(sys.argv) > 3 else "T"  # macro prefix, one per traced network on the slides
    tex = [t.replace("\\newcommand{\\T", "\\newcommand{\\" + prefix) for t in tex]
    (out / "trace_numbers.tex").write_text("\n".join(tex) + "\n")
    print("ok", {k: runs[k]["expanded"] for k in names})


if __name__ == "__main__":
    main()
