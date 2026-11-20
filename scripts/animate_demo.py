"""Animated search traces for the visual demo (imported by visual_demo.py).

Each frame shows A* after k expansions: the red halo on a host grows with the share of expanded states that hold
code execution on it, and the red ring marks the hosts held in the state expanded at that moment. When a search
finishes, its returned attack is drawn step by step. A race puts two searches side by side on one counter k, so
the one that needs fewer expansions finishes first.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import FancyArrowPatch

import draw_demo

FPS = 5
HOLD = 15  # frames held at the end so the final picture can be read


def schedule(n_max: int) -> list[int]:
    """Expansion counts shown per frame: every count up to 12, then growing by 25% (12,987 expansions -> ~45 frames)."""
    ks, k = [], 1
    while k < n_max:
        ks.append(k)
        k = k + 1 if k < 12 else max(k + 1, int(k * 1.25))
    return ks + [n_max]


def draw_step(ax, d, pos, i, g, by_name):
    """One plan step: exploit arrow (with its coupling arrow) or a credential label; same style as draw_demo."""
    arts = []
    if g.get("cve"):
        a, b = by_name[g["src"]], by_name[g["dst"]]
        p = FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=12, color="#0a5aa0", lw=2.2, zorder=7,
                            shrinkA=11, shrinkB=11, connectionstyle="arc3,rad=0.12")
        ax.add_patch(p)
        arts.append(p)
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        arts.append(ax.text(mx, my, str(i), fontsize=8, color="white", ha="center", va="center", zorder=8,
                            bbox=dict(boxstyle="circle,pad=0.18", fc="#0a5aa0", ec="none")))
        if g.get("need"):
            c = by_name[g["need"]]
            q = FancyArrowPatch(pos[c], pos[b], arrowstyle="-|>", mutation_scale=9, color="#8a2be2", lw=1.4,
                                ls="--", zorder=7, shrinkA=11, shrinkB=11, connectionstyle="arc3,rad=-0.2")
            ax.add_patch(q)
            arts.append(q)
    elif g.get("k"):
        x, y = pos[by_name[g["at"]]]
        arts.append(ax.text(x + 0.09, y + 0.09, f"{i}: cred({g['k']})", fontsize=7, color="#b35900", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.15", fc="#fff4e0", ec="#b35900", lw=0.6)))
    return arts


def draw_demo_legend_items():
    """Legend entries of draw_demo plus the two animation marks, for one legend under the whole figure."""
    from matplotlib.lines import Line2D
    items = [Line2D([], [], marker="o", ls="", color=draw_demo.LEVEL_COL[k], markeredgecolor="k", label=f"tier {k}")
             for k in range(4)]
    items += [Line2D([], [], marker="*", ls="", color=draw_demo.LEVEL_COL[3], markeredgecolor="k", markersize=12,
                     label="goal"),
              Line2D([], [], color="#0a5aa0", lw=2, label="attack step"),
              Line2D([], [], color="#8a2be2", lw=1.4, ls="--", label="also needs (AND)"),
              Line2D([], [], marker="o", ls="", color="#e03030", alpha=0.3, markersize=14, label="explored often"),
              Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#d00000", markersize=12,
                     markeredgewidth=2, label="state being expanded")]
    return items, [i.get_label() for i in items]


class Panel:
    """One search on one axes: static network, growing heat, current-state ring, then the returned plan."""

    def __init__(self, ax, d, pos, key, opt):
        self.ax, self.d, self.pos = ax, d, pos
        self.run = d["runs"][key]
        self.name = draw_demo.NAMES[key]
        self.opt = opt
        self.n = self.run["expanded"]
        self.hosts = [h["id"] for h in d["hosts"] if h["id"] != 0]
        self.by_name = {h["name"]: h["id"] for h in d["hosts"]}
        draw_demo.draw_network(ax, d, pos, [])  # network only; plan comes at the end
        held = np.zeros((max(len(self.run["trace"]), 1), len(self.hosts)), dtype=np.int32)
        col = {h: j for j, h in enumerate(self.hosts)}
        for t, s in enumerate(self.run["trace"]):
            for f in s:
                if f in col:  # fact index f = execCode on host f; credentials and the internet fact are skipped
                    held[t, col[f]] = 1
        self.held = held
        self.cum = held.cumsum(axis=0)
        xy = np.array([pos[h] for h in self.hosts])
        self.halo = ax.scatter(xy[:, 0], xy[:, 1], s=np.zeros(len(xy)), color="#e03030", alpha=0.2, zorder=2, lw=0)
        self.ring = ax.scatter(xy[:, 0], xy[:, 1], s=np.zeros(len(xy)), facecolors="none", edgecolors="#d00000",
                               lw=2.5, zorder=4)
        self.title = ax.set_title("", fontsize=13, weight="bold")
        self.steps = draw_demo.parse(self.run["plan"] or [])
        self.shown = 0

    def update(self, k, after_done):
        """k: global expansion counter; after_done: frames since this search finished (plan steps revealed)."""
        m = min(k, self.n)
        if m > 0 and len(self.run["trace"]):
            c = self.cum[m - 1]
            self.halo.set_sizes(np.where(c > 0, 3200 * np.sqrt(c / m) + 250, 0))
            self.ring.set_sizes(self.held[m - 1] * 900.0 if k < self.n else np.zeros(len(self.hosts)))
        if k < self.n:
            self.title.set_text(f"{self.name}\nstates expanded: {m:,}")
            self.title.set_color("#111111")
            return
        cost = self.run["cost"]
        tag = "optimal" if cost == self.opt else f"{cost / self.opt:.2f}x optimal"
        self.title.set_text(f"{self.name}\ndone: {self.n:,} states, attack {tag}")
        self.title.set_color("#0a7a2a" if cost == self.opt else "#b35900")
        while self.shown < min(after_done, len(self.steps)):
            i, _, g = self.steps[self.shown]
            draw_step(self.ax, self.d, self.pos, i, g, self.by_name)
            self.shown += 1


def animate(d, pos, keys, path, opt, size):
    """Write a GIF of the searches in keys, side by side, on one shared expansion counter."""
    fig, axes = plt.subplots(1, len(keys), figsize=size)
    axes = np.atleast_1d(axes)
    panels = [Panel(ax, d, pos, k, opt) for ax, k in zip(axes, keys)]
    handles, labels = draw_demo_legend_items()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07, wspace=0.03)
    ks = schedule(max(p.n for p in panels))
    longest_plan = max(len(p.steps) for p in panels)
    frames = ks + [ks[-1]] * (longest_plan + HOLD)
    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, str(path), dpi=90):
        for t, k in enumerate(frames):
            for p in panels:
                done_at = next((j for j, kk in enumerate(frames) if kk >= p.n), len(frames))
                p.update(k, t - done_at + 1 if t >= done_at else 0)
            writer.grab_frame()
    plt.close(fig)
