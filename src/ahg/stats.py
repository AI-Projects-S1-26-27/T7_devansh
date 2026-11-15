"""The D23 §2 statistics: nested bootstrap, Wilcoxon signed-rank, Holm correction, rank-biserial effect size.

Pure Python plus torch (the only numeric dependency of the project), so no new package is pinned. Every
function is deterministic: bootstrap resamples come from a torch.Generator seeded by a string key.

Conventions (fixed by D23 §2 and D24 §6):
- Ratios are analysed as logs. The point estimate is exp(mean of log ratios over instances and seeds).
- Nested bootstrap: each resample draws instances with replacement, then, within each drawn instance,
  seeds with replacement from that model's seeds. A plain instance bootstrap is the special case of one seed.
- Wilcoxon signed-rank, two-sided, on per-instance paired differences d_i; zero differences are dropped
  (Wilcoxon's original treatment, scipy's default "wilcox"). Exact null distribution when n <= 50 and the
  |d| have no ties; otherwise the normal approximation with tie correction and continuity correction
  (scipy's defaults, so values can be checked against it).
- Effect size: matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-), on the same nonzero pairs; r > 0 means
  the differences are mostly positive.
"""

import hashlib
import math

import torch

BOOTSTRAP = 10_000


def _generator(key: str) -> torch.Generator:
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 1)


def nested_bootstrap(logs: list[list[float]], key: str, n: int = BOOTSTRAP) -> list[float] | None:
    """Geometric mean and 95% percentile CI of exp(logs), logs[instance][seed] (NaN = missing, ignored).

    Instances are the outer unit and seeds the inner one (D23 §2); with one seed per instance this is the
    instance bootstrap of evaluate._geo_ci, but with a different random stream.
    """
    if not logs:
        return None
    x = torch.tensor(logs, dtype=torch.float64)  # [I, S]
    if torch.isnan(x).all():
        return None
    I, S = x.shape
    g = _generator(key)
    ii = torch.randint(I, (n, I), generator=g)
    ss = torch.randint(S, (n, I, S), generator=g)
    draws = x[ii[..., None], ss]  # [n, I, S]
    means = torch.nanmean(draws.reshape(n, -1), dim=1).sort().values
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n) - 1]
    return [math.exp(float(torch.nanmean(x))), math.exp(float(lo)), math.exp(float(hi))]


def _ranks(values: list[float]) -> list[float]:
    """Ranks 1..n of values, ties given their average rank."""
    order = sorted(range(len(values)), key=lambda k: values[k])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _exact_two_sided(w_plus: float, n: int) -> float:
    """P(W+ as extreme as observed) under H0, no ties: W+ is the sum of a random subset of 1..n."""
    counts = [1] + [0] * (n * (n + 1) // 2)
    for k in range(1, n + 1):
        for t in range(len(counts) - 1, k - 1, -1):
            counts[t] += counts[t - k]
    total = 2 ** n
    lo = min(w_plus, n * (n + 1) / 2 - w_plus)
    p = 2 * sum(counts[t] for t in range(int(math.floor(lo)) + 1)) / total
    return min(1.0, p)


def wilcoxon(d: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test of median(d) = 0, plus the matched-pairs rank-biserial r."""
    nz = [x for x in d if x != 0]
    n = len(nz)
    out = {"n": len(d), "n_nonzero": n, "zeros": len(d) - n}
    if n == 0:
        return {**out, "w_plus": 0.0, "w_minus": 0.0, "p": 1.0, "method": "none", "rank_biserial": 0.0}
    r = _ranks([abs(x) for x in nz])
    w_plus = sum(rk for rk, x in zip(r, nz) if x > 0)
    w_minus = sum(rk for rk, x in zip(r, nz) if x < 0)
    ties = len(set(abs(x) for x in nz)) < n
    if n <= 50 and not ties:
        p, method = _exact_two_sided(w_plus, n), "exact"
    else:
        mean = n * (n + 1) / 4
        tie_term = 0.0
        for v in set(abs(x) for x in nz):
            t = sum(abs(x) == v for x in nz)
            tie_term += t ** 3 - t
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48)
        z = (abs(w_plus - mean) - 0.5) / sd if sd > 0 else 0.0
        p, method = min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2))), "normal"
    return {**out, "w_plus": w_plus, "w_minus": w_minus, "p": p, "method": method,
            "rank_biserial": (w_plus - w_minus) / (w_plus + w_minus)}


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order (family-wise error control)."""
    m = len(p)
    order = sorted(range(m), key=lambda k: p[k])
    adj, running = [0.0] * m, 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[k]))
        adj[k] = running
    return adj


BEATS, LOSES, NO_DIFFERENCE = "beats", "loses", "no difference shown"


def verdict(ci: list[float], p_holm: float, threshold: float = 0.90, alpha: float = 0.01) -> str:
    """D17 §2 / D23 §2: beats if ratio <= 0.90, CI upper < 1 and Holm p < 0.01; loses if ratio >= 1/0.90,
    CI lower > 1 and Holm p < 0.01; otherwise no difference shown."""
    est, lo, hi = ci
    if p_holm < alpha and est <= threshold and hi < 1:
        return BEATS
    if p_holm < alpha and est >= 1 / threshold and lo > 1:
        return LOSES
    return NO_DIFFERENCE


# ---------------------------------------------------------------------------------------------------------
# D28 Round 12 verdict rule (decided 15 Nov 2026, P0b). The D17/D23 verdict above belongs to the superseded
# study; it is kept unchanged for the old analyses and is not used for D28.
#
#   Data per comparison (GNN vs X): log(E_GNN / E_X) on the tasks solved by every method of the comparison,
#   arranged [network][seed][task]. Geometric-mean ratio = exp(mean of those logs).
#   CI: 95% nested percentile bootstrap, 10,000 resamples: networks with replacement, then, inside each
#       drawn network, D23 seeds with replacement; a drawn (network, seed) keeps all its tasks.
#   Test: two-sided Wilcoxon signed-rank on the per-task paired values; Holm across the whole supplied
#       family (3 comparisons x every primary split run).
#   SUPPORTED: ratio <= 0.90, CI upper < 1, Holm p < 0.01, and GNN coverage >= the compared method's.
#   REFUTED:   CI lower > 0.90.
#   INCONCLUSIVE: otherwise.
# The per-task value the Wilcoxon test uses (how a task's seeds are combined) is not fixed by D28
# (open_questions OQ-P0b-6), so the caller supplies it; this module does not choose.
# ---------------------------------------------------------------------------------------------------------

SUPPORTED, REFUTED, INCONCLUSIVE = "SUPPORTED", "REFUTED", "INCONCLUSIVE"
D28_THRESHOLD, D28_ALPHA, D28_LEVEL = 0.90, 0.01, 0.95


def commonly_solved(expansions: dict[str, dict]) -> list:
    """Tasks solved by every method: expansions[method][task] is the count, None if unsolved or out of
    budget. Every method must list the same tasks, so no task can drop out silently."""
    keys = [set(v) for v in expansions.values()]
    if not keys or any(k != keys[0] for k in keys):
        raise ValueError("every method must report the same task set")
    return sorted(t for t in keys[0] if all(expansions[m][t] is not None for m in expansions))


def coverage(expansions: dict, solvable: list) -> float:
    """Fraction of the solvable tasks a method solved within budget (reported apart from the verdict)."""
    if not solvable:
        raise ValueError("no solvable tasks")
    return sum(expansions[t] is not None for t in solvable) / len(solvable)


def nested_bootstrap_d28(logs: list[list[list[float]]], key: str, n: int = BOOTSTRAP,
                         level: float = D28_LEVEL, chunk: int = 500) -> list[float]:
    """[ratio, lo, hi]: exp of the mean log and its percentile CI. logs[network][seed][task], NaN = absent
    (ragged lists are padded with NaN). Networks are the outer unit, seeds the inner one."""
    n_net = len(logs)
    if n_net == 0:
        raise ValueError("no networks")
    n_seed = max(len(s) for s in logs)
    n_task = max((len(t) for s in logs for t in s), default=0)
    x = torch.full((n_net, n_seed, n_task), float("nan"), dtype=torch.float64)
    for i, seeds in enumerate(logs):
        for j, tasks in enumerate(seeds):
            x[i, j, : len(tasks)] = torch.tensor(tasks, dtype=torch.float64)
    finite = x[~torch.isnan(x)]
    if finite.numel() == 0 or not torch.isfinite(finite).all():
        raise ValueError("log ratios must be finite, and at least one must be present")
    gen = _generator(key)
    means = []
    for start in range(0, n, chunk):  # chunks bound the [resamples, networks, seeds, tasks] tensor
        k = min(chunk, n - start)
        ii = torch.randint(n_net, (k, n_net), generator=gen)
        ss = torch.randint(n_seed, (k, n_net, n_seed), generator=gen)
        draws = x[ii[..., None], ss]  # [k, networks, seeds, tasks]
        means.append(torch.nanmean(draws.reshape(k, -1), dim=1))
    means = torch.cat(means).sort().values
    tail = (1 - level) / 2
    lo, hi = means[int(tail * n)], means[int((1 - tail) * n) - 1]
    if torch.isnan(lo) or torch.isnan(hi):
        raise ValueError("a resample drew no values; the missing-data pattern is too sparse")
    return [math.exp(float(finite.mean())), math.exp(float(lo)), math.exp(float(hi))]


def verdict_d28(ci: list[float], p_holm: float, coverage_gnn: float, coverage_other: float) -> str:
    """The D28 Round 12 outcome of one comparison, from [ratio, lo, hi], its Holm-adjusted p and coverages."""
    ratio, lo, hi = ci
    if ratio <= D28_THRESHOLD and hi < 1 and p_holm < D28_ALPHA and coverage_gnn >= coverage_other:
        return SUPPORTED
    if lo > D28_THRESHOLD:
        return REFUTED
    return INCONCLUSIVE


def decide_family(family: list[dict], n: int = BOOTSTRAP) -> list[dict]:
    """Verdicts for one confirmatory family. Each entry: name, logs ([network][seed][task] log ratios on
    commonly solved tasks), task_values (one paired value per task, for Wilcoxon), coverage_gnn,
    coverage_other. Holm runs across every entry supplied, so pass the whole family at once."""
    names = [c["name"] for c in family]
    if len(set(names)) != len(names):
        raise ValueError("comparison names must be unique")
    tests = [wilcoxon(c["task_values"]) for c in family]
    p_adj = holm([t["p"] for t in tests])
    out = []
    for c, t, p in zip(family, tests, p_adj):
        ci = nested_bootstrap_d28(c["logs"], f"d28:{c['name']}", n=n)
        out.append({"name": c["name"], "ratio": ci[0], "ci": ci[1:], "wilcoxon": t, "p_holm": p,
                    "coverage_gnn": c["coverage_gnn"], "coverage_other": c["coverage_other"],
                    "verdict": verdict_d28(ci, p, c["coverage_gnn"], c["coverage_other"])})
    return out
