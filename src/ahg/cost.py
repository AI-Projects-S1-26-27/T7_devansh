"""D9: the exact integer encoding of the frozen cost model.

The model is c(a) = 1 + 9·(1 − x_a), with x_a = (k − 1)/38 and k = round(10·E),
where E is the NVD Primary CVSS v3.x exploitability sub-score. Costs are
stored in units of 1/38 effort unit:

    C = 38·c = 389 − 9k,  so C ∈ [38, 380] ↔ c ∈ [1, 10].

Integers keep search and LM-cut comparisons exact (D9).
"""

SCALE = 38  # stored cost units per effort unit
K_MIN, K_MAX = 1, 39  # E ∈ [0.1, 3.9] on NVD's one-decimal grid


def exploitability_k(e: float) -> int:
    """k = round(10·E), checked to be on the one-decimal grid.

    It must be round(), not int(). 10·E is not exact in floating point, and
    int() truncates a value like 31.999999999999996 down to 31.
    """
    k = round(10 * e)
    if abs(10 * e - k) >= 1e-9:
        raise ValueError(f"exploitability {e!r} is not on NVD's 0.1 grid")
    if not K_MIN <= k <= K_MAX:
        raise ValueError(f"exploitability {e!r} outside the CVSS v3 range [0.1, 3.9]")
    return k


def cost_units(e: float) -> int:
    """Integer action cost C = 389 − 9k for exploitability sub-score E."""
    return 389 - 9 * exploitability_k(e)


# ---------------------------------------------------------------------------------------------------------
# D28 cost regimes (Round 5 amendment, 15 Nov 2026). The names are fixed by the spec: "ac" is the primary
# regime, "uniform" the comparison. Only AC enters cost; AV, PR and UI are prerequisites, never cost.
# ---------------------------------------------------------------------------------------------------------

REGIMES = ("ac", "uniform")
PRIMARY_REGIME = "ac"
AC_COSTS = {"L": 4, "H": 7}  # inverse CVSS v3 AC weights 0.77 / 0.44, scaled; ratio 1.75


def check_regime(regime: str) -> str:
    """The regime name, or ValueError if it is not one of REGIMES."""
    if regime not in REGIMES:
        raise ValueError(f"cost regime must be one of {REGIMES}, not {regime!r}")
    return regime


def regime_cost(regime: str, ac: str) -> int:
    """c(a) of a CVE-tied action under `regime`, from its explicit CVSS v3.x AC ("L" or "H").

    The AC is required under both regimes, although `uniform` ignores its value: only CVEs with an
    explicit AC enter the generator, so both regimes cost the same instances (Round 5). It is never
    defaulted, so a missing AC raises instead of becoming the old neutral cost of 4.
    """
    check_regime(regime)
    if ac not in AC_COSTS:
        raise ValueError(f"AC must be an explicit 'L' or 'H', not {ac!r}; it is never defaulted (D28 Round 5)")
    return 1 if regime == "uniform" else AC_COSTS[ac]


def check_priced(task, regime: str) -> None:
    """Raise unless every action cost of `task` is one `regime` can produce.

    The two regimes' cost sets are disjoint ({1} and {4, 7}) and neither meets the pre-D28 D9 range
    [38, 380], so a task priced under any other regime is refused rather than labelled or searched
    as if it were this one (labels of one regime are never used for the other).
    """
    allowed = {1} if check_regime(regime) == "uniform" else set(AC_COSTS.values())
    bad = sorted({a.cost for a in task.actions} - allowed)
    if bad:
        raise ValueError(f"task has action costs {bad}, which the {regime!r} regime cannot produce")
