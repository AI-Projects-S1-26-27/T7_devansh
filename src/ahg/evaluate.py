"""The evaluation harness (D21 §4): every heuristic in A* on every labelled instance of one split.

evaluate() takes already-opened batches. For a test split, opening them went
through the test-access log (dataset.open_batch with purpose="evaluation"),
so this module has no per-split branches: validation and test go through the
same code.

Per instance it runs A* from the initial state with each baseline and each
learned heuristic, and the D23 §2 h_add searches (GBFS+h_add, weighted A*+h_add
at w = 1.5 and 2), all with the same best_first_search, tie-breaking, duplicate
detection, reopening and budget, and checks that every admissible baseline returns the
label h*(init). Per learned model it also evaluates h_θ on the instance's
labelled states, for overestimation statistics by source, and the sibling-ordering
accuracy (D22 §3.2) on the labelled successors of labelled A*-expanded states.
summarise() turns the raw records into ratios with bootstrap CIs; it reads
nothing but the raw records, so aggregates can always be regenerated.
"""

import math
import random
from itertools import combinations

from .cost import SCALE
from .heuristics import ADMISSIBLE, INF, h0, hadd, hff, hmax, hsp, lmcut
from .measure import HardFailure
from .search import best_first_search
from .task import plan_cost
from .timing import REPEATS, WARMUP, InvalidMeasurement, Timing, log_ratio, timed

# Each in A*. h_FF added by D28 Round 7 (H1). Every one records expansions; h_add and h_FF are inadmissible,
# so their plan cost is measured and never checked against h* (M1).
BASELINES = {"h0": h0, "hmax": hmax, "hsp": hsp, "hadd": hadd, "hff": hff, "lmcut": lmcut}
# D23 §2: h_add in the other best-first searches, f = w_g*g + w_h*h (search.py): GBFS and weighted A*.
# Reopening as Fast Downward 24.06.1 (D28 Round 7, search.py): GBFS never reopens, weighted A* does.
HADD_SEARCHES = {"gbfs_hadd": {"w_g": 0.0, "w_h": 1.0, "reopen": False},
                 "wastar1.5_hadd": {"w_g": 1.0, "w_h": 1.5, "reopen": True},
                 "wastar2_hadd": {"w_g": 1.0, "w_h": 2.0, "reopen": True}}
COMPARATORS = (*BASELINES, *HADD_SEARCHES)
INADMISSIBLE_SEARCHES = ("hadd", "hff", *HADD_SEARCHES)  # plan cost measured, never bounded (rule 8)
# Candidates for the strongest classical baseline (D28 Round 11, P0c): A* with each of these. All of them
# run through _run, so each records expansions; the inadmissible ones (h_add, h_FF) are simply not checked
# against h* (M1).
STRONGEST_CANDIDATES = ("hmax", "hadd", "hff", "lmcut", "hsp")
STATE_GROUPS = {"walk": {"walk"}, "astar_expanded": {"expanded_lmcut", "expanded_hmax"}, "successor": {"successor"}}
BOOTSTRAP = 10_000


def sibling_groups(task, exs) -> list[list[tuple[int, int, int]]]:
    """For each labelled A*-expanded state: its labelled successors as (action cost, state, h*), D9 units.

    A* orders siblings by c(a) + h(s'), since they share g; a successor reached by several actions
    keeps the cheapest. Only labelled successors are used, so no new labels are computed.
    """
    lab = {ex.state: ex.hstar for ex in exs}
    groups = []
    for ex in exs:
        if not set(ex.sources) & STATE_GROUPS["astar_expanded"]:
            continue
        best: dict[int, int] = {}
        for ai, s2 in task.successors(ex.state):
            if s2 in lab:
                best[s2] = min(task.actions[ai].cost, best.get(s2, INF))
        if len(best) > 1:
            groups.append([(c, s2, lab[s2]) for s2, c in best.items()])
    return groups


def sibling_order(groups, h: dict) -> dict:
    """Pairs of siblings whose c + h* differ: concordant if c + h orders them the same way, tied if c + h is
    equal (counted 0.5 in the accuracy). h: state -> value in D9 units."""
    pairs = concordant = ties = 0
    for grp in groups:
        for (c1, s1, t1), (c2, s2, t2) in combinations(grp, 2):
            true = (c1 + t1) - (c2 + t2)
            if true == 0:
                continue
            est = (c1 + h[s1]) - (c2 + h[s2])
            pairs += 1
            if abs(est) < 1e-9:
                ties += 1
            elif (est > 0) == (true > 0):
                concordant += 1
    return {"pairs": pairs, "concordant": concordant, "ties": ties}


def _run(task, make_h, budget, repeats: int, w_g=1.0, w_h=1.0, reopen=True, tie_break="fifo", warmup: int = 0):
    """One method on one task: `warmup` discarded runs, then `repeats` timed ones (D28 Round 12, M1).

    make_h() returns a fresh heuristic each repeat, so per-instance set-up a
    learned heuristic does lazily (its static graph) is inside every timed
    repeat, not only the first. The search is deterministic: every repeat must
    give the same plan and counts, or the run aborts.
    """
    runs, timing = timed(lambda: best_first_search(task, make_h(), w_g=w_g, w_h=w_h, reopen=reopen,
                                                   tie_break=tie_break, max_expansions=budget), repeats, warmup)
    key = lambda r: (r.plan, r.expanded, r.generated, r.evaluations, r.peak_open, r.peak_closed, r.reopened)
    if any(key(r) != key(runs[0]) for r in runs):
        raise HardFailure("a repeated search gave different results: the timing repeats are not the same run")
    r = runs[0]
    if r.solved:
        assert plan_cost(task, r.plan) == r.cost  # the plan is executable and its cost is what search says
    return {"expanded": r.expanded, "generated": r.generated, "evaluations": r.evaluations,
            "cost": r.cost if r.solved else None, "time": timing.record(), "budget_hit": r.budget_hit,
            "peak_open": r.peak_open, "peak_closed": r.peak_closed, "reopened": r.reopened}


def evaluate(batches, learned: list, budget: int = 200_000, tie_break: str = "fifo", *,
             repeats: int = REPEATS, warmup: int = WARMUP) -> list[dict]:
    """One record per labelled instance. learned: list of learning.heuristic.Learned.

    tie_break: search.TIE_RULES, the same for every method of the run (D28 Round 7); recorded per record.
    repeats, warmup: timed and discarded runs per search; defaults 5 and 1 (M1, timing.py). Recorded."""
    records = []
    for batch in batches:
        by_inst: dict[int, list] = {}
        for ex in batch.examples:
            by_inst.setdefault(ex.instance, []).append(ex)
        for i, inst_rec in enumerate(batch.manifest["instances"]):
            if inst_rec["stats"] is None:
                continue
            inst, hstar = batch.instances[i], inst_rec["stats"]["hstar_init"]
            task = inst.task
            rec = {"batch": batch.manifest["batch_id"], "seed": inst_rec["spec"]["seed"],
                   "cell": {k: inst_rec["spec"][k] for k in ("variant", "n", "D", "rho", "m")},
                   "hstar": hstar, "tie_break": tie_break, "repeats": repeats, "warmup": warmup,
                   "runs": {}, "over": {}, "order": {}}
            for name, h in BASELINES.items():
                rec["runs"][name] = _run(task, lambda h=h: h, budget, repeats, tie_break=tie_break, warmup=warmup)
                got = rec["runs"][name]["cost"]
                if name in ADMISSIBLE and not rec["runs"][name]["budget_hit"] and got != hstar:
                    raise HardFailure(f"seed {rec['seed']}: A*+{name} cost {got} != label h* {hstar}")
            for name, w in HADD_SEARCHES.items():
                rec["runs"][name] = _run(task, lambda: hadd, budget, repeats, tie_break=tie_break, warmup=warmup, **w)
            exs = by_inst[i]
            groups = sibling_groups(task, exs)
            for model in learned:
                rec["runs"][model.name] = _run(task, lambda: model.bind(inst), budget, repeats, tie_break=tie_break,
                                              warmup=warmup)
                preds = model.predict_states(inst, [ex.state for ex in exs])
                rec["over"][model.name] = {
                    g: [p - ex.hstar / SCALE for p, ex in zip(preds, exs) if set(ex.sources) & srcs]
                    for g, srcs in STATE_GROUPS.items()}
                h_of = {ex.state: p * SCALE for p, ex in zip(preds, exs)}
                rec["order"][model.name] = sibling_order(groups, h_of)
            records.append(rec)
    return records


def _geo_ci(logs: list[float], key: str):
    """Geometric mean of exp(logs) with a bootstrap 95% CI over instances. logs must be finite."""
    n = len(logs)
    if not n:
        return None
    if not all(math.isfinite(x) for x in logs):
        raise InvalidMeasurement(f"{key}: non-finite log value")
    rng = random.Random(f"bootstrap:{key}")
    means = sorted(sum(rng.choices(logs, k=n)) / n for _ in range(BOOTSTRAP))
    return [math.exp(sum(logs) / n), math.exp(means[int(0.025 * BOOTSTRAP)]), math.exp(means[int(0.975 * BOOTSTRAP) - 1])]


def comparators(records: list[dict]) -> list[str]:
    """The comparators every record ran (runs before D24 have no h_add searches)."""
    return [c for c in COMPARATORS if all(c in r["runs"] for r in records)]


def strongest_baseline(records: list[dict]) -> dict:
    """The strongest classical baseline on validation records (D28 Round 11, decided P0c).

    Among A* + STRONGEST_CANDIDATES: the lowest geometric-mean expansions on the tasks every candidate
    solved; ties go to LM-cut. Admissibility plays no part, so h_add and h_FF can be picked. Compared on
    the mean of log expansions (the log of the geometric mean), so equal means are an exact tie.
    Trivial tasks (s0 already a goal: an empty plan, 0 expansions for every method) are counted and left
    out, as in the primary ratio (D28 §13.1, P0d)."""
    missing = [c for c in STRONGEST_CANDIDATES if not all(c in r["runs"] for r in records)]
    if missing:
        raise ValueError(f"records lack the A* runs of {missing}")
    solved = [r for r in records if all(r["runs"][c]["cost"] is not None for c in STRONGEST_CANDIDATES)]
    trivial = [r for r in solved if r["hstar"] == 0]
    common = [r for r in solved if r["hstar"] != 0]
    if not common:
        raise InvalidMeasurement("no task solved by every candidate")
    mean_log = {c: math.fsum(math.log(_positive_count(r["runs"][c]["expanded"], c)) for r in common) / len(common)
                for c in STRONGEST_CANDIDATES}
    best = min(mean_log.values())
    tied = [c for c in STRONGEST_CANDIDATES if mean_log[c] == best]
    pick = "lmcut" if "lmcut" in tied else tied[0]
    if len(tied) > 1 and pick != "lmcut":
        # The rule names only LM-cut as the tie-winner; a tie without LM-cut is not settled by it.
        raise ValueError(f"tie among {tied} without LM-cut: not covered by the D28 rule")
    return {"baseline": pick, "common": len(common), "trivial": len(trivial), "tied": tied,
            "geo_mean_expansions": {c: math.exp(v) for c, v in mean_log.items()}}


def _positive_count(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
        raise InvalidMeasurement(f"{what}: expansions {x!r} is not a positive integer")
    return x


def summarise_searches(records: list[dict]) -> dict:
    """Plan cost and coverage of each inadmissible classical search (A*+h_add and the HADD_SEARCHES)."""
    out = {}
    for m in [c for c in INADMISSIBLE_SEARCHES if c in comparators(records)]:
        solved = [r for r in records if r["runs"][m]["cost"] is not None]
        out[m] = {"instances": len(records), "solved": len(solved),
                  "budget_hits": sum(r["runs"][m]["budget_hit"] for r in records),
                  "optimal": sum(r["runs"][m]["cost"] == r["hstar"] for r in solved),
                  "plan_cost": _geo_ci([log_ratio(r["runs"][m]["cost"], r["hstar"], m) for r in solved], f"{m}:cost")}
    return out


def summarise(records: list[dict], models: list[str], budget: int) -> dict:
    """Per model: expansion and time ratios against every comparator, plan-cost ratio, overestimation by group.

    Ratios are over the tasks both methods solved; a zero count, a zero h* or a censored time in a log
    ratio raises InvalidMeasurement rather than being clamped (D28 Round 12, AUDIT_4 inventory #5)."""
    out = {}
    comps = comparators(records)
    for m in models:
        s = {"instances": len(records), "budget_hits": sum(r["runs"][m]["budget_hit"] for r in records),
             "vs": {}}
        for b in comps:
            # Commonly solved tasks only (D28 Round 12): an unsolved or budget-stopped run has no expansion
            # count to compare, and is never replaced by the budget or any other value.
            both = [r for r in records if r["runs"][m]["cost"] is not None and r["runs"][b]["cost"] is not None]
            timed_ = [r for r in both if not (Timing.from_record(r["runs"][m]["time"]).censored
                                              or Timing.from_record(r["runs"][b]["time"]).censored)]
            s["vs"][b] = {
                "common": len(both),
                "expansions": _geo_ci([log_ratio(r["runs"][m]["expanded"], r["runs"][b]["expanded"], f"{m}/{b} expansions")
                                       for r in both], f"{m}:{b}:exp"),
                "time": _geo_ci([log_ratio(Timing.from_record(r["runs"][m]["time"]),
                                           Timing.from_record(r["runs"][b]["time"]), f"{m}/{b} time")
                                 for r in timed_], f"{m}:{b}:time"),
                "time_censored": len(both) - len(timed_),  # left out explicitly, and counted
                "fewer": sum(r["runs"][m]["expanded"] < r["runs"][b]["expanded"] for r in both),
                "more": sum(r["runs"][m]["expanded"] > r["runs"][b]["expanded"] for r in both)}
        solved = [r for r in records if r["runs"][m]["cost"] is not None]
        s["plan_cost"] = _geo_ci([log_ratio(r["runs"][m]["cost"], r["hstar"], m) for r in solved], f"{m}:cost")
        s["optimal"] = sum(r["runs"][m]["cost"] == r["hstar"] for r in solved)
        s["solved"] = len(solved)
        s["overestimation"] = {}
        for g in STATE_GROUPS:
            d = [x for r in records for x in r["over"][m][g]]
            viol = [x for x in d if x > 1e-9]
            s["overestimation"][g] = {"states": len(d), "rate": len(viol) / len(d) if d else None,
                                      "mean_violation": sum(viol) / len(viol) if viol else 0.0,
                                      "max_violation": max(viol, default=0.0),
                                      "mean_abs_error": sum(abs(x) for x in d) / len(d) if d else None}
        o = [r["order"][m] for r in records if m in r.get("order", {})]
        n = sum(x["pairs"] for x in o)
        s["sibling_order"] = {"pairs": n, "accuracy": (sum(x["concordant"] for x in o) + 0.5 * sum(x["ties"] for x in o)) / n
                              if n else None, "tie_rate": sum(x["ties"] for x in o) / n if n else None}
        out[m] = s
    return out
