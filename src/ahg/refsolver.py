"""Independent exact reference solver (D28 Round 8, C2).

Labels and admissibility checks are compared against this module, so it must
not be able to share a bug with the code it checks. It therefore imports
nothing from ahg: not task.py (bitmask states, Action.applicable/useful,
add_mask, successors), not search.py, not heuristics.py. It works on a
neutral serialized task, a plain dict that names facts by string:

    {"facts":   ["f0", "f1", ...],
     "actions": [{"name": "a0", "pre": ["f0"], "add": ["f1"], "cost": 3}, ...],   # optional "del": [...]
     "init":    ["f0"],
     "goal":    ["f1"]}

A state is a frozenset of fact names. An action is applicable in s iff its
preconditions are a subset of s; applying it gives (s - del) | add, where
"del" is an optional delete list (absent on every delete-free task; the M3
access-loss model, AL1, uses it). Nothing here assumes the task is
delete-free: there is no "useless action" pruning beyond skipping exact
self-loops, and no use of state-size ordering.

The algorithm is exhaustive Dijkstra: it explores every state reachable from
the start before answering, rather than stopping at the first goal popped,
so the answer does not depend on goal-test placement. h* of every reachable
state comes from a second exhaustive Dijkstra, backwards over the explicit
reachable state graph from all goal states.
"""

import heapq
import itertools

INF = float("inf")


class ReferenceTaskError(ValueError):
    """A serialized task that is malformed (unknown fact, duplicate name, negative cost)."""


def validate(stask: dict) -> None:
    facts = stask["facts"]
    if len(set(facts)) != len(facts):
        raise ReferenceTaskError("duplicate fact names")
    known = set(facts)
    for a in stask["actions"]:
        if a["cost"] < 0:
            raise ReferenceTaskError(f"action {a['name']} has negative cost {a['cost']}")
        if set(a["add"]) & set(a.get("del", ())):
            raise ReferenceTaskError(f"action {a['name']} both adds and deletes a fact")
        for f in (*a["pre"], *a["add"], *a.get("del", ())):
            if f not in known:
                raise ReferenceTaskError(f"action {a['name']} names unknown fact {f!r}")
    for f in (*stask["init"], *stask["goal"]):
        if f not in known:
            raise ReferenceTaskError(f"init/goal names unknown fact {f!r}")


def _compiled(stask: dict):
    validate(stask)
    return ([(frozenset(a["pre"]), frozenset(a["add"]), frozenset(a.get("del", ())), a["cost"]) for a in stask["actions"]],
            frozenset(stask["goal"]))


def _successors(acts, s):
    for k, (pre, add, dele, cost) in enumerate(acts):
        if pre <= s:
            s2 = (s - dele) | add
            if s2 != s:  # a self-loop never shortens a path; skipping it changes no distance
                yield k, s2, cost


def reachable(stask: dict, start=None, cap: int | None = None) -> set[frozenset] | None:
    """Every state reachable from start (default init); None if there are more than cap."""
    acts, _ = _compiled(stask)
    s0 = frozenset(stask["init"] if start is None else start)
    seen, stack = {s0}, [s0]
    while stack:
        for _, s2, _ in _successors(acts, stack.pop()):
            if s2 not in seen:
                if cap is not None and len(seen) >= cap:
                    return None
                seen.add(s2)
                stack.append(s2)
    return seen


def solve(stask: dict, start=None) -> tuple[float, list[str] | None]:
    """(optimal cost, one optimal plan as action names) from start; (INF, None) if the goal is unreachable.

    Exhaustive forward Dijkstra: every reachable state is settled, then the
    cheapest goal state is taken. Ties between equal-cost plans are broken by
    insertion order, which only affects which optimal plan is returned.
    """
    acts, goal = _compiled(stask)
    s0 = frozenset(stask["init"] if start is None else start)
    dist, parent = {s0: 0}, {s0: None}
    tie = itertools.count()  # frozensets have no total order, so the heap never compares them
    heap = [(0, next(tie), s0)]
    settled = set()
    while heap:
        g, _, s = heapq.heappop(heap)
        if s in settled:
            continue
        settled.add(s)
        for k, s2, cost in _successors(acts, s):
            if g + cost < dist.get(s2, INF):
                dist[s2], parent[s2] = g + cost, (s, k)
                heapq.heappush(heap, (g + cost, next(tie), s2))
    goals = [s for s in settled if goal <= s]
    if not goals:
        return INF, None
    best = min(goals, key=lambda s: dist[s])
    plan, s = [], best
    while parent[s] is not None:
        s, k = parent[s]
        plan.append(stask["actions"][k]["name"])
    return dist[best], plan[::-1]


def hstar_all(stask: dict, cap: int | None = None) -> dict[frozenset, float] | None:
    """h* of every state reachable from init (INF for dead ends); None if more than cap states.

    Exhaustive backward Dijkstra on the explicit reachable state graph, from
    every goal state at distance 0 along reversed edges.
    """
    acts, goal = _compiled(stask)
    states = reachable(stask, cap=cap)
    if states is None:
        return None
    rev: dict[frozenset, list[tuple[frozenset, int]]] = {s: [] for s in states}
    for s in states:
        for _, s2, cost in _successors(acts, s):
            rev[s2].append((s, cost))
    h = {s: INF for s in states}
    tie = itertools.count()
    heap = []
    for s in states:
        if goal <= s:
            h[s] = 0
            heap.append((0, next(tie), s))
    heapq.heapify(heap)
    done = set()
    while heap:
        d, _, s = heapq.heappop(heap)
        if s in done:
            continue
        done.add(s)
        for p, cost in rev[s]:
            if d + cost < h[p]:
                h[p] = d + cost
                heapq.heappush(heap, (d + cost, next(tie), p))
    return h


def serialize(task, start=None) -> dict:
    """The neutral format of an ahg Task (duck-typed, so this module still imports nothing from ahg).

    Reads only the data fields facts, actions[i].name/.pre/.add/.cost/.dele, init
    and goal. The init (or start) bitmask is decoded here by plain bit tests, not
    by any task.py helper. "del" is written only for actions that delete
    something, so a delete-free task serializes exactly as before.
    """
    names = list(task.facts)
    bits = task.init if start is None else start

    def act(a):
        d = {"name": a.name, "pre": [names[i] for i in a.pre], "add": [names[i] for i in a.add], "cost": a.cost}
        dele = getattr(a, "dele", ())
        return d | ({"del": [names[i] for i in dele]} if dele else {})

    return {"facts": names,
            "actions": [act(a) for a in task.actions],
            "init": [names[i] for i in range(len(names)) if bits >> i & 1],
            "goal": [names[i] for i in task.goal]}


def to_bits(stask: dict, state: frozenset) -> int:
    """A reference state as the bitmask ahg uses (fact i of stask["facts"] is bit i), for comparisons."""
    index = {f: i for i, f in enumerate(stask["facts"])}
    return sum(1 << index[f] for f in state)
