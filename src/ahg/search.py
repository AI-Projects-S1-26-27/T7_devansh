"""Best-first search over the state space of a delete-free task.

One function covers every algorithm in the study, because they differ only in
how a node's priority f is computed from its path cost g and heuristic h:

    uniform-cost search   f = g          (pass h = zero)
    A*                    f = g + h
    weighted A*           f = g + w*h    (w > 1)
    greedy best-first     f = h          (w_g = 0)

Ties on f are broken towards lower h (nodes believed closer to the goal),
then by generation order. Tie-breaking changes node counts a lot on these
tasks, so it is fixed here and documented. Two rules are available (D28
Round 7: a second, pre-registered tie rule, conclusions checked under both):
    "fifo"  (f, h, FIFO)  the D3 rule: the node generated first
    "lifo"  (f, h, LIFO)  the node generated last

Reopening (D28 Round 7) follows the pinned Fast Downward 24.06.1 defaults:
A* and weighted A* reopen closed nodes, GBFS does not. Sources, in the pinned
release-24.06.1 checkout (third_party/downward/src/search/search_algorithms/):
  - astar: plugin_astar.cc sets reopen_closed=true (its docs entry: "eager(...,
    reopen_closed=true, f_eval=sum([g(), h]))").
  - eager_wastar: plugin_eager_wastar.cc, option reopen_closed, default "true".
  - eager_greedy: plugin_eager_greedy.cc passes reopen_closed=false to EagerSearch;
    it has no option to change it.
FD documentation of these plugins (not version-pinned; checked 15 Nov 2026):
https://www.fast-downward.org/latest/documentation/search/SearchAlgorithm/
The update rule copies eager_search.cc: a cheaper path to an open node updates
its g and parent and reinserts it; to a closed node it reopens it if reopening
is on, and otherwise only updates its g and parent pointer, without
reinsertion. Without reopening the g of a closed node's descendants can then be
stale, so the returned cost is always the cost of the extracted plan, as FD
reports it, never the g at which the goal was popped.
"""

import heapq
import itertools
import time
from dataclasses import dataclass

from .task import State, Task

INF = float("inf")


@dataclass
class SearchResult:
    plan: list[int] | None  # action indices, None if unsolved
    cost: float  # INF if unsolved
    expanded: int  # nodes taken off the open list and expanded
    generated: int  # successor nodes created
    evaluations: int  # heuristic calls
    ns: int  # time.perf_counter_ns() inside the search, one run (evaluate.py repeats whole searches)
    budget_hit: bool  # stopped by max_expansions, not by exhausting the space
    reopened: int  # closed nodes reinserted after a cheaper path was found (0 when reopen=False)
    # Peak sizes (D28 Round 7, memory). Open: heap entries, stale duplicates included, as in Fast Downward's
    # open list, which also keeps superseded entries. Closed: distinct states expanded so far.
    peak_open: int
    peak_closed: int

    @property
    def solved(self) -> bool:
        return self.plan is not None

    @property
    def seconds(self) -> float:
        return self.ns / 1e9


TIE_RULES = ("fifo", "lifo")


def best_first_search(task: Task, h, *, reopen: bool, w_g=1.0, w_h=1.0, tie_break: str = "fifo",
                      start: State | None = None, max_expansions: int | None = None,
                      trace: list[State] | None = None) -> SearchResult:
    """h(task, state) -> estimated remaining cost (INF means dead end).

    reopen: re-expand a closed node reached by a cheaper path (module docstring);
    required, so no caller gets a reopening policy by accident.
    tie_break: "fifo" (the D3 rule) or "lifo", the last key after f and h.

    trace: if given, every expanded state is appended to it in expansion
    order, at the point where it is counted, so len(trace) == expanded. A
    reopened state appears once per expansion. Tracing only observes: it
    never changes the order, the counts or the result.
    """
    if tie_break not in TIE_RULES:
        raise ValueError(f"tie_break must be one of {TIE_RULES}, not {tie_break!r}")
    t0 = time.perf_counter_ns()
    s0 = task.init if start is None else start
    h_cache = {s0: h(task, s0)}
    # The third heap key: increasing for FIFO, decreasing for LIFO, so heapq pops the oldest or newest.
    tie = itertools.count() if tie_break == "fifo" else itertools.count(0, -1)
    g_best = {s0: 0}
    parent: dict[State, tuple[State, int] | None] = {s0: None}
    closed: dict[State, float] = {}  # state -> g at which it was expanded
    open_list = [(w_h * h_cache[s0], h_cache[s0], next(tie), 0, s0)]
    expanded = generated = reopened = 0
    peak_open = len(open_list)

    def result(plan, cost, budget_hit=False):
        return SearchResult(plan, cost, expanded, generated, len(h_cache),
                            time.perf_counter_ns() - t0, budget_hit, reopened, peak_open, len(closed))

    if h_cache[s0] == INF:
        return result(None, INF)

    while open_list:
        _, _, _, g, s = heapq.heappop(open_list)
        if g > g_best[s] or closed.get(s, INF) <= g:
            continue  # stale entry: a cheaper path to s is already known
        # Goal test on expansion, not generation: required for A* optimality.
        if task.is_goal(s):
            plan = _extract_plan(parent, s)
            return result(plan, sum(task.actions[ai].cost for ai in plan))
        if max_expansions is not None and expanded >= max_expansions:
            return result(None, INF, budget_hit=True)
        closed[s] = g
        expanded += 1
        if trace is not None:
            trace.append(s)
        for ai, s2 in task.successors(s):
            g2 = g + task.actions[ai].cost
            generated += 1
            if g2 >= g_best.get(s2, INF):
                continue
            # A cheaper path to s2: record it (FD updates g and parent in every case).
            g_best[s2] = g2
            parent[s2] = (s, ai)
            if s2 in closed:
                if not reopen:
                    continue  # FD update_closed_node_parent: no reinsertion, s2 stays closed
                reopened += 1
            # Open s2: reinsert with the new g. Closed s2 with reopening: reopen it,
            # which A* needs for optimality when h is admissible but inconsistent.
            if s2 not in h_cache:
                h_cache[s2] = h(task, s2)
            h2 = h_cache[s2]
            if h2 == INF:
                continue
            heapq.heappush(open_list, (w_g * g2 + w_h * h2, h2, next(tie), g2, s2))
            peak_open = max(peak_open, len(open_list))

    return result(None, INF)


def _extract_plan(parent, s):
    plan = []
    while parent[s] is not None:
        s, ai = parent[s]
        plan.append(ai)
    plan.reverse()
    return plan


def astar(task, h, **kw):
    return best_first_search(task, h, reopen=True, **kw)


def uniform_cost(task, **kw):
    return best_first_search(task, lambda t, s: 0, reopen=True, **kw)


def weighted_astar(task, h, w, **kw):
    return best_first_search(task, h, w_h=w, reopen=True, **kw)


def gbfs(task, h, **kw):
    return best_first_search(task, h, w_g=0.0, reopen=False, **kw)
