"""AL2 structural metrics of one planning task (decisions.md §21; attack-graph metrics after Noel and Jajodia).

Every function is pure: it reads an ahg.task.Task (never changed) and, where needed, the evaluated states
{state bitmask: h*} that ahg.al1pilot.evaluated_states returns. The metrics are descriptive covariates for
where a heuristic gap can exist; they never feed a gate except through the pre-fixed AL2 decision rule.

Fact graph: an edge p -> q for every action with p in its preconditions and q in its add effects. It is the
dependency graph of the delete relaxation: holding p can (with other facts) lead to holding q.
"""

from collections import deque
from statistics import fmean

from ahg.heuristics import lmcut
from ahg.search import astar
from ahg.task import Task, facts_of

INF = float("inf")


def fact_edges(task: Task) -> set[tuple[int, int]]:
    """Distinct fact-graph edges (p, q), self-loops included."""
    return {(p, q) for a in task.actions for p in a.pre for q in a.add}


def _adjacency(task: Task) -> list[list[int]]:
    adj = [[] for _ in task.facts]
    for p, q in sorted(fact_edges(task)):  # sorted: a fixed visiting order, so the result never depends on hashing
        adj[p].append(q)
    return adj


def depth(task: Task) -> int:
    """Largest finite BFS hop distance in the fact graph from the facts held in task.init (0 if none is reachable)."""
    adj = _adjacency(task)
    dist = {f: 0 for f in facts_of(task.init)}
    queue = deque(dist)
    while queue:
        p = queue.popleft()
        for q in adj[p]:
            if q not in dist:
                dist[q] = dist[p] + 1
                queue.append(q)
    return max(dist.values(), default=0)


def edge_density(task: Task) -> float:
    """Distinct fact-graph edges over n(n-1), n = number of facts. Self-loops are left out of the count so the
    value stays in [0, 1] (n(n-1) counts ordered pairs of distinct facts; DECIDED-BY-AGENT, OQ-AL2-1)."""
    n = len(task.facts)
    if n < 2:
        return 0.0
    return sum(p != q for p, q in fact_edges(task)) / (n * (n - 1))


def _tarjan(adj: list[list[int]]) -> list[list[int]]:
    """Strongly connected components by Tarjan's algorithm, iterative (an explicit stack of (node, next edge)
    replaces the recursion, so a deep fact graph cannot hit Python's recursion limit)."""
    n = len(adj)
    index, low, on_stack = [-1] * n, [0] * n, [False] * n
    stack, comps, counter = [], [], 0
    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        work = [(root, 0)]
        while work:
            v, i = work[-1]
            if i < len(adj[v]):
                work[-1] = (v, i + 1)
                w = adj[v][i]
                if index[w] == -1:  # tree edge: descend, as the recursive version would
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w]:  # back or cross edge into the current component
                    low[v] = min(low[v], index[w])
            else:  # every edge of v done: the "return" of the recursive call
                work.pop()
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
                if low[v] == index[v]:  # v is the root of a component: pop it
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp.append(w)
                        if w == v:
                            break
                    comps.append(comp)
    return comps


def sccs(task: Task) -> tuple[int, int]:
    """(number of strongly connected components of the fact graph, number of non-trivial ones). A component is
    non-trivial if it has more than one fact or its one fact has a self-loop: exactly the components on a cycle."""
    edges = fact_edges(task)
    comps = _tarjan(_adjacency(task))
    nontrivial = sum(len(c) > 1 or (c[0], c[0]) in edges for c in comps)
    return len(comps), nontrivial


def and_share(task: Task) -> float:
    """Share of actions with 2 or more preconditions (AND nodes of the attack graph); 0.0 with no actions."""
    return sum(len(a.pre) >= 2 for a in task.actions) / len(task.actions) if task.actions else 0.0


def delete_interactions(task: Task) -> int:
    """Number of actions whose delete list hits a precondition of some other action. An action deleting its
    own precondition only is not counted: that alone cannot force another action to be redone."""
    count = 0
    for i, a in enumerate(task.actions):
        if a.del_mask and any(j != i and b.pre_mask & a.del_mask for j, b in enumerate(task.actions)):
            count += 1
    return count


def branching(task: Task, states) -> float | None:
    """Mean number of successors (state-changing applicable actions) over the given states; None if empty."""
    states = list(states)
    return fmean(sum(1 for _ in task.successors(s)) for s in states) if states else None


def optimal_first_moves(task: Task, hstar_s0: float, cap: int = 10) -> int:
    """Number of first moves from task.init that start an optimal plan (alternative routes), capped at cap.

    A successor s2 reached by action a starts an optimal plan iff cost(a) + h*(s2) == h*(s0). h*(s2) comes
    from A* with LM-cut, which is optimal because LM-cut is admissible (also with delete effects, tested).
    """
    count = 0
    for i, s2 in task.successors(task.init):
        h = astar(task, lmcut, start=s2).cost
        if h < INF and task.actions[i].cost + h == hstar_s0:
            count += 1
            if count >= cap:
                break
    return count


def lmcut_gap_states(task: Task, hs: dict[int, float]) -> int:
    """Number of evaluated states where LM-cut is below h* (LM-cut not exact there)."""
    return sum(1 for s, h in hs.items() if lmcut(task, s) < h)


def lmcut_gap_share(task: Task, hs: dict[int, float]) -> float | None:
    """Share of evaluated states where LM-cut is below h*; None if there are no evaluated states."""
    return lmcut_gap_states(task, hs) / len(hs) if hs else None


def metrics(task: Task, hstar_s0: float, hs: dict[int, float]) -> dict:
    """Every AL2 metric of one solvable task. LM-cut runs once per evaluated state (count and share share it)."""
    count, nontrivial = sccs(task)
    gaps = lmcut_gap_states(task, hs)
    return {"depth": depth(task), "edge_density": edge_density(task), "scc_count": count,
            "scc_nontrivial": nontrivial, "and_share": and_share(task),
            "delete_interactions": delete_interactions(task), "branching": branching(task, hs),
            "optimal_first_moves": optimal_first_moves(task, hstar_s0),
            "lmcut_gap_states": gaps, "lmcut_gap_share": gaps / len(hs) if hs else None}
