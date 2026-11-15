import itertools
import random

from ahg.task import Task


def random_task(seed, n_facts=9, n_actions=13, max_pre=2, max_add=2, goal_size=2) -> Task:
    """Small random delete-free task; may be unsolvable.

    Defaults chosen so hmax is usually below h*, so the tests exercise the
    cases where heuristics disagree.
    """
    rng = random.Random(seed)
    facts = [f"f{i}" for i in range(n_facts)]
    init = rng.sample(facts, rng.randint(1, 2))
    goal = rng.sample([f for f in facts if f not in init], goal_size)
    actions = []
    for i in range(n_actions):
        pre = rng.sample(facts, rng.randint(0, max_pre))
        add = rng.sample(facts, rng.randint(1, max_add))
        actions.append((f"a{i}", pre, add, rng.randint(1, 20)))
    return Task.from_names(facts, init, goal, actions)


# The brute force and reachability below read only the Task's data fields (pre/add index tuples, cost,
# init bits, goal tuple). They use none of task.py's transition code (Action.applicable, useful,
# add_mask, Task.successors, is_goal), so a bug there cannot hide in an agreement test (inventory #1).


def _bits_to_set(bits: int, n: int) -> frozenset:
    return frozenset(i for i in range(n) if bits >> i & 1)


def _set_to_bits(s) -> int:
    return sum(1 << i for i in s)


def brute_force_optimal(task: Task, start=None) -> float:
    """Optimal cost by trying every subset of actions.

    In a delete-free task order only matters for applicability, so a subset S
    is a valid plan iff repeatedly applying applicable actions of S from the
    start reaches the goal. Independent of all search and transition code.
    """
    n_facts = len(task.facts)
    s0 = _bits_to_set(task.init if start is None else start, n_facts)
    goal = frozenset(task.goal)
    acts = [(frozenset(a.pre), frozenset(a.add), a.cost) for a in task.actions]
    best = float("inf")
    for r in range(len(acts) + 1):
        for subset in itertools.combinations(acts, r):
            cost = sum(c for _, _, c in subset)
            if cost >= best:
                continue
            state, changed = s0, True
            while changed:
                changed = False
                for pre, add, _ in subset:
                    if pre <= state and not add <= state:
                        state = state | add
                        changed = True
            if goal <= state:
                best = cost
    return best


def reachable_states(task: Task) -> set[int]:
    """Every state reachable from init, as bitmasks; computed on fact sets, not with Task.successors."""
    n_facts = len(task.facts)
    acts = [(frozenset(a.pre), frozenset(a.add)) for a in task.actions]
    s0 = _bits_to_set(task.init, n_facts)
    seen, stack = {s0}, [s0]
    while stack:
        s = stack.pop()
        for pre, add in acts:
            if pre <= s:
                s2 = s | add
                if s2 not in seen:
                    seen.add(s2)
                    stack.append(s2)
    return {_set_to_bits(s) for s in seen}
