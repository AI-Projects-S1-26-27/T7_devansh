"""STRIPS planning tasks: the formal object every other module works on.

A task has a set of facts (attacker privileges), actions (exploits) with
conjunctive preconditions, add effects and optional delete effects, an
initial state and a goal. Every frozen D28 task is delete-free: once the
attacker holds a privilege it is never lost (access monotonicity, the usual
attack-graph assumption). Delete effects exist only for the exploratory M3
access-loss model (AL1, decisions.md §21). Applying an action gives
(state minus deletes) plus adds, the standard STRIPS order, so an add wins
over a delete of the same fact; Action refuses that overlap anyway.

A state is a set of facts, stored as an int bitmask: bit i set means fact i
is held. Bitmasks are hashable, cheap to copy, and make "are all
preconditions held?" a single AND.
"""

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property

State = int


@dataclass(frozen=True)
class Action:
    name: str
    pre: tuple[int, ...]
    add: tuple[int, ...]
    cost: int
    dele: tuple[int, ...] = ()  # delete effects ("del" is a Python keyword); empty on every delete-free task
    pre_mask: int = field(init=False, repr=False, compare=False)
    add_mask: int = field(init=False, repr=False, compare=False)
    del_mask: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.cost < 0:
            raise ValueError(f"action {self.name} has negative cost {self.cost}")
        if any(len(set(x)) != len(x) for x in (self.pre, self.add, self.dele)):
            raise ValueError(f"action {self.name} lists a fact twice")
        if set(self.add) & set(self.dele):
            # STRIPS would let the add win; refusing keeps every delete set exactly what its rule declares.
            raise ValueError(f"action {self.name} both adds and deletes a fact")
        object.__setattr__(self, "pre_mask", to_mask(self.pre))
        object.__setattr__(self, "add_mask", to_mask(self.add))
        object.__setattr__(self, "del_mask", to_mask(self.dele))

    def applicable(self, state: State) -> bool:
        return self.pre_mask & ~state == 0

    def apply(self, state: State) -> State:
        """The successor state: deletes removed first, then adds set."""
        return (state & ~self.del_mask) | self.add_mask

    def useful(self, state: State) -> bool:
        """True if applying the action changes the state (on a delete-free action: adds a new fact)."""
        return self.apply(state) != state


@dataclass
class Task:
    facts: list[str]
    actions: list[Action]
    init: State
    goal: tuple[int, ...]

    @cached_property
    def goal_mask(self) -> int:
        return to_mask(self.goal)

    @cached_property
    def pre_of(self) -> list[list[int]]:
        """pre_of[f] = indices of actions that have fact f as a precondition."""
        out = [[] for _ in self.facts]
        for i, a in enumerate(self.actions):
            for f in a.pre:
                out[f].append(i)
        return out

    @cached_property
    def achievers(self) -> list[list[int]]:
        """achievers[f] = indices of actions that add fact f."""
        out = [[] for _ in self.facts]
        for i, a in enumerate(self.actions):
            for f in a.add:
                out[f].append(i)
        return out

    @cached_property
    def delete_free(self) -> bool:
        return not any(a.dele for a in self.actions)

    def is_goal(self, state: State) -> bool:
        return self.goal_mask & ~state == 0

    def successors(self, state: State):
        """Yield (action index, next state) for applicable actions that change the state.

        Skipping an action whose successor equals the state is safe with or without deletes: it is a
        self-loop with nonnegative cost, so it never shortens a path. On a delete-free task this is the
        old "adds something new" test exactly.
        """
        for i, a in enumerate(self.actions):
            if a.applicable(state):
                s2 = a.apply(state)
                if s2 != state:
                    yield i, s2

    def describe(self, state: State) -> set[str]:
        return {self.facts[f] for f in facts_of(state)}

    @classmethod
    def from_names(cls, facts, init, goal, actions):
        """Build a task from readable names.

        actions: iterable of (name, pre_names, add_names, cost) or (name, pre_names, add_names, cost, del_names).
        """
        index = {name: i for i, name in enumerate(facts)}
        if len(index) != len(facts):
            raise ValueError("duplicate fact names")
        acts = []
        for name, pre, add, cost, *rest in actions:
            (dele,) = rest or [()]
            acts.append(Action(name, tuple(index[p] for p in pre), tuple(index[e] for e in add), cost,
                               tuple(index[d] for d in dele)))
        return cls(list(facts), acts, to_mask(index[f] for f in init), tuple(index[g] for g in goal))


def to_mask(facts) -> int:
    mask = 0
    for f in facts:
        mask |= 1 << f
    return mask


def facts_of(state: State) -> list[int]:
    out = []
    i = 0
    while state:
        if state & 1:
            out.append(i)
        state >>= 1
        i += 1
    return out


def plan_cost(task: Task, plan, start: State | None = None) -> int:
    """Check that the plan is executable and reaches the goal; return its cost."""
    state = task.init if start is None else start
    total = 0
    for i in plan:
        a = task.actions[i]
        if not a.applicable(state):
            raise ValueError(f"action {a.name} not applicable")
        state = a.apply(state)
        total += a.cost
    if not task.is_goal(state):
        raise ValueError("plan does not reach the goal")
    return total


def task_digest(task: Task) -> str:
    """SHA-256 over facts, actions (in order: name, pre, add, cost[, del]), init and goal.

    The delete list is appended only when non-empty, so every delete-free task keeps the digest it had
    before delete effects existed (frozen label records and cohorts are keyed by it).
    """
    blob = json.dumps({"facts": task.facts,
                       "actions": [[a.name, a.pre, a.add, a.cost] + ([a.dele] if a.dele else []) for a in task.actions],
                       "init": task.init, "goal": task.goal})
    return hashlib.sha256(blob.encode()).hexdigest()
