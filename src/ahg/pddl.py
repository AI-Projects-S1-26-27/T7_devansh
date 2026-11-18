"""Export a Task to grounded PDDL, for external planners such as Fast Downward (CLAUDE.md §5).

The export is grounded: each fact becomes a 0-ary predicate (f<i>), each action
a parameterless action (a<i>) with its conjunctive preconditions, add effects,
delete effects as (not (f<j>)) (only on M3 access-loss tasks, AL1; a
delete-free Task exports exactly as before) and its integer D9 cost as an
(increase (total-cost) c) effect. Names are indices, so the mapping back
to the Task is the identity; the readable names go in comments.
"""

from ahg.task import State, Task, facts_of


def _conj(atoms):
    return "(and " + " ".join(f"(f{i})" for i in atoms) + ")"


def domain_pddl(task: Task, name: str = "attackgraph") -> str:
    lines = [f"(define (domain {name})", "  (:requirements :strips :action-costs)",
             "  (:predicates " + " ".join(f"(f{i})" for i in range(len(task.facts))) + ")",
             "  (:functions (total-cost))"]
    for i, a in enumerate(task.actions):
        lines += [f"  ; {a.name}", f"  (:action a{i}", "   :parameters ()",
                  f"   :precondition {_conj(a.pre)}",
                  f"   :effect (and {' '.join([f'(f{j})' for j in a.add] + [f'(not (f{j}))' for j in a.dele])}"
                  f" (increase (total-cost) {a.cost})))"]
    return "\n".join(lines + [")"]) + "\n"


def problem_pddl(task: Task, state: State | None = None, name: str = "p", domain: str = "attackgraph") -> str:
    """The problem from `state` (default: the task's initial state) to the task's goal."""
    s = task.init if state is None else state
    init = " ".join(f"(f{i})" for i in facts_of(s))
    return (f"(define (problem {name}) (:domain {domain})\n  (:init {init} (= (total-cost) 0))\n"
            f"  (:goal {_conj(task.goal)})\n  (:metric minimize (total-cost)))\n")


# ---------------------------------------------------------------- parsing back (D28 P6 round-trip check)

def _sexp(text: str):
    """Parse PDDL text into nested lists of atoms; comments (';' to end of line) are dropped."""
    import re

    toks = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack: list[list] = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t.lower())
    if len(stack) != 1:
        raise ValueError("unbalanced parentheses")
    return stack[0]


def _atoms(conj) -> list[int]:
    """Fact indices of an (and (f1) (f2) ...) or a single (fK)."""
    items = conj[1:] if conj and conj[0] == "and" else [conj]
    return [int(a[0][1:]) for a in items]


def parse_grounded(domain: str, problem: str) -> dict:
    """Inverse of domain_pddl/problem_pddl.

    Returns {"facts": n, "actions": [(pre, add, cost)], "dels": [del per action], "init": [..], "goal": [..]};
    "dels" holds an empty list for every action without delete effects. Only the grounded form this module
    writes is accepted; anything else raises ValueError.
    """
    (dom,) = _sexp(domain)
    (prob,) = _sexp(problem)
    if dom[0] != "define" or prob[0] != "define":
        raise ValueError("not a PDDL define")
    preds = next(x for x in dom if isinstance(x, list) and x and x[0] == ":predicates")[1:]
    actions, dels = [], []
    for x in dom:
        if isinstance(x, list) and x and x[0] == ":action":
            body = dict(zip(x[2::2], x[3::2]))
            if body[":parameters"] != []:
                raise ValueError("only parameterless actions are accepted")
            eff = body[":effect"][1:]
            adds = [int(e[0][1:]) for e in eff if e[0] not in ("increase", "not")]
            dels.append([int(e[1][0][1:]) for e in eff if e[0] == "not"])
            (inc,) = [e for e in eff if e[0] == "increase"]
            actions.append((_atoms(body[":precondition"]), adds, int(inc[2])))
    init = next(x for x in prob if isinstance(x, list) and x and x[0] == ":init")[1:]
    goal = next(x for x in prob if isinstance(x, list) and x and x[0] == ":goal")[1]
    return {"facts": len(preds), "actions": actions, "dels": dels,
            "init": [int(a[0][1:]) for a in init if a[0] != "="], "goal": _atoms(goal)}
