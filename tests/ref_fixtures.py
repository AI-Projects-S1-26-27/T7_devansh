"""Hand-written tiny tasks with hand-derived optima, for the reference solver (D28 Round 8, C2) and C1.

Each fixture is written directly in the neutral serialized format (no ahg code builds it), and its
optimum, and where useful h* of named states, was worked out by hand. The derivation is in the comment.
"""

INF = float("inf")


def _t(facts, actions, init, goal):
    return {"facts": facts, "actions": [{"name": n, "pre": p, "add": a, "cost": c} for n, p, a, c in actions],
            "init": init, "goal": goal}


FIXTURES = {
    # Cycle in the fact graph: a -> b -> a, b -> c -> b. Cheapest: a->b (1), b->c (2), c->g (3) = 6;
    # the direct b->g costs 10, so 1 + 10 = 11 loses.
    "cycle": (_t(["a", "b", "c", "g"],
                 [("ab", ["a"], ["b"], 1), ("ba", ["b"], ["a"], 1), ("bc", ["b"], ["c"], 2),
                  ("cb", ["c"], ["b"], 1), ("cg", ["c"], ["g"], 3), ("bg", ["b"], ["g"], 10)],
                 ["a"], ["g"]), 6,
              # h* by hand: from {a,b}: bc + cg = 5; from {a,b,c}: cg = 3.
              {("a", "b"): 5, ("a", "b", "c"): 3, ("a", "b", "c", "g"): 0}),
    # Duplicate transitions: two identical actions s->t cost 2, and a third with the same effect at cost 5.
    "duplicates": (_t(["s", "t"], [("x1", ["s"], ["t"], 2), ("x2", ["s"], ["t"], 2), ("x3", ["s"], ["t"], 5)],
                      ["s"], ["t"]), 2, {}),
    # Multiple achievers of g: directly from a at 5, or via b (3) then b->g (1) = 4, or via {a, b} at 7.
    "multiple_achievers": (_t(["a", "b", "g"],
                              [("ag", ["a"], ["g"], 5), ("ab", ["a"], ["b"], 3), ("bg", ["b"], ["g"], 1),
                               ("abg", ["a", "b"], ["g"], 7)],
                              ["a"], ["g"]), 4, {("a", "b"): 1}),
    # Impossible: g has an achiever, but it needs h, which nothing adds.
    "impossible": (_t(["a", "b", "h", "g"], [("ab", ["a"], ["b"], 1), ("hg", ["h"], ["g"], 1)], ["a"], ["g"]),
                   INF, {("a", "b"): INF}),
    # Impossible: the goal fact has no achiever at all.
    "no_achiever": (_t(["a", "g"], [("aa", ["a"], ["a"], 1)], ["a"], ["g"]), INF, {}),
    # Two disjoint plans of equal cost 4: a->p->g (1 + 3) and a->q->g (2 + 2).
    "equal_cost_plans": (_t(["a", "p", "q", "g"],
                            [("ap", ["a"], ["p"], 1), ("pg", ["p"], ["g"], 3), ("aq", ["a"], ["q"], 2),
                             ("qg", ["q"], ["g"], 2)],
                            ["a"], ["g"]), 4, {("a", "p"): 3, ("a", "q"): 2}),
    # Cost ties inside an AND: goal {p, q}. p and q each cost 2 separately (total 4), or one action adds
    # both for 4: tied. A zero-cost action z adds an irrelevant fact.
    "cost_ties": (_t(["a", "p", "q", "z"],
                     [("ap", ["a"], ["p"], 2), ("aq", ["a"], ["q"], 2), ("apq", ["a"], ["p", "q"], 4),
                      ("az", ["a"], ["z"], 0)],
                     ["a"], ["p", "q"]), 4, {("a", "p"): 2, ("a", "z"): 4}),
    # Shared precondition: r (2) enables p (1) and q (1); g needs p and q (0). Optimum 2 + 1 + 1 = 4.
    # h_add counts r twice (6), h_max sees only one branch (3): a case where heuristics disagree with h*.
    "shared_subgoal": (_t(["a", "r", "p", "q", "g"],
                          [("ar", ["a"], ["r"], 2), ("rp", ["r"], ["p"], 1), ("rq", ["r"], ["q"], 1),
                           ("pqg", ["p", "q"], ["g"], 0)],
                          ["a"], ["g"]), 4, {("a", "r"): 2, ("a", "p", "r"): 1}),
    # The goal already holds in the initial state.
    "goal_in_init": (_t(["a", "b"], [("ab", ["a"], ["b"], 1)], ["a", "b"], ["b"]), 0, {}),
    # A side branch: d costs 1 and enables nothing useful, so h*({a, d}) is still mg's route, 1 + 2 = 3.
    # (Delete-free: no state reachable from a solvable init is a dead end; dead ends need an impossible goal.)
    "side_branch": (_t(["a", "d", "m", "g"],
                              [("ad", ["a"], ["d"], 1), ("am", ["a"], ["m"], 1), ("mg", ["m"], ["g"], 2),
                               ("dd", ["d"], ["d"], 1)],
                              ["a"], ["g"]), 3, {("a", "d"): 3, ("a", "m"): 2}),
}
