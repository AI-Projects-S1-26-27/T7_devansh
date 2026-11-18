"""AL1 step 1: delete effects in the planning core (decisions.md §21; exploratory M3 access-loss model).

Covers successor generation, the unchanged delete-free path (digests, serialization, successors), the
independent oracle and reference solver on tasks with deletes, PDDL export and parse-back, Fast Downward,
and admissibility of the delete-relaxation heuristics. The two hand-built tasks have hand-derived optima
and h_max values (worked out in the comments).
"""

import hashlib
import json
import random

import pytest

from ahg import fd_timing, oracle, pddl, refsolver
from ahg.heuristics import hadd, hmax, lmcut
from ahg.search import astar
from ahg.task import Action, Task, plan_cost, task_digest

from ref_fixtures import FIXTURES

INF = float("inf")

# Signature block (diagnostic 5). Exploit group X into monitored hosts A and B; each detected use deletes
# sig_free(X) ("sig"); evade (5) restores it. An alternative CVE y into B costs 7 and is not gated.
#   h*:   xA 4 + yB 7 + esc 4 = 15   (xA + evade 5 + xB 4 + esc 4 = 17 is worse)
#   h_max(init): the relaxation never loses sig, so a = 4, b = 8 via xB, g = 12  -> gap 3
#   M0 (deletes dropped): 12, via xB, so the deletes change both the cost and the route.
SIG = (["s", "sig", "a", "b", "g"],
       [("xA", ["s", "sig"], ["a"], 4, ["sig"]), ("xB", ["a", "sig"], ["b"], 4, ["sig"]),
        ("yB", ["a"], ["b"], 7), ("evade", [], ["sig"], 5), ("esc", ["b"], ["g"], 4)],
       ["s", "sig"], ["g"])

# Re-acquisition (diagnostic 2). x1 gains a volatile foothold f; x2 (into a monitored host) is detected and
# kills f; the goal action needs q and f together, so f must be re-exploited.
#   h*: x1 3 + x2 4 + x1 3 + fin 1 = 11.  h_max(init) = max(f 3, q 7) + 1 = 8.  h_add(init) = 3 + 7 + 1 = 11.
REACQ = (["s", "f", "q", "g"],
         [("x1", ["s"], ["f"], 3), ("x2", ["f"], ["q"], 4, ["f"]), ("fin", ["q", "f"], ["g"], 1)],
         ["s"], ["g"])


def build(spec) -> Task:
    facts, actions, init, goal = spec
    return Task.from_names(facts, init, goal, actions)


def bits(task, names):
    return sum(1 << task.facts.index(n) for n in names)


# ---------------------------------------------------------------- Action / successor mechanics

def test_action_refuses_add_delete_overlap_and_repeats():
    with pytest.raises(ValueError, match="adds and deletes"):
        Action("a", (0,), (1,), 1, (1,))
    with pytest.raises(ValueError, match="twice"):
        Action("a", (0,), (1,), 1, (2, 2))


def test_successor_is_state_minus_deletes_plus_adds():
    rng = random.Random(0)
    for _ in range(300):
        n = 8
        pre = tuple(rng.sample(range(n), rng.randint(0, 2)))
        add = tuple(rng.sample(range(n), rng.randint(0, 2)))
        dele = tuple(f for f in rng.sample(range(n), rng.randint(0, 3)) if f not in add)
        a = Action("a", pre, add, 1, dele)
        s = rng.getrandbits(n)
        s2 = a.apply(s)
        assert s2 == (s & ~a.del_mask) | a.add_mask
        assert all(not s2 >> f & 1 for f in dele)          # every deleted fact is gone
        assert all(s2 >> f & 1 for f in add)               # every added fact is held
        changed = s ^ s2                                    # only declared facts change
        assert changed & ~(a.del_mask | a.add_mask) == 0


def test_delete_free_successors_match_the_old_rule():
    """On every delete-free fixture, successors are exactly the old 'applicable and adds something new' set."""
    for name, (st, _, _) in FIXTURES.items():
        t = Task.from_names(st["facts"], st["init"], st["goal"],
                            [(a["name"], a["pre"], a["add"], a["cost"]) for a in st["actions"]])
        assert t.delete_free
        for s in range(1 << len(t.facts)):
            old = [(i, s | a.add_mask) for i, a in enumerate(t.actions) if a.pre_mask & ~s == 0 and a.add_mask & ~s]
            assert list(t.successors(s)) == old, name


def test_successors_skip_only_exact_self_loops():
    t = build(REACQ)
    s = bits(t, ["s", "f"])
    names = {t.actions[i].name: s2 for i, s2 in t.successors(s)}
    assert set(names) == {"x2"}  # x1 would re-add f (a self-loop); fin is not applicable
    assert names["x2"] == bits(t, ["s", "q"])


# ---------------------------------------------------------------- frozen delete-free artefacts unchanged

def _old_digest(task):
    blob = json.dumps({"facts": task.facts, "actions": [[a.name, a.pre, a.add, a.cost] for a in task.actions],
                       "init": task.init, "goal": task.goal})
    return hashlib.sha256(blob.encode()).hexdigest()


def test_delete_free_digest_and_serialization_unchanged():
    for st, _, _ in FIXTURES.values():
        t = Task.from_names(st["facts"], st["init"], st["goal"],
                            [(a["name"], a["pre"], a["add"], a["cost"]) for a in st["actions"]])
        assert task_digest(t) == _old_digest(t)
        assert refsolver.serialize(t)["actions"] == st["actions"]  # no "del" key on delete-free actions


def test_digest_sees_deletes():
    t = build(SIG)
    stripped = Task(t.facts, [Action(a.name, a.pre, a.add, a.cost) for a in t.actions], t.init, t.goal)
    assert task_digest(t) != task_digest(stripped) == _old_digest(stripped)


# ---------------------------------------------------------------- hand-derived optima and gaps

def test_signature_block_task_by_hand():
    t = build(SIG)
    r = astar(t, lmcut)
    assert r.cost == 15 and [t.actions[i].name for i in r.plan] == ["xA", "yB", "esc"]
    assert plan_cost(t, r.plan) == 15
    assert oracle.optimal_cost(t) == 15 and refsolver.solve(refsolver.serialize(t))[0] == 15
    assert hmax(t, t.init) == 12  # gap 3: the relaxation keeps sig_free(X)
    m0 = Task(t.facts, [Action(a.name, a.pre, a.add, a.cost) for a in t.actions], t.init, t.goal)
    r0 = astar(m0, lmcut)
    assert r0.cost == 12 and [m0.actions[i].name for i in r0.plan] == ["xA", "xB", "esc"]


def test_reacquisition_task_by_hand():
    t = build(REACQ)
    r = astar(t, lmcut)
    assert r.cost == 11 and [t.actions[i].name for i in r.plan] == ["x1", "x2", "x1", "fin"]
    assert hmax(t, t.init) == 8 and hadd(t, t.init) == 11
    with pytest.raises(ValueError):  # without re-exploiting, fin is not applicable: f was deleted
        plan_cost(t, [0, 1, 2])


# ---------------------------------------------------------------- independent agreement on every reachable state

def _random_delete_task(rng, n=6, m=9):
    facts = [f"f{i}" for i in range(n)]
    acts = []
    for k in range(m):
        pre = rng.sample(facts, rng.randint(1, 2))
        add = rng.sample(facts, 1)
        dele = [f for f in rng.sample(facts, rng.randint(0, 2)) if f not in add]
        acts.append((f"a{k}", pre, add, rng.randint(1, 5), dele))
    return Task.from_names(facts, ["f0"], [facts[-1]], acts)


@pytest.mark.parametrize("spec", [SIG, REACQ], ids=["signature", "reacquire"])
def test_oracle_refsolver_astar_agree_and_relaxations_admissible(spec):
    t = build(spec)
    st = refsolver.serialize(t)
    hs = refsolver.hstar_all(st)
    for state, h in hs.items():
        b = refsolver.to_bits(st, state)
        assert oracle.optimal_cost(t, b) == h
        r = astar(t, lmcut, start=b)
        assert r.cost == h
        if r.plan is not None:
            assert plan_cost(t, r.plan, start=b) == h
        for f in (hmax, lmcut):
            assert f(t, b) <= h  # the delete relaxation stays admissible with deletes


def test_random_delete_tasks_agree():
    rng = random.Random(7)
    for _ in range(40):
        t = _random_delete_task(rng)
        st = refsolver.serialize(t)
        for state, h in refsolver.hstar_all(st).items():
            b = refsolver.to_bits(st, state)
            assert oracle.optimal_cost(t, b) == h == astar(t, lmcut, start=b).cost
            assert hmax(t, b) <= h and lmcut(t, b) <= h


def test_refsolver_validates_deletes():
    st = refsolver.serialize(build(SIG))
    st["actions"][0]["del"] = ["nope"]
    with pytest.raises(refsolver.ReferenceTaskError):
        refsolver.solve(st)
    st["actions"][0]["del"] = ["a"]  # xA adds a
    with pytest.raises(refsolver.ReferenceTaskError, match="adds and deletes"):
        refsolver.solve(st)


# ---------------------------------------------------------------- PDDL and Fast Downward

def test_pddl_round_trip_with_deletes():
    for spec in (SIG, REACQ):
        t = build(spec)
        back = pddl.parse_grounded(pddl.domain_pddl(t), pddl.problem_pddl(t))
        assert back["actions"] == [(list(a.pre), list(a.add), a.cost) for a in t.actions]
        assert back["dels"] == [list(a.dele) for a in t.actions]
        assert "(not (f1))" in pddl.domain_pddl(t)


def test_pddl_delete_free_export_has_no_negation():
    st = FIXTURES["cycle"][0]
    t = Task.from_names(st["facts"], st["init"], st["goal"], [(a["name"], a["pre"], a["add"], a["cost"]) for a in st["actions"]])
    assert "(not " not in pddl.domain_pddl(t)
    assert pddl.parse_grounded(pddl.domain_pddl(t), pddl.problem_pddl(t))["dels"] == [[]] * len(t.actions)


FD_OK = (fd_timing.FD_DIR / "fast-downward.py").exists() and fd_timing.fd_version() == fd_timing.FD_VERSION


@pytest.mark.skipif(not FD_OK, reason="Fast Downward release-24.06.1 not set up")
@pytest.mark.parametrize("spec,opt", [(SIG, 15), (REACQ, 11)])
def test_fast_downward_agrees_with_deletes(tmp_path, spec, opt):
    t = build(spec)
    (tmp_path / "d.pddl").write_text(pddl.domain_pddl(t))
    (tmp_path / "p.pddl").write_text(pddl.problem_pddl(t))
    r = fd_timing.run(tmp_path / "d.pddl", tmp_path / "p.pddl", "lmcut", 1, warmup=0)
    assert r["status"] == "solved" and r["cost"] == opt
