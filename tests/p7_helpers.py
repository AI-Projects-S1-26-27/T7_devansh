"""Tiny hand-written task documents and result fixtures for the P7 tests (no generator inputs needed)."""

from ahg.gen.generate import dumps  # noqa: F401


def act(name, pre, add, ac=4, uniform=1):
    return {"name": name, "pre": pre, "add": add, "cost": {"ac": ac, "uniform": uniform}}


def tiny_doc(seed="t/0", solvable=True, n=3, archetype="tree"):
    """facts 0..3; goal 3 needs 1 and 2; one cheap and one dear way to 1; unsolvable variant lacks 2's action.
    D28 fact names and the planner-visible host OS and principal classes, so the D28 encoder can read it."""
    acts = [act("remote a", [0], [1], 4), act("remote b", [0], [1], 7), act("local c", [1], [2], 4),
            act("local d", [1, 2], [3], 4)]
    if not solvable:
        acts = acts[:2] + acts[3:]
    return {"schema": "ahg.gen/1", "params": {"archetype": archetype, "seed": seed, "n": n},
            "solvable": solvable, "unsolvable_reason": None if solvable else "no_foothold_on_goal_host_reachable",
            "task": {"facts": ["foothold(h0,p0)", "foothold(h1,p1)", "admin(h1)", "admin(h2)"], "init": [0],
                     "goal": [3], "actions": acts},
            "network": {"hosts": [{"id": 0, "os": None}, {"id": 1, "os": {"family": "linux", "version": None}},
                                  {"id": 2, "os": {"family": "windows", "version": "Windows Server 2019"}}],
                        "principal_classes": ["attacker", "service"]},
            "metadata": {}}


def timing(ns):
    return {"ns": [ns] * 5, "resolution_ns": 1, "censored": False}


def run(expanded, cost=10, ns=2000):
    return {"expanded": expanded, "cost": cost, "budget_hit": False, "python_time": timing(ns), "fd_time": timing(ns * 3),
            "peak_memory_bytes": 2**20, "overestimation": {"states": 10, "violations": 1}}


def record(i, n=12, g=10, base=20, mlp=15, wl=18, solvable=True, certified=True):
    runs = {"h0": run(100), "hmax": run(base + 5), "hadd": run(base + 9), "hff": run(base + 7), "lmcut": run(base),
            "hsp": run(base + 3)}
    for s in range(3):
        runs |= {f"gnn@{s}": run(g + s), f"mlp@{s}": run(mlp + s), f"wl@{s}": run(wl + s)}
    return {"task": f"d{i}", "network": f"net{i // 2}", "n": n, "solvable": solvable,
            "unsolvable_reason": None if solvable else "no_foothold_on_goal_host_reachable",
            "certified": certified, "hstar": 10 if solvable else None, "runs": runs if solvable else {}}


def checkpoint(path, kind, seed, regime, hidden=8, features="d28"):
    """An untrained, seeded checkpoint of a confirmatory model (gnn, mlp, wl) on the D28 encoder for pipeline
    tests; regime None writes a pre-D28 config without cost_regime, features None one without the D28 encoder.
    Returns the path. Never a trained model."""
    import torch
    from ahg.learning import d28encode, train
    from ahg.learning.models import MLP, WLLinear
    from ahg.learning.runs import make_config
    torch.manual_seed(seed)
    over = ({} if regime is None else {"cost_regime": regime}) | ({} if features is None else {"features": features})
    cfg = make_config(kind=kind, seed=seed, hidden=hidden, layers=2, **over)
    if kind == "gnn":
        model = train.new_gnn(cfg, 4.0, 4.0)
    elif kind == "mlp":
        k = len(d28encode.MLP_FEATURES) if features == "d28" else 7  # 7 = len(dataset.mlp_features)
        model = MLP(k, hidden, torch.zeros(k), torch.ones(k), 4.0)
    else:
        model = WLLinear({}, torch.zeros(0), 1.0, 4.0)
    train.save(model, cfg, path)
    return path
