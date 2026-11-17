"""D28 learned inputs (P7c, OQ-P7b-1; Round 11 LRN6): planner-view-only encoder for GNN, MLP and WL, its
leakage audit, and identical encodings and values in training and search.

Generated documents need the pinned generator inputs (skipped without them); they get the planner context
through the d28_context stand-in until the generator writes it (d28-gen, OQ-P7c-1)."""

import copy
import json
import re
from pathlib import Path

import pytest
import torch

from ahg.gen import generate as gen
from ahg.learning import d28encode as de, train
from ahg.learning.encode import gnn_inputs
from ahg.learning.heuristic import Learned
from ahg.learning.leakage import FeaturePolicy, LeakageError
from ahg.learning.models import MLP, WLLinear, goal_mask, output_transform, wl_transform
from ahg.learning.runs import make_config
from ahg.dataset import WLVocab
from d28_context import add_planner_context
from p7_helpers import tiny_doc

ROOT = Path(__file__).resolve().parents[1]
GCFG = json.loads((ROOT / "configs" / "gen" / "p6.json").read_text())
HAVE = all((ROOT / p).exists() for p in (GCFG["master"]["path"], GCFG["catalogue"]["path"]))
POLICY = de.load_policy()


@pytest.fixture(scope="module")
def docs():
    if not HAVE:
        pytest.skip("pinned generator inputs not in this checkout")
    ctx = gen.Context.load()
    out = []
    for arch, n, linux, cohort in (("tree", 12, .5, "core"), ("tree", 20, .25, "core"), ("tree", 20, .75, "core"),
                                   ("chain", 12, .5, "core"), ("tree", 12, .5, "temporal"), ("tree", 30, .5, "core")):
        for k in range(3):
            out.append(gen.generate(ctx, arch, f"p7-smoke.p7c/{arch}{n}{cohort}{linux}/{k}", n, cohort=cohort,
                                    os_mix={"linux": linux, "windows": 1 - linux}, p=1.0))
    return out


def solvable(docs):
    """Learned models are trained and searched on solvable tasks, where every reachable state has finite h_max."""
    out = [add_planner_context(d) for d in docs if d["solvable"]]
    assert len(out) >= 3
    return out


def states_of(task, limit=40):
    """The initial state and states reachable from it, breadth first (up to limit)."""
    seen, frontier = [task.init], [task.init]
    while frontier and len(seen) < limit:
        nxt = []
        for s in frontier:
            for _, s2 in task.successors(s):
                if s2 not in seen and len(seen) < limit:
                    seen.append(s2)
                    nxt.append(s2)
        frontier = nxt
    return seen


# ------------------------------------------------------------------ the feature policy

def test_policy_allows_exactly_the_encoders_names_and_none_it_denies():
    assert sorted(POLICY.allow) == sorted(de.feature_names(residual=True))
    assert len(de.ACTION_FEATURES) == de.ACTION_DIM == 1 + sum(len(v) for v in de.GROUPS.values())
    assert de.audit_docs([tiny_doc()], "t")["ok"]


def test_policy_denies_every_cited_account_product_and_forbidden_kind():
    table = json.loads((ROOT / "configs" / "gen" / "principals.json").read_text())["entries"]
    accounts = {e["principal"] for e in table.values()} | {c["principal"] for e in table.values()
                                                           for c in e.get("cves", {}).values()}
    accounts = (accounts | set(GCFG["admin_principals"])) - {"UNKNOWN"}
    products = {p for key in table for part in key.split("|") for p in [part, *part.split(":")]}
    for name in sorted(accounts | products):
        assert POLICY.denied_by(name), name
        assert POLICY.denied_by(f"principal_class={name}"), name
    for name in ("CVE-2019-15846", "cve_id", "cpe:2.3:a:exim:exim:4.92:*", "https://nvd.nist.gov/vuln", "2024-06-01",
                 "version=4.92", "KB5005565", "os=ubuntu-22.04", "os=windows_server_2019", "seed", "row_index",
                 "vendor=openbsd", "product_id", "release_line", "family=http", "advisory_text"):
        assert POLICY.denied_by(name), name


# ------------------------------------------------------------------ the encoder

def test_every_one_hot_has_exactly_one_bit_on_d28_actions(docs):
    n_actions, seen = 0, {g: set() for g in de.GROUPS}
    for d in docs:
        inst = de.instance(add_planner_context(d), "ac")
        assert len(inst.action_x) == len(inst.task.actions) == len(d["task"]["actions"])
        for row in inst.action_x:
            j = 1
            for g, vocab in de.GROUPS.items():
                bits = row[j:j + len(vocab)]
                assert sorted(bits) == [0.0] * (len(vocab) - 1) + [1.0], (g, row)  # never all zero
                seen[g].add(vocab[bits.index(1.0)])
                j += len(vocab)
        n_actions += len(inst.action_x)
    assert n_actions > 100
    assert seen["type"] == {"remote_exploit", "local_escalation"} and seen["os"] == {"linux", "windows"}
    assert seen["grants"] == {"foothold", "admin"} and seen["ac"] == {"L", "H"}


def test_values_follow_the_generators_semantics(docs):
    for d in docs:
        c = add_planner_context(d)
        meta = {m["action"]: m for m in d["metadata"]["actions"]}
        os_of = {h["id"]: h["os"] for h in d["metadata"]["hosts"]}
        for i, (a, v) in enumerate(zip(d["task"]["actions"], de.action_values(de.planner_view(c)))):
            kind = a["name"].split()[0]
            assert v["type"] == {"remote": "remote_exploit", "local": "local_escalation"}[kind] == \
                {"remote": "remote_exploit", "local": "local_escalation"}[meta[i]["kind"]]
            assert v["ac"] == meta[i]["ac"]  # the merged action's AC (cheapest), as the generator costs it
            assert v["grants"] == ("admin" if any(d["task"]["facts"][f].startswith("admin(") for f in a["add"])
                                   else "foothold")
            host = {int(re.search(r"\(h(\d+)", d["task"]["facts"][f])[1]) for f in a["add"]}.pop()
            assert v["os"] == os_of[host]
        # the principal classes name no account
        assert set(c["network"]["principal_classes"]) <= set(de.PRINCIPAL_CLASSES)


def test_the_encoding_reads_nothing_outside_the_planner_view(docs):
    for d in docs[:6]:
        c = add_planner_context(d)
        bare = {k: c[k] for k in ("task", "network")}
        scrambled = copy.deepcopy(c)
        scrambled["metadata"] = {"principals": ["x"] * 9, "hosts": [], "actions": [], "matches": []}
        scrambled["params"] = {"seed": "other"}
        ref = de.instance(c, "uniform")
        assert de.instance(bare, "uniform") == ref == de.instance(scrambled, "uniform")
        assert de.exposed_names([c]) == de.exposed_names([bare])
        assert set(de.planner_view(c)) == {"task", "network"}


def test_current_generator_documents_are_refused_until_the_generator_writes_the_context(docs):
    with pytest.raises(de.EncodingError, match="network.principal_classes"):
        de.instance(docs[0], "ac")


def test_costs_follow_the_regime_and_ac_stays_a_feature():
    doc = tiny_doc()
    ac, un = de.instance(doc, "ac"), de.instance(doc, "uniform")
    assert [r[0] for r in ac.action_x] == [4.0, 7.0, 4.0, 4.0] and [r[0] for r in un.action_x] == [1.0] * 4
    assert [r[1:] for r in ac.action_x] == [r[1:] for r in un.action_x]
    col = de.ACTION_FEATURES.index("ac=H")
    assert [r[col] for r in un.action_x] == [0.0, 1.0, 0.0, 0.0]


# ------------------------------------------------------------------ the leakage audit on encoded tasks

def test_audit_passes_on_encoded_d28_tasks_and_keeps_its_report(docs, tmp_path):
    rep = de.audit_docs([add_planner_context(d) for d in docs], "p7c_smoke", tmp_path / "audit.json")
    assert rep["ok"] and json.loads((tmp_path / "audit.json").read_text())["ok"]
    assert set(rep["features"]) <= POLICY.allow


@pytest.mark.parametrize("field,value,reason", [
    ("principal_classes", "nobody", "denied"),         # an account name instead of its class
    ("principal_classes", "root", "denied"),
    ("principal_classes", "superuser", "not on the allowlist"),
    ("os", "ubuntu-22.04", "denied"),                  # an exact OS release instead of the family
    ("os", "windows_server_2019", "denied"),
])
def test_audit_fails_on_any_forbidden_feature(field, value, reason, tmp_path):
    doc = tiny_doc()
    if field == "os":
        doc["network"]["hosts"][1]["os"] = value
    else:
        doc["network"]["principal_classes"][1] = value
    with pytest.raises(LeakageError):
        de.audit_docs([doc], "bad", tmp_path / "audit.json")
    rep = json.loads((tmp_path / "audit.json").read_text())
    assert not rep["ok"] and any(reason in v["reason"] for v in rep["violations"])
    with pytest.raises(de.EncodingError):  # and the encoder itself refuses the value
        de.instance(doc, "ac")


def test_audit_fails_on_a_forbidden_name_in_the_feature_list():
    for bad in ("cve_id", "product=exim:exim", "principal=root", "seed"):
        with pytest.raises(LeakageError):
            from ahg.learning.leakage import run_audit
            run_audit(de.feature_names() + [bad], POLICY, "x", "/dev/null")
    with pytest.raises(ValueError, match="denylist forbids"):
        FeaturePolicy(frozenset(de.feature_names() + ["cve_id"]), POLICY.deny)


# ------------------------------------------------------------------ training and search: same inputs, same values

def models(seed=0):
    torch.manual_seed(seed)
    n_mlp = len(de.MLP_FEATURES)
    out = {}
    for kind in ("gnn", "gnn_res", "mlp"):
        cfg = make_config(kind=kind.split("_")[0], seed=seed, hidden=8, layers=2, cost_regime="ac", features="d28",
                          **({"target": "residual"} if kind == "gnn_res" else {}))
        if kind == "mlp":
            model = MLP(n_mlp, 8, torch.rand(n_mlp), torch.rand(n_mlp) + .5, 4.0)
        else:
            model = train.new_gnn(cfg, 4.0, 4.0)
        out[kind] = (model.eval(), cfg)
    return out


def training_side(kind, model, inst, states):
    """h as a training loop computes it: the batch encodings, the model, the shared output transform."""
    s = float(model.label_scale)
    with torch.no_grad():
        if kind == "mlp":
            raw = model(de.mlp_matrix(inst, states))
            return (s * output_transform(raw, torch.tensor([inst.task.is_goal(x) for x in states]))).tolist()
        g = de.gnn_group(inst, states, residual=kind == "gnn_res")
        if kind == "gnn_res":
            raw = model(g["fx"], g["graph"], g["glob"])
            return (s * output_transform(raw, goal_mask(g["fx"]), g["hm"] / model.label_scale)).tolist()
        return (s * output_transform(model(g["fx"], g["graph"]), goal_mask(g["fx"]))).tolist()


def test_training_and_search_see_identical_encodings(docs):
    for d in [tiny_doc()] + solvable(docs)[:4]:
        for regime in ("ac", "uniform"):
            inst = de.instance(d, regime)
            states = states_of(inst.task)
            g = de.gnn_group(inst, states)
            g1 = de.static_graph(inst)
            assert g["graph"].keys() == g1.keys() and all(torch.equal(g["graph"][k], g1[k]) for k in g1)
            assert g["graph"]["act_x"].shape == (len(inst.task.actions), de.ACTION_DIM)
            assert torch.equal(g["fx"], gnn_inputs(inst.task, states, False)["fx"])
            assert de.mlp_matrix(inst, states).tolist() == [
                torch.tensor(de.mlp_features(inst, s), dtype=torch.float32).tolist() for s in states]
            assert de.wl_histograms(inst, states) == [de.wl_colours(inst, s) for s in states]


def test_the_same_checkpoint_gives_the_same_value_in_training_and_search(docs, tmp_path):
    insts = [de.instance(tiny_doc(), "ac")] + [de.instance(d, "ac") for d in solvable(docs)[:3]]
    for kind, (model, cfg) in models().items():
        path = tmp_path / f"{kind}.pt"
        train.save(model, cfg, path)
        loaded, lcfg = train.load(path)
        for inst in insts:
            states = states_of(inst.task)
            want = training_side(kind, model, inst, states)
            for m, c in ((model, cfg), (loaded, lcfg)):
                h = Learned(m, c, kind).bind(inst)
                got = [h(inst.task, s) for s in states]
                assert got == pytest.approx(want, rel=1e-6, abs=1e-6), kind
            assert any(v > 0 for v in want)
    # WL: a vocabulary fitted on training histograms, the same colours at search time
    inst = insts[1] if len(insts) > 1 else insts[0]
    states = states_of(inst.task)
    hists = de.wl_histograms(inst, states)
    vocab = WLVocab().fit(hists).index
    torch.manual_seed(1)
    wl = WLLinear(vocab, torch.rand(len(vocab), dtype=torch.float64), 0.5, 4.0)
    cfg = make_config(kind="wl", seed=0, cost_regime="ac", features="d28")
    train.save(wl, cfg, tmp_path / "wl.pt")
    wl2, cfg2 = train.load(tmp_path / "wl.pt")
    want = [4.0 * float(wl_transform(torch.tensor([wl.raw_one(hh)], dtype=torch.float64),
                                     torch.tensor([inst.task.is_goal(s)]))[0]) for hh, s in zip(hists, states)]
    for m, c in ((wl, cfg), (wl2, cfg2)):
        h = Learned(m, c, "wl").bind(inst)
        assert [h(inst.task, s) for s in states] == want


def test_a_checkpoint_binds_only_to_the_feature_set_it_was_trained_on():
    model, cfg = models()["gnn"]
    with pytest.raises(ValueError, match="d28"):
        Learned(model, {k: v for k, v in cfg.items() if k != "features"}, "x").bind(de.instance(tiny_doc(), "ac"))
    with pytest.raises(ValueError, match="d28"):
        Learned(model, cfg, "x").bind(type("D19", (), {"task": de.instance(tiny_doc(), "ac").task})())
    with pytest.raises(ValueError, match="features"):
        make_config(kind="gnn", seed=0, features="d27")
