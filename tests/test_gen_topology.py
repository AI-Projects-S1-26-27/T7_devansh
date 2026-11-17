"""P6 topology archetypes: zone sizes, subnets, links, firewall draws and reachability (no data needed)."""

import json
import random
from pathlib import Path

import pytest

from ahg.gen import topology

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "gen" / "p6.json").read_text())["topology"]


def zone_sizes(lay):
    out = {}
    for s in lay.subnets:
        out[s.zone] = out.get(s.zone, 0) + len(s.hosts)
    return out


@pytest.mark.parametrize("n,per", [(12, 1), (20, 1), (30, 1), (60, 2), (100, 3), (140, 4)])
def test_tree_zone_sizes(n, per):
    """DMZ and sensitive get max(1, round(n/40)) each, halves rounded up (OQ-P6-1); the rest are users."""
    lay = topology.layout("tree", n, CFG)
    assert zone_sizes(lay) == {"internet": 1, "dmz": per, "sensitive": per, "users": n - 2 * per}


@pytest.mark.parametrize("n", [3, 7, 12, 20, 30, 60])
def test_tree_user_subnets_form_a_binary_tree(n):
    lay = topology.layout("tree", n, CFG)
    users = [i for i, s in enumerate(lay.subnets) if s.zone == "users"]
    assert all(1 <= len(lay.subnets[i].hosts) <= 5 for i in users)
    sizes = [len(lay.subnets[i].hosts) for i in users]
    assert max(sizes) - min(sizes) <= 1 and len(users) == -(-(n - 2 * max(1, (n + 20) // 40)) // 5)
    dmz = next(i for i, s in enumerate(lay.subnets) if s.zone == "dmz")
    sens = next(i for i, s in enumerate(lay.subnets) if s.zone == "sensitive")
    want = {(0, dmz), (dmz, sens), (dmz, users[0])}
    for k, s in enumerate(users):
        if k:
            want.add(tuple(sorted((users[(k - 1) // 2], s))))
        want.add(tuple(sorted((sens, s))))
    assert set(lay.links) == want


@pytest.mark.parametrize("n,ent", [(4, 1), (12, 4), (20, 6), (30, 10), (60, 20)])
def test_chain_zones_and_links(n, ent):
    lay = topology.layout("chain", n, CFG)
    assert zone_sizes(lay) == {"internet": 1, "dmz": 1, "users": n - 2 - ent, "enterprise": ent, "sensitive": 1}
    assert [s.zone for s in lay.subnets] == ["internet", "dmz", "users", "enterprise", "sensitive"]
    assert lay.links == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_too_small_and_unknown_archetypes_raise():
    with pytest.raises(ValueError):
        topology.layout("tree", 2, CFG)
    with pytest.raises(ValueError):
        topology.layout("chain", 3, CFG)
    with pytest.raises(ValueError):
        topology.layout("flat", 12, CFG)


def test_firewall_allows_only_far_side_families_and_nothing_to_the_internet():
    lay = topology.layout("tree", 12, CFG)
    rng = random.Random(1)
    fams = {0: []} | {h: rng.sample(["http", "smb", "ssh", "dns"], 2) for s in lay.subnets[1:] for h in s.hosts}
    fw = topology.draw_firewall(lay, fams, random.Random(7), 0.5)
    assert set(fw) == {d for a, b in lay.links for d in ((a, b), (b, a))}
    for (a, b), allow in fw.items():
        far = {f for h in lay.subnets[b].hosts for f in fams[h]}
        assert set(allow) <= far and allow == sorted(allow)
        if b == 0:
            assert allow == []
    # each direction drawn independently: over many draws every family of a far side running two or more is
    # sometimes allowed, sometimes not; a far side running one family always allows it (NASim rule, OQ-P6-18)
    seen = {}
    for s in range(200):
        for k, allow in topology.draw_firewall(lay, fams, random.Random(s), 0.5).items():
            far = {f for h in lay.subnets[k[1]].hosts for f in fams[h]}
            for f in far:
                seen.setdefault((k, f, len(far) >= 2), set()).add(f in allow)
    assert any(multi for _, _, multi in seen)
    assert all(v == ({True, False} if multi else {True}) for (_, _, multi), v in seen.items())


def test_firewall_allows_at_least_one_running_family_per_direction():
    """§18, 17 Nov 2026 (OQ-P6-18): NASim's guarantee. Every direction towards a subnet that runs something
    allows at least one of its families; the internet direction stays empty."""
    for arch, n in (("tree", 12), ("tree", 30), ("chain", 20)):
        lay = topology.layout(arch, n, CFG)
        for s in range(100):
            rng = random.Random(s)
            fams = {0: []} | {h: rng.sample(["http", "smb", "ssh", "dns", "sql"], rng.randint(1, 3))
                              for sn in lay.subnets[1:] for h in sn.hosts}
            for (a, b), allow in topology.draw_firewall(lay, fams, random.Random(1000 + s), 0.5).items():
                far = {f for h in lay.subnets[b].hosts for f in fams[h]}
                assert bool(allow) == bool(far) and set(allow) <= far


def test_firewall_independent_draw_is_kept_and_fallback_is_uniform():
    """The 0.5 draws are those of the old rule (same stream, same order); only a direction whose draws allowed
    nothing gets one extra draw: a uniform choice among the families running on the far side."""
    lay = topology.layout("chain", 12, CFG)
    fams = {0: []} | {h: ["dns", "http", "smb"] for sn in lay.subnets[1:] for h in sn.hosts}
    forced = {}
    for s in range(3000):
        rng, ref = random.Random(s), random.Random(s)
        fw = topology.draw_firewall(lay, fams, rng, 0.5)
        for a, b in lay.links:  # replay the documented draw order with the reference stream
            for src, dst in ((a, b), (b, a)):
                far = sorted({f for h in lay.subnets[dst].hosts for f in fams[h]})
                first = [f for f in far if ref.random() < 0.5]
                if first:
                    assert fw[(src, dst)] == first
                elif far:
                    pick = ref.choice(far)
                    assert fw[(src, dst)] == [pick]
                    forced[pick] = forced.get(pick, 0) + 1
                else:
                    assert fw[(src, dst)] == []
    # 3 families, each forced pick about a third of the time (P(empty) = 1/8 per direction)
    total = sum(forced.values())
    assert set(forced) == {"dns", "http", "smb"} and all(abs(v / total - 1 / 3) < 0.05 for v in forced.values())


def test_firewall_never_sees_vulnerabilities():
    """The rule's only inputs are the layout, the families each host runs, the stream and the probability."""
    import inspect

    assert list(inspect.signature(topology.draw_firewall).parameters) == ["lay", "families_of", "rng", "prob"]


def test_can_reach_same_subnet_or_one_allowing_link_only():
    lay = topology.layout("chain", 12, CFG)
    fw = {(0, 1): ["http"], (1, 0): [], (1, 2): ["smb"], (2, 1): [], (2, 3): [], (3, 2): [], (3, 4): ["sql"], (4, 3): []}
    dmz, user_a, user_b = lay.subnets[1].hosts[0], lay.subnets[2].hosts[0], lay.subnets[2].hosts[1]
    ent, sens = lay.subnets[3].hosts[0], lay.subnets[4].hosts[0]
    assert topology.can_reach(lay, fw, 0, dmz, "http") and not topology.can_reach(lay, fw, 0, dmz, "dns")
    assert topology.can_reach(lay, fw, user_a, user_b, "dns")  # inside a subnet everything is open
    assert not topology.can_reach(lay, fw, user_a, user_a, "dns")
    assert topology.can_reach(lay, fw, dmz, user_a, "smb") and not topology.can_reach(lay, fw, user_a, dmz, "http")
    assert not topology.can_reach(lay, fw, 0, user_a, "smb")  # no transit through the DMZ
    assert topology.can_reach(lay, fw, ent, sens, "sql") and not topology.can_reach(lay, fw, user_a, sens, "sql")
