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
    # each direction drawn independently: over many draws every family is sometimes allowed, sometimes not
    seen = {}
    for s in range(200):
        for k, allow in topology.draw_firewall(lay, fams, random.Random(s), 0.5).items():
            for f in {f for h in lay.subnets[k[1]].hosts for f in fams[h]}:
                seen.setdefault((k, f), set()).add(f in allow)
    assert all(v == {True, False} for v in seen.values())


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
