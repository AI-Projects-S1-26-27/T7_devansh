"""Unseen-CVE holdout rule (decisions.md §18 item 6, 17 Nov 2026; Round 10), on synthetic pool units."""

import pytest

from ahg.gen import pool
from ahg.gen.pool import Unit


def unit(cve, ac, product="p:a"):
    return Unit(cve, product, "http", "linux", "remote", ac, "core")


def pool_units():
    us = [unit(f"CVE-2020-{i}", "L") for i in range(1, 31)]
    us += [unit("CVE-2021-1", "H"), unit("CVE-2021-2", "H"), unit("CVE-2021-2", "L", "q:b")]  # mixed CVE: H wins
    us += [unit("CVE-2020-1", "L", "q:b")]  # a second product unit of an AC:L CVE: still one ID
    return us


def test_holdout_takes_every_ac_h_cve_and_fills_to_the_share_by_id():
    held, c = pool.cve_holdout(pool_units(), 0.3, "cve-holdout")
    assert c["pool_cve_ids"] == 32 and c["target"] == 10  # round_half_up(9.6)
    assert {"CVE-2021-1", "CVE-2021-2"} <= set(held) and len(held) == 10 == len(set(held))
    assert c["ac_h_cves"] == 2 and c["ac_h_cves_with_an_ac_l_unit"] == 1 and c["ac_l_drawn"] == 8
    assert held == sorted(held, key=pool.cve_key)


def test_holdout_depends_only_on_seed_and_pool():
    a, _ = pool.cve_holdout(pool_units(), 0.3, "cve-holdout")
    b, _ = pool.cve_holdout(list(reversed(pool_units())), 0.3, "cve-holdout")
    assert a == b
    others = {tuple(pool.cve_holdout(pool_units(), 0.3, f"s{i}")[0]) for i in range(20)}
    assert len(others) > 1


def test_holdout_refuses_when_ac_h_alone_exceeds_the_share():
    us = [unit(f"CVE-2020-{i}", "H") for i in range(1, 5)] + [unit(f"CVE-2021-{i}", "L") for i in range(1, 7)]
    with pytest.raises(ValueError, match="exceed"):
        pool.cve_holdout(us, 0.3, "cve-holdout")


def p6d_units():
    """40 AC:L-only CVE IDs (one with a second product unit) and 5 AC:H IDs (one mixed H/L, two units)."""
    us = [unit(f"CVE-2020-{i}", "L") for i in range(1, 41)] + [unit("CVE-2020-1", "L", "q:b")]
    us += [unit(f"CVE-2021-{i}", "H") for i in range(1, 6)] + [unit("CVE-2021-5", "L", "q:b")]
    return us


def test_p6d_holdout_takes_half_the_ac_h_ids_and_fills_to_the_share():
    """§18 P6d item 1: round_half_up(0.5 x 5 AC:H IDs) = 3 AC:H IDs, then AC:L IDs up to round_half_up(0.3 x 45) = 14."""
    held, c = pool.cve_holdout(p6d_units(), 0.3, "cve-holdout", 0.5, "cve-holdout-ach")
    high = {f"CVE-2021-{i}" for i in range(1, 6)}
    assert c["pool_cve_ids"] == 45 and c["target"] == 14 and c["ac_h_cves"] == 5 and c["ac_h_drawn"] == 3
    assert c["ac_h_share"] == 0.5 and c["ac_l_drawn"] == 11 and c["ac_h_cves_with_an_ac_l_unit"] == 1
    assert len(held) == len(set(held)) == 14 and len(set(held) & high) == 3
    assert held == sorted(held, key=pool.cve_key)


def test_p6d_holdout_draws_ac_h_and_ac_l_from_separate_seeds():
    """The AC:H half depends only on the AC:H seed; the AC:L draw only on its own seed (and the pool)."""
    high = {f"CVE-2021-{i}" for i in range(1, 6)}
    us = p6d_units()
    base = set(pool.cve_holdout(us, 0.3, "cve-holdout", 0.5, "cve-holdout-ach")[0])
    other_l = set(pool.cve_holdout(us, 0.3, "other", 0.5, "cve-holdout-ach")[0])
    assert base & high == other_l & high
    hs = {frozenset(set(pool.cve_holdout(us, 0.3, "cve-holdout", 0.5, f"a{i}")[0]) & high) for i in range(20)}
    assert len(hs) > 1
    ls = {frozenset(set(pool.cve_holdout(us, 0.3, "cve-holdout", 0.5, f"a{i}")[0]) - high) for i in range(20)}
    assert len(ls) == 1  # same AC:L seed, same AC:L-only IDs, same number drawn: same AC:L part
    assert set(pool.cve_holdout(list(reversed(us)), 0.3, "cve-holdout", 0.5, "cve-holdout-ach")[0]) == base


def test_p6d_holdout_argument_checks():
    with pytest.raises(ValueError, match="seed"):
        pool.cve_holdout(p6d_units(), 0.3, "cve-holdout", 0.5)
    with pytest.raises(ValueError, match="ach_share"):
        pool.cve_holdout(p6d_units(), 0.3, "cve-holdout", 1.5, "x")
    # ach_share = 1 is the P6c rule: same IDs and the same (stored) count keys as before
    assert pool.cve_holdout(p6d_units(), 0.3, "cve-holdout", 1.0, "ignored") == pool.cve_holdout(p6d_units(), 0.3, "cve-holdout")


def test_cve_set_hash_is_order_free_and_numeric():
    assert pool.cve_set_sha256(["CVE-2020-10", "CVE-2020-9"]) == pool.cve_set_sha256(("CVE-2020-9", "CVE-2020-10"))
    assert pool.cve_set_sha256(["CVE-2020-9"]) != pool.cve_set_sha256(["CVE-2020-10"])


def test_stored_cve_holdout_matches_its_hash_and_the_rule():
    """configs/gen/cve_holdout.json: the IDs hash to the stored SHA-256 and are exactly what the rule gives on
    the training cohort's pool (core, no CVE touching a held-out product)."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs" / "gen" / "cve_holdout.json").read_text())
    assert cfg["seed"] == "cve-holdout" and cfg["share"] == 0.3 and cfg["heldout_products"] == ["exim:exim"]
    assert pool.cve_set_sha256(cfg["cves"]) == cfg["sha256"] and cfg["cves"] == sorted(cfg["cves"], key=pool.cve_key)
    if not (root / "data" / "d28" / "master").exists():
        pytest.skip("pinned P6 inputs not in this checkout")
    from ahg.gen import generate as gen

    ctx = gen.Context.load()
    units = ctx.run_units(cfg["cohort"], frozenset(cfg["heldout_products"]))
    held, counts = pool.cve_holdout(units, cfg["share"], cfg["seed"])
    assert held == cfg["cves"] and counts == cfg["counts"]
    assert {u.cve_id for u in units if u.ac == "H"} <= set(held)


def _script():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_heldout_sets", root / "scripts" / "make_heldout_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_heldout_script_versions_keep_v1_and_name_v2():
    """§18 P6d item 1: v2 is a new config file; v1's parameters are those stored in configs/gen/cve_holdout.json."""
    import json

    m = _script()
    v1, v2 = m.VERSIONS["v1"], m.VERSIONS["v2"]
    assert v1["config"] == "configs/gen/cve_holdout.json" != v2["config"] == "configs/gen/cve_holdout_v2.json"
    old = json.loads((m.ROOT / v1["config"]).read_text())
    assert (old["share"], old["seed"], old["decision"]) == (v1["share"], v1["seed"], v1["decision"]) and v1["ach_share"] == 1
    assert (v2["share"], v2["seed"], v2["ach_share"], v2["ach_seed"]) == (0.3, "cve-holdout", 0.5, "cve-holdout-ach")


def test_heldout_script_refuses_to_overwrite_a_stored_holdout(monkeypatch):
    import sys

    m = _script()
    monkeypatch.setattr(sys, "argv", ["make_heldout_sets.py", "--version", "v1"])
    with pytest.raises(SystemExit, match="exists"):
        m.main()


def test_lost_groups_and_ac_h_shares():
    m = _script()
    us = [Unit("CVE-2020-1", "a:x", "mail", "linux", "remote", "H", "core"),
          Unit("CVE-2020-2", "a:x", "mail", "linux", "local", "L", "core"),
          Unit("CVE-2020-3", "b:y", "http", "linux", "remote", "L", "core"),
          Unit("CVE-2020-4", "b:y", "http", "windows", "remote", "L", "core"),
          Unit("CVE-2020-3", "c:z", "http", "windows", "remote", "L", "core")]
    held = frozenset({"CVE-2020-1", "CVE-2020-2", "CVE-2020-4"})
    g = m.lost_groups(us, held)
    assert g["os_product_lost"] == ["linux|a:x", "windows|b:y"]  # windows|b:y has only CVE-2020-4
    assert g["os_family_lost"] == ["linux|mail"] and g["family_lost"] == ["mail"]
    assert g["os_family"]["windows|http"] == {"cve_ids": 2, "held_out": 1}  # CVE-2020-3 stays in training
    mix = {k: m.ac_mix([u for u in us if f(u)]) for k, f in
           (("pool", lambda u: True), ("heldout", lambda u: u.cve_id in held), ("training", lambda u: u.cve_id not in held))}
    sh = m.ac_h_shares(mix["heldout"], mix["pool"])
    assert sh["share_of_pool_ac_h_cves"] == 1.0 and sh["ac_h_share_of_side_cves"] == 1 / 3
    assert m.ac_h_shares(mix["training"], mix["pool"])["share_of_pool_ac_h_units"] == 0.0
