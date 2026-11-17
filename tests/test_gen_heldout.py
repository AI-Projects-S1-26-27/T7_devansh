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


def test_cve_set_hash_is_order_free_and_numeric():
    assert pool.cve_set_sha256(["CVE-2020-10", "CVE-2020-9"]) == pool.cve_set_sha256(("CVE-2020-9", "CVE-2020-10"))
    assert pool.cve_set_sha256(["CVE-2020-9"]) != pool.cve_set_sha256(["CVE-2020-10"])
