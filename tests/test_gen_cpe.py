"""P6: the two CPE evaluators (ahg.gen.cpe, ahg.gen.cpe_check) on hand-written and fuzzed configurations."""

import ast
import random
from pathlib import Path

import pytest

from ahg.fixtures import version_key
from ahg.gen import cpe, cpe_check

SRC = Path(__file__).resolve().parents[1] / "src" / "ahg" / "gen"


def m(criteria, vulnerable=True, **bounds):
    return {"criteria": criteria, "vulnerable": vulnerable, **bounds}


def cfg(*nodes, operator=None):
    out = {"nodes": list(nodes)}
    if operator:
        out["operator"] = operator
    return out


def node(*matches, operator="OR", negate=False):
    return {"cpeMatch": list(matches), "operator": operator, "negate": negate}


def both(configs, host, comp):
    """Configuration indices from both evaluators; the test fails if they differ."""
    a = [e["configuration"] for e in cpe.component_matches(configs, host, comp)]
    b = cpe_check.applies(configs, host, comp)
    assert a == b, (a, b)
    return a


HTTPD = "cpe:2.3:a:apache:http_server:2.4.49:-:*:*:*:*:*:*"
KERNEL = "cpe:2.3:o:linux:linux_kernel:5.10.1:-:*:*:*:*:*:*"
WIN19 = "cpe:2.3:o:microsoft:windows_server_2019:10.0.17763.3046:-:*:*:*:*:x64:*"
WIN12R2 = "cpe:2.3:o:microsoft:windows_server_2012:r2:-:*:*:*:*:x64:*"


def test_exact_version_and_any_value():
    c = [cfg(node(m("cpe:2.3:a:apache:http_server:2.4.49:*:*:*:*:*:*:*")))]
    assert both(c, [HTTPD], [HTTPD]) == [0]
    assert both(c, ["cpe:2.3:a:apache:http_server:2.4.50:-:*:*:*:*:*:*"],
                ["cpe:2.3:a:apache:http_server:2.4.50:-:*:*:*:*:*:*"]) == []


def test_na_version_matches_only_na():
    """Strict CPE 2.3: criteria '-' (NA) does not match a build-numbered host (OQ-P6-6)."""
    c = [cfg(node(m("cpe:2.3:o:microsoft:windows_server_2019:-:*:*:*:*:*:*:*")))]
    assert both(c, [WIN19], [WIN19]) == []


def test_specific_criteria_value_needs_equal_host_value():
    c = [cfg(node(m("cpe:2.3:o:microsoft:windows_server_2019:*:*:*:*:*:*:x86:*", versionEndExcluding="10.0.17763.9999")))]
    assert both(c, [WIN19], [WIN19]) == []  # x86 criteria vs x64 host
    c = [cfg(node(m("cpe:2.3:a:apache:http_server:*:*:*:*:*:*:x64:*", versionEndExcluding="2.4.50")))]
    assert both(c, [HTTPD], [HTTPD]) == []  # host target_hw '*' is not x64


@pytest.mark.parametrize("bounds,inside", [
    ({"versionStartIncluding": "10.0.17763.3046"}, True),
    ({"versionStartExcluding": "10.0.17763.3046"}, False),
    ({"versionEndIncluding": "10.0.17763.3046"}, True),
    ({"versionEndExcluding": "10.0.17763.3046"}, False),
    ({"versionEndExcluding": "10.0.17763.3165"}, True),
    ({"versionStartIncluding": "10.0.17763.3047", "versionEndExcluding": "10.0.17763.4000"}, False),
])
def test_range_bounds(bounds, inside):
    c = [cfg(node(m("cpe:2.3:o:microsoft:windows_server_2019:*:*:*:*:*:*:*:*", **bounds)))]
    assert both(c, [WIN19], [WIN19]) == ([0] if inside else [])


def test_non_numeric_version_is_never_inside_a_range():
    """Server 2012 R2 carries version 'r2'; a numeric range on windows_server_2012 never contains it."""
    c = [cfg(node(m("cpe:2.3:o:microsoft:windows_server_2012:*:*:*:*:*:*:*:*", versionEndExcluding="6.2.9200.24710")))]
    assert both(c, [WIN12R2], [WIN12R2]) == []
    c = [cfg(node(m("cpe:2.3:o:microsoft:windows_server_2012:r2:*:*:*:*:*:*:*")))]
    assert both(c, [WIN12R2], [WIN12R2]) == [0]


def test_platform_co_condition_needs_the_platform_and_never_marks_it():
    ubuntu = "cpe:2.3:o:canonical:ubuntu_linux:16.04:-:*:*:*:*:*:*"
    c = [cfg(node(m("cpe:2.3:a:apache:http_server:*:*:*:*:*:*:*:*", versionEndIncluding="2.4.49")),
             node(m("cpe:2.3:o:canonical:ubuntu_linux:16.04:*:*:*:*:*:*:*", vulnerable=False)), operator="AND")]
    assert both(c, [HTTPD], [HTTPD]) == []  # no Ubuntu on the host
    assert both(c, [HTTPD, ubuntu], [HTTPD]) == [0]
    assert both(c, [HTTPD, ubuntu], [ubuntu]) == []  # a platform match never creates a vulnerability
    assert cpe.config_true(c[0], [HTTPD, ubuntu]) and not cpe.config_true(c[0], [HTTPD])


def test_negated_node_and_negated_match_are_not_vulnerable():
    c = [cfg(node(m("cpe:2.3:a:apache:http_server:*:*:*:*:*:*:*:*", versionEndIncluding="2.4.49")),
             node(m(KERNEL.replace("5.10.1:-", "5.10.1:*"), vulnerable=True), negate=True), operator="AND")]
    assert both(c, [HTTPD], [HTTPD]) == [0]  # kernel absent: the negated node is true
    assert both(c, [HTTPD, KERNEL], [HTTPD]) == []
    assert both(c, [HTTPD], [KERNEL]) == []  # the negated match never marks the kernel vulnerable


def test_any_of_several_configurations():
    c = [cfg(node(m("cpe:2.3:a:apache:http_server:2.4.48:*:*:*:*:*:*:*"))),
         cfg(node(m("cpe:2.3:a:apache:http_server:2.4.49:*:*:*:*:*:*:*")))]
    assert both(c, [HTTPD], [HTTPD]) == [1]
    assert both([], [HTTPD], [HTTPD]) == []  # no configuration: never applies (CPE-only)


VERSION_PAIRS = ["2.4.49", "2.4.50", "2.4.5", "5.10", "5.10.0", "5.10.0.0", "11.0.0-M1", "11.0.0", "11.0.0-rc1",
                 "9.6p1", "9.6", "1.3.8b", "1.3.8", "9.11.3-P1", "9.11.3", "10.0.17763.3046", "10.0.17763.999",
                 "0", "0.0", "1.0beta2", "1.0alpha", "1.0dev", "4.89", "4.9", "15.02.1544.044", "15.02.1544.46",
                 "6.3.9600.19968", "r2", "1.003", "3.2.85", "4.19-rc3", "4.19"]


@pytest.mark.parametrize("a", VERSION_PAIRS)
def test_version_orders_agree(a):
    for b in VERSION_PAIRS:
        ka, kb = version_key(a), version_key(b)
        want = (ka > kb) - (ka < kb)
        assert cpe_check.compare_versions(a, b) == want, (a, b)


def test_evaluators_agree_on_fuzzed_configurations():
    """Seeded random configurations over a small vocabulary: both evaluators give the same indices."""
    rng = random.Random(20261116)
    prods = [("a", "apache", "http_server"), ("o", "linux", "linux_kernel"), ("o", "canonical", "ubuntu_linux"),
             ("a", "samba", "samba")]
    versions = ["1.0", "1.2", "2.0", "2.0.1", "2.1-rc1", "3", "r2", "-"]

    def rand_match():
        part, v, p = rng.choice(prods)
        if rng.random() < 0.5:
            b = {}
            for k in cpe.BOUND_KEYS:
                if rng.random() < 0.35:
                    b[k] = rng.choice(versions[:6])
            return m(f"cpe:2.3:{part}:{v}:{p}:*:*:*:*:*:*:{rng.choice(['*', 'x64'])}:*", rng.random() < 0.7, **b)
        return m(f"cpe:2.3:{part}:{v}:{p}:{rng.choice(versions + ['*'])}:*:*:*:*:*:*:*", rng.random() < 0.7)

    def rand_node(depth=0):
        nd = node(*[rand_match() for _ in range(rng.randint(0, 3))], operator=rng.choice(["OR", "AND"]),
                  negate=rng.random() < 0.2)
        if depth < 2 and rng.random() < 0.3:
            nd["children"] = [rand_node(depth + 1) for _ in range(rng.randint(1, 2))]
        return nd

    agree = 0
    for _ in range(1500):
        configs = [cfg(*[rand_node() for _ in range(rng.randint(1, 3))], operator=rng.choice([None, "AND", "OR"]))
                   for _ in range(rng.randint(1, 3))]
        host = [f"cpe:2.3:{pt}:{v}:{p}:{rng.choice(versions[:7])}:-:*:*:*:*:{rng.choice(['*', 'x64'])}:*"
                for pt, v, p in rng.sample(prods, rng.randint(1, 3))]
        comp = [rng.choice(host)]
        agree += bool(both(configs, host, comp))
    assert agree > 50  # the fuzz reaches positive matches, not only empty ones


def test_second_evaluator_imports_nothing_from_ahg():
    tree = ast.parse((SRC / "cpe_check.py").read_text())
    mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(x.startswith("ahg") or x.startswith(".") for x in mods), mods
