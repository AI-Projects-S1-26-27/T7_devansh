"""Host configurations: OS, roles' service families, products and fixtures (P6, decided 16 Nov 2026).

Draw order for a fresh configuration (one rng stream, consumed in this fixed order):
1. OS from the OS mix (Linux probability first).
2. Windows: the Server version, uniform among available versions.
3. The OS component's status: vulnerable at p = 1, else vulnerable with probability p; then its fixture
   (kernel release or Windows build), uniform among the fixtures of that status.
4. The families: k uniform in [1, 3], then min(k, m) families uniformly without replacement from the role's
   families intersected with the OS-compatible ones (m of them; sorted before drawing).
5. Per family, in sorted order: the product, uniform among the available products of that OS and family;
   then its status (as in 3) and fixture. A Windows built-in service has no draw of its own: it is a
   component of the OS and shares its build and status (OQ-P6-4).

Configuration reuse (Speicher et al.'s Dirichlet process, alpha = 2): within a role, the k-th host
(k = 0, 1, ...) copies the full configuration of an earlier host of that role, chosen uniformly, with
probability k / (k + 2); otherwise it draws a fresh one. Uniform choice among earlier hosts is the Chinese
restaurant process: a configuration is copied in proportion to how many hosts already use it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ahg.gen.inventory import Fixture, Inventory, Product


class InfeasibleCell(ValueError):
    """The inventory cannot realise this (cohort, p, OS mix, exclusions) cell: e.g. no Windows version."""


@dataclass(frozen=True)
class Component:
    """One installed component of a host: the OS ("os") or a service of one family."""

    family: str  # "os" or a service family
    product: str
    fixture: Fixture
    status: str  # "vulnerable" | "unaffected"
    builtin: bool = False


@dataclass(frozen=True)
class HostConfig:
    os: str
    components: tuple[Component, ...]  # OS first, then services by family
    copied_from: int | None = None  # host id whose configuration was copied (reuse), else None

    @property
    def families(self) -> list[str]:
        return [c.family for c in self.components if c.family != "os"]

    def cpe_names(self) -> list[str]:
        """Every CPE name on the host (for co-conditions), deduplicated, in component order."""
        out: list[str] = []
        for c in self.components:
            for n in c.fixture.cpe_names:
                if n not in out:
                    out.append(n)
        return out


def _status(rng: random.Random, p: float) -> str:
    return "vulnerable" if p == 1.0 or rng.random() < p else "unaffected"


def _fixture(rng: random.Random, inv: Inventory, prod: Product, status: str) -> Fixture:
    pool = [f for f in prod.fixtures if inv.status[(prod.name, f.version)] == status
            and (inv.p == 1.0 or not f.unverified)]  # unverified releases only at p = 1
    if not pool:
        raise InfeasibleCell(f"{prod.name}: no {status} fixture in this cell")
    return rng.choice(pool)


def compatible(inv: Inventory, os: str, version: Product | None) -> dict[str, list[Product]]:
    """Available products per family that can run on this OS (and Windows version)."""
    out = {}
    for (o, fam), prods in inv.products.items():
        if o != os:
            continue
        ok = [q for q in prods if not q.builtin or (version is not None and q.name in version.fixtures[0].builtins)]
        if ok:
            out[fam] = ok
    return out


def fresh(rng: random.Random, inv: Inventory, role_families: list[str], os_mix: dict[str, float],
          k_range: tuple[int, int]) -> HostConfig:
    """Draw one new configuration (module docstring, steps 1-5)."""
    os = "linux" if rng.random() < os_mix["linux"] else "windows"
    if os == "windows":
        if not inv.windows_versions:
            raise InfeasibleCell("no Windows Server version available in this cell")
        version = rng.choice(inv.windows_versions)
    else:
        if inv.kernel is None:
            raise InfeasibleCell("no kernel fixture available in this cell")
        version = inv.kernel
    st = _status(rng, inv.p)
    os_fix = _fixture(rng, inv, version, st)
    comps = [Component("os", version.name, os_fix, st)]
    compat = compatible(inv, os, version if os == "windows" else None)
    cands = sorted(set(role_families) & set(compat))
    k = rng.randint(*k_range)
    chosen = sorted(rng.sample(cands, min(k, len(cands))))
    for fam in chosen:
        prod = rng.choice(compat[fam])
        if prod.builtin:
            comps.append(Component(fam, prod.name, os_fix, st, builtin=True))
            continue
        s = _status(rng, inv.p)
        comps.append(Component(fam, prod.name, _fixture(rng, inv, prod, s), s))
    return HostConfig(os, tuple(comps))


def configure(rng: random.Random, inv: Inventory, roles: list[tuple[int, str]], role_families: dict[str, list[str]],
              os_mix: dict[str, float], k_range: tuple[int, int], alpha: float) -> dict[int, HostConfig]:
    """Configurations for (host id, role) pairs, in the given order, with configuration reuse per role."""
    out: dict[int, HostConfig] = {}
    seen: dict[str, list[int]] = {}
    for host, role in roles:
        earlier = seen.setdefault(role, [])
        k = len(earlier)
        if k and rng.random() < k / (k + alpha):
            src = rng.choice(earlier)
            cfg = out[src]
            out[host] = HostConfig(cfg.os, cfg.components, copied_from=src)
        else:
            out[host] = fresh(rng, inv, role_families[role], os_mix, k_range)
        earlier.append(host)
    return out
