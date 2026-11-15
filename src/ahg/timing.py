"""Wall-clock measurement and the summaries built on it (D28 Round 12, AUDIT_4 timing fix, inventory #5).

Rules:
- Every duration is time.perf_counter_ns(), an integer, measured over repeated runs.
- A repeat shorter than the timer's resolution carries no information. Such a
  measurement is marked censored and kept as recorded; it is never replaced by
  some small value.
- A duration <= 0 is impossible for a monotonic clock around real work, so it is
  invalid and raises at once.
- Log ratios and geometric means raise InvalidMeasurement on any value that is
  not a finite positive number (zero, negative, NaN, inf, None, censored); a
  caller that wants to leave censored cases out must do so explicitly and
  report how many it left out.

The per-measurement summary of the repeats is their median (robust to one-off
scheduler stalls); the raw repeats are always kept in the record.
"""

import math
import statistics
import time
from dataclasses import dataclass

RESOLUTION_NS = max(1, math.ceil(time.get_clock_info("perf_counter").resolution * 1e9))


class InvalidMeasurement(ValueError):
    pass


@dataclass(frozen=True)
class Timing:
    ns: tuple[int, ...]  # one duration per repeat, in measurement order
    resolution_ns: int = RESOLUTION_NS

    def __post_init__(self):
        if not self.ns:
            raise InvalidMeasurement("a timing needs at least one repeat")
        bad = [t for t in self.ns if not isinstance(t, int) or t <= 0]
        if bad:
            raise InvalidMeasurement(f"non-positive or non-integer duration(s) {bad} ns")

    @property
    def censored(self) -> bool:
        """True if any repeat is below the timer resolution (D28: marked, never replaced)."""
        return any(t < self.resolution_ns for t in self.ns)

    @property
    def median_ns(self) -> float:
        if self.censored:
            raise InvalidMeasurement(f"censored timing {self.ns} ns (resolution {self.resolution_ns} ns)")
        return statistics.median(self.ns)

    def record(self) -> dict:
        return {"ns": list(self.ns), "resolution_ns": self.resolution_ns, "censored": self.censored}

    @classmethod
    def from_record(cls, rec: dict) -> "Timing":
        return cls(tuple(rec["ns"]), rec["resolution_ns"])


def timed(fn, repeats: int):
    """Run fn() `repeats` times; return (the list of results, Timing). Each repeat is timed on its own."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    results, ns = [], []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        results.append(fn())
        ns.append(time.perf_counter_ns() - t0)
    return results, Timing(tuple(ns))


def _positive(x, what: str) -> float:
    if isinstance(x, Timing):
        x = x.median_ns  # raises if censored
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise InvalidMeasurement(f"{what}: {x!r} is not a finite positive number")
    return float(x)


def log_ratio(num, den, what: str = "ratio") -> float:
    """log(num / den) for positive numbers or uncensored Timings; raises InvalidMeasurement otherwise."""
    return math.log(_positive(num, f"{what} numerator")) - math.log(_positive(den, f"{what} denominator"))


def geometric_mean(values, what: str = "geometric mean") -> float:
    values = list(values)
    if not values:
        raise InvalidMeasurement(f"{what}: no values")
    return math.exp(sum(math.log(_positive(v, what)) for v in values) / len(values))
