"""Versioned golden digests of D28 generator documents (D28 Round 8 C3; P6).

    uv run python tests/golden_d28.py --version v1 --change "first D28 generator goldens (P6)"

writes tests/data/golden_d28_gen_<version>.json: the dataset version (input hashes), rule version (schema),
the generator config, the seed namespace, each case and the SHA-256 of its canonical document. An existing
file is never overwritten: a regeneration is a new version with its own change record (the old D10/D15
goldens and earlier D28 versions stay as they are). The digest leaves out metadata.inputs, which the file
records once in its header and the test compares separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
NAMESPACE = "golden"
CASES = ([(a, n, c, p, s) for a in ("tree", "chain") for n in (6, 12) for c in ("core",) for p in (1.0, 0.5)
          for s in range(3)]
         + [(a, 6, "temporal", 1.0, s) for a in ("tree", "chain") for s in range(3)])


def key(arch, n, cohort, p, s) -> str:
    return f"{arch}|n={n}|{cohort}|p={p}|{NAMESPACE}-{s}"


def case_digest(ctx, arch, n, cohort, p, s) -> str:
    from ahg.gen.generate import dumps, generate

    doc = generate(ctx, arch, f"{NAMESPACE}-{s}", n, cohort=cohort, os_mix={"linux": 0.5, "windows": 0.5}, p=p)
    doc["metadata"].pop("inputs")
    return hashlib.sha256(dumps(doc).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--change", required=True, help="change record: why this version exists")
    a = ap.parse_args()
    out = DATA / f"golden_d28_gen_{a.version}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; goldens are never overwritten, write a new version")
    from ahg import runlog
    from ahg.gen.generate import Context

    ctx = Context.load()
    body = {"version": a.version, "change_record": a.change, "git_commit": runlog._git("rev-parse", "HEAD"),
            "git_dirty": runlog.git_dirty(), "schema": ctx.cfg["schema"], "inputs": ctx.input_hashes,
            "namespace": NAMESPACE, "digests": {key(*c): case_digest(ctx, *c) for c in CASES}}
    out.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    print(out, len(body["digests"]))


if __name__ == "__main__":
    main()
