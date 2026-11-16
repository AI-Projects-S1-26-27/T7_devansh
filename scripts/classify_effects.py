"""P3 effect classifier driver (D28 Rounds 1-2). Subagents do the labelling; this script does the rest.

Subcommands (all work inside one run folder results/d28_effect/<stamp>-<commit>/, append-only):
  init CONFIG                      pool + OS sample + manifest; per family x OS x cohort pool counts
  pilot RUN                        select the 40-CVE pilot and write batch files per annotator slot and model
  ingest RUN STAGE                 validate subagent outputs of one stage, checkpoint the valid ones
  usage RUN STAGE TOKENS [--calls N]  record the tokens a stage's subagent calls used (as the harness reported)
  adjudicate-prep RUN STAGE_A STAGE_B NAME   batches of disputed fields for the adjudicator
  adjudicate-ingest RUN NAME       validate rulings
  pilot-report RUN                 kappa, disagreements, tokens per CVE, model-choice rule; writes report.md

A stage is "<slot>-<model>", slot A or B, e.g. A-haiku. Checkpoints are files, so every step resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ahg import effect as ef  # noqa: E402
from ahg import effect_rules as er  # noqa: E402
from ahg import runlog  # noqa: E402

OUT = ROOT / "results" / "d28_effect"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jload(p: Path):
    return json.loads(p.read_text())


def jdump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + "\n")


def jdump_compact(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, separators=(",", ":")) + "\n")


def work(run: Path) -> Path:
    return run / "work"


def ghsa_by_id(folder: Path) -> dict[str, str]:
    """GHSA id -> advisory text, from the archived pages (hash-checked against the fetch log)."""
    from ahg import cvedata as cd

    log = {json.loads(l)["name"]: json.loads(l) for l in (folder / "fetch_log.jsonl").read_text().splitlines() if l}
    out = {}
    for name, rec in log.items():
        p = folder / name
        if rec.get("http") == 200 and p.exists() and sha(p) == rec["sha256"]:
            out[name.split("__")[0]] = cd.ghsa_text(p.read_text(errors="replace"))
    return out


# ---------------------------------------------------------------- init

def pool_table(items: list[dict]) -> dict[str, Counter]:
    """CVE counts per label set x family x OS x cohort (a CVE counts once per unit group it appears in)."""
    t: dict[str, Counter] = {"remote": Counter(), "local": Counter()}
    for i in items:
        seen = {(u["family"], u["os"], u["cohort"]) for u in i["units"]}
        for k in seen:
            t[i["label_set"]][k] += 1
    return t


def cmd_init(a) -> None:
    cfg = jload(a.config)
    master = (ROOT / cfg["master_dataset"]).resolve()
    rows = ef.load_master(master)
    full, os_local = ef.pool_units(rows, cfg)
    sample = ef.sample_os(os_local, cfg)
    ghsa = ghsa_by_id((ROOT / cfg["ghsa_folder"]).resolve())
    items = ef.build_items(rows, full + sample, ghsa, cfg["faq_max_chars"])
    prov = runlog.provenance([cfg["os_sample"]["seed"], cfg["pilot"]["seed"]], cfg)
    stamp = prov["date_utc"].replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    run = OUT / f"{stamp}-{prov['git_commit'][:7]}"
    run.mkdir(parents=True, exist_ok=False)
    w = work(run)
    jdump(w / "pool.json", items)
    jdump(w / "sample.json", {"seed": cfg["os_sample"]["seed"], "per_group": cfg["os_sample"]["per_group"],
                              "n_units": len(sample)})
    verdicts = {i["cve_id"]: er.apply_rules(i) for i in items}
    jdump(w / "rules.json", verdicts)
    sel = ef.stage2_select(items, verdicts, cfg)
    jdump(w / "stage2_select.json", sel)
    fam = cfg["os_sample"]["family"]
    rule_counts = Counter(f"{ef.pool_of(i, fam)}|{i['label_set']}|{verdicts[i['cve_id']]['label']}" for i in items)
    sel_counts = Counter(r for rs in sel.values() for r in rs)
    tab = pool_table(items)
    manifest = {"master_dataset": str(master), "master_sha256": sha(master), "n_items": len(items),
                "n_units_full_pool": len(full), "n_units_os_local_candidates": len(os_local), "n_units_os_sample": len(sample),
                "rubric_sha256": sha(ROOT / cfg["rubric"]), "adjudicator_rubric_sha256": sha(ROOT / cfg["adjudicator_rubric"]),
                "pool_cves": {s: sum(1 for i in items if i["label_set"] == s) for s in ("remote", "local")},
                "rules_version": er.RULES_VERSION, "rule_counts": dict(sorted(rule_counts.items())),
                "stage2_n_cves": len(sel), "stage2_reasons": dict(sorted(sel_counts.items())),
                "models": {}, "pool_counts": {s: {"|".join(map(str, k)): n for k, n in sorted(c.items())} for s, c in tab.items()}}
    jdump(run / "manifest.json", manifest)
    jdump(run / "provenance.json", prov)
    print(run)
    print(json.dumps({k: manifest[k] for k in manifest if k not in ("pool_counts", "models")}))


# ---------------------------------------------------------------- pilot / batches

def write_batches(run: Path, stage_dir: Path, items: list[dict], size: int, rubric: str, reverse: bool) -> list[Path]:
    seq = items[::-1] if reverse else items  # slot B sees a different batching, so batch context differs
    paths = []
    for n, chunk in enumerate(ef.batches(seq, size)):
        p = stage_dir / f"batch_{n:02d}.json"
        jdump_compact(p, {"rubric": rubric, "items": [ef.annotator_view(i) for i in chunk]})
        paths.append(p)
    return paths


def cmd_stage2_batches(a) -> None:
    """Batches for stage 2: slots A and B over the selected CVEs; a batch is only written if it has no checkpoint yet."""
    run = Path(a.run)
    cfg = jload(ROOT / "configs/effect/p3.json")
    idx = {i["cve_id"]: i for i in jload(work(run) / "pool.json")}
    sel = sorted(jload(work(run) / "stage2_select.json"))
    items = [idx[c] for c in sel]
    rubric = (ROOT / cfg["rubric"]).read_text()
    for slot in ("A", "B"):
        d = work(run) / "stage2" / slot
        ps = write_batches(run, d, items, a.size or cfg["batch_size"], rubric, reverse=slot == "B")
        print(d.name, len(ps), "batches", len(items), "CVEs")


def cmd_audit_unres_prep(a) -> None:
    """OQ-P3-5: seeded audit of OS rule-unresolved CVEs; writes select.json and blinded A/B batches under work/audit_os_unres."""
    run = Path(a.run)
    cfg = jload(ROOT / "configs/effect/p3.json")
    items = jload(work(run) / "pool.json")
    rules = jload(work(run) / "rules.json")
    used = set(jload(work(run) / "stage2_select.json"))
    pilot = work(run) / "pilot" / "items.json"
    used |= {i["cve_id"] for i in jload(pilot)} if pilot.exists() else set()
    ids = ef.audit_unresolved_sample(items, rules, used, cfg)
    d = work(run) / "audit_os_unres"
    jdump(d / "select.json", {"seed": cfg["audit_unresolved"]["seed"], "cves": ids})
    idx = {i["cve_id"]: i for i in items}
    rubric = (ROOT / cfg["rubric"]).read_text()
    for slot in ("A", "B"):
        ps = write_batches(run, d / slot, [idx[c] for c in ids], cfg["audit_unresolved"]["batch_size"], rubric, reverse=slot == "B")
        print(slot, len(ps), "batches", len(ids), "CVEs")


def cmd_pilot(a) -> None:
    run = Path(a.run)
    cfg = jload(ROOT / "configs/effect/p3.json")
    items = jload(work(run) / "pool.json")
    pilot = ef.pilot_select(items, cfg)
    jdump(work(run) / "pilot" / "items.json", pilot)
    rubric = (ROOT / cfg["rubric"]).read_text()
    for slot in a.slots:
        for model in a.models:
            d = work(run) / "pilot" / f"{slot}-{model}"
            ps = write_batches(run, d, pilot, a.size or cfg["batch_size"], rubric, reverse=slot == "B")
            print(d.name, len(ps), "batches")
    strata = Counter((i["label_set"], any(u["role_question"] for u in i["units"])) for i in pilot)
    print(len(pilot), "pilot CVEs;", dict(strata))


# ---------------------------------------------------------------- ingest

def item_index(run: Path, scope: str) -> dict[str, dict]:
    src = work(run) / ("pilot/items.json" if scope == "pilot" else "pool.json")
    return {i["cve_id"]: i for i in jload(src)}


def cmd_ingest(a) -> None:
    """Validate outputs per CVE: valid annotations are checkpointed, invalid ones listed for `repair` (no whole-batch reruns)."""
    run = Path(a.run)
    d = work(run) / a.scope / a.stage
    idx = item_index(run, a.scope)
    ok_b = missing = n_ok = n_bad = 0
    for b in sorted(d.glob("batch_*.json")):
        n = b.stem.split("_")[1]
        out = d / f"out_{n}.json"
        if not out.exists():
            missing += 1
            continue
        ids = [i["cve_id"] for i in jload(b)["items"]]
        try:
            anns = {x.get("cve_id"): x for x in jload(out)["annotations"]}
        except (ValueError, KeyError, TypeError) as e:
            print(f"{out.name}: unreadable ({e})")
            continue
        good, errs = [], {}
        for cid in ids:
            x = anns.get(cid)
            e = ["no annotation"] if x is None else ef.check_annotation(x, idx[cid])
            if e:
                errs[cid] = e
            else:
                good.append(x)
        jdump(d / f"ann_{n}.json", good)
        jdump(d / f"errors_{n}.json", errs)
        ok_b += not errs
        n_ok, n_bad = n_ok + len(good), n_bad + len(errs)
        if errs:
            first = next(iter(errs))
            print(f"{out.name}: {len(errs)} invalid, e.g. {first}: {errs[first][0]}")
    print(f"{a.stage}: batches complete {ok_b}, missing outputs {missing}; CVEs valid {n_ok}, invalid {n_bad}")


def cmd_repair(a) -> None:
    """New batches (batch_rNN) for CVEs of this stage that have an output but no valid annotation yet."""
    run = Path(a.run)
    cfg = jload(ROOT / "configs/effect/p3.json")
    d = work(run) / a.scope / a.stage
    idx = item_index(run, a.scope)
    have = set(annotations(run, a.scope, a.stage))
    want = []
    for b in sorted(d.glob("batch_[0-9]*.json")) + sorted(d.glob("batch_r*.json")):
        n = b.stem.split("_")[1]
        if (d / f"out_{n}.json").exists():
            want += [i["cve_id"] for i in jload(b)["items"] if i["cve_id"] not in have]
    want = sorted(set(want))
    k = len(list(d.glob("batch_r*.json")))
    rubric = (ROOT / cfg["rubric"]).read_text()
    for m, chunk in enumerate(ef.batches([idx[c] for c in want], a.size or cfg["batch_size"])):
        jdump_compact(d / f"batch_r{k + m:02d}.json", {"rubric": rubric, "items": [ef.annotator_view(i) for i in chunk]})
    print(f"{a.stage}: {len(want)} CVEs to repair in {len(list(ef.batches(want, a.size or cfg['batch_size'])))} new batches")


def annotations(run: Path, scope: str, stage: str) -> dict[str, dict]:
    out = {}
    for p in sorted((work(run) / scope / stage).glob("ann_*.json")):
        for x in jload(p):
            out[x["cve_id"]] = x
    return out


def cmd_usage(a) -> None:
    with (work(Path(a.run)) / "usage.jsonl").open("a") as f:
        f.write(json.dumps({"scope": a.scope, "stage": a.stage, "tokens": a.tokens, "calls": a.calls}) + "\n")


# ---------------------------------------------------------------- adjudication

def cmd_adjudicate_prep(a) -> None:
    run = Path(a.run)
    idx = item_index(run, a.scope)
    A, B = annotations(run, a.scope, a.stage_a), annotations(run, a.scope, a.stage_b)
    jobs = []
    for cid in sorted(set(A) & set(B)):
        ls = idx[cid]["label_set"]
        dis = ef.disputes(A[cid], B[cid], ls)
        if dis:
            jobs.append({"cve_id": cid, "label_set": ls, "texts": idx[cid]["texts"], "disputes": dis,
                         "A": {k: A[cid][k] for k in ("label", "target_privilege", "roles", "evidence")},
                         "B": {k: B[cid][k] for k in ("label", "target_privilege", "roles", "evidence")}})
    cfg = jload(ROOT / "configs/effect/p3.json")
    d = work(run) / a.scope / f"adj-{a.name}"
    for n, chunk in enumerate(ef.batches(jobs, cfg["batch_size"])):
        jdump_compact(d / f"batch_{n:02d}.json", {"rubric": (ROOT / cfg["adjudicator_rubric"]).read_text(), "items": chunk})
    print(f"{len(jobs)} CVEs with disputes of {len(set(A) & set(B))}; {len(list(d.glob('batch_*.json')))} batches in {d}")


def cmd_adjudicate_ingest(a) -> None:
    """Keep every valid ruling; invalid or missing ones leave their dispute unadjudicated (counted, then REVIEW_REQUIRED)."""
    run = Path(a.run)
    idx = item_index(run, a.scope)
    d = work(run) / a.scope / f"adj-{a.name}"
    n_ok = n_bad = n_missing = 0
    for b in sorted(d.glob("batch_*.json")):
        n = b.stem.split("_")[1]
        out = d / f"out_{n}.json"
        want = {(i["cve_id"], f) for i in jload(b)["items"] for f in i["disputes"]}
        if not out.exists():
            n_missing += len(want)
            continue
        try:
            rs = jload(out)["rulings"]
        except (ValueError, KeyError, TypeError):
            n_missing += len(want)
            continue
        good, bad = [], []
        for r in rs:
            key = (r.get("cve_id"), r.get("field"))
            e = ["not a dispute of this batch"] if key not in want else ef.check_ruling(r, idx[r["cve_id"]])
            (bad if e else good).append((r, e))
        jdump(d / f"ruling_{n}.json", [r for r, _ in good])
        jdump(d / f"ruling_errors_{n}.json", [{"cve_id": r.get("cve_id"), "field": r.get("field"), "errors": e} for r, e in bad])
        n_ok += len(good)
        n_bad += len(bad)
        n_missing += len(want - {(r["cve_id"], r["field"]) for r, _ in good})
    print(f"adj-{a.name}: rulings valid {n_ok}, invalid {n_bad}, disputes without a valid ruling {n_missing}")


# ---------------------------------------------------------------- pilot report

def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def cmd_pilot_report(a) -> None:
    run = Path(a.run)
    cfg = jload(ROOT / "configs/effect/p3.json")
    idx = item_index(run, "pilot")
    S = {s: annotations(run, "pilot", s) for s in ("A-haiku", "B-haiku", "A-sonnet", "B-sonnet")}
    comparisons = {"within haiku (A vs B)": ("A-haiku", "B-haiku"), "within sonnet (A vs B)": ("A-sonnet", "B-sonnet"),
                   "haiku vs sonnet, slot A": ("A-haiku", "A-sonnet"), "haiku vs sonnet, slot B": ("B-haiku", "B-sonnet")}
    lines = ["# P3 pilot report (40 CVEs): Haiku vs Sonnet annotators", "",
             "Pilot items are not gold labels. Kappa is over the pilot only; the role field has few items.", ""]
    agr, dis_rows = {}, []
    for name, (x, y) in comparisons.items():
        common = sorted(set(S[x]) & set(S[y]))
        agr[name] = ef.agreement([(S[x][c], S[y][c], idx[c]["label_set"]) for c in common])
        for c in common:
            for f in ef.disputes(S[x][c], S[y][c], idx[c]["label_set"]):
                fx, fy = ef.fields_of(S[x][c], idx[c]["label_set"]), ef.fields_of(S[y][c], idx[c]["label_set"])
                dis_rows.append((name, c, f, fx[f], fy.get(f), S[x][c].get("note", ""), S[y][c].get("note", "")))
        lines += [f"## {name} ({len(common)} CVEs)", "", "| field | n | agreement | kappa |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['n']} | {fmt(v['agree'])} | {fmt(v['kappa'])} |" for k, v in agr[name].items()] + [""]
    usage = defaultdict(lambda: [0, 0])
    up = work(run) / "usage.jsonl"
    for l in (up.read_text().splitlines() if up.exists() else []):
        r = json.loads(l)
        usage[r["stage"]][0] += r["tokens"]
        usage[r["stage"]][1] += r["calls"]
    lines += ["## Tokens per CVE (as reported by the harness for each subagent call)", "", "| stage | tokens | calls | tokens per CVE |", "|---|---|---|---|"]
    for s, (t, c) in sorted(usage.items()):
        n = len(S.get(s, {})) or 40
        lines.append(f"| {s} | {t} | {c} | {t / n:.0f} |")
    lines += ["", "## Disagreements (all comparisons)", "", "| comparison | CVE | field | first | second | first note | second note |", "|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in dis_rows]
    cross = [agr["haiku vs sonnet, slot A"], agr["haiku vs sonnet, slot B"]]
    kmin = cfg["model_rule"]["kappa_min"]
    verdict = {f: [c.get(f, {}).get("kappa") for c in cross] for f in cfg["model_rule"]["fields"]}
    meets = all(k is not None and k >= kmin for ks in verdict.values() for k in ks)
    lines += ["", "## Model-choice rule (Devansh, 15 Nov 2026)", "",
              f"Haiku for both annotators and Sonnet for the adjudicator only, if kappa(haiku, sonnet) >= {kmin} on label and role in both slots and every disagreement is explainable; else Sonnet for all.",
              "", f"Kappa condition on the numbers: {'MET' if meets else 'NOT MET'} ({json.dumps({k: [fmt(x) for x in v] for k, v in verdict.items()})}).",
              "Explainability of each disagreement is a judgement and is stated in the task report, not computed here."]
    (run / "pilot_report.md").write_text("\n".join(lines) + "\n")
    jdump(run / "pilot_agreement.json", agr)
    print(run / "pilot_report.md", "kappa condition", "MET" if meets else "NOT MET", {k: [fmt(x) for x in v] for k, v in verdict.items()})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval for a proportion; None for n = 0."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_rulings(run: Path, scope: str, name: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in sorted((work(run) / scope / f"adj-{name}").glob("ruling_*.json")):
        if "errors" in p.name:
            continue
        for r in jload(p):
            out[r["cve_id"]][r["field"]] = r
    return out


def cmd_stage2_report(a) -> None:
    """Final labels, rule precision and audit miss rate, counts per family x OS x cohort, kappa; writes stage2_report.md and final.json."""
    run = Path(a.run)
    cfg = jload(ROOT / "configs/effect/p3.json")
    fam = cfg["os_sample"]["family"]
    idx = {i["cve_id"]: i for i in jload(work(run) / "pool.json")}
    rules = jload(work(run) / "rules.json")
    sel = jload(work(run) / "stage2_select.json")
    A, B = annotations(run, "stage2", "A"), annotations(run, "stage2", "B")
    rul = load_rulings(run, "stage2", "s2")
    final, how_count = {}, Counter()
    for cid in sel:
        if cid not in A or cid not in B:
            continue
        m = ef.merge(A[cid], B[cid], idx[cid]["label_set"], rul.get(cid, {}))
        final[cid] = {"fields": m["fields"], "how": m["how"], "A": A[cid]["label"], "B": B[cid]["label"],
                      "mismatch": [A[cid]["text_product_mismatch"], B[cid]["text_product_mismatch"]]}
        how_count.update(m["how"].values())
    missing = sorted(set(sel) - set(final))
    agr = ef.agreement([(A[c], B[c], idx[c]["label_set"]) for c in final])
    pos = {"RCE_CONFIRMED", "ADMIN_CONFIRMED"}

    def stats(reason_pool: str | None, kind: str):
        ids = [c for c, r in sel.items() if c in final and (reason_pool in r if reason_pool else True)]
        return ids

    L = ["# P3 stage 2 report (rules " + er.RULES_VERSION + ", Sonnet annotators and adjudicator)", "",
         f"Stage-2 CVEs selected {len(sel)}, annotated by both slots {len(final)}, missing {len(missing)}.", ""]
    # rule precision
    L += ["## Rule precision (stage-2 final label equals the rule's positive label)", "", "| pool | label set | rule positives | confirmed | precision | 95% CI | A agrees | B agrees |", "|---|---|---|---|---|---|---|---|"]
    prec = {}
    for pool in ("svc", "os"):
        for ls in ("remote", "local"):
            ids = [c for c, r in sel.items() if "rule_positive" in r and c in final and ef.pool_of(idx[c], fam) == pool and idx[c]["label_set"] == ls]
            if not ids:
                continue
            k = sum(final[c]["fields"]["label"] == rules[c]["label"] for c in ids)
            ka = sum(final[c]["A"] == rules[c]["label"] for c in ids)
            kb = sum(final[c]["B"] == rules[c]["label"] for c in ids)
            ci = wilson(k, len(ids))
            prec[f"{pool}|{ls}"] = {"n": len(ids), "confirmed": k, "A": ka, "B": kb, "ci": ci}
            L.append(f"| {pool} | {ls} | {len(ids)} | {k} | {k/len(ids):.3f} | {ci[0]:.3f}-{ci[1]:.3f} | {ka/len(ids):.3f} | {kb/len(ids):.3f} |")
    # audit miss rate
    L += ["", "## Audit of rule-negatives (rules missed a positive = stage-2 final label is RCE_CONFIRMED / ADMIN_CONFIRMED)", "",
          "| pool | audited | stage-2 positive (missed) | miss rate | 95% CI | rule negatives in pool | projected missed positives |", "|---|---|---|---|---|---|---|"]
    audit = {}
    for pool in ("svc", "os"):
        ids = [c for c, r in sel.items() if r == [f"audit_{pool}"] and c in final]
        k = sum(final[c]["fields"]["label"] in pos for c in ids)
        negs = sum(1 for c, v in rules.items() if v["label"] in ("NON_RCE", "NON_ADMIN") and ef.pool_of(idx[c], fam) == pool)
        ci = wilson(k, len(ids))
        audit[pool] = {"n": len(ids), "missed": k, "negatives": negs, "ci": ci}
        if ids:
            L.append(f"| {pool} | {len(ids)} | {k} | {k/len(ids):.3f} | {ci[0]:.3f}-{ci[1]:.3f} | {negs} | {k/len(ids)*negs:.0f} (range {ci[0]*negs:.0f}-{ci[1]*negs:.0f}) |")
    L += ["", "The audit covers only rule-negatives (NON_*). OS CVEs the rules leave unresolved are not annotated, so rule recall over the OS unresolved remainder is not measured.", ""]
    # agreement
    L += ["## Annotator agreement A vs B (all stage-2 CVEs)", "", "| field | n | agreement | kappa |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {fmt(v['agree'])} | {fmt(v['kappa'])} |" for k, v in agr.items()]
    L += ["", f"Field outcomes: {dict(how_count)}", ""]
    # counts
    L += ["## Final label counts per label set x family x OS x cohort", "",
          "Source: stage2 = stage-2 final label (agreed or adjudicated); rule = rule NON_* label never sent to stage 2; unannotated = rules left it unresolved and it was not sent (OS only).", "",
          "| label set | family | OS | cohort | source | label | CVEs |", "|---|---|---|---|---|---|---|"]
    cnt: Counter = Counter()
    for cid, it in idx.items():
        if cid in final:
            src, lab = "stage2", final[cid]["fields"]["label"]
        elif cid in sel:
            src, lab = "stage2", "MISSING"
        elif rules[cid]["label"] != "unresolved":
            src, lab = "rule", rules[cid]["label"]
        else:
            src, lab = "unannotated", "unresolved"
        for g in {(u["family"], u["os"], u["cohort"]) for u in it["units"]}:
            cnt[(it["label_set"], *g, src, lab)] += 1
    for k, n in sorted(cnt.items()):
        L.append("| " + " | ".join(map(str, k)) + f" | {n} |")
    # roles + mismatch
    roles = Counter()
    for cid, f in final.items():
        for k, v in f["fields"].items():
            if k.startswith("role:"):
                roles[v] += 1
    mm = sum(1 for f in final.values() if any(f["mismatch"]))
    L += ["", f"## Roles (units asked in stage 2)", "", f"Resolved server/client/library vs unclear: {dict(roles)}.", "",
          f"text_product_mismatch flagged by at least one annotator: {mm} of {len(final)} CVEs.", ""]
    usage = defaultdict(lambda: [0, 0])
    up = work(run) / "usage.jsonl"
    for l in (up.read_text().splitlines() if up.exists() else []):
        r = json.loads(l)
        if r["scope"] == "stage2":
            usage[r["stage"]][0] += r["tokens"]
            usage[r["stage"]][1] += r["calls"]
    L += ["## Tokens (as reported by the harness per subagent call; calls killed by the 5-hour limit report no usage)", "", "| stage | tokens | calls reported |", "|---|---|---|"]
    L += [f"| {s} | {t} | {c} |" for s, (t, c) in sorted(usage.items())]
    L += ["", f"Total reported: {sum(t for t, _ in usage.values())}."]
    (run / "stage2_report.md").write_text("\n".join(L) + "\n")
    jdump(run / "final.json", {"rules_version": er.RULES_VERSION, "final": final, "missing": missing, "precision": prec, "audit": audit})
    print(run / "stage2_report.md", "precision", {k: (v["confirmed"], v["n"]) for k, v in prec.items()}, "audit", {k: (v["missed"], v["n"]) for k, v in audit.items()})


def resolve(run: Path, scope: str, sa: str, sb: str, adj: str | None) -> dict[str, dict]:
    """CVE -> merged fields for every CVE annotated by both slots (agreed, adjudicated, or REVIEW_REQUIRED/unclear), with both annotations."""
    idx = item_index(run, scope)
    A, B = annotations(run, scope, sa), annotations(run, scope, sb)
    rul = load_rulings(run, scope, adj) if adj and (work(run) / scope / f"adj-{adj}").exists() else {}
    out = {}
    for cid in sorted(set(A) & set(B)):
        m = ef.merge(A[cid], B[cid], idx[cid]["label_set"], rul.get(cid, {}))
        out[cid] = {"fields": m["fields"], "how": m["how"], "A": A[cid], "B": B[cid]}
    return out


def cmd_audit_unres_report(a) -> None:
    """OQ-P3-5: share of the audited OS rule-unresolved CVEs that the annotators confirm as ADMIN_CONFIRMED."""
    run = Path(a.run)
    cfg = jload(ROOT / "configs/effect/p3.json")
    sel = jload(work(run) / "audit_os_unres" / "select.json")["cves"]
    res = resolve(run, "audit_os_unres", "A", "B", "au")
    idx = {i["cve_id"]: i for i in jload(work(run) / "pool.json")}
    n_unres = sum(1 for v in jload(work(run) / "rules.json").values() if v["label"] == "unresolved") - 0
    fam = cfg["os_sample"]["family"]
    n_os_unres = sum(1 for c, v in jload(work(run) / "rules.json").items() if v["label"] == "unresolved" and ef.pool_of(idx[c], fam) == "os")
    labs = Counter(res[c]["fields"]["label"] for c in res)
    k, n = labs.get("ADMIN_CONFIRMED", 0), len(res)
    ci = wilson(k, n)
    stop = n > 0 and k / n > cfg["audit_unresolved"]["stop_fraction"]
    agr = ef.agreement([(res[c]["A"], res[c]["B"], idx[c]["label_set"]) for c in res])
    L = ["# OQ-P3-5: audit of OS rule-unresolved CVEs (rules " + er.RULES_VERSION + ")", "",
         f"Seeded sample {len(sel)} of {n_os_unres} OS rule-unresolved CVEs; annotated by both slots: {n}.", "",
         f"Final labels (agreed or adjudicated): {dict(labs)}.", "",
         f"ADMIN_CONFIRMED among the audited: {k} of {n} = {k / n:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f}); projected over the unresolved OS pool: {k / n * n_os_unres:.0f} (range {ci[0] * n_os_unres:.0f}-{ci[1] * n_os_unres:.0f}).",
         f"Stop rule (more than {cfg['audit_unresolved']['stop_fraction']:.0%}): {'TRIGGERED' if stop else 'not triggered'}.", "",
         "| field | n | agreement | kappa |", "|---|---|---|---|"]
    L += [f"| {f} | {v['n']} | {fmt(v['agree'])} | {fmt(v['kappa'])} |" for f, v in agr.items()]
    L += ["", "## ADMIN_CONFIRMED cases (evidence the rules missed)", ""]
    for c in res:
        if res[c]["fields"]["label"] == "ADMIN_CONFIRMED":
            ev = (res[c]["A"] if res[c]["A"]["label"] == "ADMIN_CONFIRMED" else res[c]["B"])["evidence"][:1]
            L.append(f"- {c}: {json.dumps(ev)[:260]}")
    (run / "audit_os_unres_report.md").write_text("\n".join(L) + "\n")
    jdump(run / "audit_os_unres_final.json", {"labels": dict(labs), "k": k, "n": n, "ci": ci, "stop": stop, "n_os_unresolved": n_os_unres})
    print(run / "audit_os_unres_report.md", f"ADMIN_CONFIRMED {k}/{n}", "STOP" if stop else "ok")


def freeze_rows(p3: Path, master_rows: list[dict], cfg: dict) -> list[dict]:
    """One row per (CVE, unit) with the final P3 label, role, target_privilege, evidence, who labelled it, mismatch flag."""
    idx = {i["cve_id"]: i for i in jload(work(p3) / "pool.json")}
    rules = jload(work(p3) / "rules.json")
    res = {**resolve(p3, "stage2", "A", "B", "s2")}
    res.update(resolve(p3, "audit_os_unres", "A", "B", "au") if (work(p3) / "audit_os_unres" / "A").exists() else {})
    munit = {(r["cve_id"], u["product"], u["family"], u["os"]): u for r in master_rows for u in r["units"]}
    rows = []
    for cid in sorted(idx):
        it = idx[cid]
        for u in it["units"]:
            mu = munit[(cid, u["product"], u["family"], u["os"])]
            row = {"cve_id": cid, "product": u["product"], "family": u["family"], "os": u["os"], "cohort": u["cohort"],
                   "label_set": it["label_set"], "funnel_stage": u["stage"], "ac": mu["ac"], "ac_status": mu["ac_status"],
                   "rule": {"label": rules[cid]["label"], "id": rules[cid]["rule"], "version": rules[cid]["version"]}}
            if cid in res:
                r = res[cid]
                f, how = r["fields"], r["how"]
                row["label"] = f["label"]
                row["labelled_by"] = "annotator" if how["label"] == "agreed" else "adjudicator"
                row["target_privilege"] = f.get("target_privilege")
                src = r["A"] if r["A"]["label"] == f["label"] else r["B"]
                row["evidence"] = src["evidence"]
                row["text_product_mismatch"] = bool(r["A"]["text_product_mismatch"] or r["B"]["text_product_mismatch"])
                rk = f"role:{u['product']}"
                if rk in f:
                    row["role"], row["role_by"] = f[rk], "annotator" if how[rk] == "agreed" else "adjudicator"
                else:
                    row["role"], row["role_by"] = mu["role"], "master"
            elif rules[cid]["label"] in ("NON_RCE", "NON_ADMIN"):
                row.update(label=rules[cid]["label"], labelled_by="rule", target_privilege="unspecified" if it["label_set"] == "local" else None,
                           evidence=[{"source": rules[cid]["source"], "span": rules[cid]["span"]}], text_product_mismatch=False,
                           role=mu["role"], role_by="master")
            else:  # rules left it unresolved and it was not annotated (OS only)
                row.update(label="UNRESOLVED", labelled_by="rule", target_privilege=None, evidence=[], text_product_mismatch=False,
                           role=mu["role"], role_by="master")
            rows.append(row)
    return rows


def cmd_p4_init(a) -> None:
    """Freeze the P3 labels, draw the validation sample, write batches, human_check.csv and the eligible counts."""
    import csv
    import gzip

    cfg = jload(ROOT / "configs/effect/p3.json")
    p3 = Path(a.p3run)
    master = (ROOT / cfg["master_dataset"]).resolve()
    mrows = ef.load_master(master)
    rows = freeze_rows(p3, mrows, cfg)
    prov = runlog.provenance([cfg["p4"]["sample_seed"], cfg["p4"]["human_seed"]], cfg)
    stamp = prov["date_utc"].replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    run = OUT / f"{stamp}-{prov['git_commit'][:7]}"
    run.mkdir(parents=True, exist_ok=False)
    fz = run / "frozen"
    fz.mkdir()
    data = gzip.compress("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode(), mtime=0)
    name = cfg["p4"]["frozen_version"]
    (fz / f"{name}.jsonl.gz").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (fz / f"{name}.sha256").write_text(f"{digest}  {name}.jsonl.gz\n")
    # eligible counts per family x OS x cohort: unit funnel positive -> role rule -> AC explicit
    pos = [r for r in rows if r["label"] in ("RCE_CONFIRMED", "ADMIN_CONFIRMED")]
    table: dict = defaultdict(lambda: Counter())
    for r in pos:
        k = f"{r['label']}|{r['family']}|{r['os']}|{r['cohort']}"
        table[k]["positive_units"] += 1
        role_ok = r["label_set"] == "local" or r["role"] == "server"
        table[k]["after_role_rule"] += role_ok
        if ef.eligible(r):
            table[k]["eligible_units"] += 1
            table[k]["_cves"] = table[k].get("_cves", 0)
    cves: dict = defaultdict(set)
    for r in rows:
        if ef.eligible(r):
            cves[f"{r['label']}|{r['family']}|{r['os']}|{r['cohort']}"].add(r["cve_id"])
    elig = {k: {**dict(v), "eligible_cves": len(cves.get(k, ()))} for k, v in sorted(table.items())}
    for v in elig.values():
        v.pop("_cves", None)
    jdump(run / "eligible.json", elig)
    # validation sample: annotated or rule-NON labelled, never in a pilot or audit
    excl = set()
    for d in sorted(OUT.glob("*/work/pilot/items.json")):
        excl |= {i["cve_id"] for i in jload(d)}
    sel2 = jload(work(p3) / "stage2_select.json")
    excl |= {c for c, r in sel2.items() if any(x.startswith("audit_") for x in r)}
    au = work(p3) / "audit_os_unres" / "select.json"
    excl |= set(jload(au)["cves"]) if au.exists() else set()
    idx = {i["cve_id"]: i for i in jload(work(p3) / "pool.json")}
    fam = cfg["os_sample"]["family"]
    cves_label: dict[str, str] = {}
    by: dict = {}
    for r in rows:
        by.setdefault(r["cve_id"], r)
    cells: dict = defaultdict(list)
    for cid, r in by.items():
        if cid in excl or r["label"] == "UNRESOLVED":
            continue
        cells[(ef.pool_of(idx[cid], fam), r["label_set"], r["label"])].append(cid)
        cves_label[cid] = r["label"]
    samp = ef.sample_cells(cells, cfg["p4"]["sample_n"], cfg["p4"]["sample_seed"])
    items = [idx[c] for c in sorted(samp)]
    jdump(work(run) / "pool.json", items)
    jdump(work(run) / "val_select.json", {c: {"cell": list(k), "p3_label": cves_label[c], "labelled_by": by[c]["labelled_by"]} for c, k in samp.items()})
    rubric = (ROOT / cfg["p4"]["rubric"]).read_text()
    for slot in ("A", "B"):
        ps = write_batches(run, work(run) / "val" / slot, items, cfg["p4"]["batch_size"], rubric, reverse=slot == "B")
        print(slot, len(ps), "batches")
    # human check: 60 of the 300, stratified by the same cells, text only
    hc = ef.sample_cells({k: [c for c in samp if samp[c] == k] for k in set(samp.values())}, cfg["p4"]["human_n"], cfg["p4"]["human_seed"])
    with (run / "human_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ef.HUMAN_COLUMNS)
        w.writeheader()
        for c in sorted(hc):
            w.writerow(ef.human_check_row(ef.annotator_view(idx[c]) | {"units": idx[c]["units"]}))
    jdump(run / "manifest.json", {"p3_run": str(p3), "frozen": f"{name}.jsonl.gz", "frozen_sha256": digest, "frozen_rows": len(rows),
                                  "sample": len(samp), "excluded_cves": len(excl), "human_check": len(hc),
                                  "cells": {"|".join(k): len(v) for k, v in sorted(cells.items())},
                                  "sample_cells": dict(Counter("|".join(v) for v in samp.values())), "master_sha256": sha(master)})
    jdump(run / "provenance.json", prov)
    print(run, digest, len(rows), "rows;", len(samp), "sampled;", len(hc), "human check")


def cmd_p4_report(a) -> None:
    """Frozen P3 labels against the LLM-adjudicated reference set: per-class precision, recall, kappa."""
    run = Path(a.run)
    sel = jload(work(run) / "val_select.json")
    idx = {i["cve_id"]: i for i in jload(work(run) / "pool.json")}
    res = resolve(run, "val", "A", "B", "v")
    man = jload(run / "manifest.json")
    L = ["# P4: frozen P3 labels vs the LLM-adjudicated reference set", "",
         f"Frozen dataset {man['frozen']} (SHA-256 {man['frozen_sha256']}); validation sample {len(sel)} CVEs, reference labels for {len(res)}.",
         "The reference set is LLM-adjudicated (two blinded Sonnet annotators with a differently worded rubric, Sonnet adjudicator); it is not gold and not human-verified.", ""]
    agr = ef.agreement([(res[c]["A"], res[c]["B"], idx[c]["label_set"]) for c in res])
    L += ["## Reference-set annotator agreement (A vs B, before adjudication)", "", "| field | n | agreement | kappa |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {fmt(v['agree'])} | {fmt(v['kappa'])} |" for k, v in agr.items()]
    out = {}
    for ls, classes in (("remote", ef.REMOTE_LABELS), ("local", ef.LOCAL_LABELS)):
        for pool in ("all", "svc", "os"):
            ids = [c for c in res if idx[c]["label_set"] == ls and (pool == "all" or sel[c]["cell"][0] == pool)]
            if not ids:
                continue
            m = ef.class_metrics([sel[c]["p3_label"] for c in ids], [res[c]["fields"]["label"] for c in ids], classes)
            out[f"{ls}|{pool}"] = m
            L += ["", f"## {ls} label set, pool {pool}: n {m['n']}, accuracy {fmt(m['accuracy'])}, kappa {fmt(m['kappa'])}", "",
                  "| class | P3 predicted | reference | true positive | precision | recall |", "|---|---|---|---|---|---|"]
            L += [f"| {c} | {v['predicted']} | {v['reference']} | {v['tp']} | {fmt(v['precision'])} | {fmt(v['recall'])} |" for c, v in m["classes"].items()]
    L += ["", "## By who set the P3 label", "", "| labelled by | n | agree with reference | rate |", "|---|---|---|---|"]
    for lb in ("rule", "annotator", "adjudicator"):
        ids = [c for c in res if sel[c]["labelled_by"] == lb]
        k = sum(sel[c]["p3_label"] == res[c]["fields"]["label"] for c in ids)
        if ids:
            L.append(f"| {lb} | {len(ids)} | {k} | {k / len(ids):.3f} |")
    mism = [c for c in res if sel[c]["p3_label"] != res[c]["fields"]["label"]]
    L += ["", f"Disagreements P3 vs reference: {len(mism)} of {len(res)}.", "", "| CVE | cell | P3 label | reference |", "|---|---|---|---|"]
    L += [f"| {c} | {'/'.join(sel[c]['cell'][:2])} | {sel[c]['p3_label']} | {res[c]['fields']['label']} |" for c in mism]
    (run / "p4_validation_report.md").write_text("\n".join(L) + "\n")
    jdump(run / "reference.json", {"labels": {c: res[c]["fields"]["label"] for c in res}, "metrics": out})
    print(run / "p4_validation_report.md", "disagreements", len(mism), "of", len(res))


# ---------------------------------------------------------------- v2 (decided 16 Nov 2026)

def load_frozen(path: Path) -> list[dict]:
    import gzip

    return [json.loads(line) for line in gzip.open(path, "rt")]


def v2_candidates(rows: list[dict], pool: dict[str, dict]) -> dict[str, str]:
    """CVE -> reason it is re-verified for v2: r2 would demote a positive label, or the label is REVIEW_REQUIRED."""
    out: dict[str, str] = {}
    for r in rows:
        c = r["cve_id"]
        if c in out:
            continue
        if r["label"] == "REVIEW_REQUIRED":
            out[c] = "readjudicate"
        elif er.r2_demotes(pool[c], r["label"], r["evidence"]):
            out[c] = "r2_hedge"
    return out


def cmd_v2_init(a) -> None:
    """New run folder: the CVEs r2 would change plus every REVIEW_REQUIRED, with batches for two blinded slots."""
    cfg = jload(ROOT / "configs/effect/p3.json")
    p4 = Path(a.p4run)
    man = jload(p4 / "manifest.json")
    rows = load_frozen(p4 / "frozen" / man["frozen"])
    pool = {i["cve_id"]: i for i in jload(work(Path(man["p3_run"])) / "pool.json")}
    cand = v2_candidates(rows, pool)
    prov = runlog.provenance([], cfg)
    stamp = prov["date_utc"].replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    run = OUT / f"{stamp}-{prov['git_commit'][:7]}"
    run.mkdir(parents=True, exist_ok=False)
    items = [pool[c] for c in sorted(cand)]
    jdump(work(run) / "pool.json", items)
    jdump(work(run) / "v2_select.json", {c: {"reason": cand[c], "v1": next(r for r in rows if r["cve_id"] == c)["label"]} for c in sorted(cand)})
    rubric = (ROOT / cfg["v2"]["rubric"]).read_text()
    for slot in ("A", "B"):
        write_batches(run, work(run) / "v2" / slot, items, cfg["p4"]["batch_size"], rubric, reverse=slot == "B")
    jdump(run / "manifest.json", {"p4_run": str(p4), "p3_run": man["p3_run"], "v1": man["frozen"], "v1_sha256": man["frozen_sha256"],
                                  "candidates": dict(Counter(cand.values())), "rules": er.RULES_VERSION_R2})
    jdump(run / "provenance.json", prov)
    print(run, dict(Counter(cand.values())))


def finish_freeze(run: Path, rows: list[dict], name: str, man: dict, changes: list[tuple], title: str) -> None:
    """Write the frozen file, eligible counts (per cell and per remote product) and the validation table against the reference sample."""
    import gzip

    p4 = Path(man["p4_run"])
    fz = run / "frozen"
    fz.mkdir(exist_ok=True)
    data = gzip.compress("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode(), mtime=0)
    (fz / f"{name}.jsonl.gz").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (fz / f"{name}.sha256").write_text(f"{digest}  {name}.jsonl.gz\n")
    # eligible units: per family x OS x cohort, and per product for the remote set
    table: dict = defaultdict(Counter)
    prod: dict = defaultdict(Counter)
    cves: dict = defaultdict(set)
    for r in rows:
        if r["label"] not in ("RCE_CONFIRMED", "ADMIN_CONFIRMED"):
            continue
        k = f"{r['label']}|{r['family']}|{r['os']}|{r['cohort']}"
        table[k]["positive_units"] += 1
        table[k]["after_role_rule"] += r["label_set"] == "local" or r["role"] == "server"
        if ef.eligible(r):
            table[k]["eligible_units"] += 1
            cves[k].add(r["cve_id"])
            if r["label_set"] == "remote":
                prod[f"{r['family']}|{r['os']}|{r['cohort']}|{r['product']}"]["eligible_units"] += 1
    elig = {k: {**dict(v), "eligible_cves": len(cves[k])} for k, v in sorted(table.items())}
    jdump(run / "eligible.json", elig)
    jdump(run / "eligible_remote_by_product.json", {k: dict(v) for k, v in sorted(prod.items())})
    # validation against the same reference sample
    ref = jload(p4 / "reference.json")
    vsel = jload(work(p4) / "val_select.json")
    pidx = {i["cve_id"]: i for i in jload(work(p4) / "pool.json")}
    by = {}
    for r in rows:
        by.setdefault(r["cve_id"], r)
    metrics, L = {}, [f"# {title} validation: frozen P3 labels vs the same LLM-adjudicated reference sample (not gold)", "",
                      f"{name}.jsonl.gz SHA-256 {digest}; v1 SHA-256 {man['v1_sha256']} (untouched). Reference sample {len(ref['labels'])} CVEs, the same as v1's.", ""]
    for ls, classes in (("remote", ef.REMOTE_LABELS), ("local", ef.LOCAL_LABELS)):
        for pool in ("all", "svc", "os"):
            ids = [c for c in ref["labels"] if pidx[c]["label_set"] == ls and (pool == "all" or vsel[c]["cell"][0] == pool)]
            if not ids:
                continue
            m = ef.class_metrics([by[c]["label"] for c in ids], [ref["labels"][c] for c in ids], classes)
            metrics[f"{ls}|{pool}"] = m
            m1 = ref["metrics"][f"{ls}|{pool}"]
            L += ["", f"## {ls}, pool {pool}: n {m['n']}; accuracy v1 {fmt(m1['accuracy'])} -> {title} {fmt(m['accuracy'])}; kappa v1 {fmt(m1['kappa'])} -> {title} {fmt(m['kappa'])}", "",
                  f"| class | {title} predicted | reference | true positive | precision v1 | precision {title} | recall v1 | recall {title} |", "|---|---|---|---|---|---|---|---|"]
            L += [f"| {c} | {v['predicted']} | {v['reference']} | {v['tp']} | {fmt(m1['classes'][c]['precision'])} | {fmt(v['precision'])} | {fmt(m1['classes'][c]['recall'])} | {fmt(v['recall'])} |" for c, v in m["classes"].items()]
    mism = [c for c in ref["labels"] if by[c]["label"] != ref["labels"][c]]
    L += ["", f"Disagreements vs reference: {len(mism)} of {len(ref['labels'])} (v1: {sum(vsel[c]['p3_label'] != ref['labels'][c] for c in ref['labels'])}).", "",
          f"| CVE | v1 label | {title} label | reference |", "|---|---|---|---|"]
    L += [f"| {c} | {vsel[c]['p3_label']} | {by[c]['label']} | {ref['labels'][c]} |" for c in mism]
    L += ["", f"## Labels changed by {title}", "", f"| CVE | reason | before | {title} |", "|---|---|---|---|"] + [f"| {c} | {w} | {x} | {y} |" for c, w, x, y in changes]
    (run / f"{title}_validation_report.md").write_text("\n".join(L) + "\n")
    jdump(run / f"{title}_changes.json", [{"cve_id": c, "reason": w, "before": x, title: y} for c, w, x, y in changes])
    man.update({name: f"{name}.jsonl.gz", name + "_sha256": digest, name + "_rows": len(rows)})
    jdump(run / "manifest.json", man)
    print(run, name, digest, len(rows), "rows;", sum(x[2] != x[3] for x in changes), "of", len(changes), "candidates changed label")



def cmd_v21_init(a) -> None:
    """v2.1 (decided 16 Nov 2026): CVEs r2 demoted in v2 that the stock-"allow" exemption no longer demotes, re-verified blind."""
    cfg = jload(ROOT / "configs/effect/p3.json")
    v2run = Path(a.v2run)
    vm = jload(v2run / "manifest.json")
    p4 = Path(vm["p4_run"])
    pool = {i["cve_id"]: i for i in jload(work(Path(vm["p3_run"])) / "pool.json")}
    rows = {}
    for r in load_frozen(v2run / "frozen" / vm["v2"]):
        rows.setdefault(r["cve_id"], r)
    sel = jload(work(v2run) / "v2_select.json")
    v1 = {}
    for r in load_frozen(p4 / "frozen" / vm["v1"]):
        v1.setdefault(r["cve_id"], r)
    cand = sorted(c for c, v in sel.items() if v["reason"] == "r2_hedge" and not er.r2_demotes(pool[c], v1[c]["label"], v1[c]["evidence"]))
    prov = runlog.provenance([], cfg)
    stamp = prov["date_utc"].replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    run = OUT / f"{stamp}-{prov['git_commit'][:7]}"
    run.mkdir(parents=True, exist_ok=False)
    items = [pool[c] for c in cand]
    jdump(work(run) / "pool.json", items)
    jdump(work(run) / "v21_select.json", {c: {"reason": "stock_allow_exempt", "v2": rows[c]["label"]} for c in cand})
    rubric = (ROOT / cfg["v2"]["rubric_v21"]).read_text()
    for slot in ("A", "B"):
        write_batches(run, work(run) / "v21" / slot, items, cfg["p4"]["batch_size"], rubric, reverse=slot == "B")
    jdump(run / "manifest.json", {**{k: vm[k] for k in ("p4_run", "p3_run", "v1", "v1_sha256")}, "v2_run": str(v2run), "v2": vm["v2"], "v2_sha256": vm["v2_sha256"], "candidates": cand})
    jdump(run / "provenance.json", prov)
    print(run, cand)


def cmd_v21_freeze(a) -> None:
    """Apply the verified labels to the v2 rows, freeze v2.1, recount eligible units, redo the validation table."""
    cfg = jload(ROOT / "configs/effect/p3.json")
    run = Path(a.run)
    man = jload(run / "manifest.json")
    v2run = Path(man["v2_run"])
    assert sha(v2run / "frozen" / man["v2"]) == man["v2_sha256"], "v2 changed on disk"
    sel = jload(work(run) / "v21_select.json")
    res = resolve(run, "v21", "A", "B", "v21")
    missing = sorted(set(sel) - set(res))
    if missing:
        sys.exit(f"candidate CVEs without two valid annotations: {missing}")
    rows, changes = [], []
    for r in load_frozen(v2run / "frozen" / man["v2"]):
        r = dict(r)
        c = r["cve_id"]
        if c in sel:
            m = res[c]
            f, how = m["fields"], m["how"]
            src = m["A"] if m["A"]["label"] == f["label"] else m["B"]
            r["v21_from"] = {"label": r["label"], "role": r["role"]}
            r["label"] = f["label"]
            r["labelled_by"] = "annotator" if how["label"] == "agreed" else "adjudicator"
            r["target_privilege"] = f.get("target_privilege")
            r["evidence"] = src["evidence"] if f["label"] != "UNKNOWN" else []
            rk = f"role:{r['product']}"
            if rk in f:
                r["role"], r["role_by"] = f[rk], "annotator" if how[rk] == "agreed" else "adjudicator"
            r["v21_reason"] = sel[c]["reason"]
            if c not in {x[0] for x in changes}:
                changes.append((c, sel[c]["reason"], r["v21_from"]["label"], r["label"]))
        rows.append(r)
    finish_freeze(run, rows, cfg["v2"]["frozen_version_v21"], man, changes, "v2.1")


def cmd_v2_freeze(a) -> None:
    """Apply the verified labels to the v1 rows, freeze v2, count eligible units, redo the validation table."""
    import gzip

    cfg = jload(ROOT / "configs/effect/p3.json")
    run = Path(a.run)
    man = jload(run / "manifest.json")
    p4 = Path(man["p4_run"])
    v1 = load_frozen(p4 / "frozen" / man["v1"])
    assert sha(p4 / "frozen" / man["v1"]) == man["v1_sha256"], "v1 changed on disk"
    sel = jload(work(run) / "v2_select.json")
    res = resolve(run, "v2", "A", "B", "v2")
    missing = sorted(set(sel) - set(res))
    if missing:
        sys.exit(f"{len(missing)} candidate CVEs without two valid annotations: {missing[:5]}")
    rows, changes = [], []
    for r in v1:
        r = dict(r)
        c = r["cve_id"]
        if c in sel:
            m = res[c]
            f, how = m["fields"], m["how"]
            src = m["A"] if m["A"]["label"] == f["label"] else m["B"]
            r["v2_from"] = {"label": r["label"], "role": r["role"]}
            label = f["label"]
            if sel[c]["reason"] == "r2_hedge" and label in er.NEGATIVE:
                # OQ-P4-3 says demote to UNKNOWN; a clear other effect next to a hedged one is not a NON_* label.
                r["v2_annotator_label"], label = label, "UNKNOWN"
            r["label"] = label
            r["labelled_by"] = "annotator" if how["label"] == "agreed" else "adjudicator"
            r["target_privilege"] = f.get("target_privilege") if label != "UNKNOWN" else None
            r["evidence"] = src["evidence"] if label not in ("UNKNOWN",) else []
            rk = f"role:{r['product']}"
            if rk in f:
                r["role"], r["role_by"] = f[rk], "annotator" if how[rk] == "agreed" else "adjudicator"
            r["v2_reason"] = sel[c]["reason"]
        rows.append(r)
    for r in rows:
        if "v2_from" in r and r["cve_id"] not in {x[0] for x in changes}:
            changes.append((r["cve_id"], r["v2_reason"], r["v2_from"]["label"], r["label"]))
    finish_freeze(run, rows, cfg["v2"]["frozen_version"], man, changes, "v2")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("config", type=Path); p.set_defaults(f=cmd_init)
    p = sub.add_parser("pilot"); p.add_argument("run"); p.add_argument("--size", type=int)
    p.add_argument("--slots", nargs="+", default=["A", "B"]); p.add_argument("--models", nargs="+", default=["haiku", "sonnet"]); p.set_defaults(f=cmd_pilot)
    p = sub.add_parser("stage2-batches"); p.add_argument("run"); p.add_argument("--size", type=int); p.set_defaults(f=cmd_stage2_batches)
    p = sub.add_parser("ingest"); p.add_argument("run"); p.add_argument("stage"); p.add_argument("--scope", default="pilot"); p.set_defaults(f=cmd_ingest)
    p = sub.add_parser("repair"); p.add_argument("run"); p.add_argument("stage"); p.add_argument("--scope", default="pilot")
    p.add_argument("--size", type=int); p.set_defaults(f=cmd_repair)
    p = sub.add_parser("usage"); p.add_argument("run"); p.add_argument("stage"); p.add_argument("tokens", type=int)
    p.add_argument("--calls", type=int, default=1); p.add_argument("--scope", default="pilot"); p.set_defaults(f=cmd_usage)
    p = sub.add_parser("adjudicate-prep"); p.add_argument("run"); p.add_argument("stage_a"); p.add_argument("stage_b"); p.add_argument("name")
    p.add_argument("--scope", default="pilot"); p.set_defaults(f=cmd_adjudicate_prep)
    p = sub.add_parser("adjudicate-ingest"); p.add_argument("run"); p.add_argument("name"); p.add_argument("--scope", default="pilot"); p.set_defaults(f=cmd_adjudicate_ingest)
    p = sub.add_parser("audit-unres-prep"); p.add_argument("run"); p.set_defaults(f=cmd_audit_unres_prep)
    p = sub.add_parser("audit-unres-report"); p.add_argument("run"); p.set_defaults(f=cmd_audit_unres_report)
    p = sub.add_parser("p4-init"); p.add_argument("p3run"); p.set_defaults(f=cmd_p4_init)
    p = sub.add_parser("v2-init"); p.add_argument("p4run"); p.set_defaults(f=cmd_v2_init)
    p = sub.add_parser("v21-init"); p.add_argument("v2run"); p.set_defaults(f=cmd_v21_init)
    p = sub.add_parser("v21-freeze"); p.add_argument("run"); p.set_defaults(f=cmd_v21_freeze)
    p = sub.add_parser("v2-freeze"); p.add_argument("run"); p.set_defaults(f=cmd_v2_freeze)
    p = sub.add_parser("p4-report"); p.add_argument("run"); p.set_defaults(f=cmd_p4_report)
    p = sub.add_parser("stage2-report"); p.add_argument("run"); p.set_defaults(f=cmd_stage2_report)
    p = sub.add_parser("pilot-report"); p.add_argument("run"); p.set_defaults(f=cmd_pilot_report)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
