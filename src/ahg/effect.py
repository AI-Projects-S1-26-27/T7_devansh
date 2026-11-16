"""Effect classifier support (D28 Rounds 1-2, task P3): pool, evidence packets, annotation checks, agreement.

The labelling itself is done by separate Claude subagents (no external model APIs, Round 1). This module
holds everything around them that must be deterministic and testable: which CVEs are classified, what
text each annotator sees, whether an answer is well formed and cites a verbatim span, how two annotators
and an adjudicator combine, and Cohen's kappa. Nothing here is a gold label (P4 validates).

Label sets (Round 1; OQ-P3-1 option (a), decided 15 Nov 2026):
  remote (AV:N): RCE_CONFIRMED / NON_RCE / UNKNOWN / REVIEW_REQUIRED
  local  (AV:L): ADMIN_CONFIRMED / NON_ADMIN / UNKNOWN / REVIEW_REQUIRED, plus target_privilege.
Only RCE_CONFIRMED (remote) and ADMIN_CONFIRMED (local) create actions that raise privilege.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REMOTE_LABELS = ("RCE_CONFIRMED", "NON_RCE", "UNKNOWN", "REVIEW_REQUIRED")
LOCAL_LABELS = ("ADMIN_CONFIRMED", "NON_ADMIN", "UNKNOWN", "REVIEW_REQUIRED")
LABELS = {"remote": REMOTE_LABELS, "local": LOCAL_LABELS}
ADMIN_PRIVILEGES = ("root", "SYSTEM", "administrator")
TARGET_PRIVILEGES = (*ADMIN_PRIVILEGES, "other", "unspecified")
ROLES = ("server", "client", "library", "unclear")
EFFECTS = ("code_execution", "privilege_escalation", "denial_of_service", "information_disclosure",
           "authentication_bypass", "memory_corruption", "other")  # fmt: skip
MAX_SPAN = 300
ROOT = Path(__file__).resolve().parents[2]
LABELS_CONFIG = ROOT / "configs" / "d28_effect_labels.json"
GHSA_URL = re.compile(r"github\.com/[^/\s]+/[^/\s]+/security/advisories/(GHSA(?:-[0-9a-z]{4}){3})")


def norm(text: str) -> str:
    """Whitespace-collapsed text; spans are compared after this, so line wrapping never breaks a quote."""
    return " ".join(text.split())


# ---------------------------------------------------------------- pool

def label_set_of(row: dict) -> str | None:
    """'remote' or 'local' from the resolved attack vector; None if the CVE has no single AV:N / AV:L value."""
    av = set(row["vector_resolution"].get("candidates", {}).get("AV", []))
    return {"N": "remote", "L": "local"}.get("".join(av)) if len(av) == 1 else None


def load_master(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def pool_units(rows: list[dict], cfg: dict) -> tuple[list[tuple[str, dict]], list[tuple[str, dict]]]:
    """(full pool, local OS candidates) as (cve_id, unit) pairs, in cohort only.

    Full pool (task P3): every unit that reached remote_candidate or local_candidate, plus units whose
    role is ambiguous at the stages that depend on the role (role_not_server for remote CVEs,
    local_candidate), minus the OS family: kernel and Windows OS local candidates are far too many, so
    they enter only through the seeded sample.
    """
    os_cfg = cfg["os_sample"]
    full, os_local = [], []
    for r in rows:
        for u in r["units"]:
            if u["cohort"] is None:
                continue
            if u["family"] == os_cfg["family"]:
                if u["stage"] == os_cfg["stage"]:
                    os_local.append((r["cve_id"], u))
                continue
            amb = u["role"] == "ambiguous" and u["stage"] in cfg["role_ambiguous_stages"]
            if u["stage"] in cfg["pool_stages"] or amb:
                full.append((r["cve_id"], u))
    return full, os_local


def sample_os(os_local: list[tuple[str, dict]], cfg: dict) -> list[tuple[str, dict]]:
    """Seeded sample of per_group units per (os, cohort); a group smaller than that is taken whole.

    per_group null means no sampling: every OS local candidate is kept (rules-first pipeline, 15 Nov 2026).
    """
    os_cfg = cfg["os_sample"]
    if os_cfg["per_group"] is None:
        return list(os_local)
    groups: dict[tuple, list] = defaultdict(list)
    for cid, u in sorted(os_local, key=lambda t: (t[0], t[1]["product"])):
        groups[(u["os"], u["cohort"])].append((cid, u))
    rng = random.Random(os_cfg["seed"])
    out = []
    for key in sorted(groups):
        g = groups[key]
        out += g if len(g) <= os_cfg["per_group"] else rng.sample(g, os_cfg["per_group"])
    return out


def unit_view(u: dict) -> dict:
    """The unit fields an annotator or the report needs; role_question marks the units whose role is open."""
    return {"product": u["product"], "family": u["family"], "os": u["os"], "cohort": u["cohort"],
            "stage": u["stage"], "prior_role": u["role"], "role_question": u["role"] == "ambiguous",
            "role_reason": u.get("role_reason", ""), "role_evidence": u.get("role_evidence", [])}


# ---------------------------------------------------------------- evidence packet

def ghsa_texts(row: dict, ghsa_by_id: dict[str, str]) -> dict[str, str]:
    """GHSA advisory texts already archived for this CVE's references (no new fetching in P3)."""
    out = {}
    for ref in row["references"]:
        m = GHSA_URL.search(ref["url"])
        if m and m.group(1) in ghsa_by_id:
            out[f"ghsa:{m.group(1)}"] = ghsa_by_id[m.group(1)]
    return out


def packet(row: dict, units: list[dict], ghsa_by_id: dict[str, str], faq_max: int) -> dict:
    """What an annotator sees for one CVE: evidence texts (quotable) and supporting signals (not quotable)."""
    texts = {"nvd_description": row["description"]}
    m = row.get("msrc")
    if m:
        if m["impacts"]:
            texts["msrc_impact"] = "; ".join(m["impacts"])
        if m["title"]:
            texts["msrc_title"] = m["title"]
        if m["faq"]:
            texts["msrc_faq"] = m["faq"][:faq_max]
        if m["cna_note"]:
            texts["msrc_cna_note"] = "; ".join(m["cna_note"])
    texts.update({k: v[:faq_max] for k, v in ghsa_texts(row, ghsa_by_id).items()})  # same cap as the FAQ: token cost
    vr = row["vector_resolution"]
    supporting = {"cwe": [c[0] for c in row["cwe"]],
                  "cvss_v3": {k: v for k, v in vr.get("values", {}).items() if k in ("AV", "PR", "UI", "AC", "C", "I", "A")}}
    return {"cve_id": row["cve_id"], "label_set": label_set_of(row), "texts": texts, "supporting": supporting,
            "units": [unit_view(u) for u in units]}


def build_items(rows: list[dict], units: list[tuple[str, dict]], ghsa_by_id: dict[str, str], faq_max: int) -> list[dict]:
    """One item per CVE, with all its pool units; items sorted by CVE ID. CVEs with no single AV are dropped."""
    by_cve: dict[str, list[dict]] = defaultdict(list)
    for cid, u in units:
        by_cve[cid].append(u)
    row_of = {r["cve_id"]: r for r in rows}
    items = [packet(row_of[c], by_cve[c], ghsa_by_id, faq_max) for c in sorted(by_cve)]
    return [i for i in items if i["label_set"]]


def annotator_view(item: dict) -> dict:
    """Trimmed item for annotators (token cost): no CWE/CVSS (never decisive), no duplicated title, short unit info.

    Quotable texts are unchanged except msrc_title, dropped when the description already says the same.
    """
    texts = dict(item["texts"])
    if norm(texts.get("msrc_title", "")) == norm(texts.get("nvd_description", "")):
        texts.pop("msrc_title", None)
    return {"cve_id": item["cve_id"], "label_set": item["label_set"], "texts": texts,
            "products": sorted({u["product"] for u in item["units"]}),
            "role_question": sorted({u["product"] for u in item["units"] if u["role_question"]})}


# ---------------------------------------------------------------- annotation checks

def check_annotation(ann: dict, item: dict) -> list[str]:
    """Problems with one annotation; empty if it is well formed and every cited span is quoted verbatim."""
    errs = []
    if ann.get("cve_id") != item["cve_id"]:
        return [f"cve_id {ann.get('cve_id')!r} != {item['cve_id']!r}"]
    ls = item["label_set"]
    label, tp = ann.get("label"), ann.get("target_privilege")
    if label not in LABELS[ls]:
        errs.append(f"label {label!r} not in the {ls} set")
    if ls == "remote" and tp is not None:
        errs.append("target_privilege must be null for remote items")
    if ls == "local":
        if tp not in TARGET_PRIVILEGES:
            errs.append(f"target_privilege {tp!r} invalid")
        if label == "ADMIN_CONFIRMED" and tp not in ADMIN_PRIVILEGES:
            errs.append("ADMIN_CONFIRMED needs target_privilege root, SYSTEM or administrator")
        if label != "ADMIN_CONFIRMED" and tp in ADMIN_PRIVILEGES:
            errs.append(f"{label} cannot carry admin target_privilege {tp}")
    if not isinstance(ann.get("text_product_mismatch"), bool):
        errs.append("text_product_mismatch must be true or false")
    if not isinstance(ann.get("effects"), list) or any(e not in EFFECTS for e in ann["effects"]):
        errs.append("effects must be a list drawn from the effect vocabulary")
    ev = ann.get("evidence")
    if not isinstance(ev, list):
        return errs + ["evidence must be a list"]
    if label in LABELS.get(ls, ()) and label != "UNKNOWN" and not ev:
        errs.append(f"{label} needs at least one evidence span")
    errs += _span_errors(ev, item["texts"])
    asked = {u["product"] for u in item["units"] if u["role_question"]}
    roles = ann.get("roles") or {}
    if set(roles) != asked:
        errs.append(f"roles must cover exactly {sorted(asked)}, got {sorted(roles)}")
    for prod, r in roles.items():
        if r.get("role") not in ROLES:
            errs.append(f"role of {prod} invalid: {r.get('role')!r}")
        elif r["role"] != "unclear":
            errs += _span_errors([r], item["texts"], f"role {prod}")
    return errs


def _span_errors(entries: list[dict], texts: dict[str, str], where: str = "evidence") -> list[str]:
    errs = []
    for e in entries:
        src, span = e.get("source"), e.get("span")
        if src not in texts:
            errs.append(f"{where}: source {src!r} is not a quotable text")
        elif not isinstance(span, str) or not span.strip() or len(span) > MAX_SPAN:
            errs.append(f"{where}: span missing or longer than {MAX_SPAN}")
        elif norm(span) not in norm(texts[src]):
            errs.append(f"{where}: span not verbatim in {src}: {span[:60]!r}")
    return errs


# ---------------------------------------------------------------- agreement and adjudication

def fields_of(ann: dict, ls: str) -> dict[str, str | None]:
    """The comparable fields of one annotation: label, target_privilege (local), role:<product>."""
    out: dict[str, str | None] = {"label": ann["label"]}
    if ls == "local":
        out["target_privilege"] = ann["target_privilege"]
    for prod, r in (ann.get("roles") or {}).items():
        out[f"role:{prod}"] = r["role"]
    return out


def cohen_kappa(a: list, b: list) -> float | None:
    """Cohen's kappa of two equal-length label lists; None if undefined (no items, or chance agreement = 1)."""
    if len(a) != len(b):
        raise ValueError("label lists differ in length")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(pairs: list[tuple[dict, dict, str]]) -> dict[str, dict]:
    """Per field family (label, target_privilege, role): n, raw agreement, kappa over (annotation, annotation, label_set) pairs.

    Labels are compared within a label set only, so kappa is also reported per label set under "label/<set>".
    """
    vals: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for x, y, ls in pairs:
        fx, fy = fields_of(x, ls), fields_of(y, ls)
        for k in fx:
            fam = k.split(":")[0]
            keys = [fam] + ([f"label/{ls}"] if fam == "label" else [])
            for key in keys:
                vals[key][0].append(fx[k])
                vals[key][1].append(fy.get(k))
    return {k: {"n": len(a), "agree": sum(p == q for p, q in zip(a, b)) / len(a) if a else None,
                "kappa": cohen_kappa(a, b)} for k, (a, b) in sorted(vals.items())}


def disputes(a: dict, b: dict, ls: str) -> list[str]:
    """Fields on which two annotations differ."""
    fa, fb = fields_of(a, ls), fields_of(b, ls)
    return [k for k in fa if fa[k] != fb.get(k)]


def merge(a: dict, b: dict, ls: str, rulings: dict[str, dict]) -> dict:
    """Final fields for one CVE: the agreed value, else the adjudicator's pick (A or B), else REVIEW_REQUIRED.

    The adjudicator never creates a value: "REVIEW_REQUIRED" for a label stays that label; for a role
    or target_privilege it becomes "unclear" / "unspecified" and the CVE stays excluded or unrated.
    """
    fa, fb = fields_of(a, ls), fields_of(b, ls)
    final, how = {}, {}
    for k in fa:
        if fa[k] == fb.get(k):
            final[k], how[k] = fa[k], "agreed"
            continue
        pick = rulings.get(k, {}).get("pick")
        if pick == "A":
            final[k], how[k] = fa[k], "adjudicated_A"
        elif pick == "B":
            final[k], how[k] = fb[k], "adjudicated_B"
        else:
            how[k] = "adjudicated_review" if pick else "unadjudicated"
            final[k] = {"label": "REVIEW_REQUIRED", "target_privilege": "unspecified"}.get(k.split(":")[0], "unclear")
    return {"fields": final, "how": how}


def check_ruling(r: dict, item: dict) -> list[str]:
    """Problems with one adjudicator ruling."""
    if r.get("pick") not in ("A", "B", "REVIEW_REQUIRED"):
        return [f"pick {r.get('pick')!r} invalid"]
    if r["pick"] == "REVIEW_REQUIRED":
        return []
    return _span_errors([r], item["texts"], "ruling")


# ---------------------------------------------------------------- batching

def batches(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def pilot_select(items: list[dict], cfg: dict) -> list[dict]:
    """Seeded stratified 40-CVE pilot: remote, role-ambiguous, local service, local OS (per OS); no CVE twice."""
    p = cfg["pilot"]
    rng = random.Random(p["seed"])

    def has_amb(i):
        return any(u["role_question"] for u in i["units"])

    def is_os(i):
        return any(u["family"] == cfg["os_sample"]["family"] for u in i["units"])

    taken: set[str] = set()
    out: list[dict] = []

    def draw(pred, n):
        cand = [i for i in items if i["cve_id"] not in taken and pred(i)]
        for i in rng.sample(cand, min(n, len(cand))):
            taken.add(i["cve_id"])
            out.append(i)

    draw(lambda i: has_amb(i) and not is_os(i), p["role_ambiguous"])
    for os_name in ("linux", "windows"):
        draw(lambda i, o=os_name: is_os(i) and any(u["os"] == o for u in i["units"]), p["local_os_per_os"])
    draw(lambda i: i["label_set"] == "local" and not is_os(i) and not has_amb(i), p["local_service"])
    draw(lambda i: i["label_set"] == "remote" and not has_amb(i), p["remote"])
    return sorted(out, key=lambda i: i["cve_id"])


# ---------------------------------------------------------------- stage 2 selection (rules-first pipeline)

def pool_of(item: dict, os_family: str) -> str:
    """'os' for kernel and Windows OS candidates, else 'svc' (service products)."""
    return "os" if any(u["family"] == os_family for u in item["units"]) else "svc"


def stage2_select(items: list[dict], verdicts: dict[str, dict], cfg: dict) -> dict[str, list[str]]:
    """CVE -> reasons it goes to stage 2 (Sonnet annotators), for the rules-first pipeline (decided 15 Nov 2026).

    (a) rule-positive: verification of every RCE_CONFIRMED / ADMIN_CONFIRMED claim;
    (b) ambiguous role: any unit whose role is open;
    (c) rule-unresolved service unit: OS unresolved CVEs are not annotated;
    (d) audit: a seeded sample of rule-negatives per pool (service, OS) not selected for another reason,
        to measure how often the rules miss a positive.
    """
    fam = cfg["os_sample"]["family"]
    sel: dict[str, list[str]] = {}
    for it in items:
        v = verdicts[it["cve_id"]]["label"]
        why = []
        if v in ("RCE_CONFIRMED", "ADMIN_CONFIRMED"):
            why.append("rule_positive")
        if any(u["role_question"] for u in it["units"]):
            why.append("ambiguous_role")
        if v == "unresolved" and pool_of(it, fam) == "svc":
            why.append("rule_unresolved_service")
        if why:
            sel[it["cve_id"]] = why
    rng = random.Random(cfg["audit"]["seed"])
    for pool in ("svc", "os"):
        neg = sorted(it["cve_id"] for it in items if verdicts[it["cve_id"]]["label"] in ("NON_RCE", "NON_ADMIN")
                     and pool_of(it, fam) == pool and it["cve_id"] not in sel)
        for c in rng.sample(neg, min(cfg["audit"]["per_pool"], len(neg))):
            sel[c] = [f"audit_{pool}"]
    return sel


def audit_unresolved_sample(items: list[dict], verdicts: dict[str, dict], exclude: set[str], cfg: dict) -> list[str]:
    """Seeded sample of OS CVEs that the rules left unresolved, to measure the rules' recall over that remainder (OQ-P3-5).

    exclude: CVEs already used (pilot, other audits), never drawn again.
    """
    fam = cfg["os_sample"]["family"]
    pool = sorted(i["cve_id"] for i in items if pool_of(i, fam) == "os" and verdicts[i["cve_id"]]["label"] == "unresolved"
                  and i["cve_id"] not in exclude)
    return sorted(random.Random(cfg["audit_unresolved"]["seed"]).sample(pool, min(cfg["audit_unresolved"]["n"], len(pool))))


# ---------------------------------------------------------------- P4: freezing, validation sample, metrics, eligibility

def sample_cells(cells: dict[tuple, list[str]], n: int, seed: int) -> dict[str, tuple]:
    """Seeded stratified sample: equal quota per cell, a small cell is taken whole and its shortfall is spread over the rest.

    Returns CVE -> cell. Deterministic for a given seed (cells and members are visited in sorted order).
    """
    keys = sorted(cells)
    take = {k: 0 for k in keys}
    left = n
    while left > 0:
        open_keys = [k for k in keys if take[k] < len(cells[k])]
        if not open_keys:
            break
        quota = max(1, left // len(open_keys))
        for k in open_keys:
            add = min(quota, len(cells[k]) - take[k], left)
            take[k] += add
            left -= add
            if left == 0:
                break
    rng = random.Random(seed)
    out: dict[str, tuple] = {}
    for k in keys:
        for c in rng.sample(sorted(cells[k]), take[k]):
            out[c] = k
    return out


def class_metrics(pred: list[str], ref: list[str], classes: tuple[str, ...]) -> dict:
    """Per-class precision, recall, support (in ref), plus accuracy and Cohen's kappa of pred against ref."""
    out: dict = {"n": len(ref), "accuracy": (sum(p == r for p, r in zip(pred, ref)) / len(ref)) if ref else None,
                 "kappa": cohen_kappa(pred, ref), "classes": {}}
    for c in classes:
        tp = sum(p == c and r == c for p, r in zip(pred, ref))
        npred, nref = sum(p == c for p in pred), sum(r == c for r in ref)
        out["classes"][c] = {"precision": tp / npred if npred else None, "recall": tp / nref if nref else None,
                             "predicted": npred, "reference": nref, "tp": tp}
    return out


def eligible(row: dict) -> bool:
    """Benchmark-eligible unit: positive label, AC explicit, and (remote) a server role after the role rule.

    Remote: RCE_CONFIRMED and role server (D28 E5: only server-side CVEs become inbound exploits).
    Local: ADMIN_CONFIRMED; the role rule is about inbound exploits, so no role filter (OQ-P4-1).
    """
    if row["ac_status"] != "explicit":
        return False
    if row["label_set"] == "remote":
        return row["label"] == "RCE_CONFIRMED" and row["role"] == "server"
    return row["label"] == "ADMIN_CONFIRMED"


HUMAN_COLUMNS = ["cve_id", "label_set", "products", "nvd_description", "msrc_impact", "msrc_title", "msrc_faq",
                 "msrc_cna_note", "other_texts", "role_question_products", "answer_effect", "answer_target_privilege",
                 "answer_roles", "answer_evidence_span", "notes"]  # fmt: skip


def human_check_row(item: dict) -> dict:
    """One CSV row of text only: no P3 label, no rule output, no annotator output; the answer columns are empty."""
    t = item["texts"]
    other = " || ".join(f"[{k}] {v}" for k, v in t.items() if k.startswith("ghsa:"))
    return {"cve_id": item["cve_id"], "label_set": item["label_set"],
            "products": "; ".join(sorted({u["product"] for u in item["units"]})),
            "nvd_description": t.get("nvd_description", ""), "msrc_impact": t.get("msrc_impact", ""),
            "msrc_title": t.get("msrc_title", ""), "msrc_faq": t.get("msrc_faq", ""), "msrc_cna_note": t.get("msrc_cna_note", ""),
            "other_texts": other, "role_question_products": "; ".join(sorted({u["product"] for u in item["units"] if u["role_question"]})),
            "answer_effect": "", "answer_target_privilege": "", "answer_roles": "", "answer_evidence_span": "", "notes": ""}


# ---------------------------------------------------------------- frozen label sets (OQ-M2-1, decided 16 Nov 2026)

def load_frozen_labels(version: str, config: Path = LABELS_CONFIG, root: Path = ROOT) -> list[dict]:
    """Rows of a frozen effect-label set, e.g. "p3-labels-v2.1", read from its untracked copy under data/.

    The file lives outside git, so the tracked config's SHA-256 is the only thing tying it to the frozen
    labels P4 validated; a file that differs (edited, truncated, the superseded draft) is refused, never read.
    """
    entry = json.loads(config.read_text())["versions"][version]  # KeyError: unknown version
    blob = (root / entry["path"]).read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != entry["sha256"]:
        raise ValueError(f"{entry['path']}: SHA-256 {got} does not match {entry['sha256']} in {config.name}")
    return [json.loads(line) for line in gzip.decompress(blob).decode().splitlines() if line.strip()]
