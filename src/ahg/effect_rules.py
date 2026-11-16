"""Stage 1 of the P3 effect classifier: deterministic phrase rules, zero tokens (decided 15 Nov 2026).

A rule claims a label only on explicit wording in a quotable text (NVD description, MSRC impact or title).
Everything else is "unresolved" and goes to stage 2 (Sonnet annotators) or stays unlabelled. Strict
flaw-type rule: a flaw type alone ("use-after-free", "out-of-bounds read") is never a consequence, so it
never produces a label. CWE and CVSS are not read at all. Bump RULES_VERSION on any change to a pattern.

Labels follow the packet's label_set: remote -> RCE_CONFIRMED / NON_RCE, local -> ADMIN_CONFIRMED / NON_ADMIN.
"""

from __future__ import annotations

import re

RULES_VERSION = "r1"
UNRESOLVED = "unresolved"
POSITIVE = {"RCE_CONFIRMED", "ADMIN_CONFIRMED"}
NEGATIVE = {"NON_RCE", "NON_ADMIN"}

I = re.IGNORECASE
# Wording that states code or command execution by the attacker.
EXEC = re.compile(
    r"remote code execution|arbitrary code execution|code execution|"
    r"execut\w+ (?:\w+ ){0,3}(?:code|commands?|programs?|scripts?|shell|php|binaries)|"
    r"run (?:\w+ ){0,3}(?:code|commands?|programs?)|arbitrary (?:os |system |shell )?commands?|spawn(?:s|ing)? a shell|remote shell",
    I)
# A match is dropped when one of these appears shortly before it ("does not allow code execution").
NEGATION = re.compile(r"\b(?:not|cannot|can't|unable|no|without|prevents?|mitigat\w+|unlikely|nor|never)\b|n't\b", I)
HEDGE = re.compile(r"\b(?:potentially|possibly|theoretically|might)\b", I)
# Words that mean the consequence may be more than the "clear non-" wording alone; they block NON_* labels.
BLOCK = re.compile(
    r"code|execut|command|shell|corrupt|overflow|use[- ]after[- ]free|inject|privilege|escalat|elevat|memory|script|"
    r"deserializ|upload|bypass|arbitrary|traversal|write|root|administrator|\bsystem\b|takeover|hijack|crash", I)
NON_WORDING = re.compile(
    r"denial[- ]of[- ]service|\bDoS\b|infinite loop|resource exhaustion|(?:memory|cpu|disk) (?:consumption|exhaustion)|"
    r"information disclosure|disclos\w+ (?:of )?(?:sensitive|potentially sensitive)? ?(?:information|data)|"
    r"obtain sensitive information|read (?:arbitrary |sensitive )?files?|cross-site scripting|\bXSS\b|open redirect|spoof", I)
NON_MSRC = {"denial of service", "information disclosure", "spoofing", "tampering"}  # security feature bypass: too broad
# Gaining a named admin identity: an escalation verb then a target, or code running as that identity.
GAIN = r"(?:gain\w*|obtain\w*|escalat\w+|elevat\w+|achiev\w+|acquir\w+|becom\w+|attain\w*|get(?:s|ting)?)"
TARGET = (r"(?:root|SYSTEM|NT AUTHORITY\\SYSTEM|LocalSystem|administrators?|administrative|admin)[ -](?:level )?"
          r"(?:privileges?|access|permissions?|rights|user|account|control)|full control (?:of|over) the (?:host|system|machine|computer)")
ADMIN = re.compile(rf"\b{GAIN}\b[^.;]{{0,60}}?\b(?:{TARGET})|"
                   r"execut\w+ [^.;]{0,50}\b(?:as|with)\b[^.;]{0,15}\b(?:root|SYSTEM)\b|\broot shell\b", I)


def _spans(pattern: re.Pattern, text: str) -> list[re.Match]:
    """Matches that are not negated or hedged by the 70 characters before them."""
    out = []
    for m in pattern.finditer(text):
        before = text[max(0, m.start() - 70) : m.start()]
        if NEGATION.search(before) or HEDGE.search(before):
            continue
        out.append(m)
    return out


def _hit(label: str, rule: str, source: str, text: str, m: re.Match) -> dict:
    lo, hi = max(0, m.start() - 30), min(len(text), m.end() + 30)
    return {"label": label, "rule": rule, "source": source, "span": " ".join(text[m.start() : m.end()].split())[:300],
            "context": " ".join(text[lo:hi].split())}


def apply_rules(item: dict) -> dict:
    """Rule verdict for one packet: {"label": <label or "unresolved">, "rule": id, "source", "span", "version"}."""
    texts, ls = item["texts"], item["label_set"]
    desc = texts.get("nvd_description", "")
    msrc = [x.strip().lower() for x in texts.get("msrc_impact", "").split(";") if x.strip()]
    title = texts.get("msrc_title", "")
    quotable = {"nvd_description": desc, "msrc_title": title}
    verdict: dict = {"label": UNRESOLVED, "rule": "none", "source": None, "span": None}

    if ls == "remote":
        pos = None
        if "remote code execution" in msrc:  # authoritative MSRC Impact value
            pos = {"label": "RCE_CONFIRMED", "rule": "remote.msrc_impact_rce", "source": "msrc_impact", "span": texts["msrc_impact"][:300]}
        else:
            for src, t in quotable.items():
                ms = _spans(EXEC, t)
                if ms:
                    pos = _hit("RCE_CONFIRMED", "remote.exec_wording", src, t, ms[0])
                    break
        non = _non(texts, "NON_RCE", "remote")
        if pos and non:
            verdict = {"label": UNRESOLVED, "rule": "conflict.rce_vs_non", "source": None, "span": None}  # sources disagree: stage 2
        else:
            verdict = pos or non or verdict
    else:
        pos = None
        # MSRC FAQs state the gained identity ("could gain SYSTEM privileges"); admin wording is read there too.
        for src, t in {**quotable, "msrc_faq": texts.get("msrc_faq", "")}.items():
            ms = _spans(ADMIN, t)
            if ms:
                pos = _hit("ADMIN_CONFIRMED", "local.admin_wording", src, t, ms[0])
                break
        non = _non(texts, "NON_ADMIN", "local")
        verdict = ({"label": UNRESOLVED, "rule": "conflict.admin_vs_non", "source": None, "span": None} if pos and non
                   else pos or non or verdict)
    return {k: verdict[k] for k in ("label", "rule", "source", "span")} | {"version": RULES_VERSION}


def _non(texts: dict, label: str, ls: str) -> dict | None:
    """NON_* only on clear single-effect wording: DoS / disclosure / spoofing and no wording of anything more."""
    desc, title = texts.get("nvd_description", ""), texts.get("msrc_title", "")
    msrc = [x.strip().lower() for x in texts.get("msrc_impact", "").split(";") if x.strip()]
    text = desc if desc else title
    if BLOCK.search(re.sub(NON_WORDING, "", text)):  # remove the clear wording itself ("denial of service" is not "crash")
        return None
    if any(m in ("elevation of privilege", "remote code execution") for m in msrc):
        return None
    m = NON_WORDING.search(text)
    if m and not NEGATION.search(text[max(0, m.start() - 40) : m.start()]):
        return _hit(label, f"{ls}.non_wording", "nvd_description" if desc else "msrc_title", text, m)
    if msrc and all(x in NON_MSRC for x in msrc):
        return {"label": label, "rule": f"{ls}.non_msrc_impact", "source": "msrc_impact", "span": texts["msrc_impact"][:300]}
    return None


# ---- r2 (decided 16 Nov 2026, OQ-P4-3): hedge demotion layered on top of r1; r1 above is unchanged ----
RULES_VERSION_R2 = "r2"
# "could allow" is deliberately not here: it is the standard NVD/MSRC way to state an effect, not a doubt about it.
HEDGE_R2 = re.compile(r"\b(?:probably|possible|possibly|likely|might|perhaps|potentially|theoretically)\b", I)


def _find(texts: dict, source: str, span: str) -> tuple[str, int] | None:
    """The text holding an evidence span (whitespace-normalised) and the span's start offset in it."""
    t = texts.get(source, "")
    i = t.find(span)
    if i >= 0:
        return t, i
    norm = " ".join(t.split())
    j = norm.find(" ".join(span.split()))
    return (norm, j) if j >= 0 else None


# Stock NVD/Debian phrase "might/may/could allow remote attackers to ...": not a hedge (decided 16 Nov 2026, v2.1).
STOCK_ALLOW = re.compile(r"\b(?:might|may|could)\s+allow\b", I)


def _hedged(text: str, start: int, end: int) -> bool:
    """A hedge word inside the effect phrase or earlier in its sentence, at most 70 characters before it."""
    before = STOCK_ALLOW.sub(" ", re.split(r"[.;]\s", text[max(0, start - 70) : start])[-1])
    return bool(HEDGE_R2.search(before) or HEDGE_R2.search(STOCK_ALLOW.sub(" ", text[start:end])))


def r2_demotes(item: dict, label: str, evidence: list[dict]) -> bool:
    """True when r2 demotes a positive label to UNKNOWN: every evidence span is hedged and no unhedged
    explicit wording (MSRC Impact value, or an r1 phrase match) exists elsewhere in the texts."""
    texts = item["texts"]
    if label not in POSITIVE or not evidence:
        return False
    msrc = [x.strip().lower() for x in texts.get("msrc_impact", "").split(";") if x.strip()]
    if label == "RCE_CONFIRMED" and "remote code execution" in msrc:
        return False
    spans = []
    for e in evidence:
        loc = _find(texts, e.get("source", ""), e.get("span", "")) if e.get("span") else None
        if loc is None:
            return False  # cannot place the span: keep the label, stage 2 already validated it verbatim
        spans.append((loc[0], loc[1], loc[1] + len(e["span"])))
    if not all(_hedged(t, s, e) for t, s, e in spans):
        return False
    pat = EXEC if label == "RCE_CONFIRMED" else ADMIN
    for src in ("nvd_description", "msrc_title", "msrc_faq"):
        t = texts.get(src, "")
        for m in pat.finditer(t):
            if not _hedged(t, m.start(), m.end()) and not NEGATION.search(t[max(0, m.start() - 70) : m.start()]):
                return False
    return True
