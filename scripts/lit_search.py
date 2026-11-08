"""Run the Phase-1 literature queries of docs/related_work_v2.md §1 against Semantic Scholar and arXiv.

Usage: python scripts/lit_search.py

For each query: the Semantic Scholar Graph API paper search (`total` and the first 20 hits) and the arXiv API
(`opensearch:totalResults` and the first 20 hits). Raw responses are stored under
data/raw/sources/<date>/lit/ (gitignored); counts and hit lists go to results/lit_search/<stamp>-<commit>/ through
ahg.runlog. Google Scholar is not queried by script (its terms do not allow automated queries).
"""

import json
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ahg import runlog  # noqa: E402

RAW = ROOT / "data" / "raw" / "sources" / date.today().isoformat() / "lit"

QUERIES = {  # id: (topic, query string)
    "Q1": ("T1 Stackelberg planning / mitigation", "Stackelberg planning"),
    "Q2": ("T1 Stackelberg planning / mitigation", "penetration testing mitigation analysis planning"),
    "Q3": ("T1 Stackelberg planning / mitigation", "Stackelberg planning heuristic search leader follower"),
    "Q4": ("T2 learned follower models in defender search", "Stackelberg planning neural network"),
    "Q5": ("T2 learned follower models in defender search", "learned attacker model defender optimization attack graph"),
    "Q6": ("T2 learned follower models in defender search", "bilevel optimization neural network follower value function"),
    "Q7": ("T2 learned follower models in defender search", "network interdiction neural network learning"),
    "Q8": ("T2 learned follower models in defender search", "Active Directory defense graph neural network"),
    "Q9": ("T2 learned follower models in defender search", "honeypot placement Active Directory attack graph"),
    "Q10": ("T3 attack planning with delete effects / access loss", "attack graph non-monotonic privileges"),
    "Q11": ("T3 attack planning with delete effects / access loss", "penetration testing planning dynamic network"),
    "Q12": ("T4 learned heuristics / learning from search", "learning heuristic functions classical planning"),
    "Q13": ("T4 learned heuristics / learning from search", "bootstrap learning heuristic search planning"),
    "Q14": ("T5 AD attack paths and defence", "Active Directory attack path"),
    "Q15": ("T5 AD attack paths and defence", "Active Directory attack graph edge blocking"),
}


def get(url: str) -> bytes:
    r = subprocess.run(["curl", "-sS", "-L", "--max-time", "60", url], capture_output=True)
    return r.stdout


def s2(q: str, qid: str) -> dict:
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?" +
           urllib.parse.urlencode({"query": q, "limit": 20, "fields": "title,year,venue,externalIds,citationCount"}))
    for attempt in range(6):  # unauthenticated API: back off on 429
        body = get(url)
        try:
            d = json.loads(body)
        except json.JSONDecodeError:
            d = {}
        if "total" in d:
            (RAW / f"{qid}_s2.json").write_bytes(body)
            return {"url": url, "total": d["total"], "hits": [
                {"title": p["title"], "year": p.get("year"), "venue": p.get("venue"),
                 "arxiv": (p.get("externalIds") or {}).get("ArXiv"), "doi": (p.get("externalIds") or {}).get("DOI")}
                for p in d.get("data", [])]}
        time.sleep(5 * (attempt + 1))
    return {"url": url, "total": None, "hits": [], "error": body[:200].decode("utf-8", "replace")}


def arxiv(q: str, qid: str) -> dict:
    terms = " AND ".join(f"all:{w}" for w in q.split())
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": terms, "start": 0, "max_results": 20})
    body = get(url)
    (RAW / f"{qid}_arxiv.xml").write_bytes(body)
    t = body.decode("utf-8", "replace")
    tot = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", t)
    hits = []
    for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
        title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip()
        aid = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1)
        hits.append({"title": title, "arxiv": aid, "year": re.search(r"<published>(\d{4})", e).group(1)})
    return {"url": url, "total": int(tot.group(1)) if tot else None, "hits": hits}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for qid, (topic, q) in QUERIES.items():
        out[qid] = {"topic": topic, "query": q, "semantic_scholar": s2(q, qid), "arxiv": arxiv(q, qid)}
        time.sleep(3.5)  # arXiv asks for at least 3 s between calls
        print(qid, out[qid]["semantic_scholar"]["total"], out[qid]["arxiv"]["total"], flush=True)
    rep = ["# Phase-1 literature queries (docs/related_work_v2.md §1)", "",
           f"Run date {date.today().isoformat()}. Totals as returned by each API; hit lists are the first 20.", "",
           "| id | topic | query | Semantic Scholar total | arXiv total |", "|---|---|---|---|---|"]
    rep += [f"| {k} | {v['topic']} | `{v['query']}` | {v['semantic_scholar']['total']} | {v['arxiv']['total']} |"
            for k, v in out.items()]
    for k, v in out.items():
        rep += ["", f"## {k}: `{v['query']}`", "", "Semantic Scholar:", ""]
        rep += [f"- {h['title']} ({h.get('year')}, {h.get('venue') or '—'}; arXiv {h.get('arxiv') or '—'})"
                for h in v["semantic_scholar"]["hits"]]
        rep += ["", "arXiv:", ""] + [f"- {h['title']} ({h['year']}; {h['arxiv']})" for h in v["arxiv"]["hits"]]
    path = runlog.write_run("lit_search", out, "\n".join(rep) + "\n", [], {"queries": QUERIES}, root=ROOT / "results")
    print(path)


if __name__ == "__main__":
    main()
