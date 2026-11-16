"""N4: can graph statistics tell the topology archetypes apart? (D28 Round 6, protocol P0c, classifier P0d)

The directed host reach graph of a generated network has one node per host, the internet node 0 included, and
an edge u -> v when host u can open a connection to some service of host v: same subnet, or the firewall
direction (subnet(u), subnet(v)) allows one of v's service families (the generator's reach rule, OQ-P6-2,
read here from the finished document only). The internet runs nothing, so no edge enters node 0.

Statistics (Round 6, P0c/P0d), in FEATURES order:
- density |E| / (N (N - 1));
- in- and out-degree mean, population variance and skewness m3 / m2^1.5 (0 when the variance is 0);
- BFS hops from the internet node over directed edges, internet included at 0: mean, population variance, max;
- on the undirected version: diameter, and the average local clustering coefficient (a node with fewer than
  two neighbours counts 0).
"Connected" = every host reachable from node 0 in the directed graph; otherwise DisconnectedNetwork (P0d).

Classifier (P0d): a random forest only (500 trees, random_state 0), 5-fold stratified CV per size; PASS if the
mean out-of-fold balanced accuracy over the folds is >= 0.80. Fold assignment shuffles with random_state 0
(OQ-P7-9).
"""

from __future__ import annotations

from collections import deque

FEATURES = ("density", "in_mean", "in_var", "in_skew", "out_mean", "out_var", "out_skew",
            "hops_mean", "hops_var", "hops_max", "diameter", "clustering")
PASS_THRESHOLD = 0.80
TREES, RANDOM_STATE, FOLDS = 500, 0, 5


class DisconnectedNetwork(ValueError):
    """A host is not reachable from the internet node (Round 6: connectivity holds by construction)."""


def reach_graph(doc: dict) -> tuple[list[int], set[tuple[int, int]]]:
    """(nodes, directed edges) of a generated task document's network."""
    net = doc["network"]
    hosts = {h["id"]: h for h in net["hosts"]}
    allow = {(f["from"], f["to"]): set(f["allow"]) for f in net["firewall"]}
    edges = set()
    for u, hu in hosts.items():
        for v, hv in hosts.items():
            if u == v or not hv["families"]:
                continue
            if hu["subnet"] == hv["subnet"] or allow.get((hu["subnet"], hv["subnet"]), set()) & set(hv["families"]):
                edges.add((u, v))
    return sorted(hosts), edges


def moments(xs: list[float]) -> tuple[float, float, float]:
    """(mean, population variance, skewness m3 / m2^1.5); skewness is 0 when the variance is 0 (P0d)."""
    n = len(xs)
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs) / n
    m3 = sum((x - mean) ** 3 for x in xs) / n
    return mean, m2, (m3 / m2 ** 1.5 if m2 > 0 else 0.0)


def _bfs(adj: dict[int, set[int]], src: int) -> dict[int, int]:
    dist, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def graph_stats(nodes: list[int], edges: set[tuple[int, int]], source: int = 0) -> dict[str, float]:
    n = len(nodes)
    out = {u: set() for u in nodes}
    und = {u: set() for u in nodes}
    for u, v in edges:
        out[u].add(v)
        und[u].add(v)
        und[v].add(u)
    hops = _bfs(out, source)
    if len(hops) != n:
        raise DisconnectedNetwork(f"hosts {sorted(set(nodes) - set(hops))} are unreachable from the internet")
    indeg = [sum((u, v) in edges for u in nodes) for v in nodes]
    outdeg = [len(out[u]) for u in nodes]
    hm, hv, _ = moments([hops[u] for u in nodes])
    diameter = max(max(_bfs(und, u).values()) for u in nodes)  # undirected and connected, so finite
    clus = []
    for u in nodes:
        nb = sorted(und[u])
        k = len(nb)
        if k < 2:
            clus.append(0.0)
            continue
        links = sum(nb[j] in und[nb[i]] for i in range(k) for j in range(i + 1, k))
        clus.append(links / (k * (k - 1) / 2))
    im, iv, isk = moments(indeg)
    om, ov, osk = moments(outdeg)
    return {"density": len(edges) / (n * (n - 1)), "in_mean": im, "in_var": iv, "in_skew": isk,
            "out_mean": om, "out_var": ov, "out_skew": osk, "hops_mean": hm, "hops_var": hv,
            "hops_max": float(max(hops.values())), "diameter": float(diameter), "clustering": sum(clus) / n}


def doc_stats(doc: dict) -> dict[str, float]:
    return graph_stats(*reach_graph(doc))


def classify(rows: list[dict]) -> dict:
    """rows: {"archetype", "n", **FEATURES}. Per size: fold balanced accuracies, their mean, PASS/FAIL."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    out = {}
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        X = np.array([[r[f] for f in FEATURES] for r in rs], dtype=float)
        labels = sorted({r["archetype"] for r in rs})
        per_class = {a: sum(r["archetype"] == a for r in rs) for a in labels}
        if len(labels) < 2 or min(per_class.values()) < FOLDS:  # stratified 5-fold needs 5 per class
            out[str(n)] = {"networks": per_class, "verdict": "NOT_ASSESSABLE", "mean_balanced_accuracy": None,
                           "fold_balanced_accuracy": [], "pooled_oof_balanced_accuracy": None}
            continue
        y = np.array([labels.index(r["archetype"]) for r in rs])
        folds, oof = [], np.full(len(y), -1)
        for train, test in StratifiedKFold(FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y):
            rf = RandomForestClassifier(n_estimators=TREES, random_state=RANDOM_STATE).fit(X[train], y[train])
            pred = rf.predict(X[test])
            oof[test] = pred
            folds.append(float(balanced_accuracy_score(y[test], pred)))
        mean = sum(folds) / len(folds)
        out[str(n)] = {"networks": {a: int((y == i).sum()) for i, a in enumerate(labels)}, "fold_balanced_accuracy": folds,
                       "mean_balanced_accuracy": mean, "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(y, oof)),
                       "verdict": "PASS" if mean >= PASS_THRESHOLD else "FAIL"}
    return out
