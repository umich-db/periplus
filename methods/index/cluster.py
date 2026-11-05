"""Shared, seeded clustering used by BOTH merge (over category labels) and pack (over
subjects). One routine, so there is nothing new to test and nothing to drift.

partition(vectors, f, seed): split rows into clusters each with <= f members, using
k = ceil(n / f) seeded MiniBatchKMeans, then a ceiling (recursively split any cluster
with more than f items on its own embedding boundary) and a floor (merge any cluster
smaller than about f/2 into its nearest neighbor by centroid similarity, as long as the
neighbor stays <= f). No LLM, no tuned threshold. Fully deterministic given the seed.

Vocabulary: root, category, subject, field, source document.
"""
import math
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _floor_merge(vectors, clusters, f):
    """Merge clusters smaller than f/2 into the nearest cluster by centroid cosine,
    provided the target stays <= f. Deterministic."""
    floor = max(1, f // 2)
    while len(clusters) > 1:
        cents = np.array([vectors[c].mean(axis=0) for c in clusters])
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        # smallest cluster first (stable order)
        order = sorted(range(len(clusters)), key=lambda i: (len(clusters[i]), i))
        merged = False
        for i in order:
            if len(clusters[i]) >= floor:
                break
            sims = cents @ cents[i]
            cand = [(sims[j], -abs(j - i), j) for j in range(len(clusters))
                    if j != i and len(clusters[j]) + len(clusters[i]) <= f]
            if cand:
                j = max(cand)[2]
                clusters[j] = clusters[j] + clusters[i]
                del clusters[i]
                merged = True
                break
        if not merged:
            break
    return clusters


def partition(vectors, f, seed, _depth=0):
    """Return list[list[int]] of row indices, each cluster with <= f members."""
    vectors = np.asarray(vectors, dtype="float32")
    n = len(vectors)
    if n <= f:
        return [list(range(n))]
    k = max(2, math.ceil(n / f))
    labels = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(vectors)
    clusters = [c for c in ([i for i in range(n) if labels[i] == g] for g in range(k)) if c]
    # ceiling: recursively split clusters that still exceed f on their own boundary
    out = []
    for c in clusters:
        if len(c) > f and _depth < 6:
            for sub in partition(vectors[c], f, seed, _depth + 1):
                out.append([c[i] for i in sub])
        else:
            out.append(c)
    # floor: fold tiny clusters into their nearest neighbor
    return _floor_merge(vectors, out, f)
