"""Topic/event-based 70/15/15 splits (Phase 2).

A random split lets near-identical stories about one event land in train and
test, so the model is graded on topics it has already seen. Splitting whole
groups (one event, or one topic cluster) keeps every story about that topic on
one side. Test then measures how well the model handles *new* events, which is
what it faces in deployment.

Groups come from:
- MediaEval: the released `event_name` (the benchmark's own split is by event).
- Everything else: TF-IDF -> LSA (TruncatedSVD, 100 dims) -> k-means topic
  clusters over the cleaned text. LSA matters: k-means on raw sparse TF-IDF put
  51% of AVeriTeC into one cluster and left most others as singletons, which
  made "topic split" a random split in disguise.
  Fakeddit's `subreddit` is deliberately NOT used: its labels were assigned per
  subreddit, so splitting on it would put whole classes into test only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)
SEED = 42
LSA_DIMS = 100
FULL_KMEANS_MAX_ROWS = 50_000  # above this, MiniBatchKMeans for speed


def n_topics_for(n_rows: int) -> int:
    """About 20 items per topic, between 10 and 200 topics (never more than rows)."""
    return max(min(n_rows, 10), min(200, n_rows // 20))


def assign_topics(texts: list[str], seed: int = SEED) -> np.ndarray:
    """Cluster texts into topics; returns one integer topic id per text.

    Falls back to one group per row when the text carries no vocabulary
    (e.g. image-only rows), which degrades gracefully to a random split.
    """
    n = len(texts)
    k = n_topics_for(n)
    if n <= 1:
        return np.zeros(n, dtype=int)
    try:
        X = TfidfVectorizer(stop_words="english", max_features=50_000, min_df=1, sublinear_tf=True).fit_transform(
            texts
        )
    except ValueError:  # empty vocabulary
        return np.arange(n)
    if k >= n:
        return np.arange(n)
    dims = min(LSA_DIMS, X.shape[1] - 1, n - 1)
    if dims >= 2:
        # Dense, L2-normalised LSA space: Euclidean k-means there ~ cosine clustering.
        X = normalize(TruncatedSVD(n_components=dims, random_state=seed).fit_transform(X))
    if n <= FULL_KMEANS_MAX_ROWS:
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
    else:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)
    return km.fit_predict(X)


def group_split(groups: pd.Series, ratios=RATIOS, seed: int = SEED) -> pd.Series:
    """Assign every group wholesale to train/val/test, hitting `ratios` by row count.

    Groups are visited in a seeded random order; each goes to the split that is
    currently furthest below its target size. Random order (not largest-first)
    keeps test from being made only of small, niche topics.
    """
    sizes = groups.value_counts()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    total = len(groups)
    targets = np.array(ratios) * total
    filled = np.zeros(len(ratios))
    assignment: dict = {}
    for g in order:
        i = int(np.argmax(targets - filled))
        assignment[g] = SPLITS[i]
        filled[i] += sizes[g]
    return groups.map(assignment)
