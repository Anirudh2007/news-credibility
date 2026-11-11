"""Exact and near-duplicate removal (Phase 2).

Why: if the same story (or a lightly reworded copy) sits in both train and
test, the model is graded on something it memorised. Test scores go up, real
performance does not. This is called train/test leakage.

Near-duplicates are found with TF-IDF cosine similarity: two texts whose word
and bigram weight vectors point the same way (cosine >= threshold) are treated
as one story. This plays the role MinHash plays at larger scale, without an
extra dependency; it compares rows in chunks so memory stays bounded.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

NEAR_DUP_THRESHOLD = 0.9
_CHUNK = 2000
_BIG_SET = 20_000
_CHUNK_BIG = 300


def _dedup_key(text: str) -> str:
    """Case, punctuation and spacing ignored: 'Hello, World!' == 'hello world'."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def drop_exact_duplicates(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Keep the first row of each normalised text. Empty texts are never merged
    (image-only rows share the empty string but are different items)."""
    keys = df[text_col].map(_dedup_key)
    dup = keys.duplicated(keep="first") & (keys != "")
    return df.loc[~dup]


def drop_near_duplicates(
    df: pd.DataFrame, text_col: str = "text", threshold: float = NEAR_DUP_THRESHOLD
) -> pd.DataFrame:
    """Drop row j if an earlier row i has cosine(i, j) >= threshold."""
    texts = df[text_col].tolist()
    nonempty = np.array([bool(t.strip()) for t in texts])
    idx = np.flatnonzero(nonempty)
    if len(idx) < 2:
        return df
    # Large sets: ignore words in over half the rows. They add almost nothing to
    # similarity but make the sparse product dense and the memory use explode.
    big = len(idx) > _BIG_SET
    try:
        X = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_df=0.5 if big else 1.0
        ).fit_transform([texts[i] for i in idx])
    except ValueError:  # empty vocabulary (e.g. only stop words)
        return df

    chunk = _CHUNK_BIG if big else _CHUNK
    drop = np.zeros(len(idx), dtype=bool)
    for start in range(0, len(idx), chunk):
        stop = min(start + chunk, len(idx))
        sims = (X[start:stop] @ X[:stop].T).tocoo()  # rows are L2-normalised: dot = cosine
        for r, c, v in zip(sims.row, sims.col, sims.data):
            j = start + r
            if c < j and v >= threshold and not drop[c]:
                drop[j] = True
    keep = np.ones(len(df), dtype=bool)
    keep[idx[drop]] = False
    return df.loc[keep]
