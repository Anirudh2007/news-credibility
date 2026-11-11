"""Phase 4: build the meta-learner table, train LightGBM, calibrate, evaluate.

Stacking without leakage
    Channel models (3A-3G) were trained on the Phase 2 *train* split. If the
    meta-learner were trained on channel outputs for those same items, it would
    see over-confident, memorised scores and learn to trust them too much.
    So: channel outputs on *val* items train the meta-learner (with grouped
    cross-validation inside val for tuning and calibration), and *test* items
    are used once, for the final evaluation.

Steps
    build : run the pipeline (same code as the API) over val + test items of all
            four datasets -> data/processed/meta/meta_features_v2.parquet (resumable)
    fit   : LightGBM, objective cross_entropy (accepts the 0.5 soft labels),
            sample weights that balance datasets and classes, monotone constraints
            on the check scores, additive across checks (interaction constraints),
            side features unused (UNUSED_FEATURES), modality dropout (every val item
            with an image or a source URL is also trained without it), small grid
            tuned by 5-fold GroupKFold on topic groups and chosen by the mean
            per-dataset ROC-AUC, isotonic calibration on out-of-fold predictions,
            refit, save models/meta_lgbm.txt + models/meta_calibrator.joblib.
            Ablation flags: --no-dropout, --interactions, --all-features.
    eval  : on test: ROC-AUC, F1, Brier, ECE (overall and per dataset), equal-weight
            baseline, reliability diagram, fairness (text-only vs multimodal
            credible items, mean gap + KS test), missing-modality test,
            inference-time channel ablation.

Run from `news-credibility/`:
    python -m src.fusion.train build [--batch-size 32] [--max-per-split N] [--channels claims source ...]
    python -m src.fusion.train fit

Build cost: `build` runs the full val + test sets by default (~26k meta-eligible
items). Items go through `Pipeline.analyze_batch` in groups of `--batch-size`: the
DeBERTa scorer, MiniLM query encoder, cross-encoder, DeBERTa verifier and CLIP each
run once per group on all of its inputs, then each item is assembled by the same
per-item code the API uses. `--max-per-split N` downsamples (shared across datasets,
cached items first). Interrupt any time: it checkpoints every 256 items and resumes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.fusion.features import CHANNEL_FEATURES, CHANNEL_SCORE, CHANNELS, FEATURE_COLUMNS, add_derived
from src.fusion.meta import MetaModel, average_channel_scores, band, to_score

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
META_DIR = PROCESSED / "meta"
# v3 (2026-11-02): 1,082 rated source domains (source model and verifier passage weights), default evidence
# index, live lookup off. v2 (before) and v1 are kept untouched. v3 also holds a column of a wording-based check
# that was removed on 2026-11-07; it is not in FEATURE_COLUMNS, so `fit` ignores it.
# The Indian index is not used here: MultiClaim passages have no dates, so the `before=` filter could
# not stop fact-checks of the benchmark claims themselves from being retrieved.
# v4 (2026-11-07, D11): the same items through the v0.9 pipeline, so claim matching is on (an answer or
# fact-check about a different claim does not count) as in serving. A meta-learner refitted on it tied on the
# benchmark but lost on the India dev set (ROC-AUC 0.960 -> 0.908): confirmations are rare in v4, so it learned a
# weak claim effect (src/fusion/meta_compare.py, reports/meta_retrain_comparison.json). v3 stays the default;
# NEWSCRED_META_TABLE picks another table.
TABLE_PATH = Path(os.environ.get("NEWSCRED_META_TABLE") or META_DIR / "meta_features_v3.parquet")
REPORTS = Path(os.environ.get("NEWSCRED_META_REPORTS") or ROOT / "reports")
DATASETS = ["averitec", "fakeddit", "fakenewsnet", "mediaeval"]
SEED = 42
PARAM_GRID = [
    {"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
    for nl in (7, 15, 31) for lr in (0.03, 0.1) for mcs in (20, 100)
]
BASE_PARAMS = dict(
    objective="cross_entropy", n_estimators=300, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
)
# +1: a higher value may only raise the score; -1: only lower it; others free.
# Stops the trees from learning a backwards rule that holds in one dataset by accident
# (e.g. "verifier says Supported -> less credible"), and makes every channel's effect
# explainable in one sentence.
MONOTONE = {
    "claim_support_ratio": 1, "claim_refute_ratio": -1, "claim_score_mean": 1, "claim_score_min": 1,
    "claim_evidence_score": 1, "text_image_score": 1, "reuse_score": 1, "source_score": 1, "source_reliability": 1,
    "clickbait_hits": -1, "temporal_score": 1,
}
# Features that exist only because an image was supplied: blanked by image dropout
# (training) and by the missing-modality test (evaluation). The earliest-copy gap
# comes from the reverse image search (3E), so it goes with the image.
IMAGE_FEATURES = [*CHANNEL_FEATURES["text_image"], *CHANNEL_FEATURES["reuse"], "gap_earliest_copy_days"]
# Inputs a user may leave out. Dropout trains on copies of rows with each one removed.
DROPOUT_GROUPS = {"image": IMAGE_FEATURES, "source": CHANNEL_FEATURES["source"]}


# --- Build ---------------------------------------------------------------------

MAX_PER_SPLIT = None  # full val + test sets by default; --max-per-split N downsamples (shared across datasets)
BATCH_SIZE = 32  # items per Pipeline.analyze_batch call; model calls inside are batched further


def _quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` equally across datasets; a dataset with fewer rows than its share
    passes the remainder to the others ("water-filling"). Deterministic."""
    q = {d: 0 for d in sizes}
    remaining = min(total, sum(sizes.values()))
    open_ = sorted(d for d, n in sizes.items() if n > 0)
    while remaining > 0 and open_:
        share = max(1, remaining // len(open_))
        for d in list(open_):
            take = min(share, sizes[d] - q[d], remaining)
            q[d] += take
            remaining -= take
            if q[d] >= sizes[d]:
                open_.remove(d)
            if remaining == 0:
                break
    return q


def load_items(
    splits: Sequence[str] = ("val", "test"),
    max_per_split: int | None = MAX_PER_SPLIT,
    max_per_dataset: int | None = None,
    seed: int = SEED,
    prefer_ids: set[str] | frozenset = frozenset(),
) -> pd.DataFrame:
    """Meta-eligible rows (credible not NaN, so AVeriTeC NEE excluded), downsampled.

    max_per_split: total per split, shared equally across datasets (None = no cap).
    max_per_dataset: optional extra cap per (dataset, split).
    prefer_ids: item_ids already in the cache; they are taken first, so a smaller
        sample reuses finished work instead of drawing mostly new items.
    """
    frames = []
    for name in DATASETS:
        path = PROCESSED / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["split"].isin(splits) & df["meta_eligible"] & df["credible"].notna()].copy()
        df["dataset"] = name
        for col in ("source_domain", "tweet_ids", "media_id", "date", "has_video"):
            if col not in df:
                df[col] = None
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)

    picked = []
    for _, g_split in pool.groupby("split"):
        groups = dict(tuple(g_split.groupby("dataset")))
        sizes = {d: len(g) if not max_per_dataset else min(len(g), max_per_dataset) for d, g in groups.items()}
        quotas = _quotas(sizes, max_per_split) if max_per_split else sizes
        for d, g in groups.items():
            n = quotas[d]
            cached = g[g["item_id"].isin(prefer_ids)]
            if len(cached) >= n:
                picked.append(cached.sample(n=n, random_state=seed))
            else:
                rest = g[~g["item_id"].isin(prefer_ids)]
                picked += [cached, rest.sample(n=n - len(cached), random_state=seed)]
    return pd.concat(picked, ignore_index=True)


def _image_for(row, mediaeval_files: dict) -> object:
    from PIL import Image

    from src.media.text_image import IMAGE_DIR as FAKEDDIT_IMAGES

    path = None
    if row["dataset"] == "fakeddit":
        path = FAKEDDIT_IMAGES / f"{row['item_id']}.jpg"
    elif row["dataset"] == "mediaeval" and _s(row.get("media_id")):
        path = mediaeval_files.get(_s(row["media_id"]).split(",")[0].strip())
    if path is None or not Path(path).exists():
        return None
    with Image.open(path) as im:
        return im.convert("RGB")


def _s(value) -> str | None:
    """Parquet gives NaN (a float) for missing strings; NewsItem expects None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    value = str(value).strip()
    return value or None


def item_from_row(row, mediaeval_files: dict):
    from src.pipeline import NewsItem

    image = _image_for(row, mediaeval_files)
    first_media = (_s(row.get("media_id")) or "").split(",")[0].strip() or None
    return NewsItem(
        text=_s(row["text"]),
        image=image,
        source_url=_s(row.get("source_domain")),
        publish_date=_s(row.get("date")),
        claimed_event=_s(row["topic_group"]) if row["dataset"] == "mediaeval" else None,
        has_video=row.get("has_video") is True or row.get("has_video") == 1,
        # tweet_ids deliberately not passed: the API never receives tweet ids, so training
        # rows must not have them either (the source model's top feature was the tweet count).
        media_id=first_media,  # excluded from reuse matching: an image always matches itself
    )


def build_table(
    channels: Sequence[str] = CHANNELS,
    splits: Sequence[str] = ("val", "test"),
    max_per_split: int | None = MAX_PER_SPLIT,
    max_per_dataset: int | None = None,
    force: bool = False,
    checkpoint_every: int = 256,
    pipeline=None,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Run the pipeline over sampled items and store one feature row each.

    Cache: rows are keyed by (item_id, channel set). Rows already computed for the
    same channel set are reused and never recomputed (unless `force`); rows built
    with a different channel set stay in the file untouched. Returns all rows for
    this channel set (cached + new), which can exceed the current sample.
    """
    from src.media.reuse import image_files
    from src.pipeline import Pipeline

    key = ",".join(sorted(channels))
    cache = pd.read_parquet(TABLE_PATH) if TABLE_PATH.exists() else pd.DataFrame(columns=["item_id", "channels"])
    other = cache[cache["channels"] != key]  # other channel sets: kept on disk as they are
    done = cache[cache["channels"] == key] if not force else cache.iloc[0:0]
    done_ids = set(done["item_id"])

    items = load_items(splits, max_per_split, max_per_dataset, prefer_ids=done_ids)
    todo = items[~items["item_id"].isin(done_ids)]
    print(f"sample: {len(items)} items ({len(items) - len(todo)} already cached, {len(todo)} to run); "
          f"cache holds {len(done)} rows for channels: {key}")
    print(items.groupby(["split", "dataset"]).size().unstack(fill_value=0).to_string())

    pipeline = pipeline or Pipeline(components={"factcheck": None, "news": None, "wiki": None, "memory": None})  # live lookups off: it would find the datasets' own fact-checks
    files = image_files()
    rows = []
    META_DIR.mkdir(parents=True, exist_ok=True)

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            pd.concat([other, done], ignore_index=True).to_parquet(TABLE_PATH, index=False)
            rows = []

    batched = hasattr(pipeline, "analyze_batch")
    t0, done_n, last_flush = time.perf_counter(), 0, 0
    for start in range(0, len(todo), batch_size):
        chunk = todo.iloc[start:start + batch_size]
        items = [item_from_row(r, files) for _, r in chunk.iterrows()]
        if batched:
            reports = pipeline.analyze_batch(items, channels=channels)
        else:
            reports = [pipeline.analyze(it, channels=channels) for it in items]
        for (_, r), report in zip(chunk.iterrows(), reports):
            feats = {c: (np.nan if report["features"][c] is None else report["features"][c]) for c in FEATURE_COLUMNS}
            rows.append({
                "item_id": r["item_id"], "dataset": r["dataset"], "split": r["split"], "topic_group": r["topic_group"],
                "credible": float(r["credible"]), "channels": key, **feats,
            })
        del items, reports  # drop the batch's images before the next one
        done_n += len(chunk)
        if done_n - last_flush >= checkpoint_every:
            flush()
            release_memory()
            last_flush = done_n
            rate = done_n / (time.perf_counter() - t0)
            print(f"  {done_n}/{len(todo)}  {rate:.1f} items/s  ETA {(len(todo) - done_n) / rate / 60:.0f} min", flush=True)
    flush()
    release_memory()
    print("component status:", json.dumps(pipeline.status, indent=1))
    return done


def release_memory() -> None:
    """At checkpoints only: free Python garbage and give cached CUDA blocks back.
    Not after every batch: emptying the cache forces the next batch to re-allocate
    the same blocks from the driver, which is slower and does not reduce
    fragmentation inside PyTorch's allocator."""
    gc.collect()
    torch = sys.modules.get("torch")  # only if a model already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main_channel_set(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of the channel set with the most val rows, so `fit` never mixes rows
    built with different channels (their NaN patterns differ)."""
    if table["channels"].nunique() <= 1:
        return table
    counts = table[table["split"] == "val"]["channels"].value_counts()
    key = counts.index[0]
    print(f"meta table holds {table['channels'].nunique()} channel sets; fitting on '{key}' ({counts.iloc[0]} val rows)")
    return table[table["channels"] == key].reset_index(drop=True)


def blank_source_without_domain(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Same rule as the pipeline: the source check runs only when the item has a URL/domain.
    Rows built before that rule scored headline style alone; their source features become
    NaN here, so training matches serving. Returns (table, rows blanked per dataset)."""
    from src.source.credibility import parse_domain

    domains = []
    for name in DATASETS:
        path = PROCESSED / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["item_id", "source_domain"])
            domains.append(df)
    if not domains:
        return table, {}
    dom = pd.concat(domains).drop_duplicates("item_id").set_index("item_id")["source_domain"]
    has = table["item_id"].map(dom).map(lambda d: isinstance(d, str) and parse_domain(d)[0] is not None)
    blank = ~has.fillna(False).to_numpy(bool)
    table = table.copy()
    table.loc[blank, CHANNEL_FEATURES["source"]] = np.nan
    return table, table.loc[blank, "dataset"].value_counts().to_dict()


def recompute_source(table: pd.DataFrame, mode: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Source columns recomputed from each item's domain and text with the serving SourceChecker
    (cheap: no model calls), so a change to the source check needs a refit, not an 80-minute rebuild.
    mode: "rating" (deployed since 2026-11-05) or "classifier" (the old FakeNewsNet model, for the
    comparison). Returns (table, rows with a source score per dataset)."""
    from src.fusion.features import source_features
    from src.source.credibility import SourceChecker, parse_domain

    frames = [pd.read_parquet(PROCESSED / f"{n}.parquet", columns=["item_id", "text", "source_domain"])
              for n in DATASETS if (PROCESSED / f"{n}.parquet").exists()]
    if not frames:
        return table, {}
    info = pd.concat(frames).drop_duplicates("item_id").set_index("item_id")
    checker = SourceChecker(mode=mode)
    cols = CHANNEL_FEATURES["source"]
    values = np.full((len(table), len(cols)), np.nan)
    for i, item_id in enumerate(table["item_id"]):
        if item_id not in info.index:
            continue
        dom, text = info.at[item_id, "source_domain"], info.at[item_id, "text"]
        if not (isinstance(dom, str) and parse_domain(dom)[0] is not None):
            continue
        f = source_features(checker.score(dom, text if isinstance(text, str) and text.strip() else None))
        values[i] = [f.get(c, np.nan) for c in cols]
    table = table.copy()
    table[cols] = values
    scored = ~np.isnan(values[:, 0])
    return table, table.loc[scored, "dataset"].value_counts().to_dict()


def _hard_classes(s: pd.Series) -> int:
    return s[s != 0.5].nunique()


def regroup_one_class_test(table: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """A dataset whose test rows hold only one class cannot be scored (ROC-AUC/F1
    undefined). Phase 2 split MediaEval by event, and its test events happened to be
    all one class. Fix, for the meta-learner only: pool that dataset's val + test
    topic groups and re-deal them (whole groups, stratified by class, same test share).
    Both sides were unseen by the channel models, so swapping them leaks nothing."""
    from sklearn.model_selection import StratifiedGroupKFold

    table, changes = table.copy(), {}
    for d in table["dataset"].unique():
        sel = (table["dataset"] == d) & table["split"].isin(["val", "test"])
        sub = table[sel]
        test_share = (sub["split"] == "test").mean()
        if _hard_classes(sub.loc[sub["split"] == "test", "credible"]) > 1 or _hard_classes(sub["credible"]) < 2 \
                or not 0 < test_share < 1:
            continue
        n_splits = max(2, int(round(1 / test_share)))
        y = (sub["credible"] >= 0.5).astype(int)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, test_idx = next(cv.split(sub, y, sub["topic_group"]))
        new = pd.Series("val", index=sub.index)
        new.iloc[test_idx] = "test"
        table.loc[sub.index, "split"] = new
        changes[d] = {"test_rows_before": int((sub["split"] == "test").sum()), "test_rows_after": int(len(test_idx)),
                      "test_credible_share_after": float(sub["credible"].iloc[test_idx].mean())}
    return table, changes


# --- Fit -------------------------------------------------------------------------

def balance_weights(datasets: Sequence[str], y: Sequence[float]) -> np.ndarray:
    """Each dataset gets equal total weight; inside a dataset, each label value
    (0, 0.5, 1) gets equal weight. Stops the largest dataset (Fakeddit) and the
    majority class from dominating, so the model learns credibility rather than
    "this looks like Fakeddit". Normalised to mean 1."""
    d = pd.DataFrame({"d": list(datasets), "y": np.round(np.asarray(y, dtype=float) * 2) / 2})
    n_d = d["d"].nunique()
    cell = d.groupby(["d", "y"])["y"].transform("size")
    bins_in_d = d.groupby("d")["y"].transform("nunique")
    w = 1.0 / (n_d * bins_in_d * cell)
    return (w / w.mean()).to_numpy()


def soft_log_loss(y, p, w=None) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(ll, weights=w))


def _lgbm(params: dict, seed: int = SEED, additive: bool = True):
    """additive: interaction constraints, one per check, so every tree reads the
    features of a single check. The model's log-odds is then a sum of one learned
    function per check (a GAM built from trees): checks add up evidence and cannot
    team up to recognise *which dataset* an item came from (e.g. "no image + many
    capital letters"), which is what made it fail on AVeriTeC."""
    from lightgbm import LGBMRegressor

    monotone = [MONOTONE.get(c, 0) for c in FEATURE_COLUMNS]
    extra = {}
    if additive:
        extra["interaction_constraints"] = [[FEATURE_COLUMNS.index(c) for c in CHANNEL_FEATURES[ch]] for ch in CHANNELS]
    return LGBMRegressor(**BASE_PARAMS, **params, monotone_constraints=monotone,
                         monotone_constraints_method="advanced", random_state=seed, **extra)


# Kept in the feature vector (reports, explanations) but never split on by the model.
# - input-type flags: an item must never score higher or lower just because it came with an image;
# - descriptive side features whose values mostly say *which dataset* an item looks like
#   (headline capitals, how check-worthy a Reddit title is, ...) rather than whether it is credible.
#   With them the model scored AVeriTeC at ROC-AUC 0.57-0.63 although the verifier alone reaches 0.76.
UNUSED_FEATURES = [
    *CHANNEL_FEATURES["modality"],
    "n_claims", "verifier_confidence_mean", "checkworthiness_max",
    "claim_support_ratio", "claim_refute_ratio", "claim_score_mean", "claim_score_min", "evidence_relevance",
    "text_image_confidence", "clip_similarity",
    "reuse_n_matches", "reuse_date_gap_days", "location_similarity", "ela",
    "capital_ratio", "gap_earliest_copy_days", "gap_publish_days",
]


def model_matrix(df: pd.DataFrame, unused: Sequence[str] | None = None) -> np.ndarray:
    """Feature matrix for training: features in `unused` (default UNUSED_FEATURES) are
    blanked to NaN, so no tree can split on them; at inference they then have no effect."""
    X = np.array(df[FEATURE_COLUMNS], dtype=float)  # a copy: to_numpy() can return a read-only view
    for c in (UNUSED_FEATURES if unused is None else unused):
        X[:, FEATURE_COLUMNS.index(c)] = np.nan
    return X


def image_dropout(tr: pd.DataFrame, groups: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Modality dropout: for each optional input (image, source URL), append a copy of
    every row that has it, with its features blanked (NaN), same label. The model then
    learns what the rest of the item says on its own, so at inference a missing image or
    URL means "check not run", not "evidence against". Copies keep their topic group
    (same CV fold as the original) and are flagged `augmented`."""
    tr = tr.assign(augmented=False)
    copies = []
    for name, cols in (DROPOUT_GROUPS if groups is None else groups).items():
        has = tr[cols].notna().any(axis=1)
        copy = tr[has].copy()
        copy[cols] = np.nan
        if name == "image":
            copy["has_image"] = 0.0
        copy["augmented"] = True
        copies.append(copy)
    return pd.concat([tr, *copies], ignore_index=True)


def mean_dataset_auc(datasets, y, p) -> tuple[float, dict]:
    """Mean of per-dataset ROC-AUCs (datasets with both classes). Unlike one pooled
    number, a small dataset (AVeriTeC) counts as much as Fakeddit."""
    from sklearn.metrics import roc_auc_score

    datasets, y, p = np.asarray(datasets), np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    per = {}
    for d in np.unique(datasets):
        m = (datasets == d) & (y != 0.5)
        if len(set(y[m])) > 1:
            per[str(d)] = float(roc_auc_score(y[m], p[m]))
    return (float(np.mean(list(per.values()))) if per else math.nan), per


def tune_and_oof(X, y, w, groups, grid: list[dict] | None = None, n_splits: int = 5, seed: int = SEED,
                 datasets=None, select_mask=None, additive: bool = True):
    """Grouped CV over the grid. Returns (best params, out-of-fold predictions for
    them, per-config results). Folds split by topic group so near-duplicate stories
    never sit on both sides of a fold.

    Selection: highest mean per-dataset ROC-AUC on the rows in `select_mask`
    (the real, non-augmented rows), ties broken by weighted log loss. Without
    `datasets` it falls back to weighted log loss alone."""
    from sklearn.model_selection import GroupKFold

    grid = grid or PARAM_GRID  # read at call time, not bound at import
    sel = np.ones(len(y), bool) if select_mask is None else np.asarray(select_mask, bool)
    n_splits = max(2, min(n_splits, len(set(groups))))
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    results, best = [], None
    for params in grid:
        oof = np.zeros(len(y))
        for tr, va in folds:
            m = _lgbm(params, seed, additive).fit(X[tr], y[tr], sample_weight=w[tr])
            oof[va] = m.predict(X[va])
        loss = soft_log_loss(y[sel], oof[sel], w[sel])
        auc, per = mean_dataset_auc(np.asarray(datasets)[sel], y[sel], oof[sel]) if datasets is not None else (math.nan, {})
        results.append({**params, "cv_weighted_log_loss": loss, "cv_mean_dataset_auc": auc, "cv_auc_by_dataset": per})
        key = (-auc if not math.isnan(auc) else 0.0, loss)
        if best is None or key < best[1]:
            best = (params, key, oof)
    return best[0], best[2], results


def fit_meta(table: pd.DataFrame, seed: int = SEED, dropout: bool = True, additive: bool = True,
             unused: Sequence[str] | None = None):
    """Train on the val rows of the meta table. Returns (MetaModel, fit report)."""
    from sklearn.isotonic import IsotonicRegression

    tr = table[table["split"] == "val"].reset_index(drop=True)
    if len(tr) < 50:
        raise ValueError(f"only {len(tr)} val rows in the meta table; run `build` first")
    tr = image_dropout(tr) if dropout else tr.assign(augmented=False)
    real = ~tr["augmented"].to_numpy()
    unused = UNUSED_FEATURES if unused is None else list(unused)
    X = model_matrix(tr, unused)
    y = tr["credible"].to_numpy(float)
    w = balance_weights(tr["dataset"], y)
    params, oof, grid = tune_and_oof(X, y, w, tr["topic_group"].to_numpy(), seed=seed,
                                     datasets=tr["dataset"].to_numpy(), select_mask=real, additive=additive)
    # Calibrate on out-of-fold predictions of real rows, unweighted: "80" should mean
    # ~80% of such items are credible. Augmented copies are synthetic, so they are left out.
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof[real], y[real])
    final = _lgbm(params, seed, additive).fit(X, y, sample_weight=w, feature_name=FEATURE_COLUMNS)  # names checked on load
    info = {"params": {**BASE_PARAMS, **params}, "train_rows": int(real.sum()), "augmented_rows": int((~real).sum()),
            "monotone": MONOTONE, "additive": additive, "image_dropout": dropout, "unused_features": unused, "seed": seed}
    model = MetaModel(final.booster_, calibrator, FEATURE_COLUMNS, info)
    importance = dict(zip(FEATURE_COLUMNS, map(int, final.booster_.feature_importance("gain"))))
    oof_auc, oof_per = mean_dataset_auc(tr["dataset"][real], y[real], oof[real])
    return model, {"best_params": params, "grid": grid, "train_rows": int(real.sum()),
                   "augmented_rows": int((~real).sum()), "oof_mean_dataset_auc": oof_auc, "oof_auc_by_dataset": oof_per,
                   "train_rows_by_dataset": tr.loc[real, "dataset"].value_counts().to_dict(),
                   "val_channel_score_auc_by_dataset": channel_score_auc(tr[real]),
                   "evidence_relevance_quantiles": {
                       d: g["evidence_relevance"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(4).to_dict()
                       for d, g in tr[real].groupby("dataset") if g["evidence_relevance"].notna().any()},
                   "gain_importance": importance}


# --- Evaluate --------------------------------------------------------------------

def ece(y_bin, p, n_bins: int = 10) -> float:
    """Expected Calibration Error: per probability bin, |mean predicted - observed
    rate|, weighted by bin size."""
    y_bin, p = np.asarray(y_bin, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y_bin[m].mean())
    return float(total)


def binary_metrics(y, p) -> dict:
    """y may contain 0.5 (conflicting); AUC/F1/ECE use only 0/1 rows, Brier uses all."""
    from sklearn.metrics import f1_score, roc_auc_score

    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    out = {"n": int(len(y)), "coverage": float(ok.mean()) if len(y) else math.nan}
    y, p = y[ok], p[ok]
    out["brier"] = float(np.mean((p - y) ** 2)) if len(y) else math.nan
    hard = y != 0.5
    yb, pb = y[hard], p[hard]
    if len(set(yb)) > 1:
        out["roc_auc"] = float(roc_auc_score(yb, pb))
        out["macro_f1"] = float(f1_score(yb, pb >= 0.5, average="macro"))
        out["ece"] = ece(yb, pb)
    return out


def fairness(test: pd.DataFrame, p: np.ndarray) -> dict:
    """Credible items only: text-only vs multimodal score distributions.
    A large gap would mean images help or hurt credible items."""
    from scipy.stats import ks_2samp

    cred = (test["credible"] == 1.0).to_numpy()
    multimodal = ((test["has_image"] == 1) | (test["has_video"] == 1)).to_numpy()
    s = 100 * np.asarray(p)
    a, b = s[cred & ~multimodal], s[cred & multimodal]
    out = {"n_text_only": int(len(a)), "n_multimodal": int(len(b))}
    if len(a) and len(b):
        ks = ks_2samp(a, b)
        out.update(mean_text_only=float(a.mean()), mean_multimodal=float(b.mean()),
                   mean_gap=float(b.mean() - a.mean()), ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue))
    # Context for the gap: the same means for NOT credible items. If text-only items sit closer
    # to 50 in *both* classes, text-only inputs simply carry less evidence (less certainty);
    # a penalty for missing media would instead push both classes down.
    fake = (test["credible"] == 0.0).to_numpy()
    for name, m in (("text_only", ~multimodal), ("multimodal", multimodal)):
        if (fake & m).any():
            out[f"mean_not_credible_{name}"] = float(s[fake & m].mean())
    return out


def _drop_image(X: np.ndarray) -> np.ndarray:
    X2 = X.copy()
    for col in IMAGE_FEATURES:
        X2[:, FEATURE_COLUMNS.index(col)] = np.nan
    X2[:, FEATURE_COLUMNS.index("has_image")] = 0.0
    return X2


def missing_modality_test(model: MetaModel, test: pd.DataFrame) -> dict:
    """Missing-modality test: take multimodal credible items, delete the image
    (image-derived features -> NaN, has_image -> 0) and re-score. A fair model's score
    should barely move.

    Also reported, to tell the two possible causes of a drop apart: the same test on
    *not credible* items. If both groups move toward the middle, the image was
    evidence and losing it costs certainty (expected). If both groups drop, the
    model punishes a missing image (unfair)."""
    has_img = (test["has_image"] == 1).to_numpy()
    out = {}
    for name, label in (("credible", 1.0), ("not_credible", 0.0)):
        sel = has_img & (test["credible"] == label).to_numpy()
        if not sel.any():
            out[name] = {"n": 0}
            continue
        X = test.loc[sel, FEATURE_COLUMNS].to_numpy(float)
        before, after = 100 * model.predict(X), 100 * model.predict(_drop_image(X))
        out[name] = {"n": int(sel.sum()), "mean_abs_change_points": float(np.mean(np.abs(after - before))),
                     "mean_change_points": float(np.mean(after - before)),
                     "mean_before": float(before.mean()), "mean_after": float(after.mean())}
    # How well the model still ranks image items once the image is gone (text-only use of the app).
    X = test.loc[has_img, FEATURE_COLUMNS].to_numpy(float)
    y = test.loc[has_img, "credible"].to_numpy(float)
    without = {"with_image": binary_metrics(y, model.predict(X)).get("roc_auc", math.nan),
               "without_image": binary_metrics(y, model.predict(_drop_image(X))).get("roc_auc", math.nan)}
    # Headline numbers stay at the top level (credible items).
    return {**out["credible"], "not_credible": out["not_credible"], "roc_auc_image_items": without}


def channel_ablation(model: MetaModel, test: pd.DataFrame) -> dict:
    """Inference-time ablation: blank one channel's features (NaN) on test and
    record the ROC-AUC drop. (Retraining without the channel is the Phase 8 version.)"""
    X = test[FEATURE_COLUMNS].to_numpy(float)
    y = test["credible"].to_numpy(float)
    base = binary_metrics(y, model.predict(X)).get("roc_auc", math.nan)
    out = {}
    for ch in CHANNELS:
        Xa = X.copy()
        for col in CHANNEL_FEATURES[ch]:
            Xa[:, FEATURE_COLUMNS.index(col)] = np.nan
        auc = binary_metrics(y, model.predict(Xa)).get("roc_auc", math.nan)
        out[ch] = {"roc_auc": auc, "drop": base - auc}
    return out


def channel_score_auc(test: pd.DataFrame) -> dict:
    """Diagnostic: ROC-AUC of each channel's own score, per dataset, on the rows where
    it ran (with coverage). Shows which channels carry signal where, independent of
    the meta-learner: a channel near 0.5 cannot be rescued by fusion."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for d, g in test.groupby("dataset"):
        g = g[g["credible"] != 0.5]
        out[d] = {}
        for ch, col in CHANNEL_SCORE.items():
            m = g[col].notna()
            entry = {"coverage": float(m.mean()) if len(g) else 0.0}
            if m.sum() >= 20 and g.loc[m, "credible"].nunique() > 1:
                entry["roc_auc"] = float(roc_auc_score(g.loc[m, "credible"], g.loc[m, col]))
            out[d][ch] = entry
    return out


def reliability_figure(y, p_raw, p_cal, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float)
    hard = y != 0.5
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", c="#999", lw=1, label="perfect")
    for p, name, c in ((p_raw, "LightGBM raw", "#e76f51"), (p_cal, "after isotonic", "#2a9d8f")):
        p = np.asarray(p)[hard]
        bins = np.minimum((p * 10).astype(int), 9)
        xs = [p[bins == b].mean() for b in range(10) if (bins == b).any()]
        ys = [y[hard][bins == b].mean() for b in range(10) if (bins == b).any()]
        ax.plot(xs, ys, marker="o", c=c, label=name)
    ax.set_xlabel("predicted probability credible")
    ax.set_ylabel("observed share credible")
    ax.set_title("Meta-learner reliability (test)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model: MetaModel, table: pd.DataFrame) -> dict:
    test = table[table["split"] == "test"].reset_index(drop=True)
    X = test[FEATURE_COLUMNS].to_numpy(float)
    y = test["credible"].to_numpy(float)
    p_raw, p_cal = model.predict_raw(X), model.predict(X)
    p_avg = average_channel_scores(X)
    report = {
        "test_rows": int(len(test)),
        "meta": binary_metrics(y, p_cal),
        "meta_uncalibrated": binary_metrics(y, p_raw),
        "baseline_equal_weight": binary_metrics(y, p_avg),
        "per_dataset": {
            d: {"meta": binary_metrics(y[m], p_cal[m]), "meta_uncalibrated": binary_metrics(y[m], p_raw[m]),
                "baseline_equal_weight": binary_metrics(y[m], p_avg[m])}
            for d in DATASETS for m in [(test["dataset"] == d).to_numpy()] if m.any()
        },
        "channel_score_auc_by_dataset": channel_score_auc(test),
        "band_counts": pd.Series([band(to_score(v)) for v in p_cal]).value_counts().to_dict(),
        "missing_modality": missing_modality_test(model, test),
        "ablation": channel_ablation(model, test),
    }
    report["fairness"] = fairness(test, p_cal)
    reliability_figure(y, p_raw, p_cal, REPORTS / "figures" / "meta_reliability.png")
    return report


# Source columns the deployed meta-learner is TRAINED on. Serving uses outlet ratings (SOURCE_MODE "rating",
# 2026-11-05), but a meta-learner refitted on rating columns lost on the India test set (fakes rated Credible
# 1.0% -> 4.7%, ROC-AUC 0.920 -> 0.824; reports/source_comparison.json): thousands of training rows lost their
# source score, which changed how text-only items are read. So the meta-learner keeps its training columns and
# only the served source value changes; that combination is measured in `evaluate source-compare`.
TRAIN_SOURCE_MODE = "classifier"


def prepare_table(path: Path = TABLE_PATH, source_mode: str | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """The meta table exactly as `fit` uses it: one channel set, derived columns, source features
    recomputed from each item's domain (mode: TRAIN_SOURCE_MODE unless given; NaN without a domain),
    one-class test sets re-dealt. Shared with src.fusion.evaluate so every Phase 8 number is
    measured on the same test rows. Returns (table, rows with a source score per dataset, regrouping)."""
    source_mode = source_mode or TRAIN_SOURCE_MODE
    table = main_channel_set(pd.read_parquet(path))
    if "evidence_relevance" in table:
        table = add_derived(table)
    missing = [c for c in FEATURE_COLUMNS if c not in table]
    if missing:  # table built before these features existed: run `build --force` to compute them
        print(f"meta table lacks {missing}; treated as not available (NaN)")
        table[missing] = np.nan
    table, blanked = recompute_source(table, source_mode)
    print(f"rows with a source score per dataset: {blanked}")
    table, regrouped = regroup_one_class_test(table)
    for d, c in regrouped.items():
        print(f"{d}: test held one class; val+test topic groups re-dealt -> {c}")
    return table, blanked, regrouped


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run channels over val+test items")
    b.add_argument("--channels", nargs="+", default=CHANNELS, choices=CHANNELS)
    b.add_argument("--max-per-split", type=int, default=0,
                   help="downsample: items per split, shared across datasets; 0 (default) = full sets")
    b.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="items per batched pipeline call (default 32)")
    b.add_argument("--max-per-dataset", type=int, default=0, help="optional extra cap per dataset and split; 0 = none")
    b.add_argument("--force", action="store_true")
    f = sub.add_parser("fit", help="train, calibrate, evaluate, save")
    f.add_argument("--no-dropout", action="store_true", help="train without image dropout (for the ablation in the report)")
    f.add_argument("--interactions", action="store_true",
                   help="let trees combine features of different checks (old, non-additive model; for the ablation)")
    f.add_argument("--all-features", action="store_true",
                   help="let the model use the side features too (only the input-type flags stay unused; for the ablation)")
    args = ap.parse_args()

    if args.cmd == "build":
        table = build_table(args.channels, max_per_split=args.max_per_split or None, batch_size=args.batch_size,
                            max_per_dataset=args.max_per_dataset or None, force=args.force)
        cov = table[FEATURE_COLUMNS].notna().groupby(table["dataset"]).mean().round(3)
        print("feature coverage by dataset:\n", cov[[*CHANNEL_SCORE.values()]].to_string())
        return

    table, blanked, regrouped = prepare_table()
    model, fit_report = fit_meta(table, dropout=not args.no_dropout, additive=not args.interactions,
                                 unused=CHANNEL_FEATURES["modality"] if args.all_features else None)
    model.save()
    report = {"fit": {**fit_report, "regrouped_val_test": regrouped, "source_scored_rows": blanked},
              "eval": evaluate(model, table)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "meta_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    (REPORTS / "fairness_report.json").write_text(json.dumps(
        {"fairness": report["eval"]["fairness"], "missing_modality": report["eval"]["missing_modality"]}, indent=2))
    ev = report["eval"]
    print(json.dumps({
        "meta": ev["meta"], "baseline_equal_weight": ev["baseline_equal_weight"],
        "fairness": ev["fairness"], "missing_modality": ev["missing_modality"],
    }, indent=2))
    print("per-dataset test ROC-AUC (meta vs equal-weight baseline):")
    for d, v in ev["per_dataset"].items():
        print(f"  {d:12s} {v['meta'].get('roc_auc', math.nan):.3f} vs {v['baseline_equal_weight'].get('roc_auc', math.nan):.3f}"
              f"  (n={v['meta']['n']})")


if __name__ == "__main__":
    main()
