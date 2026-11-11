"""Phase 2 master pipeline: raw datasets -> cleaned, labelled, split Parquet files.

Run from the `news-credibility/` folder:

    python -m src.preprocess.build_dataset

Steps per dataset (order matters):
    load -> clean text -> drop empty -> English filter -> ISO dates
    -> unify labels -> exact dedup -> near dedup -> topic groups -> 70/15/15 split

Outputs:
    data/interim/<dataset>.parquet     cleaned, not yet unified (all raw columns kept)
    data/processed/<dataset>.parquet   per-channel table (unified columns + extras)
    data/processed/unified.parquet     all datasets, common columns only
    reports/preprocess_summary.json    row counts per step, label and split balance
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.ingest.load_raw import LOADERS
from src.preprocess.clean_text import clean_text, extract_dates, is_english, parse_date
from src.preprocess.dedup import drop_exact_duplicates, drop_near_duplicates
from src.preprocess.split import SEED, assign_topics, group_split
from src.preprocess.unify_labels import unify_labels

ROOT = Path(__file__).resolve().parents[2]
RAW, INTERIM, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "interim", ROOT / "data" / "processed"
SUMMARY_PATH = ROOT / "reports" / "preprocess_summary.json"

UNIFIED_COLUMNS = [
    "item_id", "dataset", "text", "date", "dates_in_text",
    "has_text", "has_image", "has_video", "source_domain",
    "raw_label", "credible", "reason_tag", "meta_eligible",
    "topic_group", "split", "orig_split", "n_chars", "n_words",
]

log = logging.getLogger("build_dataset")


def process_dataset(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = {"raw": len(df)}

    # 1. Clean text (also the AVeriTeC justification, used in explanations later).
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    if "justification" in df:
        df["justification"] = df["justification"].map(clean_text)

    # 2. Drop rows with nothing to analyse: no text and no media.
    has_media = df["has_image"].astype(bool) | df["has_video"].astype(bool)
    df = df[(df["text"] != "") | has_media]
    counts["after_drop_empty"] = len(df)

    # 3. English only (MVP scope).
    df = df[df["text"].map(is_english)]
    counts["after_english_filter"] = len(df)

    # 4. Dates to ISO: the item's own date, plus dates mentioned in the text.
    df["date"] = df["date_raw"].map(parse_date)
    df["dates_in_text"] = df["text"].map(extract_dates)
    df.to_parquet(INTERIM / f"{name}.parquet", index=False)

    # 5. One target for every dataset.
    df = unify_labels(df, name, label_col="label")

    # 6. Deduplicate before splitting, so no story is in both train and test.
    df = drop_exact_duplicates(df)
    counts["after_exact_dedup"] = len(df)
    df = drop_near_duplicates(df)
    counts["after_near_dedup"] = len(df)

    # 7. Topic/event groups, then whole-group 70/15/15 split.
    df = df.reset_index(drop=True)
    if df["event"].notna().all():
        df["topic_group"] = name + ":event:" + df["event"].astype(str)
    else:
        topics = assign_topics(df["text"].tolist())
        df["topic_group"] = [f"{name}:topic:{t}" for t in topics]
    df["split"] = group_split(df["topic_group"])

    # 8. Derived columns.
    df.insert(0, "dataset", name)
    df.insert(0, "item_id", df.pop("id"))
    df["has_text"] = df["text"] != ""
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len().fillna(0).astype(int)
    df = df.drop(columns=["date_raw", "event"])

    counts["n_topic_groups"] = int(df["topic_group"].nunique())
    counts["split_rows"] = df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0).to_dict()
    counts["raw_label_counts"] = df["raw_label"].value_counts().to_dict()
    counts["credible_mean_by_split"] = {
        k: (None if pd.isna(v) else round(float(v), 4)) for k, v in df.groupby("split")["credible"].mean().items()
    }
    counts["meta_eligible"] = int(df["meta_eligible"].sum())
    return df, counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary: dict = {"seed": SEED, "ratios": [0.70, 0.15, 0.15], "datasets": {}}
    unified = []
    for name, loader in LOADERS.items():
        raw = loader(RAW)
        if raw.empty:
            log.warning("%s: no raw files found, skipped", name)
            continue
        df, counts = process_dataset(name, raw)
        if counts["raw"] < 100:
            counts["warning"] = "sample file only; replace with the full release before Phase 3"
            log.warning("%s: only %d rows (sample data)", name, counts["raw"])
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        unified.append(df[UNIFIED_COLUMNS])
        summary["datasets"][name] = counts
        log.info("%s: %d -> %d rows, splits %s", name, counts["raw"], len(df), counts["split_rows"])

    all_df = pd.concat(unified, ignore_index=True)
    all_df.to_parquet(PROCESSED / "unified.parquet", index=False)
    summary["unified_rows"] = len(all_df)
    summary["unified_split_rows"] = all_df["split"].value_counts().to_dict()
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("unified.parquet: %d rows -> %s", len(all_df), PROCESSED / "unified.parquet")


if __name__ == "__main__":
    main()
