"""Build the check-worthiness training table for Channel 3A, and optionally train.

Sources:
- CLEF CheckThat! 2024 Task 1 English (data/raw/checkthat/): debate and speech
  sentences labelled Yes / No. "No" sentences are the negatives.
  Splits: train -> train, dev -> val, dev-test + test_gold -> test.
- AVeriTeC claims (data/processed/averitec.parquet) as extra positives, keeping
  the Phase 2 topic split so 3A never trains on val/test claims.

Output: data/processed/checkworthiness.parquet (text, label, source, split).

Run from `news-credibility/`:
    python -m src.ingest.download_checkthat          # once
    python -m src.claims.prepare_data                # build the table
    python -m src.claims.prepare_data --train        # + fine-tune DeBERTa (GPU advised)
    python -m src.claims.prepare_data --eval         # test P/R/F1 -> reports/checkworthiness_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from src.preprocess.clean_text import clean_text

ROOT = Path(__file__).resolve().parents[2]
CHECKTHAT_DIR = ROOT / "data" / "raw" / "checkthat"
AVERITEC_PROCESSED = ROOT / "data" / "processed" / "averitec.parquet"
OUT = ROOT / "data" / "processed" / "checkworthiness.parquet"
SUMMARY = ROOT / "reports" / "checkworthiness_summary.json"
METRICS = ROOT / "reports" / "checkworthiness_metrics.json"

_SPLIT_FROM_NAME = [("dev-test", "test"), ("test_gold", "test"), ("train", "train"), ("dev", "val")]


def _split_of(path: Path) -> str | None:
    for key, split in _SPLIT_FROM_NAME:  # order matters: "dev-test" before "dev"
        if key in path.stem:
            return split
    return None


def load_checkthat(raw_dir: Path = CHECKTHAT_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.rglob("*english*.tsv")):
        if "__MACOSX" in path.parts or _split_of(path) is None:
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, quoting=3, keep_default_na=False)
        frames.append(
            pd.DataFrame(
                {
                    "text": df["Text"],
                    "label": df["class_label"].str.strip().str.lower().map({"yes": 1, "no": 0}),
                    "source": "checkthat",
                    "split": _split_of(path),
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No CheckThat! English TSVs under {raw_dir}; run src.ingest.download_checkthat")
    return pd.concat(frames, ignore_index=True)


def load_averitec_positives(path: Path = AVERITEC_PROCESSED) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["text", "split"])
    return df.assign(label=1, source="averitec")[["text", "label", "source", "split"]]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def build_table(checkthat: pd.DataFrame, averitec: pd.DataFrame | None) -> pd.DataFrame:
    """Clean, drop empty/unlabelled, dedup, and remove from train any text that
    also appears in val/test (leakage across splits)."""
    df = pd.concat([checkthat, averitec] if averitec is not None else [checkthat], ignore_index=True)
    df["text"] = df["text"].map(clean_text)
    df = df[(df["text"] != "") & df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["_k"] = df["text"].map(_key)
    df = df.drop_duplicates(["_k", "split"])
    held_out = set(df.loc[df["split"] != "train", "_k"])
    df = df[~((df["split"] == "train") & df["_k"].isin(held_out))]
    return df.drop(columns="_k").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="fine-tune DeBERTa after building the table")
    ap.add_argument("--no-averitec", action="store_true", help="CheckThat! only")
    ap.add_argument("--eval", action="store_true",
                    help="test-split P/R/F1 of the saved DeBERTa scorer vs the heuristic baseline (uses the saved table)")
    args = ap.parse_args()

    if args.eval:
        from src.claims.extract import DEFAULT_CHECKWORTHY_DIR, DebertaCheckWorthiness, HeuristicCheckWorthiness, \
            evaluate_checkworthiness

        scorers = {"heuristic": HeuristicCheckWorthiness()}
        if (DEFAULT_CHECKWORTHY_DIR / "config.json").exists():
            scorers["deberta"] = DebertaCheckWorthiness()
        report = evaluate_checkworthiness(pd.read_parquet(OUT), scorers)
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(report, indent=2))
        for name in scorers:
            r = report[name]["all"]
            print(f"{name:10s} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  macro-F1 {r.get('macro_f1', math.nan):.3f}")
        return

    averitec = None if args.no_averitec or not AVERITEC_PROCESSED.exists() else load_averitec_positives()
    table = build_table(load_checkthat(), averitec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    summary = {
        "rows": len(table),
        "by_split_label": {
            f"{s}/{lab}": int(n) for (s, lab), n in table.groupby(["split", "label"]).size().items()
        },
        "by_source": table["source"].value_counts().to_dict(),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.train:
        from src.claims.extract import train_checkworthiness

        result = train_checkworthiness(table[table["split"] == "train"], table[table["split"] == "val"])
        print("validation:", result)


if __name__ == "__main__":
    main()
