"""Retrain with claim matching (D11, 2026-11-07): the deployed meta-learner vs a candidate, same rows.

Why: the deployed meta-learner was trained on the v3 table (built 2026-11-02), before claim matching existed
(2026-11-03). Serving has used claim matching since then, so the claim features it sees in the app (answers about
a different claim do not count) are distributed differently from the ones it was trained on: a train/serve
mismatch. The candidate is fitted the same way on the v4 table (the same items run through the v0.9 pipeline,
claim matching on), into its own folder; the deployed files are untouched until the decision.

Decision rule, fixed before any result was seen (dev / benchmark only; the India test sets are scored once,
for the report, after the decision):
    1. Benchmark, as served (v4 test rows, outlet-rating source): the candidate's pooled ROC-AUC and mean
       per-dataset ROC-AUC are each at least the deployed model's minus 0.005.
    2. India dev set, deployed setting (news + Wikipedia confirm-only, time rule, claim matching): the candidate
       keeps true claims rated Low <= 5% and fakes rated Credible <= 5% (the targets of the India fix), and its
       ROC-AUC on scored items is at least the deployed model's minus 0.01.
    Both hold -> the candidate replaces the deployed model (it removes a known mismatch, so ties go to it).

    NEWSCRED_META_TABLE=... python -m src.fusion.train build                       # v4 table
    NEWSCRED_META_TABLE=... NEWSCRED_META_DIR=models/meta_v4 NEWSCRED_META_REPORTS=reports/meta_v4 \\
        python -m src.fusion.train fit                                              # candidate
    python -m src.fusion.meta_compare --candidate models/meta_v4 [--test]          # -> reports/meta_retrain_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.fusion import train as T
from src.fusion.features import FEATURE_COLUMNS
from src.fusion.meta import MetaModel

ROOT = T.ROOT
V3 = T.META_DIR / "meta_features_v3.parquet"
V4 = T.META_DIR / "meta_features_v4.parquet"
OUT = ROOT / "reports" / "meta_retrain_comparison.json"
BENCH_TOL, INDIA_AUC_TOL = 0.005, 0.01


def _load(folder: Path) -> MetaModel:
    return MetaModel.load(folder / "meta_lgbm.txt", folder / "meta_calibrator.joblib", folder / "meta_info.json")


def benchmark(models: dict[str, MetaModel]) -> dict:
    """Every model on the v3 and v4 test rows, source served as the outlet rating (as in the app)."""
    from src.fusion.evaluate import _scores, _test

    out = {}
    for name, path in (("v3_table_no_claim_matching", V3), ("v4_table_claim_matching", V4)):
        table, _, _ = T.prepare_table(path, source_mode="rating")
        test = _test(table)
        X = test[FEATURE_COLUMNS].to_numpy(float)
        out[name] = {"test_rows": int(len(test))}
        for m, model in models.items():
            s = _scores(test["credible"], model.predict(X), test["dataset"])
            out[name][m] = {"roc_auc": s["overall"]["roc_auc"], "macro_f1": s["overall"].get("macro_f1"),
                            "ece": s["overall"].get("ece"), "brier": s["overall"].get("brier"),
                            "mean_dataset_auc": s["mean_dataset_auc"], "auc_by_dataset": s["auc_by_dataset"]}
    return out


def india(models: dict[str, MetaModel], sets: tuple[str, ...]) -> dict:
    """The deployed India setting replayed from the recordings (no network): news + Wikipedia as confirm-only
    evidence, the time rule, the claim-match threshold chosen on dev for the deployed model."""
    from src.fusion import india_tune as I

    chosen = json.loads((I.CACHE / "chosen_wiki.json").read_text())["best"]["deployed"]
    out = {"setting": {"sources": ["news", "wiki"], "match_min": chosen["match_min"], "matcher": chosen.get("matcher"),
                       "time_rule": True, "support_only": True}}
    for name in sets:
        rec = I.attach_dates(I._load(f"{name}_news_wiki"), name)
        if chosen.get("matcher") == "learned":
            rec = I.rematch(rec, model=json.loads((I.CACHE / "chosen_wiki.json").read_text())["matcher"])
        y = np.array(rec["labels"], dtype=float)
        out[name] = {m: I.bands(y, I.replay(rec, chosen["match_min"], model, time_rule=True)) for m, model in models.items()}
    return out


def decide(bench: dict, ind: dict) -> dict:
    b = bench["v4_table_claim_matching"]
    dep, cand = b["deployed"], b["candidate"]
    d_dep, d_cand = ind["dev"]["deployed"], ind["dev"]["candidate"]
    checks = {
        "benchmark_pooled_auc": cand["roc_auc"] >= dep["roc_auc"] - BENCH_TOL,
        "benchmark_mean_dataset_auc": cand["mean_dataset_auc"] >= dep["mean_dataset_auc"] - BENCH_TOL,
        "dev_true_low_max_5pct": d_cand["true"]["low"] <= 0.05,
        "dev_fake_credible_max_5pct": d_cand["fake"]["credible"] <= 0.05,
        "dev_auc": d_cand.get("roc_auc_scored", 0.0) >= d_dep.get("roc_auc_scored", 0.0) - INDIA_AUC_TOL,
    }
    return {"checks": checks, "adopt_candidate": all(checks.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True, help="folder with the candidate meta-learner")
    ap.add_argument("--test", action="store_true", help="also score the India test and matched sets (after deciding)")
    args = ap.parse_args()
    models = {"deployed": _load(ROOT / "models"), "candidate": _load(args.candidate)}
    report = {"rule": __doc__.split("Decision rule")[1].split("NEWSCRED_META_TABLE")[0].strip(),
              "benchmark": benchmark(models)}
    report["india"] = india(models, ("dev", "test", "matched") if args.test else ("dev",))
    report["decision"] = decide(report["benchmark"], report["india"])
    OUT.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report["decision"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
