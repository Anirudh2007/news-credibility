"""Slide/report figures not drawn by `src.fusion.evaluate figures` (used in the slides and report).

    python -m src.fusion.extra_figures                 # all three; the SHAP chart needs the API running
    python -m src.fusion.extra_figures --only fairness

- architecture.png  pipeline from input to score, as built
- shap_examples.png points per check for two demo items, from the live API's `contributions`
- fairness_distribution.png  test-set scores of credible / not-credible items, text-only vs with image
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
BLUE, RED, GREY, INK = "#2b6cb0", "#c53030", "#a0aec0", "#1a202c"


def architecture(path: Path = FIG / "architecture.png") -> Path:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#edf2f7", ec="#4a5568", size=9.5, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, zorder=3,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.3), zorder=1)

    box(0.2, 5.9, 2.3, 1.0, "News item\ntext · URL · image\n(+ claimed date / place)", fc="#fefcbf", bold=True)
    box(0.2, 4.4, 2.3, 0.9, "Input guard\nrefuses gibberish,\nnon-English", fc="#fed7d7")
    arrow(1.35, 5.9, 1.35, 5.3)

    # claim branch
    box(3.2, 5.9, 2.2, 1.0, "Claim detection\nspaCy split +\nDeBERTa check-worthiness")
    box(5.8, 5.9, 2.6, 1.0, "Evidence search\noffline index (BM25 + MiniLM)\n+ live fact-checks + Wikipedia\n→ cross-encoder re-rank")
    box(8.8, 5.9, 1.9, 1.0, "Claim matching\ncos ≥ 0.55: a fact-check\ncounts only for its claim")
    box(11.0, 5.9, 1.8, 1.0, "Verifier\nDeBERTa NLI\nSupported / Refuted /\nNEI / Conflicting")
    arrow(2.5, 4.95, 3.2, 6.1)  # guard -> claims
    arrow(5.4, 6.4, 5.8, 6.4)
    arrow(8.4, 6.4, 8.8, 6.4)
    arrow(10.7, 6.4, 11.0, 6.4)

    # other checks
    checks = [("Caption ↔ image\nCLIP (Fakeddit)", 3.2), ("Media reuse +\nlandmark\npHash + CLIP", 5.5),
              ("Source rating\n1,082 domains", 7.8), ("Temporal\ndate gaps (rules)", 10.1)]
    for text, x in checks:
        box(x, 4.0, 1.6, 1.0, text)
    arrow(2.5, 4.85, 3.2, 4.5)

    # fusion
    box(3.2, 2.2, 9.6, 1.0, "Meta-learner: LightGBM, monotone + one check per tree, missing check = not run (NaN),\n"
                            "isotonic calibration → 0–100 score · abstain (\"Unverified\") if no check found evidence",
        fc="#bee3f8", bold=True, size=10)
    for _, x in checks:
        arrow(x + 0.8, 4.0, x + 0.8, 3.2)
    arrow(12.65, 5.9, 12.65, 3.2)  # verdicts -> meta-learner

    box(0.2, 0.4, 3.6, 1.1, "Score + band\nLow · Questionable ·\nMostly credible · High · Unverified", fc="#c6f6d5")
    box(4.3, 0.4, 4.0, 1.1, "Explanation\nSHAP points per check (sum = score)\nquoted evidence · fact-check badge\ntemplate / validated LLM text", fc="#c6f6d5")
    box(8.8, 0.4, 4.0, 1.1, "FastAPI + SQLite (history, feedback, cache)\n→ Streamlit UI", fc="#e9d8fd")
    arrow(5.0, 2.2, 2.0, 1.5)
    arrow(8.0, 2.2, 6.3, 1.5)
    arrow(11.0, 2.2, 10.8, 1.5)

    ax.set_title("News Credibility and Claim Verification System: architecture", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _api_report(text: str, image: Path | None = None, api: str = "http://localhost:8000") -> dict:
    import requests

    files = {"image": (image.name, image.read_bytes(), "image/jpeg")} if image else None
    r = requests.post(f"{api}/api/v1/analyze", data={"text": text, "explain": "false"}, files=files, timeout=900)
    r.raise_for_status()
    return r.json()


LABELS = {"claim_verification": "Claim verification", "text_image_match": "Caption–image match",
          "media_reuse": "Media reuse / location", "source": "Source", "temporal": "Dates",
          "calibration": "Calibration step"}


def shap_examples(examples: list[tuple[str, str, Path | None]], path: Path = FIG / "shap_examples.png") -> Path:
    fig, axes = plt.subplots(1, len(examples), figsize=(6.5 * len(examples), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (title, text, image) in zip(axes, examples):
        rep = _api_report(text, image)
        cs = [c for c in rep["contributions"] if c["applicable"]]
        names = [LABELS.get(c["check"], c["check"]) for c in cs][::-1]
        pts = [c["points"] for c in cs][::-1]
        colours = [GREY if c["check"] == "calibration" else (BLUE if c["points"] > 0 else RED) for c in cs][::-1]
        ax.barh(names, pts, color=colours)
        ax.axvline(0, color=INK, lw=0.8)
        for i, v in enumerate(pts):
            ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
        lim = max(5, max(abs(v) for v in pts) * 1.35)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("points relative to an average item with the same inputs")
        ax.set_title(f"{title}\naverage item {rep['baseline_score']:.0f} → score {rep['score']} ({rep['band']})", fontsize=10)
    fig.suptitle("What affected the score (SHAP points per check; they add up to the score)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def fairness_distribution(path: Path = FIG / "fairness_distribution.png") -> Path:
    from src.fusion.features import FEATURE_COLUMNS
    from src.fusion.meta import MetaModel
    from src.fusion.train import prepare_table

    table, _, _ = prepare_table()
    test = table[table["split"] == "test"].reset_index(drop=True)
    s = 100 * MetaModel.load().predict(test[FEATURE_COLUMNS].to_numpy(float))
    multimodal = ((test["has_image"] == 1) | (test["has_video"] == 1)).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(0, 100, 26)
    for ax, (label, value) in zip(axes, (("credible items", 1.0), ("not-credible items", 0.0))):
        m = (test["credible"] == value).to_numpy()
        for name, sel, colour in (("text only", ~multimodal, "#dd6b20"), ("with image", multimodal, BLUE)):
            x = s[m & sel]
            ax.hist(x, bins=bins, alpha=0.55, color=colour, density=True,
                    label=f"{name} (n={len(x):,}, mean {x.mean():.1f})")
        ax.set_title(f"Test-set scores of {label}")
        ax.set_xlabel("score (0–100)")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("density")
    fig.suptitle("Fairness check: text-only items sit nearer the middle in both classes (less evidence, less certainty), "
                 "not lower", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["architecture", "shap", "fairness"])
    ap.add_argument("--image", type=Path, help="photo for the image example of the SHAP chart (e.g. Howrah Bridge)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "architecture"):
        print(architecture())
    if args.only in (None, "shap"):
        ex = [("Text: cow urine cures COVID", "Drinking cow urine and applying cow dung can cure and prevent COVID-19.", None)]
        if args.image:
            ex.append(("Image: photo posted as India Gate", "Huge crowd gathers at India Gate in Delhi for the candle march.",
                       args.image))
        print(shap_examples(ex))
    if args.only in (None, "fairness"):
        print(fairness_distribution())


if __name__ == "__main__":
    main()
