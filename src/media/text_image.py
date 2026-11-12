"""Channel 3D: text-image consistency on Fakeddit (Phase 3D).

    title + image -> CLIP text vector t, CLIP image vector v (same space)
                  -> features [t, v, t*v, |t-v|, cos(t, v)]
                  -> 6-way classifier (true + 5 fake sub-types)
    score = P(true)      confidence = max class probability
    subtype = most likely fake sub-type (reason tag for the explanation)
    no image -> score NaN (check not applicable)

CLIP was trained to put an image and its caption close together, so a caption
that does not describe its photo ("Flood in Mumbai today" on a 2015 Chennai
picture) gives a low cosine. Raw cosine alone is weak; the element-wise product
and difference let the classifier learn which dimensions matter for Fakeddit.

CLIP stays frozen: only a small classifier is trained on its features, which
runs on a CPU or the 6 GB RTX 4050 (fine-tuning CLIP itself would not fit well).

Also holds the Phase 2 image hygiene: download, drop corrupt/tiny files, resize.
"""

from __future__ import annotations

import io
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "interim" / "fakeddit_images"
EMB_DIR = ROOT / "data" / "processed" / "fakeddit_clip"
MODEL_PATH = ROOT / "models" / "text_image_clf.joblib"

CLIP_ARCH = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
IMAGE_SIZE = 224
MIN_SIDE = 32  # smaller images are icons/thumbnails, not photos

# Order matters: index 0 is "true" (credible). Names match Phase 2 reason tags.
CLASSES = ["true", "satire", "misleading_content", "imposter_content", "false_connection", "manipulated_content"]


# --- Images: download and hygiene (Phase 2) -----------------------------------

def load_image(data: bytes | str | Path):
    """Open, verify and convert to RGB; None if corrupt or tiny."""
    from PIL import Image, UnidentifiedImageError

    try:
        src = io.BytesIO(data) if isinstance(data, bytes) else data
        with Image.open(src) as im:
            im.verify()  # detects truncated/corrupt files
        im = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return None
    if min(im.size) < MIN_SIDE:
        return None
    return im


def resize_square(im, size: int = IMAGE_SIZE):
    """Shorter side to `size`, centre crop to size x size (what CLIP sees anyway)."""
    from PIL import ImageOps

    return ImageOps.fit(im, (size, size), method=3)  # 3 = BICUBIC


def _fetch(item: tuple[str, str], out_dir: Path, timeout: float) -> tuple[str, bool]:
    item_id, url = item
    dest = out_dir / f"{item_id}.jpg"
    if dest.exists():
        return item_id, True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "news-credibility-research/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            im = load_image(r.read())
    except Exception:
        return item_id, False
    if im is None:
        return item_id, False
    resize_square(im).save(dest, quality=92)
    return item_id, True


def download_images(
    ids: Sequence[str], urls: Sequence[str], out_dir: Path = IMAGE_DIR, workers: int = 16, timeout: float = 10.0
) -> dict[str, bool]:
    """Download, verify and resize to 224x224 JPEG. Returns id -> success.
    Many Fakeddit URLs are dead (imgur/reddit deletions); failures are expected
    and those items simply have no image (3D score NaN)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [(i, u) for i, u in zip(ids, urls) if isinstance(u, str) and u.startswith("http")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(lambda it: _fetch(it, out_dir, timeout), items))


# --- CLIP encoder -------------------------------------------------------------

TextEncoder = Callable[[Sequence[str]], np.ndarray]
ImageEncoder = Callable[[Sequence], np.ndarray]  # PIL images -> (n, d)


class ClipEncoder:
    """open_clip ViT-B/32. Vectors are L2-normalised, so dot product = cosine."""

    def __init__(self, arch: str = CLIP_ARCH, pretrained: str = CLIP_PRETRAINED, device: str | None = None):
        import open_clip
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        # Caches filled only by prefetch_*() during batch analysis. Images are keyed by
        # id(): valid while the caller holds the image objects; clear_memo() after the batch.
        self._tmemo: dict[str, np.ndarray] = {}
        self._imemo: dict[int, np.ndarray] = {}

    def _texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(list(texts[i : i + batch_size])).to(self.device)  # truncates at 77 tokens
            with torch.inference_mode():
                out.append(torch.nn.functional.normalize(self.model.encode_text(tok).float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def _images(self, images: Sequence, batch_size: int = 64) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i : i + batch_size]]).to(self.device)
            with torch.inference_mode():
                out.append(torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        misses = [t for t in dict.fromkeys(texts) if t not in self._tmemo]
        fresh = dict(zip(misses, self._texts(misses, batch_size))) if misses else {}
        rows = [self._tmemo[t] if t in self._tmemo else fresh[t] for t in texts]
        return np.stack(rows) if rows else np.zeros((0, 512), np.float32)

    def encode_images(self, images: Sequence, batch_size: int = 64) -> np.ndarray:
        misses = [im for im in images if id(im) not in self._imemo]
        fresh = dict(zip(map(id, misses), self._images(misses, batch_size))) if misses else {}
        rows = [self._imemo[id(im)] if id(im) in self._imemo else fresh[id(im)] for im in images]
        return np.stack(rows) if rows else np.zeros((0, 512), np.float32)

    def prefetch_texts(self, texts: Sequence[str]) -> None:
        misses = [t for t in dict.fromkeys(texts) if t not in self._tmemo]
        if misses:
            self._tmemo.update(zip(misses, self._texts(misses)))

    def prefetch_images(self, images: Sequence) -> None:
        misses = list({id(im): im for im in images if id(im) not in self._imemo}.values())
        if misses:
            self._imemo.update(zip(map(id, misses), self._images(misses)))

    def clear_memo(self) -> None:
        self._tmemo.clear()
        self._imemo.clear()


# --- Features and classifier --------------------------------------------------

def pair_features(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """[t, v, t*v, |t-v|, cos] per row. Inputs (n, d), L2-normalised."""
    t, v = np.atleast_2d(t), np.atleast_2d(v)
    cos = np.sum(t * v, axis=1, keepdims=True)
    return np.hstack([t, v, t * v, np.abs(t - v), cos]).astype(np.float32)


def labels_from_tags(reason_tags: Sequence) -> np.ndarray:
    """Phase 2 `reason_tag` (None for true items) -> class index."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([idx["true"] if t is None or (isinstance(t, float) and math.isnan(t)) else idx[t] for t in reason_tags])


def train_classifier(X_train, y_train, X_val, y_val, Cs: Sequence[float] = (0.1, 0.3, 1.0, 3.0), seed: int = 42):
    """Multinomial logistic regression on frozen CLIP features; C picked by val macro-F1.

    Balanced class weights because the 50/50 true/fake sample still leaves rare
    fake sub-types (imposter, satire) small. Returns (model, report dict).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best, report = None, {}
    for C in Cs:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed),
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        p_true = clf.predict_proba(X_val)[:, list(clf.classes_).index(0)] if 0 in clf.classes_ else np.zeros(len(pred))
        f6 = f1_score(y_val, pred, average="macro")
        f2 = f1_score(np.asarray(y_val) == 0, p_true >= 0.5, average="macro")
        report[C] = {"macro_f1_6way": float(f6), "macro_f1_2way": float(f2)}
        if best is None or f6 > report[best[0]]["macro_f1_6way"]:
            best = (C, clf)
    return best[1], {"best_C": best[0], "by_C": report}


@dataclass
class TextImageResult:
    score: float  # P(true); NaN when no image
    confidence: float
    subtype: str | None  # most likely fake sub-type when score < 0.5
    clip_similarity: float
    probs: dict[str, float]


NAN_RESULT = TextImageResult(math.nan, 0.0, None, math.nan, {})


class TextImageChecker:
    """score(text, image) for one item. `encoder` must expose encode_texts/encode_images."""

    def __init__(self, model=None, encoder=None, model_path: str | Path = MODEL_PATH):
        if model is None:
            import joblib

            model = joblib.load(model_path)
        self.model = model
        self.encoder = encoder or ClipEncoder()

    def score(self, text: str, image) -> TextImageResult:
        if image is None or not text:
            return NAN_RESULT
        t = self.encoder.encode_texts([text])
        v = self.encoder.encode_images([image])
        x = pair_features(t, v)
        proba = self.model.predict_proba(x)[0]
        probs = {CLASSES[int(c)]: float(p) for c, p in zip(self.model.classes_, proba)}
        p_true = probs.get("true", 0.0)
        fakes = {k: v for k, v in probs.items() if k != "true"}
        subtype = max(fakes, key=fakes.get) if fakes and p_true < 0.5 else None
        return TextImageResult(p_true, float(proba.max()), subtype, float(x[0, -1]), probs)


# --- Batch pipeline -----------------------------------------------------------

def embed_split(df, encoder: ClipEncoder, image_dir: Path = IMAGE_DIR):
    """Encode rows whose image downloaded. Returns (features, labels, kept item_ids)."""
    from PIL import Image

    rows = [(i, t, tag) for i, t, tag in zip(df["item_id"], df["text"], df["reason_tag"]) if (image_dir / f"{i}.jpg").exists()]
    if not rows:
        return np.zeros((0, 512 * 4 + 1), np.float32), np.zeros(0, int), []
    ids, texts, tags = map(list, zip(*rows))
    images = [Image.open(image_dir / f"{i}.jpg").convert("RGB") for i in ids]
    return pair_features(encoder.encode_texts(texts), encoder.encode_images(images)), labels_from_tags(tags), ids


def main() -> None:
    """python -m src.media.text_image : download images, embed, train, save, print val/test macro-F1."""
    import json

    import joblib
    import pandas as pd
    from sklearn.metrics import classification_report

    df = pd.read_parquet(ROOT / "data" / "processed" / "fakeddit.parquet")
    df = df[df["has_image"]]
    ok = download_images(df["item_id"].tolist(), df["image_url"].tolist())
    print(f"images ok: {sum(ok.values())} / {len(df)}")

    enc = ClipEncoder()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for split in ("train", "val", "test"):
        X, y, ids = embed_split(df[df["split"] == split], enc)
        np.save(EMB_DIR / f"{split}_X.npy", X)
        np.save(EMB_DIR / f"{split}_y.npy", y)
        data[split] = (X, y)
        print(split, X.shape)

    clf, report = train_classifier(*data["train"], *data["val"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    X_te, y_te = data["test"]
    report["test"] = classification_report(y_te, clf.predict(X_te), target_names=CLASSES, output_dict=True, zero_division=0)
    out = ROOT / "reports" / "text_image_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"best_C": report["best_C"], "test_macro_f1": report["test"]["macro avg"]["f1-score"]}, indent=2))


if __name__ == "__main__":
    main()
