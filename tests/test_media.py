import io
import math

import numpy as np
import pytest
from PIL import Image

from src.media import reuse
from src.media.reuse import (
    LocationReference,
    MediaIndex,
    MediaRecord,
    MediaReuseChecker,
    ela_score,
    hamming,
    phash,
    read_exif,
    records_from_mediaeval,
    temporal_signals,
    train_reuse_classifier,
)
from src.media.text_image import (
    CLASSES,
    TextImageChecker,
    labels_from_tags,
    load_image,
    pair_features,
    resize_square,
    train_classifier,
)


def make_image(seed: int, size=(320, 240)) -> Image.Image:
    """Smooth random pattern: structured enough for pHash to be stable."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BICUBIC)


def jpeg_roundtrip(im: Image.Image, quality=60) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def fake_image_encoder(images):
    """8x8 greyscale thumbnail, mean-centred, L2-normalised -> 64-dim vector."""
    out = []
    for im in images:
        v = np.asarray(im.convert("L").resize((8, 8)), dtype=np.float32).ravel()
        v -= v.mean()
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(out)


class FakeClip:
    def encode_images(self, images):
        return fake_image_encoder(images)

    def encode_texts(self, texts):
        rng = [np.random.default_rng(len(t)) for t in texts]
        v = np.stack([r.standard_normal(64) for r in rng]).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


# --- image hygiene ------------------------------------------------------------

def test_load_image_rejects_corrupt_and_tiny():
    buf = io.BytesIO()
    make_image(1).save(buf, "PNG")
    assert load_image(buf.getvalue()).size == (320, 240)
    assert load_image(b"not an image") is None
    tiny = io.BytesIO()
    make_image(1, (20, 20)).save(tiny, "PNG")
    assert load_image(tiny.getvalue()) is None


def test_resize_square():
    assert resize_square(make_image(1)).size == (224, 224)


# --- 3D text-image -------------------------------------------------------------

def test_pair_features_shape_and_cosine():
    t = np.array([[1.0, 0.0]], np.float32)
    v = np.array([[0.6, 0.8]], np.float32)
    f = pair_features(t, v)
    assert f.shape == (1, 2 * 4 + 1) and math.isclose(f[0, -1], 0.6, rel_tol=1e-6)


def test_labels_from_tags():
    assert labels_from_tags([None, "false_connection", float("nan"), "satire"]).tolist() == [0, 4, 0, 1]


def test_train_classifier_and_checker_on_separable_toy_data():
    rng = np.random.default_rng(0)
    n, d = 300, 8
    y = rng.integers(0, 3, n)  # true, satire, misleading
    X = rng.standard_normal((n, d)) + np.eye(3, d)[y] * 3
    clf, report = train_classifier(X[:200], y[:200], X[200:], y[200:], Cs=(1.0,))
    assert report["by_C"][1.0]["macro_f1_6way"] > 0.8

    class FixedFeatures:  # encoder whose pair_features land on a "true" point
        def encode_texts(self, texts):
            return np.zeros((1, 2), np.float32)

        def encode_images(self, images):
            return np.zeros((1, 2), np.float32)

    checker = TextImageChecker(model=clf, encoder=FixedFeatures())
    checker.model = type("M", (), {"classes_": clf.classes_, "predict_proba": lambda self, x: clf.predict_proba(X[:1])})()
    res = checker.score("caption", make_image(1))
    assert 0 <= res.score <= 1 and set(res.probs) <= set(CLASSES)


def test_text_image_nan_without_image():
    res = TextImageChecker(model=object(), encoder=FakeClip()).score("caption", None)
    assert math.isnan(res.score) and res.confidence == 0.0


# --- 3E fingerprints and forensics ---------------------------------------------

def test_phash_survives_resize_and_recompression():
    a = make_image(1)
    copy = jpeg_roundtrip(a.resize((160, 120)))
    other = make_image(2)
    assert hamming(phash(a), phash(copy)) <= reuse.PHASH_MAX_HAMMING
    assert hamming(phash(a), phash(other)) > reuse.PHASH_MAX_HAMMING


def test_ela_higher_for_pasted_region():
    base = jpeg_roundtrip(make_image(3), quality=70)
    pasted = base.copy()
    pasted.paste(make_image(4, (120, 90)), (100, 80))  # uncompressed patch
    assert ela_score(pasted) > ela_score(base)


def test_read_exif_absent_and_present():
    assert read_exif(make_image(1)) == {"capture_date": None, "gps": None}
    im = make_image(1)
    exif = Image.Exif()
    exif[306] = "2015:12:01 10:00:00"
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    assert read_exif(Image.open(io.BytesIO(buf.getvalue())))["capture_date"] == "2015-12-01"


# --- 3E index and temporal logic -----------------------------------------------

@pytest.fixture()
def index():
    imgs = [make_image(1), make_image(2)]
    recs = [
        MediaRecord("chennai_flood_01", phash(imgs[0]), "2015-12-02", "chennai_flood_2015"),
        MediaRecord("other_01", phash(imgs[1]), "2020-01-01", "other_event"),
    ]
    return MediaIndex(recs, fake_image_encoder(imgs))


def test_index_finds_reposted_copy(index):
    q = jpeg_roundtrip(make_image(1).resize((200, 150)))
    matches = index.query(phash(q), fake_image_encoder([q])[0])
    assert [m.record.media_id for m in matches][:1] == ["chennai_flood_01"]
    assert index.query(phash(q), None, exclude_ids={"chennai_flood_01"}) == []


def test_temporal_signals(index):
    q = make_image(1)
    m = index.query(phash(q))
    earliest, gap, predates, different = temporal_signals(m, "2024-07-01", "mumbai_flood_2024")
    assert earliest == "2015-12-02" and gap > 3000 and predates and different
    _, gap, predates, _ = temporal_signals(m, "2015-12-03", "chennai_flood_2015")  # 1 day later: normal spread
    assert gap == 1 and not predates
    _, gap, predates, _ = temporal_signals(m, None, None)
    assert math.isnan(gap) and not predates


def test_location_reference():
    refs = {"india_gate": fake_image_encoder([make_image(7), make_image(8)])}
    loc = LocationReference(fake_image_encoder, refs=refs)
    assert math.isclose(loc.similarity(fake_image_encoder([make_image(7)])[0], "India Gate"), 1.0, rel_tol=1e-5)
    assert math.isnan(loc.similarity(fake_image_encoder([make_image(7)])[0], "Jantar Mantar"))
    assert math.isnan(loc.similarity(fake_image_encoder([make_image(7)])[0], None))


def test_checker_flags_old_image_and_nan_without_media(index):
    checker = MediaReuseChecker(index, encoder=fake_image_encoder)
    res = checker.score(make_image(1), claimed_date="2024-07-01", claimed_event="mumbai_flood_2024")
    assert res.score < 0.5 and res.signals.predates_claim and res.signals.reasons
    assert len(res.signals.features()) == len(reuse.FEATURE_NAMES)
    fresh = checker.score(make_image(9), claimed_date="2024-07-01")
    assert fresh.score > res.score and fresh.signals.n_matches == 0
    assert math.isnan(checker.score(None).score)


def test_records_from_mediaeval_uses_earliest_post():
    import pandas as pd

    df = pd.DataFrame(
        {
            "media_id": ["a,b", "a", "c"],
            "date": ["2016-01-05", "2016-01-01", None],
            "topic_group": ["e1", "e1", "e2"],
            "text": ["t1", "t2", "t3"],
        }
    )
    recs = {r[0]: r for r in records_from_mediaeval(df)}
    assert recs["a"][1] == "2016-01-01" and recs["b"][1] == "2016-01-05" and recs["c"][1] is None


def test_train_reuse_classifier_handles_nan():
    rng = np.random.default_rng(0)
    n = 120
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, len(reuse.FEATURE_NAMES)))
    X[:, 4] = 1 - y  # predates_claim drives "fake"
    X[rng.random(n) < 0.3, 3] = np.nan
    model, cv = train_reuse_classifier(X, y, groups=[f"e{i % 6}" for i in range(n)])
    assert cv > 0.9 and model.predict_proba(X[:2]).shape == (2, 2)


def test_train_reuse_classifier_drops_all_nan_and_constant_columns():
    """Regression: an all-NaN column crashed HistGradientBoosting's binner."""
    rng = np.random.default_rng(1)
    n = 90
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, len(reuse.FEATURE_NAMES)))
    X[:, reuse.FEATURE_NAMES.index("predates_claim")] = 1 - y
    X[:, reuse.FEATURE_NAMES.index("location_similarity")] = np.nan  # no landmark refs
    X[:, reuse.FEATURE_NAMES.index("exif_date_gap_days")] = np.nan  # EXIF stripped
    X[:, reuse.FEATURE_NAMES.index("different_event")] = 0.0  # constant
    model, cv = train_reuse_classifier(X, y, groups=[f"e{i % 5}" for i in range(n)])
    assert model.named_steps["drop"].dropped_ == ["different_event", "location_similarity", "exif_date_gap_days"]
    assert cv > 0.9 and model.predict_proba(X[:3]).shape == (3, 2)
