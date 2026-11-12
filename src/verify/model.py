"""Channel 3C: claim verification on AVeriTeC (Phase 3C).

    claim + evidence passage -> DeBERTa-v3 (cross-attention over both) -> 4 labels
    many passages -> weighted vote (re-ranker score x source reliability) -> one verdict

This is Natural Language Inference: does the evidence entail (Supported),
contradict (Refuted), mix (Conflicting) or say nothing about (NEE) the claim?
The claim and passage go into one input, "[CLS] claim [SEP] passage [SEP]", so
attention can compare words across the two.

Channel output for the meta-learner (Phase 4), consistent with Phase 2 labels:
    score      = (P(S) + 0.5 * P(C)) / (P(S) + P(R) + P(C))   credibility in [0, 1]
    confidence = 1 - P(NEE)                                     how much evidence spoke
    no evidence at all -> score NaN (check not applicable), never 0.

Training pairs (`build_pairs`), from data/processed/averitec.parquet:
- "concat": claim + all its QA evidence joined -> the claim's label (clean signal).
- "passage": claim + each single QA answer -> the claim's label. Noisier (one
  answer alone may not decide the claim) but matches inference, where the
  verifier sees one retrieved passage at a time.
Default "both". Splits come from Phase 2 (topic-based), so test claims are unseen topics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "verifier-deberta"
SOURCES_CSV = ROOT / "data" / "reference" / "sources.csv"
BASE_MODEL = "microsoft/deberta-v3-base"  # -small if the GPU runs out of memory
MAX_LENGTH = 384  # claims are <= ~100 tokens; the rest is evidence
LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
S, R, NEE, C = range(4)
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
DEFAULT_RELIABILITY = 0.5  # unknown domain: neither trusted nor distrusted
RERANK_TEMPERATURE = 1.0

PredictFn = Callable[[str, Sequence[str]], np.ndarray]  # (claim, passages) -> (n, 4) probabilities


# --- Training data ------------------------------------------------------------

def qa_text(question: str | None, answer: dict) -> str | None:
    """One AVeriTeC question-answer pair as the verifier reads it: "Q: ... A: ...".
    Shared with the evidence index (3B), so the verifier sees passages at inference
    in exactly the format it was trained on. None for an empty answer."""
    ans = (answer.get("answer") or "").strip()
    if not ans:
        return None
    if answer.get("answer_type") == "Boolean" and answer.get("boolean_explanation"):
        ans = f"{ans}. {answer['boolean_explanation']}"
    return f"Q: {(question or '').strip()} A: {ans}"


def _qa_texts(questions_json: str | None) -> list[str]:
    out = []
    for q in json.loads(questions_json or "[]"):
        for a in q.get("answers") or []:
            t = qa_text(q.get("question"), a)
            if t:
                out.append(t)
    return out


def build_pairs(df: pd.DataFrame, mode: str = "both") -> pd.DataFrame:
    """(claim, evidence, label, split) rows from the processed AVeriTeC table."""
    if mode not in {"concat", "passage", "both"}:
        raise ValueError(mode)
    rows = []
    for claim, label, split, qjson in zip(df["text"], df["raw_label"], df["split"], df["questions_json"]):
        qa = _qa_texts(qjson)
        lab = LABEL2ID[label]
        if mode in {"concat", "both"}:
            rows.append((claim, " ".join(qa) if qa else "No evidence found.", lab, split, "concat"))
        if mode in {"passage", "both"}:
            rows.extend((claim, t, lab, split, "passage") for t in qa)
    return pd.DataFrame(rows, columns=["claim", "evidence", "label", "split", "kind"])


def class_weights(labels: Sequence[int], n_classes: int = 4) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1. AVeriTeC is ~59% Refuted:
    unweighted loss would let the model ignore Conflicting and NEE."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return w / w.mean()


# --- Fine-tuning --------------------------------------------------------------

def train_verifier(
    pairs: pd.DataFrame,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
    grad_accum: int = 2,
    max_length: int = MAX_LENGTH,
    seed: int = 42,
):
    """Fine-tune DeBERTa-v3 for 4-way verification with class-weighted loss.

    Uses pairs with split == "train" for training and "val" for model selection
    (macro-F1). Saves model + tokenizer to `output_dir`. GPU strongly advised.
    """
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        dtype=torch.float32,  # transformers 5 keeps checkpoint dtype (fp16) otherwise; Trainer fp16 = mixed precision
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )

    def encode(batch):
        # only_second: truncate the evidence, never the claim
        return tok(batch["claim"], batch["evidence"], truncation="only_second", max_length=max_length)

    cols = ["claim", "evidence", "label"]
    train_ds = Dataset.from_pandas(pairs.loc[pairs["split"] == "train", cols], preserve_index=False).map(
        encode, batched=True
    )
    val_ds = Dataset.from_pandas(pairs.loc[pairs["split"] == "val", cols], preserve_index=False).map(
        encode, batched=True
    )
    weights = torch.tensor(class_weights(train_ds["label"]), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def metrics(pred):
        y_hat = np.argmax(pred.predictions, axis=-1)
        per_class = f1_score(pred.label_ids, y_hat, average=None, labels=list(range(len(LABELS))), zero_division=0)
        out = {"macro_f1": f1_score(pred.label_ids, y_hat, average="macro"), "accuracy": accuracy_score(pred.label_ids, y_hat)}
        out.update({f"f1_{LABELS[i].split()[0].lower()}": float(v) for i, v in enumerate(per_class)})
        return out

    args = TrainingArguments(
        output_dir=str(Path(output_dir) / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        weight_decay=0.01,
        warmup_steps=0.1,  # transformers 5: float < 1 = fraction of steps (old warmup_ratio)
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return trainer.evaluate()


# --- Source reliability -------------------------------------------------------

def load_source_reliability(path: str | Path = SOURCES_CSV) -> dict[str, float]:
    """domain -> reliability in [0, 1] from data/reference/sources.csv."""
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(d).lower().removeprefix("www."): float(r) for d, r in zip(df["domain"], df["reliability"])}


def reliability_of(domain: str | None, table: dict[str, float], default: float = DEFAULT_RELIABILITY) -> float:
    """Exact domain, else parent domain ('edition.cnn.com' -> 'cnn.com'), else default."""
    if not domain:
        return default
    parts = domain.lower().removeprefix("www.").split(".")
    for i in range(len(parts) - 1):
        hit = table.get(".".join(parts[i:]))
        if hit is not None:
            return hit
    return default


# --- Aggregation --------------------------------------------------------------

def passage_weights(
    rerank_scores: Sequence[float], reliabilities: Sequence[float], temperature: float = RERANK_TEMPERATURE
) -> np.ndarray:
    """w_i proportional to softmax(rerank_i / T) * reliability_i, summing to 1.

    Softmax turns cross-encoder logits (any real number) into relevance shares;
    multiplying by reliability lets a relevant passage from a trusted source count
    more than an equally relevant one from a dubious site.
    """
    z = np.asarray(rerank_scores, dtype=float) / temperature
    rel = np.exp(z - z.max())
    w = rel / rel.sum() * np.clip(np.asarray(reliabilities, dtype=float), 1e-6, None)
    return w / w.sum()


def credibility_from_probs(p: np.ndarray) -> tuple[float, float]:
    """(score, confidence) from one 4-way distribution. Score uses the Phase 2
    mapping (Supported 1, Conflicting 0.5, Refuted 0); NEE mass becomes low
    confidence instead of a low score."""
    decisive = p[S] + p[R] + p[C]
    score = (p[S] + 0.5 * p[C]) / decisive if decisive > 0 else math.nan
    return float(score), float(1.0 - p[NEE])


# --- Claim matching (2026-11-03) ------------------------------------------------
# A fact-check's verdict is about the claim it checked. Retrieved for a *different* claim on the same
# topic ("images of an IED blast in Kanker" vs "a blast in Bijapur kills eight police"), its "No" was
# read as a refutation of the new claim: 76% of true 2023-2025 claims rated Low had a fact-check of
# another claim as top evidence. The same holds for AVeriTeC answers, which were written for one claim.
# Rule: a passage that carries a verdict on some claim counts only if that claim matches ours
# (cosine of sentence embeddings >= MATCH_MIN, chosen on the India dev set); otherwise it is read as
# "says nothing about this claim" (NEE). News reports carry no verdict and are not matched.
import os
import re as _re

MATCH_MIN = float(os.environ.get("NEWSCRED_MATCH_MIN", "0.55"))  # re-chosen on dev for the refitted meta-learner (2026-11-07); 0.6 before
AVERITEC_PATH = ROOT / "data" / "processed" / "averitec.parquet"
_AVERITEC_ID = _re.compile(r"^(?P<item>.+)_q\d+_a\d+$")
_averitec_claims: dict[str, str] | None = None


# The same map as a small file beside the evidence index: the Docker image and the server get models/ but not
# data/processed/, and without it claim matching was silently off for every index passage there (found
# 2026-11-06: in Docker an answer about cricketer Yasir Shah "refuted" a claim about Shah Rukh Khan).
AVERITEC_CLAIMS_JSON = ROOT / "models" / "retrieval_index" / "averitec_claims.json"


def _averitec_claim(item_id: str) -> str | None:
    global _averitec_claims
    if _averitec_claims is None:
        try:
            df = pd.read_parquet(AVERITEC_PATH, columns=["item_id", "text"])
            _averitec_claims = dict(zip(df["item_id"].astype(str), df["text"].astype(str)))
        except Exception:
            try:  # Docker / server: the exported map (export_averitec_claims)
                _averitec_claims = json.loads(AVERITEC_CLAIMS_JSON.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # neither (tests, a fresh clone): match fact-checks only
                _averitec_claims = {}
    return _averitec_claims.get(item_id)


def export_averitec_claims(path: Path = AVERITEC_CLAIMS_JSON) -> int:
    """Write item_id -> claim text for every AVeriTeC claim (python -m src.verify.model --export-claims)."""
    df = pd.read_parquet(AVERITEC_PATH, columns=["item_id", "text"])
    claims = dict(zip(df["item_id"].astype(str), df["text"].astype(str)))
    Path(path).write_text(json.dumps(claims, ensure_ascii=False), encoding="utf-8")
    return len(claims)


def claim_behind(passage) -> str | None:
    """The claim a passage gives a verdict on: the claim quoted in a fact-check passage, or the AVeriTeC
    claim an answer was written for. None for passages that are not about one claim (news reports)."""
    from src.retrieval.factcheck_api import parse_factcheck_passage

    info = parse_factcheck_passage(getattr(passage, "text", "") or "")
    if info:
        return info["claim"]
    m = _AVERITEC_ID.match(str(getattr(passage, "id", "")))
    return _averitec_claim(m.group("item")) if m else None


MATCHER_PATH = ROOT / "models" / "claim_matcher.json"
MATCH_FEATURES = ["cos", "word_jacc", "len_ratio", "num_conflict", "name_missing"]
_MATCH_STOP = frozenset("the a an of in on at to and or is are was were for by with from this that it as be has have had".split())


def match_features(claim: str, other: str, cos: float) -> list[float]:
    """Features of (our claim, the claim a passage judged) for the learned matcher (2026-11-03, second version).
    cos: embedding cosine; word_jacc: shared content words; len_ratio: shorter/longer length; num_conflict: both
    give numbers and none agree; name_missing: share of our capitalised names absent from the other claim."""
    def words(t):
        return {w for w in _re.findall(r"[a-z0-9]+", t.lower()) if w not in _MATCH_STOP and len(w) > 2}

    def names(t):
        return {w.lower() for w in _re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", t)} - {"the", "viral", "video", "photo", "claim"}

    def nums(t):
        return set(_re.findall(r"\d+(?:\.\d+)?", t.replace(",", "")))

    wa, wb, na, nb, ca, cb = words(claim), words(other), nums(claim), nums(other), names(claim), names(other)
    return [float(cos), len(wa & wb) / max(1, len(wa | wb)), min(len(claim), len(other)) / max(1, len(claim), len(other)),
            float(bool(na) and bool(nb) and not (na & nb)), len(ca - cb) / max(1, len(ca))]


def load_match_model(path: Path = MATCHER_PATH) -> dict | None:
    try:
        m = json.loads(Path(path).read_text())
        return m if m.get("features") == MATCH_FEATURES else None
    except (OSError, ValueError):
        return None


def match_probability(model: dict, feats: Sequence[float]) -> float:
    z = model["intercept"] + float(np.dot(model["coef"], feats))
    return 1.0 / (1.0 + math.exp(-z))


class ClaimMatcher:
    """similarity(claim, others) -> how likely each other claim is the same claim as ours (NaN where other is
    None). With models/claim_matcher.json: a logistic regression on MATCH_FEATURES (trained on the India dev
    set); without it: the cosine of MiniLM sentence embeddings."""

    def __init__(self, encoder=None, model: dict | None | bool = True):
        self._encoder = encoder
        self._memo: dict[str, np.ndarray] = {}
        self.model = load_match_model() if model is True else (model or None)

    def _vec(self, texts: Sequence[str]) -> list[np.ndarray]:
        if self._encoder is None:
            from src.retrieval.search import sentence_transformer_encoder

            self._encoder = sentence_transformer_encoder()
        miss = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if miss:
            self._memo.update(zip(miss, self._encoder(miss)))
        return [self._memo[t] for t in texts]

    def similarity(self, claim: str, others: Sequence[str | None]) -> np.ndarray:
        idx = [i for i, o in enumerate(others) if o]
        out = np.full(len(others), np.nan)
        if idx:
            vecs = self._vec([claim] + [others[i] for i in idx])
            q = vecs[0] / max(np.linalg.norm(vecs[0]), 1e-12)
            for i, v in zip(idx, vecs[1:]):
                cos = float(v @ q / max(np.linalg.norm(v), 1e-12))
                out[i] = cos if self.model is None else match_probability(self.model, match_features(claim, others[i], cos))
        return out

    def clear_memo(self) -> None:
        self._memo.clear()


# --- Entailment between the two claims (2026-11-12) -------------------------------------------------
# Similar is not the same. "The Eiffel Tower is located in Paris" and "Video shows the Eiffel Tower collapsing
# after being set ablaze" (AFP: False) are similar enough to pass MATCH_MIN, and the fact-check's "No" was read as
# a refutation of the true claim (also: "Ratan Tata died in October 2024" vs fake "last appearance" videos;
# "Obama was born in Hawaii" vs the Kenya birth-certificate hoax). A verdict on the judged claim B carries over
# to our claim A only where logic allows it:
#     "B is false" refutes A only if A implies B   (if A were true, B would be true too)
#     "B is true" supports A only if B implies A
# A natural-language-inference model gives P(premise entails hypothesis) for both directions. ENTAIL_MODE:
#     "off"          cosine matching only (the behaviour up to 0.10.1)
#     "both"         the passage counts only if each claim entails the other (the same claim)
#     "directional"  a passage read as speaking against our claim needs A => B, one speaking for it needs B => A
#     "same"         round 2: the two claims say the same thing (`same_claim_allows`, below)
# Measured on the India dev set by the rule in src/fusion/india_tune.py (ROOT_CAUSE_A_DECISION), 2026-11-12:
# NOT shipped, the mode stays "off". "directional" at 0.5 fixed all five known cases in every rewording and left
# 3 of 227 true dev claims with a counted refuting fact-check (20 before), but fakes rated Low fell from 68.7% to
# 52.3% (floor in the rule: 65.7%) and "Vaccines cause autism" was no longer rated Low: about a quarter of the
# fakes rated Low today are rated Low by fact-checks of related claims, not of the claim itself
# (reports/root_cause_a_eval.json). The code is kept so the measurement can be repeated.
ENTAIL_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # MIT licence; DeBERTa-v3-base on MNLI + FEVER-NLI + ANLI
ENTAIL_MODE = os.environ.get("NEWSCRED_ENTAIL_MODE", "off")
ENTAIL_MIN = float(os.environ.get("NEWSCRED_ENTAIL_MIN", "0.5"))
ENTAIL_MAX_LENGTH = 256  # two claims, each a sentence or two
# Round 2 (2026-11-12), mode "same": do the two claims say the same thing? Round 1's direction test lost real
# catches (a fact-checker words a rumour more fully than a user: "Video shows PM Modi using curse words at a
# rally in Gujarat" implies "Modi used curse words at a rally in Gujarat", not the other way round) and let
# through generic captions and expired verdicts ("Pope Francis had passed away", rated False before he died,
# is implied by "Pope Francis died in April 2025"). A verdict passage counts if
#     each claim implies the other; or
#     the checked claim implies ours and the wording is close (cosine >= SAME_CONTAINS_COS); or
#     ours implies the checked claim and the wording is very close (cosine >= SAME_FORWARD_COS);
# and a fact-check that CONFIRMED its claim counts against ours only if the two claims contradict each other
# ("Vaccines do not cause autism", rated True, against "Vaccines cause autism"). A passage that passes the
# similarity threshold but not this test is a fact-check of a RELATED claim: it does not count, and it is marked
# so that it can be shown as context. Measured in src/fusion/india_tune.py (ROOT_CAUSE_A_SCORECARD).
SAME_CONTAINS_COS = float(os.environ.get("NEWSCRED_SAME_CONTAINS_COS", "0.70"))
SAME_FORWARD_COS = float(os.environ.get("NEWSCRED_SAME_FORWARD_COS", "0.90"))


class ClaimEntailment:
    """entails(pairs) -> P(premise entails hypothesis) for (premise, hypothesis) pairs. `predict_fn` replaces the
    model in tests (no download)."""

    def __init__(self, model_name: str = ENTAIL_MODEL, predict_fn: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None,
                 device: str | None = None, batch_size: int = 32):
        self.model_name = model_name
        self._predict_fn = predict_fn
        self.device = device
        self.batch_size = batch_size
        self._tok = self._model = None
        self._entail_id = 0
        self._contra_id: int | None = None
        self._memo: dict[tuple[str, str], float] = {}
        self._memo3: dict[tuple[str, str], tuple[float, float]] = {}

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, dtype=torch.float32).to(self.device).eval()
        ids = [i for i, name in self._model.config.id2label.items() if str(name).lower().startswith("entail")]
        if len(ids) != 1:
            raise ValueError(f"{self.model_name}: no single entailment label in {self._model.config.id2label}")
        self._entail_id = int(ids[0])
        contra = [i for i, name in self._model.config.id2label.items() if str(name).lower().startswith("contra")]
        self._contra_id = int(contra[0]) if len(contra) == 1 else None  # two-class models have no such label

    def _infer2(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]:
        """(P(entailment), P(contradiction)) per pair; contradiction is 0 for a model without that label."""
        import torch

        if self._model is None:
            self._load()
        out: list[tuple[float, float]] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = list(pairs[i : i + self.batch_size])
            enc = self._tok([a for a, _ in batch], [b for _, b in batch], truncation=True, max_length=ENTAIL_MAX_LENGTH,
                            padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1).cpu()
            ent = probs[:, self._entail_id].tolist()
            con = probs[:, self._contra_id].tolist() if self._contra_id is not None else [0.0] * len(batch)
            out += list(zip(ent, con))
        return out

    def _infer(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [e for e, _ in self._infer2(pairs)]

    def entails(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        pairs = [(a, b) for a, b in pairs]
        miss = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        if miss:
            vals = self._predict_fn(miss) if self._predict_fn is not None else self._infer(miss)
            self._memo.update(zip(miss, map(float, vals)))
        return np.array([self._memo[p] for p in pairs], dtype=float)

    def relations(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """(n, 2): P(premise entails hypothesis), P(premise contradicts hypothesis). `predict_fn` (tests) may
        return one number per pair (entailment only: contradiction 0) or two."""
        pairs = [(a, b) for a, b in pairs]
        miss = [p for p in dict.fromkeys(pairs) if p not in self._memo3]
        if miss:
            vals = self._predict_fn(miss) if self._predict_fn is not None else self._infer2(miss)
            for p, v in zip(miss, vals):
                e, c = (float(v), 0.0) if np.ndim(v) == 0 else (float(v[0]), float(v[1]))
                self._memo3[p] = (e, c)
        return np.array([self._memo3[p] for p in pairs], dtype=float).reshape(-1, 2)

    def both_ways(self, claim: str, others: Sequence[str | None]) -> np.ndarray:
        """(n, 2): column 0 = P(claim entails other), column 1 = P(other entails claim); NaN where other is None."""
        out = np.full((len(others), 2), np.nan)
        idx = [i for i, o in enumerate(others) if o]
        if idx:
            vals = self.entails([(claim, others[i]) for i in idx] + [(others[i], claim) for i in idx])
            out[idx, 0], out[idx, 1] = vals[: len(idx)], vals[len(idx):]
        return out

    def four_ways(self, claim: str, others: Sequence[str | None]) -> np.ndarray:
        """(n, 4): P(claim entails other), P(claim contradicts other), P(other entails claim), P(other contradicts
        claim); NaN rows where other is None."""
        out = np.full((len(others), 4), np.nan)
        idx = [i for i, o in enumerate(others) if o]
        if idx:
            vals = self.relations([(claim, others[i]) for i in idx] + [(others[i], claim) for i in idx])
            out[idx, 0:2], out[idx, 2:4] = vals[: len(idx)], vals[len(idx):]
        return out

    def clear_memo(self) -> None:
        self._memo.clear()
        self._memo3.clear()


def entailment_allows(probs: np.ndarray, entail_row: Sequence[float], mode: str, entail_min: float) -> bool:
    """May this verdict passage count for our claim? probs: the verifier's reading of the passage against our
    claim; entail_row: [P(our claim entails the judged claim), P(the judged claim entails ours)]."""
    fwd, bwd = float(entail_row[0]), float(entail_row[1])
    if mode == "both":
        return min(fwd, bwd) >= entail_min
    score, _ = credibility_from_probs(np.asarray(probs, dtype=float))
    against = score < 0.5  # the passage would lower our claim's score: its "No" must be a "No" to our claim
    return (fwd if against else bwd) >= entail_min


def same_claim_allows(probs: np.ndarray, relations: Sequence[float], cosine: float, answer: str | None,
                      entail_min: float = ENTAIL_MIN, contains_cos: float | None = None,
                      forward_cos: float | None = None) -> bool:
    """Mode "same": may this verdict passage count for our claim? probs: the verifier's reading of the passage
    against our claim; relations: [P(ours entails checked), P(ours contradicts checked), P(checked entails ours),
    P(checked contradicts ours)]; cosine: similarity of the two claims; answer: the fact-check's answer to its own
    claim ("Yes" / "No" / "Partly"; None for a passage that is not a rated fact-check)."""
    contains_cos = SAME_CONTAINS_COS if contains_cos is None else contains_cos
    forward_cos = SAME_FORWARD_COS if forward_cos is None else forward_cos
    ef, cf, eb, cb = (float(x) for x in relations)
    score, _ = credibility_from_probs(np.asarray(probs, dtype=float))
    if answer == "Yes" and score < 0.5:  # a confirmed claim, read as speaking against ours: they must contradict
        return min(cf, cb) >= entail_min
    if min(ef, eb) >= entail_min:  # each claim says what the other says
        return True
    if eb >= entail_min and cosine >= contains_cos:  # the checked claim says everything ours says
        return True
    return bool(ef >= entail_min and cosine >= forward_cos)  # ours says everything the checked claim says


def fact_check_answer(text: str | None) -> str | None:
    """A fact-check passage's answer to its own claim ("Yes" / "No" / "Partly"), else None."""
    from src.retrieval.factcheck_api import parse_factcheck_passage

    info = parse_factcheck_passage(text or "")
    return info["answer"] if info else None


@dataclass
class EvidenceUse:
    passage_id: str
    text: str
    source_url: str | None
    domain: str | None
    weight: float
    probs: dict[str, float]
    date: str | None = None  # passage date (ISO), shown with quoted evidence
    claim_match: float = math.nan  # similarity of the claim this passage judged to ours (NaN: not a verdict)
    counted: bool = True  # False: a verdict on a different claim, read as NEE
    related: bool = False  # True: similar enough to pass the threshold, but not the same claim (mode "same")


@dataclass
class Verdict:
    label: str
    probs: dict[str, float]
    score: float  # NaN when no evidence
    confidence: float
    evidence: list[EvidenceUse] = field(default_factory=list)
    relevance: float = math.nan  # sigmoid(best re-ranker score): did retrieval find evidence about *this* claim?


def aggregate(
    passage_probs: np.ndarray, rerank_scores: Sequence[float], reliabilities: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted average of per-passage distributions. Returns (probs, weights)."""
    passage_probs = np.asarray(passage_probs, dtype=float)
    w = passage_weights(rerank_scores, reliabilities)
    return w @ passage_probs, w


# --- Inference ----------------------------------------------------------------

class ClaimVerifier:
    """verify(claim, hits) -> Verdict. `hits` are `src.retrieval.search.Hit`s;
    their `.score` should be the cross-encoder score (search with reranker=...)."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        predict_fn: PredictFn | None = None,
        reliability: dict[str, float] | None = None,
        device: str | None = None,
        batch_size: int = 16,
        matcher: ClaimMatcher | None = None,
        match_min: float | None = None,
        entailer: ClaimEntailment | None = None,
        entail_mode: str | None = None,
        entail_min: float | None = None,
    ):
        self.model_dir = Path(model_dir)
        self._predict_fn = predict_fn
        self.reliability = load_source_reliability() if reliability is None else reliability
        self.device = device
        self.batch_size = batch_size
        self._tok = self._model = None
        self._memo: dict[tuple[str, str], np.ndarray] = {}  # (claim, passage) -> probs; filled by prefetch()
        self.match_min = MATCH_MIN if match_min is None else match_min
        # Tests that inject predict_fn get no matcher unless they pass one (no model download).
        self.matcher = matcher if matcher is not None else (ClaimMatcher() if predict_fn is None and self.match_min > 0 else None)
        self.entail_mode = ENTAIL_MODE if entail_mode is None else entail_mode
        self.entail_min = ENTAIL_MIN if entail_min is None else entail_min
        # Same rule as the matcher: needs a matcher (it decides which passages are verdicts), and tests that
        # inject predict_fn get no entailment model unless they pass one.
        self.entailer = entailer if entailer is not None else (
            ClaimEntailment() if predict_fn is None and self.matcher is not None and self.entail_mode != "off" else None)

    def _load(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, dtype=torch.float32).to(self.device).eval()

    def _infer_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """(claim, passage) pairs -> (n, 4) probabilities, in batches; claims may differ."""
        import torch

        if self._model is None:
            self._load()
        out = []
        for i in range(0, len(pairs), self.batch_size):
            batch = list(pairs[i : i + self.batch_size])
            enc = self._tok(
                [c for c, _ in batch], [p for _, p in batch], truncation="only_second", max_length=MAX_LENGTH,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                out.append(torch.softmax(self._model(**enc).logits, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    def predict(self, claim: str, passages: Sequence[str]) -> np.ndarray:
        if self._predict_fn is not None:
            return np.asarray(self._predict_fn(claim, passages), dtype=float)
        pairs = [(claim, p) for p in passages]
        misses = [pr for pr in dict.fromkeys(pairs) if pr not in self._memo]
        fresh = dict(zip(misses, self._infer_pairs(misses))) if misses else {}
        return np.stack([self._memo[pr] if pr in self._memo else fresh[pr] for pr in pairs])

    def prefetch(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Verify (claim, passage) pairs from many claims in shared batches; predict() then reads the cache."""
        if self._predict_fn is not None:
            return
        misses = [pr for pr in dict.fromkeys(pairs) if pr not in self._memo]
        if misses:
            self._memo.update(zip(misses, self._infer_pairs(misses)))

    def clear_memo(self) -> None:
        self._memo.clear()
        if self.matcher is not None:
            self.matcher.clear_memo()
        if self.entailer is not None:
            self.entailer.clear_memo()

    def claim_matches(self, claim: str, hits: Sequence) -> np.ndarray:
        """Per hit: similarity of the claim it judged to `claim` (NaN: the passage is not a verdict)."""
        if self.matcher is None or not hits:
            return np.full(len(hits), np.nan)
        return self.matcher.similarity(claim, [claim_behind(h.passage) for h in hits])

    def claim_entailment(self, claim: str, hits: Sequence, matches: Sequence[float] | None = None) -> np.ndarray | None:
        """Per hit: [P(claim entails the claim it judged), P(the claim it judged entails claim)]; NaN rows for
        passages that are not verdicts or that already fail the similarity threshold (`matches` given). None
        when entailment is off."""
        if self.entailer is None or self.entail_mode == "off":
            return None
        others = [claim_behind(h.passage) for h in hits]
        if matches is not None:
            others = [o if not (m < self.match_min) else None for o, m in zip(others, matches)]
        if self.entail_mode == "same":
            return self.entailer.four_ways(claim, others)
        return self.entailer.both_ways(claim, others)

    def verify(self, claim: str, hits: Sequence, no_support: set[str] | frozenset = frozenset()) -> Verdict:
        """no_support: ids of passages that may not confirm this claim (about another time, see
        src/retrieval/coverage.py `time_mismatch_ids`); they can still refute."""
        if not hits:
            probs = np.zeros(4)
            probs[NEE] = 1.0
            return Verdict(LABELS[NEE], dict(zip(LABELS, probs.tolist())), math.nan, 0.0, [])
        passage_probs = self.predict(claim, [h.passage.text for h in hits])
        matches = self.claim_matches(claim, hits)
        return verdict_from(hits, passage_probs, matches,
                            [reliability_of(h.passage.domain, self.reliability) for h in hits], self.match_min,
                            no_support=no_support, entail=self.claim_entailment(claim, hits, matches),
                            entail_mode=self.entail_mode, entail_min=self.entail_min)


# Corroboration-only passages (2026-11-03): Wikipedia text and news reports may *confirm* a claim, but a
# "Refuted" from them is not trusted. Measured on India dev claims: relevant Wikipedia text was read as
# Supported for 20% of true claims and 0% of fakes (informative), but as Refuted for 30% of BOTH (noise:
# small differences in detail such as "eight killed" vs "ten killed"). Only fact-checks and AVeriTeC
# evidence refute. For these passages P(Refuted) and P(Conflicting) are moved to NEE.
SUPPORT_ONLY_PREFIXES = ("wiki:", "news:")
SUPPORT_MIN = 0.5  # such a passage counts only if the verifier reads it as Supported with P > 0.5


def is_confirm_only(hit, prefixes: Sequence[str] = SUPPORT_ONLY_PREFIXES) -> bool:
    return str(hit.passage.id).startswith(tuple(prefixes))


def evidence_budget(hits: Sequence, k: int = 5, k_confirm: int | None = None,
                    prefixes: Sequence[str] = SUPPORT_ONLY_PREFIXES) -> list:
    """The passages a claim is verified on: the `k` best that can refute (offline index, fact-checks) plus the
    `k_confirm` best that may only confirm (Wikipedia, news reports), each kind ranked by re-ranker score among its
    own kind, then merged by score.

    Before 2026-11-07 both kinds shared one top-k. For well-known myths the Wikipedia articles about them ranked
    highest; they cannot refute, so they did not count, and they pushed out the index passages that refute the
    claim: "The earth is flat." and "Vaccines cause autism." came back Unverified. With separate budgets the set
    always contains the old shared top-k, so the change only adds passages: refutations that were crowded out, and
    confirmations that were outranked."""
    k_confirm = k if k_confirm is None else k_confirm
    ranked = sorted(hits, key=lambda h: -h.score)
    keep = [h for h in ranked if not is_confirm_only(h, prefixes)][:k]
    keep += [h for h in ranked if is_confirm_only(h, prefixes)][:k_confirm]
    return sorted(keep, key=lambda h: -h.score)


def verdict_from(hits: Sequence, passage_probs: np.ndarray, matches: Sequence[float], reliabilities: Sequence[float],
                 match_min: float = MATCH_MIN, support_only: Sequence[str] = SUPPORT_ONLY_PREFIXES,
                 no_support: set[str] | frozenset = frozenset(), entail: np.ndarray | None = None,
                 entail_mode: str = "off", entail_min: float = ENTAIL_MIN) -> Verdict:
    """One verdict from per-passage distributions. A passage judging a different claim (match < match_min)
    is left out of the vote: it neither supports nor refutes, and it does not dilute the passages that
    are about this claim. If none is left, the verdict is NEE with no score. Relevance is the best
    re-ranker score among the passages that count. Kept separate from ClaimVerifier so threshold
    tuning reuses exactly this code.
    entail (n, 2) with entail_mode "both" / "directional": a verdict passage that passed the similarity
    threshold must also pass `entailment_allows` (NaN rows, i.e. passages that are not verdicts, are not tested).
    entail (n, 4) with entail_mode "same": it must pass `same_claim_allows`; one that does not is marked related."""
    passage_probs = np.array(passage_probs, dtype=float).reshape(-1, len(LABELS))
    matches = np.asarray(matches, dtype=float)
    counted = ~(matches < match_min)  # NaN (not a verdict) counts
    # Time rule (2026-11-05): a passage about another time cannot confirm (Wikipedia's 2016 demonetisation
    # text "confirmed" "Rs 500 notes banned from tomorrow"). Its P(Supported) becomes "says nothing".
    for i, h in enumerate(hits):
        if str(h.passage.id) in no_support:
            p = passage_probs[i]
            passage_probs[i] = [0.0, p[R], p[NEE] + p[S], p[C]]
    related = np.zeros(len(hits), dtype=bool)
    if entail is not None and entail_mode == "same":
        entail = np.asarray(entail, dtype=float).reshape(-1, 4)
        for i in np.flatnonzero(counted & ~np.isnan(matches)):
            if not np.isnan(entail[i]).any() and not same_claim_allows(
                    passage_probs[i], entail[i], matches[i], fact_check_answer(hits[i].passage.text), entail_min):
                counted[i] = False
                related[i] = True
    elif entail is not None and entail_mode != "off":
        entail = np.asarray(entail, dtype=float).reshape(-1, 2)
        for i in np.flatnonzero(counted & ~np.isnan(matches)):
            if not np.isnan(entail[i]).any() and not entailment_allows(passage_probs[i], entail[i], entail_mode, entail_min):
                counted[i] = False
    if support_only:
        for i, h in enumerate(hits):
            if str(h.passage.id).startswith(tuple(support_only)):
                p = passage_probs[i]
                if p[S] > SUPPORT_MIN:  # it confirms: keep only the support, the rest says nothing
                    passage_probs[i] = [p[S], 0.0, 1.0 - p[S], 0.0]
                else:  # it does not confirm: left out of the vote (a tiny P(S) alone would read as score 1.0)
                    counted[i] = False
    w = np.zeros(len(hits))
    if counted.any():
        idx = np.flatnonzero(counted)
        probs, w_used = aggregate(passage_probs[idx], [hits[i].score for i in idx], [reliabilities[i] for i in idx])
        w[idx] = w_used
        score, conf = credibility_from_probs(probs)
    else:
        probs = np.eye(len(LABELS))[NEE]
        score, conf = math.nan, 0.0
    evidence = [
        EvidenceUse(h.passage.id, h.passage.text, h.passage.source_url, h.passage.domain, float(wi),
                    dict(zip(LABELS, pp.tolist())), h.passage.date, float(m), bool(c), bool(r))
        for h, wi, pp, m, c, r in sorted(zip(hits, w, passage_probs, matches, counted, related),
                                         key=lambda t: (not t[4], -t[1]))
    ]
    scores = [h.score for h, c in zip(hits, counted) if c]
    relevance = 1.0 / (1.0 + math.exp(-max(scores))) if scores else math.nan
    return Verdict(LABELS[int(np.argmax(probs))], dict(zip(LABELS, probs.tolist())), score, conf, evidence, relevance)


# --- Evaluation (macro-F1, accuracy, confusion matrix) ----------------------

def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str] = LABELS) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix
    (rows = true label, columns = predicted), for label ids 0..len(labels)-1."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    idx = list(range(len(labels)))
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])}
                      for i in idx},
        "confusion_matrix": {"labels": list(labels),
                             "rows_true_cols_predicted": confusion_matrix(y_true, y_pred, labels=idx).tolist()},
    }


def evaluate_verifier(df: pd.DataFrame, verifier: ClaimVerifier, split: str = "test", searcher=None, reranker=None,
                      k: int = 5) -> dict:
    """4-way verdicts for AVeriTeC `split` claims, in three settings:

    gold_concat    claim + all its gold QA evidence in one input (the "concat" training
                   format): the verifier's ceiling when the evidence is perfect.
    gold_passages  each gold answer verified on its own, then averaged: perfect retrieval,
                   but one passage at a time as at inference.
    retrieved      end-to-end, as the pipeline runs: top-k passages from the deployed index
                   (train claims only) + cross-encoder re-rank + weighted vote. Also reported
                   with the pipeline's relevance gate (off-topic evidence -> NEE).
    """
    from src.fusion.features import RELEVANCE_MIN

    sub = df[df["split"] == split]
    claims = sub["text"].tolist()
    y = np.array([LABEL2ID[label] for label in sub["raw_label"]])
    qa = [_qa_texts(j) for j in sub["questions_json"]]
    out = {"split": split, "n_claims": len(claims),
           "label_counts": {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))},
           "majority_class_accuracy": float(np.bincount(y, minlength=len(LABELS)).max() / len(y))}

    concat = [(c, " ".join(q) if q else "No evidence found.") for c, q in zip(claims, qa)]
    verifier.prefetch(concat)  # batched; predict() below reads the cache
    out["gold_concat"] = classification_metrics(y, [int(np.argmax(verifier.predict(c, [e])[0])) for c, e in concat])

    verifier.prefetch([(c, t) for c, q in zip(claims, qa) for t in q])
    pred = [int(np.argmax(verifier.predict(c, q).mean(axis=0))) if q else NEE for c, q in zip(claims, qa)]
    out["gold_passages"] = classification_metrics(y, pred)

    if searcher is not None:
        searcher.prefetch_queries(claims)
        if hasattr(reranker, "prefetch"):
            reranker.prefetch([(c, searcher.passages[i].text) for c in claims
                               for i in searcher.candidates(c)[0][:50]])
        hits = [searcher.search(c, k=k, reranker=reranker) for c in claims]
        verifier.prefetch([(c, h.passage.text) for c, hs in zip(claims, hits) for h in hs])
        verdicts = [verifier.verify(c, hs) for c, hs in zip(claims, hits)]
        raw = [LABEL2ID[v.label] for v in verdicts]
        off_topic = np.array([not math.isnan(v.relevance) and v.relevance < RELEVANCE_MIN for v in verdicts])
        out["retrieved"] = classification_metrics(y, raw)
        out["retrieved_gated"] = classification_metrics(y, np.where(off_topic, NEE, raw))
        out["retrieved_off_topic_share"] = float(off_topic.mean())
        out["relevance_min"] = RELEVANCE_MIN
    return out


def main() -> None:
    """python -m src.verify.model  -> build pairs from Phase 2 output and fine-tune.
    --eval : test-set metrics of the saved verifier -> reports/verifier_metrics.json."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="evaluate models/verifier-deberta on the test split")
    ap.add_argument("--export-claims", action="store_true",
                    help="write models/retrieval_index/averitec_claims.json (claim matching in Docker / on a server)")
    args = ap.parse_args()
    if args.export_claims:
        print(f"wrote {export_averitec_claims()} claims to {AVERITEC_CLAIMS_JSON}")
        return

    df = pd.read_parquet(ROOT / "data" / "processed" / "averitec.parquet")
    if args.eval:
        from src.retrieval.search import HybridSearcher, cross_encoder_reranker

        searcher = HybridSearcher.load(ROOT / "models" / "retrieval_index")
        report = evaluate_verifier(df, ClaimVerifier(), searcher=searcher, reranker=cross_encoder_reranker())
        (ROOT / "reports").mkdir(parents=True, exist_ok=True)
        (ROOT / "reports" / "verifier_metrics.json").write_text(json.dumps(report, indent=2))
        for setting in ("gold_concat", "gold_passages", "retrieved", "retrieved_gated"):
            if setting in report:
                print(f"{setting:16s} macro-F1 {report[setting]['macro_f1']:.3f}  accuracy {report[setting]['accuracy']:.3f}")
        return
    pairs = build_pairs(df)
    print(pairs.groupby(["split", "kind"]).size().to_string())
    print("validation:", train_verifier(pairs))


if __name__ == "__main__":
    main()
