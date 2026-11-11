"""Channel 3A: claim extraction (Phase 3A).

    text -> sentences (spaCy) -> check-worthiness score (DeBERTa) -> keep >= threshold
         -> self-contained claim (relative dates resolved) -> entities, dates, places (spaCy NER)

Check-worthy = a sentence states a fact that could be checked against evidence
("Unemployment fell to 6.1% in March"), as opposed to opinion, questions or
filler ("This is a sad day for democracy"). Only check-worthy sentences go on to
evidence retrieval (3B) and verification (3C), which are expensive.

Two scorers share one interface (`score(sentences) -> list[float]`):
- `DebertaCheckWorthiness`: fine-tuned DeBERTa-v3 classifier (the real model;
  train it with `train_checkworthiness`, ideally on a Colab/Kaggle GPU).
- `HeuristicCheckWorthiness`: rule-based fallback used until the model exists,
  and as a baseline to beat in the evaluation.

spaCy is optional at import time: on machines where its compiled parts cannot
load, a regex sentence splitter is used and NER returns nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKWORTHY_DIR = ROOT / "models" / "checkworthy-deberta"
BASE_MODEL = "microsoft/deberta-v3-small"  # small: fits free GPUs; swap to -base if time allows
THRESHOLD = 0.5
MAX_LENGTH = 128  # Phase 2 EDA: 99% of AVeriTeC claims are <= 47 words


# --- Data classes -------------------------------------------------------------

@dataclass
class Sentence:
    text: str
    start: int  # character offsets in the cleaned input
    end: int


@dataclass
class Claim:
    text: str  # self-contained claim (relative dates resolved)
    sentence: str  # original sentence
    checkworthiness: float
    entities: list[tuple[str, str]] = field(default_factory=list)  # (text, spaCy label)
    dates: list[str] = field(default_factory=list)  # ISO dates mentioned or resolved
    locations: list[str] = field(default_factory=list)  # GPE/LOC/FAC entities


# --- Sentence splitting and NER (spaCy, with fallback) -------------------------

@lru_cache(maxsize=2)
def load_nlp(model: str = "en_core_web_sm"):
    """spaCy pipeline, or None if spaCy cannot load on this machine."""
    try:
        import spacy
    except Exception:  # ImportError, or OS policy blocking its DLLs
        return None
    try:
        return spacy.load(model)
    except OSError:  # model not downloaded: sentence splitting still works, NER does not
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


# Split after . ! ? when followed by space + capital/digit/quote. Skips common
# abbreviations and decimals ("Dr. Rao", "U.S. Army", "6.1%").
_ABBREV = "".join(
    rf"(?<!\b{a})" for a in (r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"St\.", r"vs\.", r"No\.", r"U\.S\.", r"U\.K\.", r"e\.g\.", r"i\.e\.")
)
_SENT_END_RE = re.compile(_ABBREV + r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def _regex_sentences(text: str) -> list[Sentence]:
    out, start = [], 0
    for m in _SENT_END_RE.finditer(text):
        seg = text[start : m.start() + len(m.group(0).rstrip())]
        if seg.strip():
            out.append(Sentence(seg.strip(), start, start + len(seg.rstrip())))
        start = m.end()
    if text[start:].strip():
        out.append(Sentence(text[start:].strip(), start, len(text.rstrip())))
    return out


def split_sentences(text: str) -> list[Sentence]:
    if not text or not text.strip():
        return []
    nlp = load_nlp()
    if nlp is None:
        return _regex_sentences(text)
    return [Sentence(s.text.strip(), s.start_char, s.end_char) for s in nlp(text).sents if s.text.strip()]


_LOCATION_LABELS = {"GPE", "LOC", "FAC"}


def tag_entities(text: str) -> list[tuple[str, str]]:
    nlp = load_nlp()
    if nlp is None or "ner" not in nlp.pipe_names:
        return []
    return [(e.text, e.label_) for e in nlp(text).ents]


# Entity types that name *what* a claim is about (dates and plain numbers do not).
_SUBJECT_LABELS = {"PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
_GENERIC_NOUNS = frozenset(
    "thing things something anything everything nothing stuff news story stories fact facts truth lie lies way ways "
    "time times day days week weeks month months year years today yesterday tomorrow tonight morning evening night "
    "people person someone somebody everyone everybody anyone one ones lot lots kind sort part case point place".split()
)


@lru_cache(maxsize=4096)
def needs_context(sentence: str) -> bool:
    """True if a claim names nothing evidence could be about: no person, place, organisation
    or other named thing, no proper noun and no specific common noun. "They said it was true."
    and "She confirmed it last week." cannot be verified; searching evidence for them returns
    passages about other claims (one came back 0.97 relevant in the robustness test).
    "Vaccines cause autism." is specific (common nouns). False when spaCy's tagger is missing."""
    nlp = load_nlp()
    if nlp is None or "ner" not in nlp.pipe_names:
        return False
    doc = nlp(sentence)
    if any(e.label_ in _SUBJECT_LABELS for e in doc.ents):
        return False
    return not any(t.pos_ == "PROPN" or (t.pos_ == "NOUN" and t.lemma_.lower() not in _GENERIC_NOUNS) for t in doc)


# --- Check-worthiness scorers -------------------------------------------------

class CheckWorthinessScorer(Protocol):
    def score(self, sentences: Sequence[str]) -> list[float]: ...


_OPINION_RE = re.compile(
    r"\b(i|we) (think|believe|feel|hope)\b|\bin my (opinion|view)\b|\b(should|must|ought to)\b"
    r"|\b(shameful|disgusting|amazing|terrible|wonderful|sad day)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d")
_CAPITALISED_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+")
_FACT_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|will|announced|said|says|claimed|reported|killed|died|"
    r"increased|decreased|rose|fell|passed|signed|launched|banned|approved|arrested|won|lost|cost|costs)\b",
    re.IGNORECASE,
)


class HeuristicCheckWorthiness:
    """Rule-based baseline: numbers, names and factual verbs raise the score;
    questions, opinion phrases and very short sentences lower it."""

    def score(self, sentences: Sequence[str]) -> list[float]:
        return [self._one(s) for s in sentences]

    @staticmethod
    def _one(s: str) -> float:
        words = s.split()
        if len(words) < 4:
            return 0.05
        x = 0.25
        x += 0.25 if _NUMBER_RE.search(s) else 0.0
        x += 0.2 if _CAPITALISED_RE.search(s) else 0.0
        x += 0.2 if _FACT_VERB_RE.search(s) else 0.0
        x -= 0.35 if s.rstrip().endswith("?") else 0.0
        x -= 0.35 if _OPINION_RE.search(s) else 0.0
        return float(min(max(x, 0.0), 1.0))


class DebertaCheckWorthiness:
    """Fine-tuned DeBERTa-v3 binary classifier; label 1 = check-worthy.

    Loads lazily so importing this module never pulls torch weights.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_CHECKWORTHY_DIR, device: str | None = None, batch_size: int = 32):
        self.model_dir = Path(model_dir)
        self.device = device
        self.batch_size = batch_size
        self._tok = self._model = None
        self._memo: dict[str, float] = {}  # filled only by prefetch(); see Pipeline.analyze_batch

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, dtype=torch.float32).to(self.device).eval()

    def _infer(self, sentences: Sequence[str]) -> list[float]:
        import torch

        self._load()
        out: list[float] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = list(sentences[i : i + self.batch_size])
            enc = self._tok(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self._model(**enc).logits, dim=-1)[:, 1]
            out.extend(probs.cpu().tolist())
        return out

    def score(self, sentences: Sequence[str]) -> list[float]:
        if not sentences:
            return []
        misses = [s for s in dict.fromkeys(sentences) if s not in self._memo]
        fresh = dict(zip(misses, self._infer(misses))) if misses else {}
        return [self._memo[s] if s in self._memo else fresh[s] for s in sentences]

    def prefetch(self, sentences: Sequence[str]) -> None:
        """Score many sentences in large batches now; score() then reads the cache."""
        misses = [s for s in dict.fromkeys(sentences) if s not in self._memo]
        if misses:
            self._memo.update(zip(misses, self._infer(misses)))

    def clear_memo(self) -> None:
        self._memo.clear()


def default_scorer() -> CheckWorthinessScorer:
    """DeBERTa if a fine-tuned model is saved, else the heuristic baseline."""
    if (DEFAULT_CHECKWORTHY_DIR / "config.json").exists():
        return DebertaCheckWorthiness()
    return HeuristicCheckWorthiness()


# --- Training setup for the DeBERTa scorer ------------------------------------

def build_checkworthiness_dataset(positives: Sequence[str], negatives: Sequence[str], seed: int = 42):
    """(text, label) DataFrame, shuffled. Positives: AVeriTeC claims (train split
    only, so 3A never sees val/test claims). Negatives: non-check-worthy sentences,
    e.g. CLEF CheckThat! 'no' sentences or opinion sentences from news op-eds."""
    import pandas as pd

    df = pd.DataFrame(
        {"text": list(positives) + list(negatives), "label": [1] * len(positives) + [0] * len(negatives)}
    )
    df = df[df["text"].str.strip() != ""].drop_duplicates("text")
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_checkworthiness(
    train_df,
    val_df,
    output_dir: str | Path = DEFAULT_CHECKWORTHY_DIR,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    seed: int = 42,
):
    """Fine-tune DeBERTa-v3 on (text, label). Saves model + tokenizer to `output_dir`.

    Run on a GPU (Colab/Kaggle); on CPU it works but takes hours. Selects the
    checkpoint with best validation macro-F1, not accuracy, because classes are
    imbalanced.
    """
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, dtype=torch.float32  # transformers 5 keeps checkpoint dtype (fp16) otherwise
    )

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(encode, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False).map(encode, batched=True)

    def metrics(pred):
        y_hat = np.argmax(pred.predictions, axis=-1)
        return {"macro_f1": f1_score(pred.label_ids, y_hat, average="macro")}

    args = TrainingArguments(
        output_dir=str(Path(output_dir) / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
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
    trainer = Trainer(
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


def evaluate_checkworthiness(table, scorers: dict[str, CheckWorthinessScorer], threshold: float = THRESHOLD) -> dict:
    """Test-split metrics per scorer (precision, recall, F1), on all
    test rows and per source (CheckThat! debate sentences alone; AVeriTeC claims are
    positives only, so per-source F1 there is just recall). `table` needs text, label,
    source, split columns (data/processed/checkworthiness.parquet)."""
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

    test = table[table["split"] == "test"].reset_index(drop=True)
    out: dict = {"test_rows": int(len(test)), "threshold": threshold,
                 "rows_by_source_label": {f"{s}/{lab}": int(n) for (s, lab), n in test.groupby(["source", "label"]).size().items()}}
    for name, scorer in scorers.items():
        p = np.asarray(scorer.score(test["text"].tolist()), dtype=float)
        res = {}
        for part, m in [("all", np.ones(len(test), bool)), *[(s, (test["source"] == s).to_numpy()) for s in sorted(test["source"].unique())]]:
            y, yhat = test.loc[m, "label"].to_numpy(int), (p[m] >= threshold).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, labels=[1], zero_division=0)
            entry = {"n": int(m.sum()), "precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0])}
            if len(set(y)) > 1:
                entry["macro_f1"] = float(f1_score(y, yhat, average="macro"))
                entry["roc_auc"] = float(roc_auc_score(y, p[m]))
            res[part] = entry
        out[name] = res
    return out


# --- Claim rewriting (rules) --------------------------------------------------

_RELATIVE_DAYS = {"today": 0, "yesterday": -1, "tomorrow": 1}
_RELATIVE_RE = re.compile(r"\b(today|yesterday|tomorrow|last week|next week|last month|last year|this year)\b", re.IGNORECASE)


def resolve_relative_dates(sentence: str, publish_date: date | None) -> tuple[str, list[str]]:
    """'Protest held yesterday' + 2024-03-10 -> 'Protest held on 2024-03-09'.

    A claim must stand alone: once separated from its article, "yesterday" means
    nothing. Pronoun resolution needs a seq2seq model (Flan-T5) and is left to
    that optional step; dates are handled by rules because they are exact.
    """
    if publish_date is None:
        return sentence, []
    resolved: list[str] = []

    def repl(m: re.Match) -> str:
        phrase = m.group(0).lower()
        if phrase in _RELATIVE_DAYS:
            iso = (publish_date + timedelta(days=_RELATIVE_DAYS[phrase])).isoformat()
            resolved.append(iso)
            return f"on {iso}"
        if phrase in {"last week", "next week"}:
            d = publish_date + timedelta(weeks=-1 if phrase == "last week" else 1)
            resolved.append(d.isoformat())
            return f"in the week of {d.isoformat()}"
        if phrase == "last month":
            first = publish_date.replace(day=1) - timedelta(days=1)
            return f"in {first.strftime('%B %Y')}"
        year = publish_date.year - (1 if phrase == "last year" else 0)
        return f"in {year}"

    return _RELATIVE_RE.sub(repl, sentence), resolved


# --- Public entry point -------------------------------------------------------

def extract_claims(
    text: str,
    publish_date: date | None = None,
    scorer: CheckWorthinessScorer | None = None,
    threshold: float = THRESHOLD,
) -> list[Claim]:
    """Check-worthy claims in `text`, highest score first. Empty list if none:
    the verifier channel (3C) then returns NaN for this item, not 0."""
    from src.preprocess.clean_text import clean_text, extract_dates

    sentences = split_sentences(clean_text(text))
    if not sentences:
        return []
    scorer = scorer or default_scorer()
    scores = scorer.score([s.text for s in sentences])
    claims = []
    for s, p in zip(sentences, scores):
        if p < threshold:
            continue
        claim_text, resolved = resolve_relative_dates(s.text, publish_date)
        ents = tag_entities(claim_text)
        dates = list(dict.fromkeys(extract_dates(claim_text) + resolved))
        claims.append(
            Claim(
                text=claim_text,
                sentence=s.text,
                checkworthiness=float(p),
                entities=ents,
                dates=dates,
                locations=[t for t, label in ents if label in _LOCATION_LABELS],
            )
        )
    return sorted(claims, key=lambda c: c.checkworthiness, reverse=True)
