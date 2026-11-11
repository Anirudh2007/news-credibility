"""Input guard: can the submitted text be checked at all? (Phase 8 robustness)

Before this guard, keyboard mashing, emoji, "ok" and Hindi text all came back with a score of
60 ("Questionable"): the meta-learner's prior for a text-only item, shown as if it were a
verdict (reports/robustness_before.json). The guard runs before the pipeline (api/main.py) and
refuses text that is not readable English, saying why. Rules, cheapest first:

    too_short    fewer than MIN_WORDS words or MIN_LETTERS letters
    not_text     under half of the visible characters are letters (symbols, digits, emoji)
    repetitive   one word or one letter makes up most of the text ("BREAKING BREAKING ...", "aaaa")
    not_english  mostly non-Latin script (Hindi, Tamil, ...); romanized Hindi function words
                 (hai, ne, ka, mein, ...); or few English words *and* langdetect names another
                 language (either alone misfires on short headlines)
    gibberish    letter sequences unlike English (keyboard mashing): average surprise per
                 character under a character trigram model of English above MAX_CHAR_NLL

English model: the evidence passages the app already loads (models/retrieval_index/passages.jsonl,
fact-check questions and answers, ~1.3 M characters). Without them only the first three rules run.
Thresholds were set on val texts and checked on test texts (python -m src.fusion.evaluate guard).

Readable English can still hold nothing checkable (word salad, opinions, "They said it was
true."); the pipeline then gives no score instead of a number (src/pipeline.py).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from src.preprocess.clean_text import _latin_share

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "models" / "retrieval_index" / "passages.jsonl"

MIN_WORDS = 2
MIN_LETTERS = 8
MIN_LETTER_SHARE = 0.5  # letters / visible characters
MAX_REPEAT_SHARE = 0.6  # share of the most common word (3+ words) or letter (8+ letters)
MIN_LATIN_SHARE = 0.5
MIN_ENGLISH_WORD_SHARE = 0.6  # below this share of English running words, langdetect is asked ...
MIN_LANGDETECT_PROB = 0.9  # ... and must name another language this surely; alone it misfires on headlines
MIN_WORDS_FOR_LANGUAGE = 4  # lowercase-initial (running-text) words needed before judging the language
HINGLISH_MARKERS = frozenset(
    "hai hain ne ka ki ke ko se mein nahi nahin aur kya tha thi raha rahi rahe gaya gayi diya kiya bhi yeh woh "
    "liya wala wali karo karna hua hui hoga naya nayi aaj abhi kuch bahut sirf lekin agar toh".split()
)
HINGLISH_MIN_SHARE = 0.25  # 2+ different markers making up a quarter of the words
MAX_CHAR_NLL = 3.2  # nats per character, set on val: real-text p99 is 2.5 (claims) to 3.3 (Reddit titles)
MAX_GIBBERISH_WORD_SHARE = 0.5  # gibberish also needs fewer than this share of English words
LETTER_SOUP_NLL = 3.6  # above this, refuse as gibberish before asking for a language (keyboard mash p5: 3.8)
MIN_LETTERS_FOR_NLL = 6
VOCAB_MIN_COUNT = 2

_ALPHABET = "abcdefghijklmnopqrstuvwxyz "
_INDEX = {c: i for i, c in enumerate(_ALPHABET)}
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")

MESSAGES = {
    "too_short": "The text is too short to contain a claim. Enter a headline or a sentence that states something.",
    "not_text": "The text is mostly symbols, digits or emoji, not words.",
    "repetitive": "The text repeats one word or letter over and over.",
    "not_english": "The text does not look like English. Only English text can be checked for now.",
    "gibberish": "The text does not look like real words (it reads like random letters).",
}


def _ascii_letters(text: str) -> str:
    """Lowercase a-z and single spaces; accents folded (é -> e), everything else a space."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z]+", " ", folded).split())


class CharModel:
    """Character trigram model of English: P(c | previous two characters) over a-z and space,
    add-one smoothed. `nll` is the average surprise per character in nats: English prose sits
    around 2.0-2.4, random letters near or above log(27) = 3.3."""

    def __init__(self, texts: Iterable[str]):
        counts = np.ones((27, 27, 27))
        for t in texts:
            s = " " + _ascii_letters(t) + " "
            if len(s) >= 3:
                idx = np.fromiter((_INDEX[c] for c in s), dtype=np.intp)
                np.add.at(counts, (idx[:-2], idx[1:-1], idx[2:]), 1)
        self.logp = np.log(counts / counts.sum(axis=2, keepdims=True))

    def nll(self, text: str) -> float:
        s = " " + _ascii_letters(text) + " "
        if len(s) < 4:
            return float("nan")
        idx = np.fromiter((_INDEX[c] for c in s), dtype=np.intp)
        return float(-self.logp[idx[:-2], idx[1:-1], idx[2:]].mean())


@dataclass
class GuardResult:
    ok: bool
    reason: str | None = None  # one of MESSAGES' keys
    message: str | None = None
    details: dict = field(default_factory=dict)


class InputGuard:
    """check(text) -> GuardResult.

    corpus: English texts for the character model and word list; None reads the evidence passages
        (if present), [] runs the corpus-free rules only (tests).
    words: extra English words; None adds the vocabulary of the MiniLM encoder the pipeline
        already uses (~21k whole words: the passages alone lack everyday words like "tiger")."""

    def __init__(self, corpus: Iterable[str] | None = None, words: Iterable[str] | None = None):
        texts = list(_read_corpus(CORPUS) if corpus is None else corpus)
        counts = Counter(w.lower() for t in texts for w in _WORD_RE.findall(t))
        self.vocab = {w for w, n in counts.items() if n >= VOCAB_MIN_COUNT}
        self.vocab |= {w.lower() for w in (_encoder_words() if words is None and texts else words or [])}
        self.chars = CharModel(texts) if texts else None

    @property
    def status(self) -> str:
        if self.chars is None:
            return "rules only (no English corpus: language and gibberish checks off)"
        return f"ready ({len(self.vocab)} English words)"

    def check(self, text: str | None, caption: bool = False) -> GuardResult:
        """caption=True: the text comes with an image, so a short or playful caption ("2 cats",
        "Ribbit ribbit") is fine; only letter soup and other languages are refused."""
        raw = (text or "").strip()
        text = _strip_social(raw)
        if raw and not text and not caption:  # only links, handles or emoji
            return self._no("not_text", {"letters": 0, "words": 0})
        visible = [c for c in text if not c.isspace()]
        letters = [c for c in visible if c.isalpha()]
        words = _WORD_RE.findall(text)
        d = {"letters": len(letters), "words": len(words)}

        if not visible:
            return GuardResult(True, details=d)  # no text at all: the API decides (image/URL only)
        # measure everything first (cheap), then apply the rules in order
        d["letter_share"] = round(len(letters) / len(visible), 3)
        lower = [w.lower() for w in words]
        if lower:
            d["top_word_share"] = round(Counter(lower).most_common(1)[0][1] / len(lower), 3)
        if letters:
            d["top_letter_share"] = round(Counter(c.lower() for c in letters).most_common(1)[0][1] / len(letters), 3)
            d["latin_share"] = round(_latin_share(text), 3)
        if self.chars is not None and lower:
            d["english_word_share"] = round(sum(w in self.vocab for w in lower) / len(lower), 3)
            if len(letters) >= MIN_LETTERS_FOR_NLL:
                d["char_nll"] = round(self.chars.nll(text), 3)

        if not caption:
            if d["letter_share"] < MIN_LETTER_SHARE:
                return self._no("not_text", d)
            if (len(words) >= 3 and d["top_word_share"] >= MAX_REPEAT_SHARE) or \
                    (len(letters) >= MIN_LETTERS and d["top_letter_share"] >= MAX_REPEAT_SHARE):
                return self._no("repetitive", d)
            if len(words) < MIN_WORDS or len(letters) < MIN_LETTERS:
                return self._no("too_short", d)
        if not letters:
            return GuardResult(True, details=d)  # caption without letters: the image carries the item
        if d["latin_share"] < MIN_LATIN_SHARE:
            return self._no("not_english", d)
        share = d.get("english_word_share")
        if share is None:  # no English corpus: rules above only
            return GuardResult(True, details=d)
        # Letter soup: unlike English letter by letter *and* few English words. Either alone misfires
        # on Indian names ("ISRO launched Chandrayaan-3" has rare letter sequences). A caption needs
        # 3+ words: one-word captions ("Hedgehog", "Startrek") are often missing from the word list.
        nll = d.get("char_nll", 0.0)
        soup = nll > MAX_CHAR_NLL and share < MAX_GIBBERISH_WORD_SHARE and (not caption or len(words) >= 3)
        if soup and nll > LETTER_SOUP_NLL:  # far from any language: random letters, keyboard runs
            return self._no("gibberish", d)
        # Romanized Hindi ("Modi ji ne naya rule banaya hai") reads as English letter by letter and
        # several of its words are in the English list, so it is caught by its function words.
        hindi = {w for w in lower if w in HINGLISH_MARKERS}
        d["hinglish_markers"] = len(hindi)
        if len(hindi) >= 2 and sum(w in HINGLISH_MARKERS for w in lower) / len(lower) >= HINGLISH_MIN_SHARE:
            return self._no("not_english", d)
        # Other languages: judged on running text only, i.e. words that start lowercase. Names and Title
        # Case headlines ("Gigi Hadid, Zayn Malik Spotted Kissing") fooled langdetect into "100% German".
        running = [w.lower() for w in words if w[:1].islower()]
        if len(running) >= MIN_WORDS_FOR_LANGUAGE:
            run_share = sum(w in self.vocab for w in running) / len(running)
            d["running_english_share"] = round(run_share, 3)
            if run_share < MIN_ENGLISH_WORD_SHARE:
                lang, prob = _detect(text)
                d["langdetect"] = f"{lang} {prob:.2f}" if lang else None
                if lang and lang != "en" and prob >= MIN_LANGDETECT_PROB:
                    return self._no("not_english", d)
        if soup:
            return self._no("gibberish", d)
        return GuardResult(True, details=d)

    @staticmethod
    def _no(reason: str, details: dict) -> GuardResult:
        return GuardResult(False, reason, MESSAGES[reason], details)


_SOCIAL_RE = re.compile(r"https?://\S+|www\.\S+|@\w+|\bRT\b")


def _strip_social(text: str) -> str:
    """Tweets: links, @handles and "RT" are not words; a hashtag's words are
    ("#PrayForBoston" -> "Pray For Boston")."""
    text = _SOCIAL_RE.sub(" ", text).replace("#", " ")
    return " ".join(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).split())


def _read_corpus(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def _encoder_words() -> list[str]:
    """Whole words in the MiniLM tokenizer's vocabulary (uncased BERT word list), from the local
    model cache the pipeline fills anyway. Empty if it cannot be loaded."""
    try:
        from transformers import AutoTokenizer

        from src.retrieval.search import DENSE_MODEL

        vocab = AutoTokenizer.from_pretrained(DENSE_MODEL).get_vocab()
    except Exception:  # offline without a cache, or transformers missing
        return []
    return [w for w in vocab if w.isalpha() and len(w) > 1]


def _detect(text: str) -> tuple[str | None, float]:
    """(most likely language, its probability) from langdetect (seeded in clean_text)."""
    from src.preprocess.clean_text import _HAVE_LANGDETECT

    if not _HAVE_LANGDETECT:
        return None, 0.0
    from langdetect import LangDetectException, detect_langs

    try:
        best = detect_langs(text)[0]
    except LangDetectException:
        return None, 0.0
    return best.lang, float(best.prob)


_default: InputGuard | None = None


def default_guard() -> InputGuard:
    """Process-wide guard, built once (reads the evidence passages on first use)."""
    global _default
    if _default is None:
        _default = InputGuard()
    return _default
