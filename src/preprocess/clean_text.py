"""Text cleaning and date normalisation (Phase 2).

Every function is pure (string in, string out) so the same cleaning runs at
training time and at inference time on user input. Training/serving skew is
avoided by never cleaning text any other way.
"""

from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime, timezone

from dateutil import parser as date_parser

# --- Patterns -----------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+", re.IGNORECASE)

# Emoji and pictograph blocks, plus the joiners/selectors that glue them together.
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U0001F300-\U0001F5FF"  # symbols and pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport and map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FAFF"  # chess, symbols and pictographs extended-A
    "☀-⛿"  # miscellaneous symbols
    "✀-➿"  # dingbats
    "‍"  # zero-width joiner
    "️"  # variation selector-16
    "⃣"  # combining enclosing keycap
    "]+"
)

# Line-level boilerplate common in scraped articles. Kept conservative: a false
# positive deletes real content, a false negative only leaves noise.
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"advertisement|sponsored( content)?|"
    r"(?:click|tap) here.*|read more.*|continue reading.*|"
    r"subscribe to (?:our|the) newsletter.*|sign up for .*newsletter.*|"
    r"share (?:this|on) .*|follow us on .*|"
    r"all rights reserved.*|copyright \S+.*|\(c\) \d{4}.*"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_WS_RE = re.compile(r"\s+")


# --- Cleaning -----------------------------------------------------------------

def strip_html(text: str) -> str:
    """Remove script/style blocks and tags, then decode entities (&amp; -> &)."""
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def strip_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def strip_boilerplate(text: str) -> str:
    return _BOILERPLATE_RE.sub(" ", text)


def normalise_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_text(text: object) -> str:
    """Full cleaning chain. Non-strings (None, NaN) become an empty string.

    Order matters: HTML first (tags can hide URLs and entities), boilerplate
    before whitespace collapse (it is matched line by line), whitespace last.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = strip_html(text)
    text = strip_urls(text)
    text = strip_emojis(text)
    text = strip_boilerplate(text)
    return normalise_whitespace(text)


# --- Language filter ----------------------------------------------------------

# langdetect misfires on short text: on AVeriTeC (English by construction) it
# tagged 45 claims under 100 chars as Afrikaans, Somali, Catalan, ... and
# "Senate passes bipartisan infrastructure bill" as French. Below this length
# only the script is checked.
MIN_CHARS_FOR_LANGDETECT = 100
MIN_LATIN_SHARE = 0.7

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # langdetect is random by default; fix it for reproducibility
    _HAVE_LANGDETECT = True
except ImportError:  # pragma: no cover - fallback only
    _HAVE_LANGDETECT = False


def _latin_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0
    latin = sum(ch.isascii() or "LATIN" in unicodedata.name(ch, "") for ch in letters)
    return latin / len(letters)


def is_english(text: str) -> bool:
    """True if `text` is English (long text) or Latin-script (short text).

    Short strings (headlines, tweets) only need to be mostly Latin script, which
    drops Hindi, Chinese, Arabic, ... but keeps a short Spanish headline. That
    trade-off is deliberate: wrongly dropping English headlines would bias the
    data toward long text, and the datasets used here are English-first.
    """
    if not text:
        return True
    if len(text) < MIN_CHARS_FOR_LANGDETECT:
        return _latin_share(text) >= MIN_LATIN_SHARE
    if _HAVE_LANGDETECT:
        try:
            return detect(text) == "en"
        except LangDetectException:
            return True
    ascii_ratio = sum(ch.isascii() for ch in text) / len(text)
    return ascii_ratio > 0.9


# --- Dates --------------------------------------------------------------------

_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_IN_TEXT_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"  # 2020-03-12
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{4}"  # 12/03/2020, 12-3-2020
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\.?,?\s+\d{{4}}"  # 12 March 2020
    rf"|{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"  # March 12, 2020
    r")\b",
    re.IGNORECASE,
)


def parse_date(value: object, dayfirst: bool = True) -> str | None:
    """Parse one date value into ISO `YYYY-MM-DD`; None if unparseable.

    Accepts strings in most human formats and Unix timestamps (int/float, as in
    Fakeddit `created_utc`). `dayfirst=True` because AVeriTeC and Indian sources
    write 25-8-2020 (day-month-year); ISO strings are unaffected by it.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if re.fullmatch(r"\d{9,11}(?:\.\d+)?", s):  # Unix timestamp stored as text
        return parse_date(float(s))
    try:
        is_iso = bool(re.match(r"\d{4}-\d{1,2}-\d{1,2}", s))
        dt = date_parser.parse(s, dayfirst=dayfirst and not is_iso, fuzzy=False)
    except (ValueError, OverflowError, TypeError):
        return None
    if not 1900 <= dt.year <= 2100:
        return None
    return dt.date().isoformat()


def extract_dates(text: str, dayfirst: bool = True) -> list[str]:
    """All dates mentioned inside `text`, as unique ISO strings in order of appearance.

    Used later by the temporal channel (3G) to compare claimed dates with the
    earliest copy of a story or image.
    """
    if not text:
        return []
    out: list[str] = []
    for match in _DATE_IN_TEXT_RE.finditer(text):
        iso = parse_date(match.group(0), dayfirst=dayfirst)
        if iso and iso not in out:
            out.append(iso)
    return out
