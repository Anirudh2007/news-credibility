"""When silence is evidence: expected news coverage and the absence rule (Phase 8, 2026-11-05).

"Absence of evidence is not evidence of absence" holds only when evidence would not be expected anyway.
By Bayes' rule, finding no report is evidence against a claim exactly as strongly as reports would be
expected if the claim were true:

    P(true | no trusted report) / P(false | no trusted report)
        = prior odds  x  P(no report | true) / P(no report | false)          (likelihood ratio, LR)

For "a man in a Bihar village rescued a dog" P(no report | true) is close to 1: LR ~ 1, silence says
nothing. For "Shah Rukh Khan died this morning" or "RBI bans Rs 500 notes from tomorrow" every trusted
outlet would carry the story within hours: P(no report | true) is small, so silence is strong evidence.

This module holds the parts that decide when the rule may speak:
1. `claim_time`         when the claim says the event happened (a date, a month, a year, or "today").
2. `time_mismatch_ids`  passages that cannot *confirm* a claim because they are about another time
                        (Wikipedia's 2016 demonetisation text cannot confirm "Rs 500 notes banned from
                        tomorrow"; before this rule it did, and that claim scored 67).
3. `expected_coverage`  would trusted outlets certainly have reported this if it were true? Transparent
                        rules: an event (death, resignation, ban, arrest, "Rs X to every citizen", war ...)
                        + a well-known subject (national institution, or a person / organisation with a
                        Wikipedia article) or national scale, and not a local story.
4. `absence_probability` the Bayes update above: the model's odds (or even odds if no other check found
                        evidence) times LR. LR is measured, not chosen: see ABSENCE_LR.

The rule is applied by the pipeline only when a news search of trusted outlets actually ran and
answered ("silence counts only if we listened"), and when no claim check found evidence that counted.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Iterable, Sequence

# --- Likelihood ratio of "no trusted outlet reported it" for a claim with expected coverage -------------
# LR = P(silence | true) / P(silence | false) among claims the news would cover and whose news search ran.
# 6 Nov (python -m src.fusion.india_tune absence): 35/86 true vs 10/32 fakes "silent", LR 1.26 (CI 0.80-2.36):
# "silence is not evidence". Re-checked 7 Nov (india_tune absence-recheck, cached answers): that measurement had
# three flaws. (1) Silence was read from the claim's top-5 evidence, where Wikipedia / fact-check passages push a
# relevant report out. (2) The denominator counted fakes already refuted by their own fact-check as "not silent".
# (3) The strict date rule dropped every undated report (it keeps hindsight out of the evidence, but an undated
# report still shows the outlets covered the topic): counted, 31 of the 36 "silent" true dev claims had one.
# The instrument now (`news_search.relevant_reports`, `lookup_twice`): every returned report counts, undated ones
# too (unless the URL names a later year); fact-check / debunk headlines never count (they exist because the claim
# is false); a claim that heard nothing is searched again with its keywords.
# Decided by a rule written before the numbers (india_tune absence-hoax, reports/absence_hoax_eval.json), with 60
# invented-event hoaxes added (src/ingest/live_testset.py --hoax; the India sets hold only 4-7 each):
#   dev:  5/86 true big claims silent vs 15/62 fakes, LR 0.27 (95% CI 0.10-0.60): silence IS evidence;
#   test: 11/120 vs 14/62, LR 0.42 (CI 0.21-0.83): confirmed;
#   with the deployed setting (v0.10.1: meta-learner refitted without the wording check, separate evidence budgets):
#   cost: true claims rated Low 4.4% -> 5.7% (dev, target <= 5%: FAILS), 8.7% -> 12.0% (test, target < 10%: FAILS);
#   gain: new hoaxes (own fact-check removed) rated Low 47% -> 53% (dev), 47% -> 70% (test).
# History of the decision: with the first meta-learner it failed the dev target by 3 items (4.4% -> 5.7%); after the
# 7 Nov refit it passed (3.1% -> 4.4%) and was briefly on; once refuting passages were no longer crowded out of the
# evidence (0.10.1) more true claims are rated Low before the rule applies, and it fails again. So the rule is OFF:
# silence is real evidence, but about 1 in 10 real big events is also unreported in the search, and those would be
# marked Low. Only 13 of the 60 hoaxes were silent: a report about the person or topic is "relevant" even to an
# invented claim. NEWSCRED_ABSENCE_LR=0.27 switches it on (the measured LR), for a demonstration.
MEASURED_ABSENCE_LR = 0.27  # India dev set + dev hoaxes, reports/absence_hoax_eval.json
_ABSENCE_ENV = os.environ.get("NEWSCRED_ABSENCE_LR", "").strip().lower()
ABSENCE_LR: float | None = (None if _ABSENCE_ENV in ("", "off", "0", "none", "false") else float(_ABSENCE_ENV))
ABSENCE_LR_SOURCE = ("off: silence is evidence (LR 0.27 dev, 0.42 test) but scoring it would rate 5.7% of true dev "
                     "claims Low (target 5%)" if ABSENCE_LR is None else
                     f"set by NEWSCRED_ABSENCE_LR ({ABSENCE_LR}; measured 0.27, but it fails the 5% target for true claims)")
# What counts as silence. "no_relevant_report": no trusted report about the claim at all (re-ranker relevance
# < 0.5 for every report the search returned, dated or not). "no_confirming_report": trusted reports may exist,
# but none confirms the claim.
# The second also fires when the verifier fails to read a real report as confirmation, so it is riskier for
# true claims; the India dev set decides (python -m src.fusion.india_tune absence).
ABSENCE_MODE = "no_relevant_report"
RELEVANT_REPORT = 0.5  # sigmoid of the cross-encoder logit: its own relevant / not relevant boundary
NEWS_WINDOW_DAYS = 7  # a dated claim: reports from the 7 days up to the claim date (and the day after)
OLD_SUPPORT_DAYS = 31  # a passage dated more than a month before a dated claim cannot confirm it

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_MONTH_RE = "|".join(m.capitalize() for m in MONTHS) + "|" + "|".join(m[:3].capitalize() for m in MONTHS)
_MONTH_YEAR = re.compile(rf"\b({_MONTH_RE})\.?,?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?((?:19|20)\d\d)\b", re.I)
_YEAR = re.compile(r"\b(?:in|of|during|since|by|for|from)\s+((?:19|20)\d\d)\b", re.I)
# Words that put the event at the time of writing. "today"/"yesterday"/"tomorrow" are rewritten to a date by
# 3A when the publish date is known; the rest ("this morning") stay as words.
_CURRENT = re.compile(r"\b(today|tonight|this (?:morning|afternoon|evening|week)|yesterday|last night|tomorrow|"
                      r"from tomorrow|just now|right now|hours? ago|minutes ago|breaking)\b", re.I)
_SHIFT = {"yesterday": -1, "last night": -1, "tomorrow": 1, "from tomorrow": 1}


@dataclass
class ClaimTime:
    day: date | None = None  # the event day, if known to the day
    month: tuple[int, int] | None = None  # (year, month), if known to the month
    year: int | None = None  # if only the year is known
    current: bool = False  # "today", "this morning", "from tomorrow": the event is at the time of writing

    @property
    def known(self) -> bool:
        return self.day is not None or self.month is not None or self.year is not None

    def years(self) -> set[int]:
        """Years a confirming passage may name: the event's year, and the previous one for an event in the first
        week of January (a report of 2 January may describe 31 December)."""
        if self.day is not None:
            return {self.day.year, self.day.year - 1} if (self.day.month, self.day.day) <= (1, 7) else {self.day.year}
        y = self.month[0] if self.month else self.year
        return {y} if y is not None else set()


def _iso(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def claim_time(text: str, dates: Sequence[str] = (), claimed_date=None, publish_date=None,
               today: date | None = None) -> ClaimTime:
    """When the claim says its event happened. Priority: the date the user gave, a full date in the claim
    (3A's `Claim.dates`), a month + year, a year, then words like "today" (resolved against the publish
    date, else today's date). Nothing found: unknown time (no time rule applies)."""
    today = today or date.today()
    if (d := _iso(claimed_date)) is not None:
        return ClaimTime(day=d, month=(d.year, d.month), year=d.year)
    full = [x for x in (_iso(s) for s in dates) if x is not None]
    if full:
        d = max(full)
        return ClaimTime(day=d, month=(d.year, d.month), year=d.year)
    text = text or ""
    if (m := _MONTH_YEAR.search(text)) is not None:
        mon = MONTHS.index(next(n for n in MONTHS if n.startswith(m.group(1).lower()[:3]))) + 1
        return ClaimTime(month=(int(m.group(2)), mon), year=int(m.group(2)))
    if (m := _YEAR.search(text)) is not None:
        return ClaimTime(year=int(m.group(1)))
    if (m := _CURRENT.search(text)) is not None:
        base = _iso(publish_date) or today
        d = base + timedelta(days=_SHIFT.get(m.group(1).lower(), 0))
        return ClaimTime(day=d, month=(d.year, d.month), year=d.year, current=True)
    return ClaimTime()


def news_window(ct: ClaimTime, before=None, today: date | None = None) -> tuple[date | None, date]:
    """(start, end) of the news search, never after `before` or today (no hindsight):
    - the claim names a day: the NEWS_WINDOW_DAYS before it to the day after;
    - a month or a year: that month / year;
    - no time in the claim but the item is dated (`before`): the NEWS_WINDOW_DAYS up to that date;
    - nothing dated at all: every report up to today, so an undated claim about a real past event
      ("Chandrayaan-3 landed on the Moon") still finds its reports."""
    today = today or date.today()
    given = _iso(before)
    end = min(x for x in (given, today) if x is not None)
    if ct.day is not None:
        start, stop = ct.day - timedelta(days=NEWS_WINDOW_DAYS), ct.day + timedelta(days=1)
    elif ct.month is not None:
        y, m = ct.month
        start = date(y, m, 1)
        stop = (date(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)
    elif ct.year is not None:
        start, stop = date(ct.year, 1, 1), date(ct.year, 12, 31)
    elif given is not None:
        start, stop = given - timedelta(days=NEWS_WINDOW_DAYS), given
    else:
        return None, end
    stop = min(stop, end)
    return min(start, stop), stop


# --- 2. Time-consistent support --------------------------------------------------------------------------

def mentions_time(text: str, ct: ClaimTime) -> bool:
    """Does an encyclopaedia passage place something in the claim's year? (The year, not the month: on the India
    sets a month rule also blocked 4 passages that named the right year and confirmed true claims, while the
    year alone already separates "banned from tomorrow" (2026) from the 2016 demonetisation.)"""
    years = ct.years()
    return not years or any(re.search(rf"(?<!\d){y}(?!\d)", text or "") for y in years)


def time_mismatch_ids(hits: Iterable, ct: ClaimTime) -> set[str]:
    """Ids of passages that may not count as *support* for a claim at time `ct` (they may still refute).
    - Wikipedia text: its date is the revision date (always near the claim), so the passage itself must name
      the claim's year.
    - Other dated passages except news reports (fact-checks, the evidence index, the evidence memory): for a
      claim dated to the day, a passage written more than OLD_SUPPORT_DAYS before it is about something
      earlier. A recurring rumour's old debunk still refutes; an old "yes" cannot confirm a new event.
    - News reports come from a search limited to the claim's window: they are about that time."""
    if not ct.known:
        return set()
    out = set()
    for h in hits:
        p = h.passage
        pid = str(p.id)
        if pid.startswith("news:"):
            continue
        if pid.startswith("wiki:"):
            body = p.text.split("): ", 1)[-1]
            if not mentions_time(body, ct):
                out.add(pid)
        elif ct.day is not None and (d := _iso(p.date)) is not None and (ct.day - d).days > OLD_SUPPORT_DAYS:
            out.add(pid)
    return out


# --- 3. Expected coverage ------------------------------------------------------------------------------

EVENTS: dict[str, re.Pattern] = {
    "a death": re.compile(r"\b(dies|died|is dead|found dead|death of|passed away|passes away|killed|assassinated|"
                          r"shot dead|murdered|is no more)\b", re.I),
    "a resignation or removal": re.compile(r"\b(resigns?|resigned|resignation|steps? down|stepped down|sacked|ousted|"
                                           r"impeached|removed (?:as|from))\b", re.I),
    "an arrest or court ruling": re.compile(r"\b(arrested|arrest of|jailed|sentenced|convicted|acquitted|detained|"
                                            r"struck down|strikes down|upheld)\b", re.I),
    "a ban or withdrawal": re.compile(r"\b(bans?|banned|ban on|prohibit(?:s|ed)?|scrap(?:s|ped)?|withdraw[ns]?|"
                                      r"withdrew|demoneti[sz]\w*|invalid|discontinued)\b", re.I),
    "a payment or benefit for everyone": re.compile(r"\b(will (?:give|get|receive|deposit|credit|pay)|"
                                                    r"free (?:\w+ ){0,3}(?:for|to) (?:all|every)|"
                                                    r"waive[sd]?|waiver)\b", re.I),
    "an official announcement": re.compile(r"\b(announce[sd]?|announcing|declare[sd]?|declaring|approve[sd]?|"
                                           r"order(?:s|ed)|confirm(?:s|ed)|launch(?:es|ed)?|introduce[sd]?|"
                                           r"makes? (?:it )?mandatory|lockdown|curfew|emergency|holiday)\b", re.I),
    "a war, attack or disaster": re.compile(r"\b(war|attack(?:s|ed)?|invade[sd]?|invasion|bomb(?:s|ed|ing)?|blasts?|"
                                            r"explosion|earthquake|tsunami|cyclone|floods?|crash(?:es|ed)?|"
                                            r"derail(?:s|ed)?|stampede|coup|nuclear)\b", re.I),
    "an election or contest result": re.compile(r"\b(won|wins|defeated|elected|sworn in|crowned)\b", re.I),
}
# Extraordinary claims: reported worldwide if true, whoever the subject is.
EXTRAORDINARY = re.compile(r"\b(aliens?|ufos?|extraterrestrial|cures? (?:for )?(?:cancer|diabetes|aids|hiv|covid)|"
                           r"cure for|immortal|end of the world|doomsday)\b", re.I)
INSTITUTIONS = re.compile(
    r"\b(government of india|indian government|central government|union government|the centre|union cabinet|"
    r"cabinet|prime minister|pm modi|narendra modi|modi|president of india|president|chief justice|"
    r"chief minister|union minister|finance minister|home minister|governor|rbi|reserve bank|sebi|"
    r"supreme court|high court|election commission|parliament|lok sabha|rajya sabha|ministry|isro|nasa|"
    r"world health organi[sz]ation|united nations|imf|world bank|indian army|army|air force|navy|cbse|ugc|"
    r"ncert|upsc|sbi|irctc|indian railways|railways|uidai|aadhaar|gst council|income tax department|bcci|"
    r"fifa|olympics?|google|microsoft|facebook|whatsapp|instagram|youtube|tata|reliance|adani|ambani|infosys|"
    r"pakistan|china|russia|ukraine|israel|iran|united states|us government)\b", re.I)
# Acronyms that are also ordinary words ("who", "us"): matched only in capitals.
ACRONYMS = re.compile(r"\b(WHO|UN|US|USA|UK|EU|ICC|CBI|ED|NIA)\b")
SCALE = re.compile(
    r"\b(every (?:citizen|indian|household|family|farmer|student|woman|man|person|bank account|voter|worker)s?|"
    r"all (?:citizens|indians|households|families|farmers|students|women|people|bank accounts|voters|workers|"
    r"banks|states|schools)|nationwide|across (?:india|the country)|all over (?:india|the country)|"
    r"(?:entire|whole) (?:country|nation)|worldwide|globally)\b", re.I)
LOCAL = re.compile(
    r"\b(a (?:man|woman|boy|girl|youth|teen(?:ager)?|couple|family|shopkeeper|farmer|student|teacher|doctor|"
    r"driver|resident|child|baby)|locals?|villagers?|(?:in|of|from) (?:a|my|our) (?:village|town|colony|locality|"
    r"society|area|neighbourhood|neighborhood)|my (?:neighbou?r|friend|uncle|aunt|cousin))\b", re.I)
NOTABLE_LABELS = {"PERSON", "ORG"}
MAX_NAME_LOOKUPS = 3  # Wikipedia requests per claim, at most
_NAME_SPAN = re.compile(r"\b[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,3}\b")
_NOT_NAMES = frozenset("the a an this that these those it he she they we i in on at breaking viral".split())


def candidate_names(text: str, entities: Sequence[Sequence[str]] = ()) -> list[str]:
    """Names a claim is about, for the Wikipedia lookup: spaCy PERSON / ORG entities, each widened to the run
    of capitalised words around it (the small spaCy model tags only "Khan" in "Shah Rukh Khan"), then other
    runs of 2-4 capitalised words."""
    spans = []
    for m in _NAME_SPAN.finditer(text or ""):
        words = m.group(0).split()
        while words and words[0].lower() in _NOT_NAMES:
            words = words[1:]
        if len(words) >= 2:
            spans.append(" ".join(words))
    out = []
    for name, label in entities:
        name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I)
        if label in NOTABLE_LABELS and len(name) > 2:
            out.append(next((s for s in spans if name in s), name))
    out += spans
    return list(dict.fromkeys(out))


@dataclass
class Coverage:
    expected: bool  # trusted outlets would report this within days if it were true
    event: str | None = None
    subject: str | None = None
    why: str = ""
    cues: list[str] = field(default_factory=list)
    kind: str | None = None  # why the subject counts: "extraordinary", "scale", "institution", "notable name"

    def as_dict(self) -> dict:
        return {"expected": self.expected, "event": self.event, "subject": self.subject, "why": self.why}


def expected_coverage(text: str, entities: Sequence[Sequence[str]] = (),
                      notable: Callable[[str], bool | None] | None = None) -> Coverage:
    """Would trusted outlets certainly have reported this claim if it were true?

    expected = (an extraordinary claim) or (an event of a covered kind AND a well-known subject or national
    scale), and not a local story ("a man in a village"; national scale overrides that).
    `notable(name)`: True if a person / organisation named in the claim has a Wikipedia article (the
    pipeline asks Wikipedia; None = unknown). Only asked when an event was found, to save requests."""
    text = text or ""
    if (m := EXTRAORDINARY.search(text)) is not None:
        return Coverage(True, "an extraordinary claim", m.group(0),
                        f"an extraordinary claim (\"{m.group(0)}\") would be reported worldwide if true",
                        [f"extraordinary:{m.group(0)}"], "extraordinary")
    event = next(((name, m.group(0)) for name, rx in EVENTS.items() if (m := rx.search(text))), None)
    if event is None:
        return Coverage(False, why="no event of a kind that the news always covers")
    scale = SCALE.search(text)
    local = LOCAL.search(text)
    if local and not scale:
        return Coverage(False, event[0], why=f"a local story (\"{local.group(0)}\"): it may never be reported",
                        cues=[f"local:{local.group(0)}"])
    subject = scale.group(0) if scale else None
    kind, tag = ("national scale", "scale") if scale else (None, None)
    shouting = sum(c.isupper() for c in text) > 0.5 * max(1, sum(c.isalpha() for c in text))  # "THE MAN WHO..."
    if subject is None and (m := INSTITUTIONS.search(text) or (None if shouting else ACRONYMS.search(text))):
        subject, kind, tag = m.group(0), "a national leader, institution or major organisation", "institution"
    if subject is None and notable is not None:
        for name in candidate_names(text, entities)[:MAX_NAME_LOOKUPS]:
            if notable(name):
                subject, kind, tag = name, "a well-known person or organisation (has a Wikipedia article)", "notable name"
                break
    if subject is None:
        return Coverage(False, event[0], why="its subject is not known to be prominent")
    return Coverage(True, event[0], subject,
                    f"{event[0]} involving {subject} ({kind}) would be reported by trusted outlets if true",
                    [f"event:{event[1]}", f"subject:{subject}"], tag)


# --- 4. Bayes update ---------------------------------------------------------------------------------

def is_silent(claim_coverage: dict, mode: str | None = None) -> bool:
    """A claim the rule may use: expected to be covered, its news search ran, and (mode, default ABSENCE_MODE)
    no relevant trusted report / no confirming report."""
    mode = mode or ABSENCE_MODE
    cv = claim_coverage or {}
    if not (cv.get("expected") and cv.get("searched")) or cv.get("confirmed"):
        return False
    return mode == "no_confirming_report" or not cv.get("relevant_reports")


def absence_probability(p_model: float | None, lr: float = ABSENCE_LR) -> float:
    """Posterior P(credible) after "no trusted report" with likelihood ratio `lr`. Prior: the meta-learner's
    calibrated probability when other checks found evidence, else even odds (no information either way)."""
    prior = 0.5 if p_model is None or (isinstance(p_model, float) and math.isnan(p_model)) else float(p_model)
    prior = min(max(prior, 0.01), 0.99)
    odds = prior / (1.0 - prior) * lr
    return odds / (1.0 + odds)


def likelihood_ratio(found_true: int, n_true: int, found_fake: int, n_fake: int) -> float:
    """LR of "no confirming report" from counts, with add-one smoothing so a zero count cannot give 0 or inf:
    (1 - r_true) / (1 - r_fake), r = (found + 1) / (n + 2)."""
    r_true = (found_true + 1) / (n_true + 2)
    r_fake = (found_fake + 1) / (n_fake + 2)
    return (1.0 - r_true) / (1.0 - r_fake)
