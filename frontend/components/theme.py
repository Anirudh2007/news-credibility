"""Look of the app: colours that match `.streamlit/config.toml`, one CSS block, and small HTML pieces
(page header, score, credibility meter). Colour is never the only cue: every band and verdict is also
written out in words."""

from __future__ import annotations

from html import escape

import streamlit as st

# Score bands, same cut-offs as the API (0-30, 31-60, 61-80, 81-100). Rust = low, blue = credible
# (a colour-blind-safe pair); the band name is always printed next to the colour.
BAND_RANGES = [("Low", 0, 30), ("Questionable", 30, 60), ("Mostly credible", 60, 80), ("High", 80, 100)]
BAND_COLOURS = {
    "light": {"Low": "#B23E0A", "Questionable": "#B97F06", "Mostly credible": "#5B8FC9", "High": "#1F5FA6"},
    "dark": {"Low": "#F08A4B", "Questionable": "#E9B44C", "Mostly credible": "#8FB8E8", "High": "#BFD8F5"},
}
NEUTRAL = {"light": "#6B7280", "dark": "#8E99A8"}

_VARS = {
    "light": {"ink": "#16202E", "text": "#1D2633", "muted": "#5B6270", "card": "#FFFFFF", "line": "#E3DCCD",
              "accent": "#1F5FA6", "rust": "#B23E0A", "shadow": "rgba(22, 32, 46, 0.07)"},
    "dark": {"ink": "#ECE8DF", "text": "#ECE8DF", "muted": "#A3ADBB", "card": "#172131", "line": "#2A3545",
             "accent": "#8FB8E8", "rust": "#F08A4B", "shadow": "rgba(0, 0, 0, 0.35)"},
}

_CSS = """
<style>
:root {{
  --nc-ink: {ink}; --nc-text: {text}; --nc-muted: {muted}; --nc-card: {card}; --nc-line: {line};
  --nc-accent: {accent}; --nc-rust: {rust}; --nc-shadow: {shadow};
}}
[data-testid="stMainBlockContainer"] {{ max-width: 1120px; padding-top: 2.75rem; padding-bottom: 4rem; }}
[data-testid="stSidebarContent"] {{ padding-top: 0.5rem; }}

.nc-hero {{ margin: 0 0 1.4rem 0; }}
.nc-eyebrow {{ font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; font-weight: 600;
  color: var(--nc-rust); margin: 0 0 0.4rem 0; }}
.nc-hero h1 {{ font-family: 'Libre Baskerville', Georgia, serif; font-weight: 700; font-size: 2.35rem;
  line-height: 1.15; color: var(--nc-ink); margin: 0 0 0.55rem 0; padding: 0; }}
.nc-lede {{ font-size: 1.05rem; line-height: 1.55; color: var(--nc-muted); max-width: 46rem; margin: 0; }}

.nc-steps {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.85rem; margin: 0.4rem 0 0.6rem 0; }}
@media (max-width: 820px) {{ .nc-steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.nc-step {{ background: var(--nc-card); border: 1px solid var(--nc-line); border-radius: 0.9rem; padding: 1rem 1.1rem; }}
.nc-step .nc-num {{ font-family: 'Libre Baskerville', Georgia, serif; color: var(--nc-accent); font-size: 0.95rem; }}
.nc-step .nc-title {{ display: block; font-weight: 700; color: var(--nc-ink); margin: 0.2rem 0 0.25rem 0; }}
.nc-step .nc-body {{ display: block; font-size: 0.9rem; line-height: 1.45; color: var(--nc-muted); }}

.st-key-nc_verdict, .st-key-nc_evidence {{ background: var(--nc-card);
  box-shadow: 0 1px 2px var(--nc-shadow), 0 10px 28px var(--nc-shadow); border-radius: 1rem; }}
.st-key-nc_verdict h3 {{ font-size: 1.45rem; line-height: 1.35; padding-top: 0.35rem; }}
.nc-score {{ display: flex; align-items: baseline; gap: 0.45rem; }}
.nc-score-num {{ font-family: 'Libre Baskerville', Georgia, serif; font-weight: 700; font-size: 4.4rem; line-height: 1; }}
.nc-score-of {{ font-size: 1.1rem; color: var(--nc-muted); }}
.nc-noscore {{ font-family: 'Libre Baskerville', Georgia, serif; font-weight: 700; font-size: 2.2rem; color: var(--nc-ink); }}
.nc-score-caption {{ font-size: 0.85rem; color: var(--nc-muted); margin-top: 0.35rem; line-height: 1.45; }}

.nc-meter {{ position: relative; margin: 0.35rem 0 0.25rem 0; }}
.nc-meter-track {{ display: flex; gap: 3px; height: 10px; }}
.nc-meter-track span {{ display: block; height: 100%; }}
.nc-meter-track span:first-child {{ border-radius: 999px 0 0 999px; }}
.nc-meter-track span:last-child {{ border-radius: 0 999px 999px 0; }}
.nc-meter-marker {{ position: absolute; top: -6px; width: 5px; height: 22px; border-radius: 3px;
  background: var(--nc-ink); transform: translateX(-50%); box-shadow: 0 0 0 3px var(--nc-card); }}
.nc-meter-labels {{ display: flex; gap: 3px; margin-top: 0.4rem; font-size: 0.75rem; color: var(--nc-muted); }}
.nc-meter-labels span {{ display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

.nc-brand {{ display: flex; flex-direction: column; gap: 0.15rem; margin: 0.25rem 0 0.5rem 0; }}
.nc-brand-name {{ font-family: 'Libre Baskerville', Georgia, serif; font-weight: 700; font-size: 1.3rem; line-height: 1.2; }}
.nc-brand-tag {{ font-size: 0.82rem; opacity: 0.75; }}
.nc-status {{ display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.82rem; font-weight: 600; }}
.nc-status::before {{ content: ""; width: 0.55rem; height: 0.55rem; border-radius: 50%; background: #6FCF97; }}

/* phones: last, so these win over the rules above */
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-top: 4.25rem; }}  /* clear the top bar */
  .nc-hero h1 {{ font-size: 1.9rem; }}
  .nc-score-num {{ font-size: 3.4rem; }}
}}
</style>
"""


def inject_css(mode: str = "light") -> None:
    st.html(_CSS.format(**_VARS.get(mode, _VARS["light"])))


def band_colour(band: str, mode: str = "light") -> str:
    return BAND_COLOURS.get(mode, BAND_COLOURS["light"]).get(band, NEUTRAL.get(mode, NEUTRAL["light"]))


def hero(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="nc-hero"><p class="nc-eyebrow">{escape(eyebrow)}</p><h1>{escape(title)}</h1>'
                f'<p class="nc-lede">{escape(lede)}</p></div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> None:
    """A row of numbered cards: (title, one-line description)."""
    cards = "".join(f'<div class="nc-step"><span class="nc-num">{i:02d}</span><span class="nc-title">{escape(t)}</span>'
                    f'<span class="nc-body">{escape(b)}</span></div>' for i, (t, b) in enumerate(items, 1))
    st.markdown(f'<div class="nc-steps">{cards}</div>', unsafe_allow_html=True)


def score_html(score: int | None, band: str, mode: str = "light") -> str:
    if score is None:
        return ('<div class="nc-noscore">No score</div><div class="nc-score-caption">Nothing in this input could be '
                'checked against evidence, so no number is given.</div>')
    return (f'<div class="nc-score"><span class="nc-score-num" style="color:{band_colour(band, mode)}">{int(score)}</span>'
            f'<span class="nc-score-of">/ 100</span></div>')


def meter_html(score: int, mode: str = "light") -> str:
    """Horizontal credibility meter: the four bands as segments, a marker at the score."""
    colours = BAND_COLOURS.get(mode, BAND_COLOURS["light"])
    track = "".join(f'<span style="flex:{hi - lo};background:{colours[name]}"></span>' for name, lo, hi in BAND_RANGES)
    labels = "".join(f'<span style="flex:{hi - lo}">{escape(name)}</span>' for name, lo, hi in BAND_RANGES)
    left = min(max(int(score), 0), 100)
    return (f'<div class="nc-meter" role="img" aria-label="Score {left} of 100"><div class="nc-meter-track">{track}</div>'
            f'<div class="nc-meter-marker" style="left:{left}%"></div><div class="nc-meter-labels">{labels}</div></div>')
