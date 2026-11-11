"""Plotly figure for "what affected this score" (SHAP points), and band styling.

Colour rules (colour-blind-safe, light + dark):
- points are a diverging quantity: blue = raised the score, rust = lowered it, neutral grey = not a check
  (`not_applicable`, `calibration`). Each bar also carries a signed text label, so the sign is never shown by
  colour alone.
- score bands are a status: each band has a colour, but the band name and an icon are always printed next to it.
"""

from __future__ import annotations

import plotly.graph_objects as go

from components.theme import band_colour

POSITIVE = {"light": "#1F5FA6", "dark": "#8FB8E8"}
NEGATIVE = {"light": "#C4480E", "dark": "#F08A4B"}
NEUTRAL = {"light": "#B8B2A6", "dark": "#5D6470"}
GRID = {"light": "#E3DCCD", "dark": "#2A3545"}
TEXT = {"light": "#3F4652", "dark": "#C9CED6"}
FONT = "Public Sans, Arial, sans-serif"

BAND_ICONS = {"Low": ":material/dangerous:", "Questionable": ":material/warning:",
              "Mostly credible": ":material/check_circle:", "High": ":material/verified:"}
CHECK_LABELS = {
    "claim_verification": "Claim verification",
    "text_image_match": "Caption-image match",
    "media_reuse": "Media reuse",
    "source": "Source rating",
    "temporal": "Date consistency",
    "input_type": "Input type",
    "missing_coverage": "Not reported by trusted news",
    "not_applicable": "Checks that did not run",
    "calibration": "Calibration",
}


def band_style(band: str, mode: str = "light") -> tuple[str, str]:
    """(colour, icon) for a band name; grey and a help icon for Unknown / Unverified."""
    return band_colour(band, mode), BAND_ICONS.get(band, ":material/help:")


def contributions_bar(contributions: list[dict], mode: str = "light") -> go.Figure:
    """Horizontal bars of points per check, largest effect on top."""
    rows = sorted((c for c in contributions if abs(c["points"]) >= 0.05),  # drop zero bars
                  key=lambda c: abs(c["points"]))  # plotly draws bottom-up
    labels = [CHECK_LABELS.get(c["check"], c["check"]) for c in rows]
    colours = [
        NEUTRAL[mode] if c["check"] in ("not_applicable", "calibration")
        else POSITIVE[mode] if c["points"] >= 0 else NEGATIVE[mode]
        for c in rows
    ]
    fig = go.Figure(go.Bar(
        x=[c["points"] for c in rows], y=labels, orientation="h",
        marker={"color": colours, "cornerradius": 6},
        text=[f"{c['points']:+.1f}" for c in rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f} points<extra></extra>",
    ))
    span = max([abs(c["points"]) for c in rows] + [5])
    fig.update_layout(
        height=70 + 38 * len(rows), margin=dict(l=10, r=48, t=6, b=34), bargap=0.4,
        font=dict(family=FONT, size=13, color=TEXT[mode]),
        xaxis=dict(title=dict(text="points (relative to an average item)", font=dict(size=12)),
                   range=[-span * 1.4, span * 1.4], zeroline=True,  # room for the value labels beside the bars zerolinewidth=1, zerolinecolor=TEXT[mode],
                   gridcolor=GRID[mode]),
        yaxis=dict(showgrid=False), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
