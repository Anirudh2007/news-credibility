"""Results view, top to bottom:
verdict card (score, band, one-sentence summary, credibility meter) -> article read from a link
-> what the evidence says (quoted) -> what the news says (when silence was used) -> why this score + what affected it (SHAP) -> claim cards
(with "previously fact-checked" badges) -> media -> findings -> checks skipped -> disclaimer -> feedback."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client as api
from components.charts import CHECK_LABELS, band_style, contributions_bar
from components.theme import meter_html, score_html

VERDICT_STYLE = {  # icon + label always shown together; colour is never the only cue
    "Supported": (":material/check_circle:", "blue"),
    "Refuted": (":material/cancel:", "red"),
    "Not Enough Evidence": (":material/help:", "gray"),
    "Conflicting Evidence/Cherrypicking": (":material/balance:", "orange"),
}
BADGE_COLOUR = {"Low": "red", "Questionable": "orange", "Mostly credible": "blue", "High": "blue"}
LABEL_OPTIONS = {"credible": "Credible", "not_credible": "Not credible", "mixed": "Mixed / partly true",
                 "unsure": "Not sure"}


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def render_report(report: dict, image_bytes: bytes | None = None) -> None:
    mode = theme_mode()
    score, band = report.get("score"), report.get("band", "Unknown")
    _, icon = band_style(band, mode)
    explanation = report.get("explanation") or {}

    with st.container(border=True, key="nc_verdict"):
        left, right = st.columns([2, 3], vertical_alignment="center", gap="large")
        with left:
            st.markdown(score_html(score, band, mode), unsafe_allow_html=True)
        with right:
            st.badge(band, icon=icon, color=BADGE_COLOUR.get(band, "gray"))
            st.subheader(explanation.get("summary") or (f"Score {score}/100 ({band})" if score is not None else "No score"))
            if report.get("meta_model", "").startswith("equal-weight"):
                st.caption("Meta-learner not trained: score is the plain average of the checks that ran.")
            if report.get("cached"):
                st.caption(f"Same input analysed before ({report.get('created_at', '')[:16].replace('T', ' ')} UTC).")
        if score is not None:
            st.markdown(meter_html(score, mode), unsafe_allow_html=True)

    fa = report.get("fetched_article")
    if fa and not fa.get("error") and fa.get("title"):
        with st.container(border=True):
            st.caption(":material/article: Read from the article page (headline and summary were checked)")
            st.markdown(f"**{fa['title']}**")
            if fa.get("description"):
                st.caption(fa["description"][:300])
            if fa.get("published"):
                st.caption(f"Published {fa['published']}")
    elif fa and fa.get("error"):
        st.caption(f":material/link_off: The article itself was not checked: {fa['error']}. "
                   "Paste the headline or text to check the claims.")

    evidence_box(report.get("evidence_box") or [])
    silence_box(report.get("coverage"))

    contribs = report.get("contributions") or []
    why, what = st.container(), st.container()  # stacked: the bar labels need the full width
    if explanation:
        with why:
            st.markdown("#### Why this score" if score is not None else "#### Why no score")
            st.write(explanation.get("explanation", ""))
            for ev in explanation.get("evidence", []):
                st.caption(f"[{ev['id']}] {ev.get('source') or 'unknown source'}: “{ev['snippet']}”")
            src = "written by an LLM and checked against the evidence" if explanation.get("source") == "llm" \
                else "generated from fixed templates"
            st.caption(f"Explanation {src}.")
    if contribs:
        with what:
            st.markdown("#### What affected this score")
            st.plotly_chart(contributions_bar(contribs, mode), width="stretch",
                            config={"displayModeBar": False}, key=f"shap_{report.get('analysis_id')}")
            baseline = report.get("baseline_score")
            if baseline is not None:
                st.caption(f"Bars start from an average item with the same kind of input ({baseline:.0f}/100) and "
                           "add up to the final score. Blue raised it, rust lowered it, grey is not a check; "
                           "checks that did not run add nothing.")
            with st.expander("Table view"):
                st.dataframe(pd.DataFrame([{"check": CHECK_LABELS.get(c["check"], c["check"]), "points": c["points"],
                                            "ran": c["applicable"]} for c in contribs]),
                             hide_index=True, width="stretch")

    claims = report.get("claims") or []
    if claims:
        st.markdown(f"#### Claims found ({len(claims)})")
        for c in claims:
            claim_card(c)

    media = report.get("media") or {}
    features = report.get("features") or {}
    if image_bytes or features.get("has_image") or media:
        st.markdown("#### Media")
        with st.container(border=True):
            cols = st.columns([1, 2], gap="large")
            with cols[0]:
                if image_bytes:
                    st.image(image_bytes, width="stretch")
                else:
                    st.caption("Image not stored (only its fingerprint).")
            with cols[1]:
                if "reuse_found" in media:
                    st.write(("Earlier copy found" if media["reuse_found"] else "No earlier copy found in the index")
                             + (f" (earliest {media['earliest_match_date']})" if media.get("earliest_match_date") else ""))
                if media.get("location_mismatch"):  # the location check runs silently; only a mismatch is shown
                    st.write(f":material/location_off: The image does not resemble reference photos of "
                             f"{media.get('location_checked')}.")
                if media.get("text_image_match") is not None:
                    sub = f", likely {media['text_image_subtype'].replace('_', ' ')}" if media.get("text_image_subtype") else ""
                    st.write(f"Caption matches image: {round(100 * media['text_image_match'])}%{sub}")

    if report.get("reasons"):
        st.markdown("#### Findings")
        for r in report["reasons"]:
            st.write(f"- {r}")

    if report.get("skipped_checks"):
        with st.expander(f"Checks not run ({len(report['skipped_checks'])}): these did not count against the item"):
            for s in report["skipped_checks"]:
                st.write(f"- {s}")

    st.caption(":material/info: " + report.get("disclaimer", "This score is an aid, not a final judgement. Read the evidence."))
    feedback_form(report)


def claim_card(c: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{c['text']}**")
        verdict = c.get("verdict")
        if verdict:
            v_icon, v_colour = VERDICT_STYLE.get(verdict, (":material/help:", "gray"))
            conf = f" · confidence {round(100 * c['confidence'])}%" if c.get("confidence") is not None else ""
            st.badge(f"{verdict}{conf}", icon=v_icon, color=v_colour)
        elif c.get("context") == "missing":
            st.badge("Needs context: not checked", icon=":material/help:", color="gray")
        rel = c.get("evidence_relevance")
        if c.get("evidence_off_topic"):
            st.caption(":material/info: Closest evidence shown below is about other claims; it did not count.")
        elif verdict and rel is not None and rel < 0.5:
            st.caption(":material/warning: Evidence only loosely related to this claim; weak verdict.")
        cv = c.get("coverage") or {}
        if cv.get("expected"):
            s = cv.get("search") or {}
            if cv.get("searched"):
                found = s.get("trusted_reports", 0)
                st.caption(f":material/newspaper: Would be in the news if true ({cv.get('event')}). Trusted news "
                           f"search: {found} report{'s' if found != 1 else ''} found"
                           + (", one confirms it." if cv.get("confirmed") else "."))
            else:
                st.caption(f":material/newspaper: Would be in the news if true ({cv.get('event')}); the news search "
                           f"did not run ({cv.get('news_state') or 'off'}).")
        for fc in c.get("fact_checks", [])[:2]:
            when = f", {fc['date']}" if fc.get("date") else ""
            who = f"[{fc['publisher']}]({fc['url']})" if fc.get("url") else fc["publisher"]
            st.markdown(f":material/fact_check: Previously fact-checked by {who}: **{fc['rating']}**{when}")
        if c.get("justification"):
            st.write(c["justification"])
        for e in c.get("evidence", [])[:2]:
            link = f"[{e.get('source') or 'source'}]({e['url']})" if e.get("url") else (e.get("source") or "source")
            if e.get("counted", True):
                note = ""
            elif e.get("claim_match") is not None:
                note = " *(a verdict on a different claim: not counted)*"
            else:
                note = " *(does not confirm the claim: not counted)*"
            st.caption(f"{link}: {e['snippet'][:220]}{note}")


def _quote_parts(quote: str) -> tuple[str | None, str]:
    """A "Q: ... A: ..." evidence passage -> (question, answer); other passages -> (None, text)."""
    if quote.startswith("Q: ") and " A: " in quote:
        q, a = quote[3:].split(" A: ", 1)
        return q.strip(), a.strip()
    return None, quote


def evidence_box(items: list[dict]) -> None:
    """'What the evidence says': the strongest passage per confident, on-topic verdict, quoted as found.
    Nothing here is written by a model; with no such verdict the box is not shown."""
    if not items:
        return
    with st.container(border=True, key="nc_evidence"):
        st.markdown("#### What the evidence says")
        for it in items:
            v_icon, v_colour = VERDICT_STYLE.get(it["verdict"], (":material/help:", "gray"))
            st.badge(it["verdict"], icon=v_icon, color=v_colour)
            st.markdown(f"**{it['claim']}**")
            question, answer = _quote_parts(it["quote"])
            if question:
                st.caption(f"Evidence answers: {question}")
            st.markdown(f"> {answer}")
            src = f"[{it.get('source') or 'source'}]({it['url']})" if it.get("url") else (it.get("source") or "source")
            st.caption(f"Source: {src}" + (f" · {it['date']}" if it.get("date") else ""))
        st.caption("Quoted from the evidence, not written by the system. Shown only for confident verdicts on "
                   "evidence about the claim.")


def silence_box(absence: dict | None) -> None:
    """'What the news says' when silence was used as evidence: the claim, why reports were expected, what was
    searched. Shown only when the absence rule set or moved the score."""
    if not absence or not absence.get("applied"):
        return
    s = absence.get("search") or {}
    where = f"{s['outlets']} trusted news sites" if s.get("outlets") else "trusted news sites"
    when = f"{s['from']} to {s['to']}" if s.get("from") else f"everything up to {s.get('to')}"
    with st.container(border=True, key="nc_silence"):
        st.markdown("#### What the news says")
        st.badge("No trusted report", icon=":material/newspaper:", color="red")
        st.markdown(f"**{absence['claim']}**")
        st.write(f"A claim like this would be widely reported if it were true: {absence['why']}. "
                 f"Searched {where} ({when}): nothing reported it.")
        st.caption("Silence counts as evidence only for claims that the news always covers, and it is not proof: "
                   "an event from the last few hours may not be indexed yet.")


def feedback_form(report: dict) -> None:
    aid = report.get("analysis_id")
    if not aid:
        return
    with st.container(border=True):
        st.markdown("#### Was this helpful?")
        sent = st.session_state.setdefault("feedback_sent", set())
        if aid in sent:
            st.success("Thanks, your feedback was saved.", icon=":material/thumb_up:")
            return
        with st.form(f"feedback_{aid}", clear_on_submit=True, border=False):
            agrees = st.radio("Do you agree with this result?", ["Yes", "No"], horizontal=True, index=None)
            label = st.selectbox("If not, what do you think is right?", list(LABEL_OPTIONS),
                                 format_func=LABEL_OPTIONS.get, index=None, placeholder="Optional")
            comment = st.text_area("Comment (optional)", max_chars=1000)
            if st.form_submit_button("Send feedback"):
                if agrees is None:
                    st.warning("Choose Yes or No first.")
                    return
                try:
                    api.send_feedback(aid, agrees == "Yes", label, comment.strip() or None)
                except api.ApiError as e:
                    st.error(str(e))
                    return
                sent.add(aid)
                st.rerun()
