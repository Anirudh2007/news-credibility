"""Streamlit frontend (Phase 7). Talks only to the FastAPI backend.

Run from `news-credibility/frontend/` (API first, in another terminal), so `.streamlit/config.toml` (theme)
is picked up:
    uvicorn api.main:app                     # from news-credibility/
    streamlit run app.py                     # from news-credibility/frontend/, http://localhost:8501
Backend URL: env NEWSCRED_API (default http://localhost:8000).

Pages: Analyze (text / URL / image + optional claimed date and place), History
(past analyses, reopen, delete), About (method, bands, limitations).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import api_client as api
from components.report import render_report, theme_mode
from components.theme import hero, inject_css, steps

st.set_page_config(page_title="News Credibility Checker", page_icon=":material/fact_check:", layout="wide")
inject_css(theme_mode())

EXAMPLES = {  # demo inputs only; the real-world test set is reports/india_live_eval.json (2023-2025 India claims)
    "": "",
    "Cow urine cures COVID": "Drinking cow urine and applying cow dung can cure and prevent COVID-19.",
    "RBI new Rs 1000 notes": "Reserve Bank of India announces new Rs 1000 currency notes.",
    "Chandrayaan-3 launch": "ISRO successfully launches Chandrayaan-3 lunar mission.",
    "WhatsApp subscription fee": "WhatsApp to start charging users subscription fee starting next month.",
}
PAGE_SIZE = 20
PAGE_ICONS = {"Analyze": ":material/search:", "History": ":material/history:", "About": ":material/menu_book:"}
WHAT_IS_CHECKED = [
    ("Claims", "Each check-worthy sentence is found and checked on its own."),
    ("Evidence", "Published fact-checks, an offline index, Wikipedia as of the claim date, trusted news reports."),
    ("Image, source, dates", "Caption match, earlier copies, the outlet’s rating, date gaps."),
    ("A score with reasons", "Calibrated 0–100 with points per check, or “Unverified”."),
]


@st.cache_data(ttl=15, show_spinner=False)
def api_health():
    try:
        return api.health(), None
    except api.ApiError as e:
        return None, str(e)


def llm_available() -> bool:
    """False when the server runs without an LLM (EXPLAIN_ENABLED=0, as on the Azure VM): the switch is hidden
    and explanations come from templates. A router that is only down for now keeps the switch."""
    health, _ = api_health()
    return not str((health or {}).get("explainer", "")).startswith("templates only")


def sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="nc-brand"><span class="nc-brand-name">News Credibility</span>'
                    '<span class="nc-brand-tag">Claim-level fact checking for news</span></div>', unsafe_allow_html=True)
        page = st.radio("Page", ["Analyze", "History", "About"], label_visibility="collapsed",
                        format_func=lambda p: f"{PAGE_ICONS[p]} {p}")
        st.divider()
        health, err = api_health()
        if err:
            st.error(err, icon=":material/cloud_off:")
        else:
            st.markdown('<span class="nc-status">API online</span>', unsafe_allow_html=True)
            stored = (health.get("database") or "").removeprefix("ok (").removesuffix(")")
            st.caption(f"Model {health['model_version'].split('+')[0]}" + (f" · {stored}" if stored else ""))
            if health.get("components"):
                with st.expander("Model status"):
                    for name, status in health["components"].items():
                        st.caption(f"**{name}**: {status}")
        st.space("medium")
        st.caption("Research prototype. A score is an aid for reading the evidence, not a final judgement.")
        return page


def iso(d) -> str | None:
    return d.isoformat() if d else None


def page_analyze() -> None:
    hero("Fact-check assistant", "Check a news item",
         "Paste a claim or post, add a link or an image, or combine them. Checks that need something you did "
         "not give are skipped, never counted against the item.")
    # fills the text box only when the choice changes, so later edits are kept
    st.selectbox("Try an example", list(EXAMPLES), format_func=lambda k: k or "(none)", key="example",
                 on_change=lambda: st.session_state.update(text_input=EXAMPLES[st.session_state["example"]]))

    with st.form("analyze"):
        t_text, t_url, t_image = st.tabs([":material/notes: Text", ":material/link: Link", ":material/image: Image"])
        with t_text:
            text = st.text_area("Claim, post or article text", key="text_input", height=140, max_chars=20_000,
                                placeholder="e.g. Reserve Bank of India announces new Rs 1000 currency notes.")
        with t_url:
            url = st.text_input("Source URL or domain", placeholder="https://www.example.com/article")
            st.caption("Rates the outlet (1,082 rated outlets; social-media links are not rated: the platform is not "
                       "the author). An article link with no text: its headline, summary and date are read from the "
                       "page and checked. For a social-media post, paste the post's text.")
        with t_image:
            image = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "gif", "bmp"])
        with st.expander("Optional: what does the item claim?"):
            c1, c2, c3 = st.columns(3)
            claimed_date = c1.date_input("Claimed date", value=None, format="YYYY-MM-DD")
            claimed_location = c2.text_input("Claimed place", placeholder="e.g. India Gate, Delhi",
                                             help="Optional. With an image, places named in the text are checked "
                                                  "anyway; only a mismatch is reported.")
            publish_date = c3.date_input("Published on", value=None, format="YYYY-MM-DD")
        o1, o2 = st.columns(2)
        if llm_available():
            explain = o1.toggle("Write an explanation (LLM)", value=True,
                                help="Sends the facts and your text to the LLM router; templates are used if it is off or down.")
            refresh = o2.toggle("Re-analyse even if seen before", value=False)
        else:  # no LLM on this server: the explanation is still asked for, and the API writes it from templates
            explain = True
            refresh = o1.toggle("Re-analyse even if seen before", value=False)
        submitted = st.form_submit_button("Analyze", type="primary", icon=":material/search:", width="stretch")

    if submitted:
        if not (text.strip() or url.strip() or image):
            st.warning("Provide text, a URL or an image.")
        else:
            img = (image.name, image.getvalue(), image.type) if image else None
            with st.spinner("Checking claims, evidence, media and source... (the first run loads the models)"):
                try:
                    report = api.analyze(text=text.strip() or None, url=url.strip() or None, image=img,
                                         claimed_date=iso(claimed_date), claimed_location=claimed_location.strip() or None,
                                         publish_date=iso(publish_date), explain=explain, refresh=refresh)
                except api.ApiError as e:
                    if e.status == 422:  # refused by the input guard or validation: a reason, not a failure
                        st.warning(str(e), icon=":material/block:")
                    else:
                        st.error(str(e))
                    return
            st.session_state["report"] = report
            st.session_state["report_image"] = img[1] if img else None

    if st.session_state.get("report"):
        st.space("small")
        render_report(st.session_state["report"], st.session_state.get("report_image"))
    else:
        st.markdown('<p class="nc-eyebrow" style="margin-top:1.6rem">What gets checked</p>', unsafe_allow_html=True)
        steps(WHAT_IS_CHECKED)


def page_history() -> None:
    hero("Analyses", "History", "Every analysis is stored with its result. Open one to see the full report, or "
                                "delete it. The history is shared by everyone who uses this site.")
    page_no = st.session_state.setdefault("history_page", 0)
    try:
        data = api.history(limit=PAGE_SIZE, offset=page_no * PAGE_SIZE)
    except api.ApiError as e:
        st.error(str(e))
        return
    if not data["items"]:
        if data["total"] and page_no > 0:  # the page emptied (last item deleted): go to the new last page
            st.session_state["history_page"] = (data["total"] - 1) // PAGE_SIZE
            st.rerun()
        st.info("No analyses yet. Run one on the Analyze page.")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Analyses", data["total"], border=True)
    try:
        fb = api.feedback_summary()
        m2.metric("Feedback received", fb["feedback"], border=True)
        m3.metric("Agreed with the result", f"{fb['agree_rate']:.0%}" if fb.get("agree_rate") is not None else "–",
                  border=True)
    except api.ApiError:
        pass
    df = pd.DataFrame(data["items"])
    df["created_at"] = df["created_at"].str[:16].str.replace("T", " ")
    st.dataframe(df[["created_at", "preview", "input_type", "score", "band", "feedback_count"]],
                 hide_index=True, width="stretch",
                 column_config={
                     "created_at": "When (UTC)", "preview": st.column_config.TextColumn("Input", width="large"),
                     "input_type": "Type",
                     "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                     "band": "Band", "feedback_count": "Feedback"})
    n_pages = max(1, -(-data["total"] // PAGE_SIZE))
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="center")
    if c1.button("Previous", icon=":material/chevron_left:", disabled=page_no == 0):
        st.session_state["history_page"] -= 1
        st.rerun()
    c2.caption(f"Page {page_no + 1} of {n_pages} · {data['total']} analyses")
    if c3.button("Next", icon=":material/chevron_right:", disabled=page_no + 1 >= n_pages):
        st.session_state["history_page"] += 1
        st.rerun()

    labels = {i["analysis_id"]: f"{i['created_at'][:16].replace('T', ' ')} · {i['preview'][:60]}" for i in data["items"]}
    selected = st.selectbox("Open an analysis", list(labels), format_func=labels.get, index=None)
    if selected:
        try:
            report = api.get_analysis(selected)
        except api.ApiError as e:
            st.error(str(e))
            return
        if st.button("Delete this analysis", icon=":material/delete:"):
            try:
                api.delete_analysis(selected)
            except api.ApiError as e:
                st.error(str(e))
                return
            st.rerun()
        if report.get("feedback"):
            st.caption(f"{len(report['feedback'])} feedback entries on this analysis.")
        render_report(report)


def page_about() -> None:
    hero("Method", "How it works", "A news item is checked the way a fact-checker would: claim by claim, against "
                                   "evidence, with the image, the source and the dates looked at too.")
    steps(WHAT_IS_CHECKED)
    st.space("small")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("""
#### The checks
Each check-worthy claim is compared with **evidence** (an offline index, published fact-checks found live through
the Google Fact Check Tools API, Wikipedia as it stood on the claim date, and reports from about 300 trusted news
outlets found through a news search) by a fine-tuned DeBERTa verifier. Wikipedia and news reports can only
*confirm* a claim; only fact-checks and the index can refute one.
Images are checked for **caption mismatch** (CLIP) and **earlier copies** (perceptual hashing); if the text names a
landmark with reference photos, the image is compared with them and only a mismatch is reported. The **source** is
rated (1,082 outlets from public lists; platforms such as X or YouTube and unrated sites get no rating), and
**dates** are compared. The score comes from evidence only: how a text is worded is never scored.

A LightGBM **meta-learner** combines whichever checks ran into a calibrated 0-100 score. Checks that do not apply
(no image, no date) are skipped and do **not** lower the score. **What the evidence says** quotes the strongest
passage for confident verdicts; nothing there is written by a model. **What affected this score** shows SHAP points:
how much each check moved the score away from an average item.
""")
    with right:
        st.markdown("""
#### Score bands

| Score | Band |
|---|---|
| 0-30 | Low |
| 31-60 | Questionable |
| 61-80 | Mostly credible |
| 81-100 | High |
| no score | Unverified |

#### Limitations
English only; no video analysis (both future work). Published fact-checks are mostly about false claims, so true
news is confirmed less often than false news is caught. Media reuse is detected only against a small local image
index. The score is an aid for reading the evidence, not a final judgement.

#### Privacy
Analyses are stored (text, URL, score; images only as a fingerprint), are visible to everyone using this site on
the History page, and can be deleted there. To find evidence, the claims are sent to the Google Fact Check Tools
API, Wikipedia and a news search service (Tavily); images never leave the server.
""" + ("When LLM explanations are on, the facts and your text are also sent through the router to an external "
       "model provider." if llm_available() else ""))


PAGES = {"Analyze": page_analyze, "History": page_history, "About": page_about}
PAGES[sidebar()]()
