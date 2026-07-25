"""
app.py -- Streamlit UI for the Resume Screening Agent.

Run with:
    streamlit run app.py

Visual concept: "the recruiter's desk" -- a dark walnut work surface holding
cream index cards (one per candidate), each stamped with its rank like a
reviewer's ink stamp. Score bands (Top Match / Solid / Needs Review) are
computed from the actual score spread of the run, not fixed thresholds, so
the color-coding always reflects real relative standing in that batch.
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import parser, pipeline, llm_reasoner

st.set_page_config(page_title="Resume Screening Agent", page_icon="🗂️", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens + global styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
:root {
    --ink-bg: #0F172A;
    --ink-panel: #20242E;
    --ink-panel-2: #262B37;
    --card: #F4EEDF;
    --card-text: #23262B;
    --gold: #D9A441;
    --teal: #3E8E7E;
    --coral: #C9694F;
    --muted: #9099AC;
    --hairline: rgba(144, 153, 172, 0.22);
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--ink-bg) !important;
    color: #E7E5DC;
    font-family: 'Public Sans', sans-serif;
}

/* Sidebar -- the "desk drawer" */
[data-testid="stSidebar"] {
    background: var(--ink-panel) !important;
    border-right: 1px solid var(--hairline);
}
[data-testid="stSidebar"] * { color: #C8CDDA !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif !important;
    color: #F4EEDF !important;
}
[data-testid="stSidebar"] code {
    background: rgba(217,164,65,0.15) !important;
    color: var(--gold) !important;
    border-radius: 4px;
}

/* Headings use the display serif */
h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
h1 { color: #F7F2E4 !important; }
h2, h3 { color: #EDE7D6 !important; }

/* Hero */
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: var(--gold);
    margin-bottom: 0.35rem;
}
.hero-title { font-size: 2.6rem; font-weight: 600; margin: 0 0 0.4rem 0; line-height: 1.05; }
.hero-sub { color: var(--muted); font-size: 1.02rem; max-width: 640px; }
.hero-rule { border: none; border-top: 1px solid var(--hairline); margin: 1.6rem 0 1.8rem 0; }

/* Buttons -- stamped, not soft */
.stButton>button, .stDownloadButton>button {
    background: var(--gold) !important;
    color: #241A05 !important;
    border: none !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    font-family: 'Public Sans', sans-serif !important;
    letter-spacing: 0.01em;
    padding: 0.5rem 1.1rem !important;
    box-shadow: 0 2px 0 rgba(0,0,0,0.35);
    transition: transform 0.08s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: #E6B25A !important;
    transform: translateY(-1px);
}

/* Text areas / uploaders -- panel cards */
[data-testid="stTextArea"] textarea {
    background: var(--ink-panel-2) !important;
    color: #E7E5DC !important;
    border: 1px solid var(--hairline) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--ink-panel-2) !important;
    border: 1px dashed var(--hairline) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #C8CDDA !important; }

/* Checkbox label color */
.stCheckbox label p { color: #C8CDDA !important; }

/* Progress bar -- inked fill */
.stProgress > div > div { background: var(--gold) !important; }

/* Divider */
hr { border-top: 1px solid var(--hairline) !important; }

/* Section labels */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.3rem;
}

/* ---- Candidate dossier card ---- */
.dossier-card {
    background: var(--card);
    color: var(--card-text);
    border-radius: 6px;
    padding: 1.1rem 1.3rem 1.2rem 1.3rem;
    margin-bottom: 0.85rem;
    position: relative;
    box-shadow: 0 6px 14px rgba(0,0,0,0.28);
    border-left: 5px solid var(--band-color, var(--teal));
}
.dossier-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.dossier-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.18rem; margin: 0; }
.dossier-file { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #6b6f76; margin-top: 0.1rem; }

.rank-stamp {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--band-color, var(--teal));
    border: 2px solid var(--band-color, var(--teal));
    border-radius: 50%;
    width: 46px; height: 46px;
    display: flex; align-items: center; justify-content: center;
    transform: rotate(-6deg);
    flex-shrink: 0;
    background: rgba(0,0,0,0.02);
}

.score-row { display: flex; align-items: center; gap: 0.7rem; margin: 0.7rem 0 0.55rem 0; }
.score-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.05rem; color: var(--band-color, var(--teal)); min-width: 3.4rem; }
.score-track { flex: 1; height: 8px; border-radius: 4px; background: rgba(0,0,0,0.10); overflow: hidden; }
.score-fill { height: 100%; border-radius: 4px; background: var(--band-color, var(--teal)); }
.band-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--band-color, var(--teal));
    border: 1px solid var(--band-color, var(--teal));
    border-radius: 3px; padding: 0.12rem 0.4rem;
}

.meta-line { font-size: 0.84rem; color: #4b4e54; margin-bottom: 0.5rem; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.3rem; }
.chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
}
.chip-match { background: rgba(62,142,126,0.16); color: #2A6659; border: 1px solid rgba(62,142,126,0.4); }
.chip-missing { background: rgba(201,105,79,0.10); color: #A24E36; border: 1px solid rgba(201,105,79,0.35); }
.reasoning-text { font-size: 0.88rem; color: #33363B; margin-top: 0.55rem; line-height: 1.45; font-style: italic; }

/* Expander styled like a folder tab */
[data-testid="stExpander"] {
    background: var(--ink-panel-2) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary { color: #E7E5DC !important; font-family: 'JetBrains Mono', monospace; }

/* Metrics */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: var(--gold) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-eyebrow">RESUME SCREENING AGENT</div>
<div class="hero-title">The shortlist, ranked and stamped.</div>
<div class="hero-sub">Drop in a job description and a stack of resumes. Every
candidate gets a relevance score from TF-IDF similarity and skill overlap,
plus a plain-language read on where they fit and where they fall short.</div>
<hr class="hero-rule"/>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙ Configuration")

    st.markdown("**LLM Reasoning Provider**")
    detected_provider = llm_reasoner.get_active_provider()
    st.write(f"Detected: `{detected_provider}`")
    st.caption(
        "Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY` as an "
        "environment variable before launching to enable LLM reasoning. "
        "Without a key, the agent still runs end-to-end using a "
        "deterministic, template-based explanation."
    )

    use_llm = st.checkbox(
        "Generate LLM reasoning per candidate",
        value=(detected_provider != "none"),
        help="If unchecked, a fast rule-based explanation is used instead.",
    )

    st.markdown("---")
    st.markdown("**Scoring weights**")
    st.caption(
        "final_score = 0.6 × TF-IDF similarity + 0.4 × skill overlap "
        "(see docs/SCORING_METHODOLOGY.md)"
    )

    st.markdown("---")
    st.markdown("**Score bands**")
    st.caption(
        "Colors below are assigned by tertile *within this run's* score "
        "spread -- gold marks the top third of this batch, teal the "
        "middle third, coral the bottom third."
    )

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload a JD (.txt)", type=["txt"], key="jd_upload")
    default_jd = ""
    jd_sample_path = Path("data/job_description.txt")
    if jd_sample_path.exists():
        default_jd = jd_sample_path.read_text(encoding="utf-8")

    if jd_file is not None:
        jd_text = jd_file.read().decode("utf-8", errors="ignore")
    else:
        jd_text = st.text_area("Or paste/edit the JD text", value=default_jd, height=300)

with col2:
    st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
    st.subheader("Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload resume files (PDF/DOCX/TXT) -- select multiple",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    use_sample_folder = st.checkbox(
        "Use bundled sample resumes (data/resumes/) instead of uploads",
        value=(not uploaded_resumes),
    )

run_button = st.button("🚀 Run Screening", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_for_index(i: int, total: int) -> tuple[str, str]:
    """Return (color, label) for a candidate's position in the ranked list,
    based on which tertile of *this run's* results they fall into."""
    if total <= 1:
        return "var(--gold)", "TOP MATCH"
    fraction = i / total
    if fraction < 1 / 3:
        return "var(--gold)", "TOP MATCH"
    elif fraction < 2 / 3:
        return "var(--teal)", "SOLID FIT"
    else:
        return "var(--coral)", "NEEDS REVIEW"


def render_candidate_card(record: dict, band_color: str, band_label: str) -> str:
    name = html.escape(Path(record["file_name"]).stem.replace("_", " ").title())
    file_name = html.escape(record["file_name"])
    score = record["final_score"]
    years = record.get("years_experience")
    education = record.get("education") or "Not detected"
    exp_str = f"{years:.0f} yrs experience" if years is not None else "Experience not detected"

    matched = (record.get("matched_skills") or [])[:8]
    missing = (record.get("missing_skills") or [])[:6]

    chips_html = "".join(f'<span class="chip chip-match">{html.escape(s)}</span>' for s in matched)
    missing_html = "".join(f'<span class="chip chip-missing">{html.escape(s)}</span>' for s in missing)

    reasoning = record.get("reasoning")
    reasoning_html = f'<div class="reasoning-text">"{html.escape(reasoning)}"</div>' if reasoning else ""

    return f"""
<div class="dossier-card" style="--band-color:{band_color};">
  <div class="dossier-top">
    <div>
      <p class="dossier-name">#{record['rank']} &nbsp; {name}</p>
      <div class="dossier-file">{file_name}</div>
    </div>
    <div class="rank-stamp">#{record['rank']}</div>
  </div>
  <div class="score-row">
    <div class="score-value">{score:.1f}</div>
    <div class="score-track"><div class="score-fill" style="width:{min(score,100)}%;"></div></div>
    <div class="band-tag">{band_label}</div>
  </div>
  <div class="meta-line">{exp_str} &nbsp;•&nbsp; {html.escape(str(education))}</div>
  <div class="chip-row">{chips_html}</div>
  <div class="chip-row">{missing_html}</div>
  {reasoning_html}
</div>
"""


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_button:
    if not jd_text.strip():
        st.error("Please provide a job description.")
        st.stop()

    resume_records = []

    if uploaded_resumes and not use_sample_folder:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for uploaded in uploaded_resumes:
                dest = tmp_path / uploaded.name
                dest.write_bytes(uploaded.read())
            resume_records = parser.load_resumes(tmp_path)
    else:
        sample_dir = Path("data/resumes")
        if not sample_dir.exists():
            st.error("No resumes uploaded and data/resumes/ sample folder not found.")
            st.stop()
        resume_records = parser.load_resumes(sample_dir)

    if not resume_records:
        st.error("No parsable resumes found.")
        st.stop()

    st.info(f"Parsed {len(resume_records)} resumes. Scoring against the JD...")

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_cb(i, total, name):
        progress_bar.progress(i / total)
        status_text.text(f"Processing {i}/{total}: {name}")

    ranked = pipeline.run_pipeline_from_records(
        jd_text=jd_text,
        resume_records=resume_records,
        use_llm_reasoning=use_llm,
        progress_callback=progress_cb,
    )

    status_text.empty()
    progress_bar.empty()

    csv_path, json_path = pipeline.save_outputs(ranked, "outputs")

    st.success(f"Ranked {len(ranked)} candidates.")

    total = len(ranked)

    # --- Ranked dossier cards -----------------------------------------------
    st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
    st.subheader("The Shortlist")

    cards_html = ""
    for idx, record in enumerate(ranked):
        color, label = _band_for_index(idx, total)
        cards_html += render_candidate_card(record, color, label)
    st.markdown(cards_html, unsafe_allow_html=True)

    # --- Downloads -----------------------------------------------------------
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        with open(csv_path, "rb") as f:
            st.download_button("⬇ Download CSV", f, file_name="ranked_results.csv", mime="text/csv")
    with dl_col2:
        with open(json_path, "rb") as f:
            st.download_button("⬇ Download JSON", f, file_name="ranked_results.json", mime="application/json")

    # --- Raw table (for spreadsheet-style scanning) --------------------------
    with st.expander("📋 View as a plain table"):
        df = pd.DataFrame([
            {
                "Rank": r["rank"],
                "Candidate": r["file_name"],
                "Final Score": r["final_score"],
                "Similarity": r["similarity_score"],
                "Skill Match": r["skill_match_score"],
                "Experience (yrs)": r["years_experience"],
                "Education": r["education"],
            }
            for r in ranked
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("Fill in the job description and resumes above, then click **Run Screening**.")