"""
app.py -- Streamlit UI for the Resume Screening Agent.

Run with:
    streamlit run app.py

Lets a user paste/upload a Job Description, upload a batch of resumes
(PDF/DOCX/TXT), and get back a ranked, explained shortlist -- viewable in
the browser and downloadable as CSV/JSON.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import parser, pipeline, llm_reasoner

st.set_page_config(page_title="Resume Screening Agent", page_icon="🧑\u200d💼", layout="wide")

st.title("🧑\u200d💼 Resume Screening Agent")
st.caption(
    "Rank resumes against a job description using NLP similarity + skill "
    "matching, with optional LLM-generated reasoning per candidate."
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

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

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
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
    st.subheader("2. Resumes")
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

    # --- Summary table -----------------------------------------------------
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

    st.subheader("📊 Ranked Shortlist")
    st.dataframe(df, use_container_width=True, hide_index=True)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        with open(csv_path, "rb") as f:
            st.download_button("⬇️ Download CSV", f, file_name="ranked_results.csv", mime="text/csv")
    with dl_col2:
        with open(json_path, "rb") as f:
            st.download_button("⬇️ Download JSON", f, file_name="ranked_results.json", mime="application/json")

    st.subheader("🔍 Candidate Details & Reasoning")
    for record in ranked:
        with st.expander(f"#{record['rank']} — {record['file_name']} — {record['final_score']}/100"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Score", f"{record['final_score']:.1f}")
            c2.metric("Text Similarity", f"{record['similarity_score']:.1f}")
            c3.metric("Skill Match", f"{record['skill_match_score']:.1f}")

            st.markdown(f"**Experience:** {record['years_experience']}  |  **Education:** {record['education']}")

            if record.get("matched_skills"):
                st.markdown(f"✅ **Matched skills:** {', '.join(record['matched_skills'])}")
            if record.get("missing_skills"):
                st.markdown(f"⚠️ **Missing skills:** {', '.join(record['missing_skills'])}")

            if record.get("reasoning"):
                st.markdown("**Reasoning:**")
                st.write(record["reasoning"])

else:
    st.info("Fill in the job description and resumes on the left, then click **Run Screening**.")
