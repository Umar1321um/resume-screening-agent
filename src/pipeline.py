"""
pipeline.py
-----------
End-to-end orchestration: load JD -> parse resumes -> score -> rank ->
attach LLM reasoning -> return/save results.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import parser, scorer, llm_reasoner


def run_pipeline(
    jd_text: str,
    resumes_folder: str | Path,
    use_llm_reasoning: bool = True,
    progress_callback=None,
) -> list[dict]:
    """Run the full resume screening pipeline.

    Args:
        jd_text: raw job description text.
        resumes_folder: path to a folder containing resume files
            (.pdf / .docx / .txt).
        use_llm_reasoning: if True, generate a natural-language reasoning
            string per candidate (LLM if configured, else template fallback).
        progress_callback: optional callable(current, total, file_name)
            invoked as each resume is processed, for UI progress bars.

    Returns:
        List of ranked candidate dicts, sorted best-to-worst.
    """
    resumes = parser.load_resumes(resumes_folder)
    if not resumes:
        raise ValueError(f"No parsable resumes found in {resumes_folder}")

    ranked = scorer.score_resumes(jd_text, resumes)

    total = len(ranked)
    for i, record in enumerate(ranked, start=1):
        if use_llm_reasoning:
            record["reasoning"] = llm_reasoner.generate_reasoning(jd_text, record)
        else:
            record["reasoning"] = None
        # Don't leak full resume text into the output payload/report.
        record.pop("text", None)
        if progress_callback:
            progress_callback(i, total, record["file_name"])

    return ranked


def run_pipeline_from_records(
    jd_text: str,
    resume_records: list[dict],
    use_llm_reasoning: bool = True,
    progress_callback=None,
) -> list[dict]:
    """Same as run_pipeline, but takes already-parsed resume records
    (list of {"file_name", "file_path", "text"}). Useful for the Streamlit
    UI where files are uploaded in-memory rather than read from disk.
    """
    if not resume_records:
        raise ValueError("No resume records provided")

    ranked = scorer.score_resumes(jd_text, resume_records)

    total = len(ranked)
    for i, record in enumerate(ranked, start=1):
        if use_llm_reasoning:
            record["reasoning"] = llm_reasoner.generate_reasoning(jd_text, record)
        else:
            record["reasoning"] = None
        record.pop("text", None)
        if progress_callback:
            progress_callback(i, total, record["file_name"])

    return ranked


def save_outputs(ranked: list[dict], output_dir: str | Path) -> tuple[str, str]:
    """Write ranked results to CSV and JSON. Returns (csv_path, json_path)."""
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ranked_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "ranked_results.csv"
    fieldnames = [
        "rank", "file_name", "final_score", "similarity_score",
        "skill_match_score", "years_experience", "education",
        "matched_skills", "missing_skills", "reasoning",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in ranked:
            row = dict(record)
            row["matched_skills"] = "; ".join(row.get("matched_skills") or [])
            row["missing_skills"] = "; ".join(row.get("missing_skills") or [])
            writer.writerow(row)

    return str(csv_path), str(json_path)
