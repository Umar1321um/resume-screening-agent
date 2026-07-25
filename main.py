#!/usr/bin/env python3
"""
main.py -- CLI entry point for the Resume Screening Agent.

Usage:
    python main.py --jd data/job_description.txt --resumes data/resumes --out outputs

Environment variables (optional, for richer LLM reasoning):
    ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY
    LLM_PROVIDER=none   # force the free/offline template fallback
    LLM_MODEL=...        # override the default model for the active provider
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import pipeline, llm_reasoner


def main() -> None:
    parser_args = argparse.ArgumentParser(description="Resume Screening Agent")
    parser_args.add_argument(
        "--jd", required=True, help="Path to the job description text file"
    )
    parser_args.add_argument(
        "--resumes", required=True, help="Path to a folder of resume files (PDF/DOCX/TXT)"
    )
    parser_args.add_argument(
        "--out", default="outputs", help="Output directory for CSV/JSON results"
    )
    parser_args.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM reasoning entirely and use the deterministic template explanations"
    )
    args = parser_args.parse_args()

    jd_path = Path(args.jd)
    if not jd_path.exists():
        print(f"Job description file not found: {jd_path}", file=sys.stderr)
        sys.exit(1)

    resumes_dir = Path(args.resumes)
    if not resumes_dir.is_dir():
        print(f"Resumes folder not found: {resumes_dir}", file=sys.stderr)
        sys.exit(1)

    jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")

    provider = "none" if args.no_llm else llm_reasoner.get_active_provider()
    print(f"[main] LLM reasoning provider: {provider}")

    def progress(i, total, name):
        print(f"[main] Scored {i}/{total}: {name}")

    ranked = pipeline.run_pipeline(
        jd_text=jd_text,
        resumes_folder=resumes_dir,
        use_llm_reasoning=not args.no_llm,
        progress_callback=progress,
    )

    csv_path, json_path = pipeline.save_outputs(ranked, args.out)

    print("\n=== Ranked Shortlist ===")
    for record in ranked:
        print(
            f"#{record['rank']:>2}  {record['final_score']:>6.2f}  "
            f"{record['file_name']}"
        )

    print(f"\nSaved CSV  -> {csv_path}")
    print(f"Saved JSON -> {json_path}")


if __name__ == "__main__":
    main()
