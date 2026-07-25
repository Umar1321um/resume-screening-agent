"""
Basic tests for the Resume Screening Agent.

Run with:  python -m pytest tests/ -v
(or simply: python tests/test_pipeline.py)

These tests avoid any network/LLM calls (use_llm_reasoning=False) so they
run fast, free, and deterministically in CI.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import extractor, scorer, parser, pipeline  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data"


class TestExtractor(unittest.TestCase):
    def test_extract_skills_finds_known_skills(self):
        text = "Experienced with Python, FastAPI, Docker, and PostgreSQL."
        skills = extractor.extract_skills(text)
        self.assertIn("Python", skills)
        self.assertIn("FastAPI", skills)
        self.assertIn("Docker", skills)
        self.assertIn("PostgreSQL", skills)

    def test_extract_skills_no_false_positive_substring(self):
        # "R" or single-letter skills aren't in our taxonomy, but make sure
        # word-boundary matching doesn't false-positive "Go" inside "Google".
        text = "Experience with Google Cloud and Golang tooling."
        skills = extractor.extract_skills(text)
        self.assertNotIn("Go", skills)

    def test_years_experience_explicit_statement(self):
        text = "I have 5+ years of experience in backend development."
        years = extractor.estimate_years_experience(text)
        self.assertEqual(years, 5.0)

    def test_education_detection(self):
        self.assertEqual(extractor.extract_education("B.Tech in Computer Science"), "Bachelor's")
        self.assertEqual(extractor.extract_education("M.S. in Computer Science"), "Master's")
        self.assertIsNone(extractor.extract_education("No degree mentioned here"))


class TestScorer(unittest.TestCase):
    def test_identical_text_scores_highest(self):
        jd = "Python backend engineer with FastAPI and PostgreSQL experience."
        resumes = [
            {"file_name": "a.txt", "file_path": "a.txt", "text": jd},
            {"file_name": "b.txt", "file_path": "b.txt", "text": "Marketing manager with SEO experience."},
        ]
        ranked = scorer.score_resumes(jd, resumes)
        self.assertEqual(ranked[0]["file_name"], "a.txt")
        self.assertGreater(ranked[0]["final_score"], ranked[1]["final_score"])

    def test_skill_overlap_bounds(self):
        overlap = scorer.compute_skill_overlap(["Python", "SQL"], ["Python"])
        self.assertAlmostEqual(overlap, 0.5)
        self.assertEqual(scorer.compute_skill_overlap([], ["Python"]), 0.0)


class TestParserAndPipelineIntegration(unittest.TestCase):
    def test_load_sample_resumes_and_rank(self):
        jd_path = FIXTURES_DIR / "job_description.txt"
        resumes_dir = FIXTURES_DIR / "resumes"
        if not jd_path.exists() or not resumes_dir.exists():
            self.skipTest("Sample data not present")

        jd_text = jd_path.read_text(encoding="utf-8")
        ranked = pipeline.run_pipeline(
            jd_text=jd_text,
            resumes_folder=resumes_dir,
            use_llm_reasoning=False,
        )
        self.assertGreaterEqual(len(ranked), 10)
        scores = [r["final_score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # The strongest backend-focused resume should clearly outrank the
        # marketing resume with no relevant skills.
        by_name = {r["file_name"]: r for r in ranked}
        self.assertGreater(
            by_name["01_aditi_sharma.txt"]["final_score"],
            by_name["07_emily_watson.txt"]["final_score"],
        )


if __name__ == "__main__":
    unittest.main()
