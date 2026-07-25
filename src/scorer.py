"""
scorer.py
---------
Computes a relevance score between a Job Description (JD) and a resume.

Method (see docs/SCORING_METHODOLOGY.md for full rationale):
1. TF-IDF vectorization of the JD + all resumes, cosine similarity between
   the JD vector and each resume vector -> "semantic overlap" component.
2. Exact skill-taxonomy overlap between JD-required skills and resume
   skills -> "skill match" component (more literal / precise signal).
3. Weighted blend of the two into a single 0-100 relevance score.

Why TF-IDF instead of a transformer embedding model?
- Zero external network/model-download dependency -> works fully offline
  and reproducibly on any reviewer's machine.
- Fast and deterministic for 10-100 resumes.
- Trade-off: purely lexical, so it can miss synonyms (e.g. "K8s" vs
  "Kubernetes"). The skill-taxonomy overlap component and the optional
  LLM reasoning step (llm_reasoner.py) compensate for this. See
  docs/TRADEOFFS.md for how this would be swapped for sentence-transformer
  embeddings given more time / infra.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import extractor

# Weight given to semantic (TF-IDF) similarity vs. exact skill overlap
# when computing the final blended score.
SIMILARITY_WEIGHT = 0.6
SKILL_WEIGHT = 0.4


def compute_tfidf_similarities(jd_text: str, resume_texts: list[str]) -> list[float]:
    """Return cosine similarity (0-1) between the JD and each resume text.

    All documents are vectorized together so the vocabulary/IDF weights are
    shared and comparable across resumes.
    """
    documents = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    return [float(s) for s in similarities]


def compute_skill_overlap(jd_skills: list[str], resume_skills: list[str]) -> float:
    """Fraction of JD-required skills also present on the resume (0-1)."""
    if not jd_skills:
        return 0.0
    jd_set = {s.lower() for s in jd_skills}
    resume_set = {s.lower() for s in resume_skills}
    matched = jd_set & resume_set
    return len(matched) / len(jd_set)


def score_resumes(jd_text: str, resumes: list[dict]) -> list[dict]:
    """Score every resume against the JD and return enriched records.

    `resumes` is a list of {"file_name", "file_path", "text"} dicts (see
    parser.load_resumes). Returns the same list with added fields:
    similarity_score, skill_match_score, final_score, matched_skills,
    missing_skills, years_experience, education.
    """
    jd_skills = extractor.extract_skills(jd_text)
    resume_texts = [r["text"] for r in resumes]
    similarities = compute_tfidf_similarities(jd_text, resume_texts)

    scored = []
    for resume, similarity in zip(resumes, similarities):
        extracted = extractor.extract_all(resume["text"])
        resume_skills = extracted["skills"]

        skill_overlap = compute_skill_overlap(jd_skills, resume_skills)
        final_score = round(
            100 * (SIMILARITY_WEIGHT * similarity + SKILL_WEIGHT * skill_overlap), 2
        )

        matched_skills = sorted(set(s.lower() for s in jd_skills) & set(s.lower() for s in resume_skills))
        missing_skills = sorted(set(s.lower() for s in jd_skills) - set(s.lower() for s in resume_skills))

        scored.append({
            **resume,
            "similarity_score": round(similarity * 100, 2),
            "skill_match_score": round(skill_overlap * 100, 2),
            "final_score": final_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "years_experience": extracted["years_experience"],
            "education": extracted["education"],
            "all_resume_skills": resume_skills,
        })

    scored.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, record in enumerate(scored, start=1):
        record["rank"] = rank

    return scored
