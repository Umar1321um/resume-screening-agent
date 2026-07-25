"""
extractor.py
------------
Pulls structured signals (skills, years of experience, education level) out
of raw resume text using deterministic regex/keyword heuristics.

This is intentionally dependency-free (no LLM call) so the pipeline has a
reliable, fast, free baseline. `llm_reasoner.py` builds on top of this for
richer natural-language extraction/explanations when an API key is present.
"""

from __future__ import annotations

import re

# A reasonably broad taxonomy covering common backend/software-engineering
# skills. Matching is case-insensitive with word boundaries so "R" doesn't
# match inside "Report", etc. Extend this list for other job families.
SKILL_TAXONOMY = [
    "Python", "Java", "JavaScript", "TypeScript", "Go", "C\\+\\+", "C#", "Ruby",
    "FastAPI", "Flask", "Django", "Spring Boot", "Node.js", "React", "Redux",
    "Angular", "Vue",
    "REST API", "GraphQL", "gRPC", "Microservices",
    "SQL", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "BigQuery",
    "Docker", "Kubernetes", "Terraform", "Jenkins", "GitHub Actions",
    "GitLab CI", "CI/CD", "ArgoCD",
    "AWS", "GCP", "Azure",
    "Kafka", "RabbitMQ", "SQS", "Spark", "Airflow",
    "Machine Learning", "NLP", "scikit-learn", "TensorFlow", "PyTorch",
    "Data Pipelines", "ETL",
    "Git", "Agile", "Scrum", "Jira",
    "Unit Testing", "pytest", "Selenium", "Postman",
    "HTML", "CSS", "Figma",
    "Excel", "SEO", "Google Analytics",
]

_PLURAL_OPTIONAL = {"REST API"}


def _pattern_for(skill: str) -> re.Pattern:
    if skill in _PLURAL_OPTIONAL:
        return re.compile(rf"\b{skill}s?\b", re.IGNORECASE)
    return re.compile(rf"\b{skill}\b", re.IGNORECASE)


_SKILL_PATTERNS = [(skill, _pattern_for(skill)) for skill in SKILL_TAXONOMY]

_YEARS_PATTERNS = [
    re.compile(r"(\d+)\+?\s*years?\s+of\s+(?:professional\s+)?experience", re.IGNORECASE),
    re.compile(r"(\d+)\+?\s*years?\s+experience", re.IGNORECASE),
]

_DATE_RANGE_PATTERN = re.compile(
    r"(19|20)\d{2}\s*[-\u2013to]+\s*(?:(19|20)\d{2}|present|current)",
    re.IGNORECASE,
)

_EDUCATION_LEVELS = [
    ("PhD", re.compile(r"\bph\.?d\.?\b", re.IGNORECASE)),
    ("Master's", re.compile(r"\b(m\.?s\.?c?|master'?s|mba)\b", re.IGNORECASE)),
    ("Bachelor's", re.compile(r"\b(b\.?tech|b\.?e\.?|b\.?sc|bachelor'?s)\b", re.IGNORECASE)),
    ("Associate", re.compile(r"\bassociate'?s?\s+degree\b", re.IGNORECASE)),
]


def extract_skills(text: str) -> list[str]:
    """Return the list of taxonomy skills found in the resume text."""
    found = []
    for skill, pattern in _SKILL_PATTERNS:
        if pattern.search(text):
            # Normalize the escaped "C\+\+" back to "C++" for display.
            found.append(skill.replace("\\+\\+", "++"))
    return found


def estimate_years_experience(text: str) -> float | None:
    """Best-effort estimate of years of professional experience.

    Strategy:
    1. Look for an explicit "X years of experience" statement (most reliable).
    2. Otherwise, fall back to spanning the earliest-to-latest year mentioned
       in date ranges (e.g. "2019 - Present") as a rough proxy.
    """
    for pattern in _YEARS_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))

    years_found = []
    for match in _DATE_RANGE_PATTERN.finditer(text):
        start_year = int(match.group(0)[:4])
        years_found.append(start_year)

    if years_found:
        import datetime
        current_year = datetime.datetime.now().year
        earliest = min(years_found)
        if 1980 <= earliest <= current_year:
            return float(current_year - earliest)

    return None


def extract_education(text: str) -> str | None:
    """Return the highest education level mentioned, if any."""
    for level, pattern in _EDUCATION_LEVELS:
        if pattern.search(text):
            return level
    return None


def extract_all(text: str) -> dict:
    """Convenience wrapper returning all extracted fields as a dict."""
    return {
        "skills": extract_skills(text),
        "years_experience": estimate_years_experience(text),
        "education": extract_education(text),
    }
