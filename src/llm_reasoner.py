"""
llm_reasoner.py
---------------
Generates a short, human-readable explanation of *why* a candidate received
their score, using an LLM. Supports Anthropic, OpenAI, or Groq (whichever
API key is present in the environment) and gracefully degrades to a
template-based explanation if no key is configured or the call fails.

Design rationale (see docs/TRADEOFFS.md):
- The numeric relevance score comes from deterministic NLP (scorer.py) so
  it's fast, reproducible, and free to run on 10+ resumes without hitting
  rate limits or costing money.
- The LLM is used only for the *reasoning/narrative* layer on top of that
  score -- turning "72.4" into a sentence a recruiter can act on. This
  keeps the agent fully functional (and free) with LLM_PROVIDER=none, while
  still demonstrating real LLM API integration when a key is supplied.
"""

from __future__ import annotations

import os


def _template_reasoning(record: dict) -> str:
    """Deterministic fallback explanation -- no API key required."""
    matched = record.get("matched_skills") or []
    missing = record.get("missing_skills") or []
    years = record.get("years_experience")
    education = record.get("education")

    parts = []
    parts.append(
        f"Scored {record['final_score']}/100 "
        f"({record['similarity_score']}/100 text similarity, "
        f"{record['skill_match_score']}/100 skill match)."
    )
    if matched:
        shown = ", ".join(matched[:6])
        parts.append(f"Matches required skills: {shown}.")
    if missing:
        shown = ", ".join(missing[:6])
        parts.append(f"Missing/unclear: {shown}.")
    if years is not None:
        parts.append(f"Estimated experience: ~{int(years)} years.")
    if education:
        parts.append(f"Highest education detected: {education}.")
    return " ".join(parts)


def _build_prompt(jd_text: str, record: dict) -> str:
    return f"""You are a resume screening assistant. Given the job description and a
candidate's resume summary below, write a concise 2-3 sentence justification
for the candidate's fit, referencing specific skills, experience, or gaps.
Be specific and factual -- do not invent details not present in the data.

JOB DESCRIPTION (truncated):
{jd_text[:1500]}

CANDIDATE: {record['file_name']}
Final score: {record['final_score']}/100
Text similarity score: {record['similarity_score']}/100
Skill match score: {record['skill_match_score']}/100
Matched required skills: {', '.join(record.get('matched_skills') or []) or 'none'}
Missing required skills: {', '.join(record.get('missing_skills') or []) or 'none'}
Estimated years of experience: {record.get('years_experience')}
Highest education detected: {record.get('education')}

Write only the justification, no preamble.
"""


def _call_anthropic(prompt: str) -> str | None:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model = os.environ.get("LLM_MODEL", "claude-sonnet-5")
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_reasoner] Anthropic call failed, using fallback: {exc}")
        return None


def _call_openai(prompt: str) -> str | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_reasoner] OpenAI call failed, using fallback: {exc}")
        return None


def _call_groq(prompt: str) -> str | None:
    try:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        model = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        response = client.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_reasoner] Groq call failed, using fallback: {exc}")
        return None


def get_active_provider() -> str:
    """Return which LLM provider will be used, based on env vars."""
    forced = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if forced == "none":
        return "none"
    if forced in ("anthropic", "openai", "groq"):
        return forced
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "none"


def generate_reasoning(jd_text: str, record: dict) -> str:
    """Generate the explanation for one candidate, using whichever LLM
    provider is configured, falling back to a deterministic template.
    """
    provider = get_active_provider()

    if provider == "none":
        return _template_reasoning(record)

    prompt = _build_prompt(jd_text, record)
    result = None
    if provider == "anthropic":
        result = _call_anthropic(prompt)
    elif provider == "openai":
        result = _call_openai(prompt)
    elif provider == "groq":
        result = _call_groq(prompt)

    return result or _template_reasoning(record)
