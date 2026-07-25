# Scoring Methodology

## Overview

Each resume gets a **final_score** from 0-100, computed as a weighted blend
of two independent signals:

```
final_score = 100 * (0.6 * text_similarity + 0.4 * skill_overlap)
```

| Component | Weight | What it measures | How it's computed |
|---|---|---|---|
| **Text similarity** | 0.6 | Overall semantic/lexical overlap between the full JD and the full resume text | TF-IDF vectorization (unigrams + bigrams) of the JD and all resumes together, then cosine similarity between the JD vector and each resume vector |
| **Skill overlap** | 0.4 | Precise overlap of *required* skills | Regex/keyword matching against a curated skill taxonomy (~70 common tech skills), applied separately to the JD and each resume, then `|matched| / |JD skills|` |

## Why this combination?

- **TF-IDF cosine similarity** captures overall topical relevance (job
  titles, responsibilities, domain vocabulary) without needing any
  internet access, model download, or API key — it works the same on any
  reviewer's machine, deterministically, in milliseconds even for 100+
  resumes.
- **Skill overlap** is added because TF-IDF alone can be fooled: a resume
  padded with JD-adjacent buzzwords can score well on similarity while
  missing the specific required skills, or a resume can phrase the same
  skill differently and score low on lexical similarity. Explicit skill
  matching acts as a precision check on top of the fuzzier similarity
  score.
- Weights (0.6 / 0.4) favor overall fit slightly over a strict skill
  checklist, since resumes rarely list every required skill verbatim even
  when the candidate is qualified. These weights are configurable
  constants in `src/scorer.py` (`SIMILARITY_WEIGHT`, `SKILL_WEIGHT`).

## Structured extraction

In addition to the score, `src/extractor.py` pulls out:

- **Skills** — matched against the taxonomy in `SKILL_TAXONOMY`.
- **Years of experience** — first tries an explicit statement like "5+
  years of experience"; falls back to the span between the earliest and
  latest year mentioned in date ranges (e.g. "2019 - Present").
- **Education level** — highest of PhD / Master's / Bachelor's / Associate
  detected via keyword/degree-abbreviation matching.

These are heuristic and best-effort — they are meant to give a recruiter
quick, scannable signal, not a legally precise parse.

## LLM reasoning layer

The **numeric score is never LLM-dependent** — it's fully deterministic
NLP. On top of that score, `src/llm_reasoner.py` optionally calls an LLM
(Anthropic / OpenAI / Groq, whichever API key is configured) to turn the
numbers into a short, readable justification, e.g.:

> "Strong match: 5 years of backend Python experience with FastAPI,
> PostgreSQL, and Docker directly overlapping the JD's core stack. Missing
> explicit Kubernetes and Kafka experience mentioned as preferred
> qualifications."

If no API key is configured (or the call fails/rate-limits), the same
explanation slot is filled by a deterministic template built from the
same structured fields — so the agent is always fully runnable end-to-end,
with or without an LLM key.

## Ranking

Candidates are sorted descending by `final_score` and assigned a `rank`
(1 = best fit). Ties are broken by original file order (stable sort).
