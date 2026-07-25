# Design Tradeoffs & What I'd Improve With More Time

## 1. TF-IDF vs. embedding-based semantic similarity

**Chosen:** TF-IDF + cosine similarity (scikit-learn).
**Alternative considered:** sentence-transformer embeddings (e.g.
`all-MiniLM-L6-v2`) or an LLM-generated embedding via the Anthropic/OpenAI
embeddings API.

- *Pro of TF-IDF:* zero network dependency, no model download, fully
  deterministic, runs in milliseconds, reproducible on any reviewer
  machine without GPU/large downloads.
- *Con of TF-IDF:* purely lexical — misses synonyms ("K8s" vs
  "Kubernetes", "ML" vs "Machine Learning") and paraphrases.
- **Mitigation implemented:** a separate exact skill-taxonomy overlap
  score compensates for the most important synonyms/keywords, and the
  optional LLM reasoning layer can catch nuance a human would.
- **With more time:** swap `compute_tfidf_similarities` in `src/scorer.py`
  for a `sentence-transformers` embedding + cosine similarity, behind the
  same function signature, and A/B the two on a labeled validation set of
  resumes with known "good fit" / "bad fit" labels before committing to
  one.

## 2. Rule-based skill/experience/education extraction vs. LLM extraction

**Chosen:** regex/keyword-based extraction (`src/extractor.py`) as the
default, LLM used only for the *reasoning narrative*, not the score.

- *Pro:* free, instant, no rate limits, works for 10, 100, or 1000 resumes
  without cost scaling, and the score is reproducible run-to-run.
- *Con:* a curated taxonomy misses skills not on the list, and free-text
  years-of-experience/education parsing can be wrong for unusual resume
  formats (e.g. non-chronological CVs, career gaps).
- **With more time:** add an LLM-based structured-extraction pass
  (JSON-mode) as an *optional upgrade* for the skills/experience/education
  fields specifically, with the regex output kept as an always-available
  fallback if the LLM call fails or isn't configured — the same pattern
  already used for reasoning.

## 3. Multi-provider LLM support (Anthropic / OpenAI / Groq) with offline fallback

**Chosen:** auto-detect provider from whichever API key is set in the
environment; if none, use a deterministic template for the "reasoning"
field instead of failing or blocking.

- *Pro:* the agent is runnable by any reviewer immediately, regardless of
  which LLM provider (or none) they have access to, satisfying "get
  access to an AI model" from any of the suggested options while not
  making the whole pipeline hard-depend on a paid API.
- *Con:* three thin provider wrappers is a bit more code than committing
  to a single SDK, and the reasoning text's tone/quality will differ
  slightly across providers.
- **With more time:** unify behind something like `litellm` for a single
  call interface across providers, and add response caching so repeated
  runs against the same JD/resume pair don't re-spend API calls.

## 4. Streamlit UI vs. a custom web frontend

**Chosen:** Streamlit (`app.py`) for the UI.

- *Pro:* one file, no separate frontend/backend split, runs anywhere with
  `streamlit run app.py`, and gives file upload, progress bars, tables,
  and CSV/JSON download out of the box — fast to build and easy for a
  reviewer to run.
- *Con:* less customizable styling/branding than a React/Flask app, and
  Streamlit re-runs the whole script on each interaction (fine at this
  scale of 10-20 resumes, but would need restructuring — e.g. caching,
  background jobs, or a proper client/server split — for very large
  batches or a production multi-user deployment).
- **With more time:** move to a small FastAPI backend (reusing
  `src/pipeline.py` unchanged) with a lightweight React frontend, enabling
  auth, persistent job history, and async processing for large batches.

## 5. Batch size / concurrency

**Chosen:** synchronous, in-process loop over resumes (both CLI and UI).

- *Pro:* simple, easy to reason about and debug, no threading bugs.
- *Con:* LLM reasoning calls are made one at a time, so wall-clock time
  scales roughly linearly with the number of resumes and the LLM
  provider's latency (TF-IDF scoring itself is effectively instant even
  for hundreds of resumes).
- **With more time:** parallelize the LLM reasoning calls (e.g.
  `concurrent.futures.ThreadPoolExecutor`) since they're independent
  per-candidate, while keeping the scoring step (already vectorized)
  as-is.

## 6. Sample data

Included 12 synthetic resumes (10 required minimum) spanning strong
backend-engineering fits, adjacent-but-not-matching profiles (frontend,
DevOps, data engineering, QA), and a clearly irrelevant profile (marketing)
to make the ranking behavior easy to sanity-check. Two of the twelve are
provided as `.docx` and `.pdf` (the rest `.txt`) to exercise all three
supported parser paths end-to-end.
