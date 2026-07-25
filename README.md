🔗 **Live App:** [resume-screening-agent-6tpz.onrender.com]
(https://resume-screening-agent-6tpz.onrender.com)

> ⚠️ Hosted on a free tier — the app may take 30–60 seconds to wake up on first visit.

📂 **GitHub Repo:** https://github.com/Umar1321um/resume-screening-agent
---

## 🚀 Live Demo

Paste a job description, upload a batch of resumes (or use the bundled sample data), and get back a ranked, explained shortlist — right in your browser.



![App Screenshot](docs/screenshot.png)



**Try it now →** [resume-screening-agent-6tpz.onrender.com](https://resume-screening-agent-6tpz.onrender.com)

---
# Resume Screening Agent

Ranks a folder of resumes (PDF / DOCX / TXT) against a Job Description using
NLP similarity + skill matching, and outputs a scored, ranked shortlist with
reasoning — via CLI or a Streamlit web UI.

- **Works with or without an LLM API key.** The relevance score is always
  deterministic NLP (TF-IDF + skill overlap). An LLM (Anthropic, OpenAI, or
  Groq — whichever key you set) is used only to generate a readable
  explanation per candidate; without a key, a template-based explanation is
  used instead so the agent is always fully runnable.
- **Handles 10+ resumes per run** (12 sample resumes included: 10 `.txt`, 1
  `.docx`, 1 `.pdf`).

---

## 1. Setup

### 1.1 Clone and install dependencies

```bash
git clone <your-fork-url>
cd resume-screening-agent
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 1.2 (Optional) Get access to an AI model for reasoning

Pick **one** provider. The agent auto-detects whichever key is present.

**Anthropic (Claude)**
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

**OpenAI**
```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

**Groq (free tier)**
```bash
pip install groq
export GROQ_API_KEY=gsk_...
```

Or copy `.env.example` to `.env` and fill in one key, then load it
(`source .env` or use `python-dotenv` / your shell's env loading of choice).

**Verify your key works (one test message):**
```bash
python3 -c "
import os, anthropic
client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
resp = client.messages.create(
    model='claude-sonnet-5',
    max_tokens=50,
    messages=[{'role': 'user', 'content': 'Say hello in 5 words.'}],
)
print(resp.content[0].text)
"
```
If you see a reply printed, you're set. Swap in the equivalent OpenAI/Groq
snippet if using those instead.

If you skip this step entirely, the agent still runs end-to-end — it just
uses rule-based explanations instead of LLM-generated ones (set
`LLM_PROVIDER=none` to force this explicitly even if a key is present).

---

## 2. Run it (CLI)

```bash
python3 main.py --jd data/job_description.txt --resumes data/resumes --out outputs
```

Flags:
- `--jd` — path to a job description `.txt` file
- `--resumes` — path to a folder of resumes (`.pdf` / `.docx` / `.txt`)
- `--out` — output directory for `ranked_results.csv` / `ranked_results.json` (default: `outputs/`)
- `--no-llm` — skip LLM calls entirely, use the fast rule-based explanation

Example output:
```
=== Ranked Shortlist ===
# 1   44.62  01_aditi_sharma.txt
# 2   34.94  09_grace_kim.docx
# 3   32.08  11_natasha_ivanova.pdf
# 4   30.00  03_sophia_lee.txt
...
Saved CSV  -> outputs/ranked_results.csv
Saved JSON -> outputs/ranked_results.json
```

---

## 3. Run it (Web UI)

```bash
streamlit run app.py
```

This opens a browser UI where you can:
1. Paste or upload a Job Description (defaults to the bundled sample JD).
2. Upload your own resumes, or use the bundled sample folder.
3. Click **Run Screening** to see a ranked table, per-candidate reasoning
   (expandable cards), and CSV/JSON download buttons.

---

## 4. Project structure

```
resume-screening-agent/
├── app.py                     # Streamlit UI
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── parser.py              # PDF/DOCX/TXT text extraction
│   ├── extractor.py           # Skills / years-experience / education extraction
│   ├── scorer.py              # TF-IDF similarity + skill-overlap scoring
│   ├── llm_reasoner.py        # Multi-provider LLM reasoning + offline fallback
│   └── pipeline.py            # Orchestrates parse -> score -> rank -> save
├── data/
│   ├── job_description.txt    # Sample JD
│   └── resumes/               # 12 sample resumes (.txt / .docx / .pdf)
├── outputs/
│   ├── ranked_results.csv     # Sample output (checked in for review)
│   └── ranked_results.json
├── docs/
│   ├── SCORING_METHODOLOGY.md
│   └── TRADEOFFS.md
└── tests/
    └── test_pipeline.py
```

---

## 5. How scoring works (summary)

```
final_score = 100 * (0.6 * TF-IDF cosine similarity + 0.4 * skill overlap)
```

- **TF-IDF similarity** — JD and all resumes vectorized together (unigrams
  + bigrams), cosine similarity between JD vector and each resume vector.
- **Skill overlap** — exact match against a ~70-term tech skill taxonomy,
  `matched / required`.
- Plus extracted **years of experience** and **education level** shown
  alongside the score (not part of the numeric score itself).

Full rationale in [`docs/SCORING_METHODOLOGY.md`](docs/SCORING_METHODOLOGY.md).
Design tradeoffs and what I'd improve with more time in
[`docs/TRADEOFFS.md`](docs/TRADEOFFS.md).

---

## 6. Testing

```bash
python3 -m pytest tests/ -v
```

Tests cover skill/education/experience extraction, scoring math, and a full
pipeline integration test against the bundled sample data — all without any
network or LLM calls, so they run fast and free in CI.

---

## 7. Adding your own JD / resumes

- Replace `data/job_description.txt` with your own JD text (or paste it
  directly into the Streamlit UI).
- Drop `.pdf` / `.docx` / `.txt` resumes into `data/resumes/` (or upload
  them directly in the UI) — no naming convention required.
- Extend the skill taxonomy in `src/extractor.py` (`SKILL_TAXONOMY`) if
  you're screening for a different role family (e.g. add `Photoshop`,
  `SEO`, `Salesforce` for a marketing role).
