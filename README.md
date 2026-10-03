# 🎓 ScholarHunter

**Free-text scholarship matcher.** Describe yourself and your goals in plain
English and get a ranked list of scholarships that match — with an explainable
score, eligibility flag, and reason for each.

The core pipeline is **stdlib-only** (no hard runtime dependencies). It runs
fully **offline by default** (deterministic keyword scoring). Add a Groq API
key to switch to **LLM-based scoring** automatically.

![CI style pipeline badge](https://img.shields.io/badge/streamlit-cloud-ready-blue)

---

## How it works

A four-stage, loosely-coupled pipeline:

| # | Stage        | What it does                                              |
|---|--------------|-----------------------------------------------------------|
| 1 | **Researcher** | Extracts a structured profile from free text             |
| 2 | **Searcher**   | Gathers candidate scholarships (bundled seed list)        |
| 3 | **PreFilter**  | Drops obvious country / level mismatches                  |
| 4 | **Matchmaker** | Scores every candidate (LLM, or keyword fallback)        |

Every LLM call is guarded by `try/except LLMError` with a deterministic
fallback, so the app never crashes — it just degrades to offline mode.

## Project layout

```
scholarhunter-app/        # repo root
├── scholarhunter/        # the importable core package (stdlib-only)
│   ├── __init__.py       # public surface + version
│   ├── schemas.py        # data models + JSON-LLM helpers
│   ├── tools.py          # search / seed / filter / keyword scoring
│   ├── llm_client.py     # stdlib Groq/OpenAI chat client
│   ├── crew.py           # the 4-stage pipeline (run_pipeline)
│   ├── __main__.py       # CLI entry point
│   └── data/seed_scholarships.json
├── app.py                # Streamlit entry point (Streamlit Cloud runs this)
├── requirements.txt      # streamlit (web UI only)
├── pyproject.toml        # packaging (core + optional `app` extra)
├── README.md
├── .env.example
└── .streamlit/config.toml
```

## 1. Install & run locally

```bash
# 1) clone
git clone <your-repo-url>
cd scholarhunter-app

# 2) create a virtualenv (optional but recommended)
python -m venv .venv
# Windows PowerShell:
#   .venv\Scripts\Activate.ps1
# Linux/macOS:
#   source .venv/bin/activate

# 3) install
pip install -r requirements.txt
```

### Console CLI (no web UI needed)

```bash
# offline demo profile
python -m scholarhunter --demo

# your own free text (uses the LLM if GROQ_API_KEY is set)
python -m scholarhunter "CS grad, ML focus, target Germany, PhD"

# limit output
python -m scholarhunter --demo --limit 3
```

### Streamlit web app

```bash
streamlit run app.py        # opens http://localhost:8501
```

## 2. Push to GitHub

```bash
cd scholarhunter-app

git init
git add .
git commit -m "Initial commit: scholarhunter web app + core package"

# on GitHub, create an empty repo `scholarhunter-app` first, then:
git branch -M main
git remote add origin https://github.com/<your-username>/scholarhunter-app.git
git push -u origin main
```

> Tip: make sure `scholarhunter/data/seed_scholarships.json` is committed —
> it is the offline seed data the app relies on.

## 3. Deploy on Streamlit Cloud

1. Go to **<https://share.streamlit.io>** and click **New app**.
2. Connect the GitHub repo (`scholarhunter-app`) and branch `main`.
3. Streamlit Cloud auto-detects `requirements.txt` **and** `app.py` —
   leave everything at the defaults.
4. Click **Deploy** and wait for the build to finish.

### Optional: enable live LLM scoring

1. In the Cloud dashboard, open your app → **Manage** → **Secrets**.
2. Add a key/value pair:
   ```
   GROQ_API_KEY=gsk_...
   ```
   (Get one at <https://groq.com/quickstart>.)
3. Redeploy. The app banner changes from
   **"Offline demo mode"** to **"Live LLM scoring enabled"**, and
   **Matchmaker** now uses the model to rank. No code change needed.

> If you set `GROQ_API_KEY` but want to force offline mode in a session, tick
> **Force offline (skip LLM…** in the sidebar.

## Configuration

| Variable               | Default              | Purpose                              |
|------------------------|----------------------|--------------------------------------|
| `GROQ_API_KEY`         | *(unset → offline)*  | Enables live LLM scoring              |
| `GROQ_MODEL`           | `llama-3.3-70b-versatile` | Primary model                  |
| `GROQ_MODEL_FALLBACKS` | `llama-3.3-70b-versatile` | Comma-separated fallbacks   |
| `GROQ_TIMEOUT_S`       | `30`                 | Per-request timeout (seconds)         |
| `GROQ_MAX_RETRIES`     | `2`                  | Retries before falling to next model  |

## Extending

* **More scholarships** — add rows to
  `scholarhunter/data/seed_scholarships.json` (see existing entries for the
  shape of each dict).
* **Live web search** — install `duckduckgo-search` if you want the `tools`
  module to augment the seed list at runtime; the pipeline already handles the
  case where the package or the network is unavailable and silently skips the
  search step.

## Troubleshooting

| Symptom                                        | Fix                                                     |
|------------------------------------------------|---------------------------------------------------------|
| `streamlit: command not found`                 | Re-run `pip install -r requirements.txt` in the venv.   |
| Nothing shows in the table on Cloud            | Confirm `app.py` is at the repo root and `main branch`  |
|                                              | is selected in the Cloud dashboard.                     |
| LLM path not kicking in                        | Add `GROQ_API_KEY` to Cloud Secrets and redeploy.       |
| `ImportError: scholarhunter`                   | Make sure `scholarhunter/` (with its subfolder) is      |
|                                              | inside the repo next to `app.py`.                       |

## License

MIT — see `pyproject.toml`.
