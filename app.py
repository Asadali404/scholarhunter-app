"""scholarhunter — Streamlit app (the entry point for Streamlit Cloud).

Run locally:
    streamlit run app.py

On Streamlit Cloud this same ``app.py`` is the web-server entry point.
Live LLM scoring turns on automatically when a ``GROQ_API_KEY`` secret is set
in the Streamlit Cloud dashboard; otherwise the app runs in offline demo mode
(deterministic keyword scoring) with zero external dependencies.
"""
from __future__ import annotations

import csv
import io

import streamlit as st

import scholarhunter
from scholarhunter import llm_client, run_pipeline

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ScholarHunter",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_TEXT = (
    "Computer Science graduate student focused on Machine Learning and "
    "Computer Vision, targeting the United Kingdom and Germany for 2026-2027. "
    "Looking for fully funded research positions with a stipend of at least "
    "$20,000/year. Strong GPA (3.8/4.0), active open-source ML contributor."
)


@st.cache_resource
def _live_available() -> bool:
    """True when a Groq API key is present (live LLM scoring possible)."""
    return not llm_client.demo_ok()


st.title("🎓 ScholarHunter")
st.caption(
    "Free-text scholarship matcher — ranks scholarships by fit to your profile. "
    "Runs fully offline; add a Groq API key to enable LLM-based scoring."
)

if _live_available():
    st.success("Live LLM scoring enabled (GROQ_API_KEY detected).")
else:
    st.info("Offline demo mode — using deterministic keyword scoring (no LLM).")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Search controls")

    st.text_area(
        "Describe yourself and your goals",
        value=DEMO_TEXT,
        height=240,
        key="profile_text",
        help="Free text about your field, level, target countries, funding needs, etc.",
    )

    if st.button("Load demo profile", use_container_width=True, key="load_demo"):
        st.session_state["profile_text"] = DEMO_TEXT
        st.rerun()

    force_offline = st.checkbox(
        "Force offline (skip LLM even if a key is set)",
        key="force_offline",
    )

    limit = st.number_input(
        "Show top N (0 = all)", min_value=0, max_value=100, value=0, key="limit"
    )

    run_now = st.button("🚀 Search scholarships", type="primary",
                        use_container_width=True, key="run")


def _run(text: str, force_offline: bool) -> None:
    result = run_pipeline(text, force_demo=force_offline)
    st.session_state["result"] = result
    st.session_state["result_text"] = text


# Auto-run on first load, or re-run when the button is pressed.
if st.session_state.get("result") is None or run_now:
    _run(st.session_state["profile_text"], force_offline)

result: scholarhunter.crew.PipelineResult = st.session_state["result"]

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
if result.warnings:
    with st.expander(f"⚠️ {len(result.warnings)} warning(s) in this run"):
        for w in result.warnings:
            st.text(f"- {w}")

# ---------------------------------------------------------------------------
# Extracted profile
# ---------------------------------------------------------------------------
st.subheader("👤 Extracted profile")
p = result.profile
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"**Name**: {p.student_name or '—'}")
c2.markdown(f"**GPA**: {p.gpa or '—'}")
c3.markdown(f"**Major**: {p.major or '—'}")
c4.markdown(f"**Level**: {p.grade_level or '—'}")
c1.markdown(f"**Target year**: {p.target_year or '—'}")
c2.markdown(f"**Funding**: {p.financial_aid or '—'}")
st.write(f"**Preferred domains**: {', '.join(p.preferred_domains) or '—'}")
st.write(f"**Target countries**: {', '.join(p.target_countries) or '—'}")
st.write(f"**Target colleges**: {', '.join(p.target_colleges) or '—'}")
if p.notes:
    st.write(f"**Notes**: {p.notes}")

st.divider()

# ---------------------------------------------------------------------------
# Ranked results
# ---------------------------------------------------------------------------
st.subheader(f"🏆 Ranked matches ({len(result.matches)})")

rows = result.ranked()
limit_int = int(st.session_state.get("limit", 0) or 0)
if limit_int > 0:
    rows = rows[:limit_int]

if not rows:
    st.warning(
        "No scholarships matched your profile. Try broadening your goals or "
        "clicking **Load demo profile**."
    )
else:
    table: list[dict] = []
    for i, m in enumerate(rows, 1):
        table.append(
            {
                "#": i,
                "Scholarship": m.name or "—",
                "Organization": m.organization or "—",
                "Level": m.level or "—",
                "Scope": m.geographic_scope or "—",
                "Award": m.award_amount or "—",
                "Deadline": (m.deadline_iso or m.deadline or "—"),
                "Eligible": "✅ yes" if m.eligible else "❌ no",
                "Score": f"{m.match_score:.2f}",
                "Why": (m.eligibility_reason or "—"),
                "URL": m.url or "—",
            }
        )
    st.dataframe(table, use_container_width=True, hide_index=True, height=520)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(table[0].keys()))
    writer.writeheader()
    writer.writerows(table)
    st.download_button(
        "⬇️ Download CSV",
        data=buf.getvalue(),
        file_name="scholarhunter_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ About & pipeline"):
    st.write(
        "scholarhunter turns a free-text scholarship goal into a ranked match list "
        f"(v{scholarhunter.__version__}) using a four-stage pipeline:\n\n"
        "1. **Researcher** — extract a structured profile from free text.\n"
        "2. **Searcher** — gather candidate scholarships (bundled seed list).\n"
        "3. **PreFilter** — drop obvious country / level mismatches.\n"
        "4. **Matchmaker** — score each candidate (LLM if a key is set; "
        "deterministic keyword fallback otherwise).\n\n"
        f"Last run used the LLM: **{'yes' if result.llm_used else 'no'}**"
    )
