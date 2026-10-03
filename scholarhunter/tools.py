"""scholarhunter.tools — web search (DuckDuckGo with cap) + local search/filter utilities.

The "tools" that our agent pipeline invokes. Every tool has a safe local
fallback so the demo can run offline (blueprint §2 "cached demo mode").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import schemas
from .schemas import UserProfile

# ------------------------------------------------------------------ constants
MAX_QUERIES_PER_RUN = 5           # hard cap (blueprint §4, Agent 2)
DDG_TIMEOUT_S = 12
DATA_DIR = Path(__file__).parent / "data"
SEED_PATH = DATA_DIR / "seed_scholarships.json"


# --------------------------------------------------------------------------
# DuckDuckGo search — capped, fails open.
# --------------------------------------------------------------------------
def ddg_search(query: str, max_results: int = 5, timeout: int = DDG_TIMEOUT_S) -> list[dict[str, Any]]:
    """One DuckDuckGo web search. Returns list of {title, url, snippet}.

    Fails open: returns [] (not raises) if the `duckduckgo_search` package is
    missing, the network is down, or a rate-limit error occurs.  The rest of the
    pipeline degrades gracefully by using the seed list.
    """
    DDGS = None
    try:
        import duckduckgo_search as _ddg
        DDGS = getattr(_ddg, "DDGS", None)
    except Exception:
        DDGS = None
    if DDGS is None:
        print("[tools] duckduckgo-search not importable; returning [] for this query")
        return []

    hits: list[dict[str, Any]] = []
    try:
        with DDGS(timeout=timeout) as ddgs:
            for row in ddgs.text(query, max_results=max_results):
                hits.append(
                    {
                        "title": row.get("title", ""),
                        "url": row.get("href") or row.get("url", ""),
                        "snippet": row.get("body") or row.get("snippet", ""),
                    }
                )
    except Exception as exc:
        print(f"[tools] ddg_search failed for {query!r}: {exc}")
    return hits[:max_results]


def build_queries(profile: UserProfile, extra_domains: Optional[Sequence[str]] = None) -> list[str]:
    """Compose up to MAX_QUERIES_PER_RUN search queries (blueprint §4)."""
    countries = [c.strip() for c in (profile.target_countries or []) if c and c.strip()]
    for c in (extra_domains or []):
        if c and c.strip() and c.strip() not in countries:
            countries.append(c.strip())
    countries = countries[:5]
    domains = [d.strip() for d in (profile.preferred_domains or []) if d and d.strip()][:3]
    major = (profile.major or "").strip()
    level = (profile.grade_level or "graduate").strip().lower()

    queries: list[str] = []
    for c in countries:
        dom = domains[0] if domains else major
        queries.append(f"fully funded {level} scholarship {dom or 'stem'} {c} 2026")
    if countries and major:
        queries.append(f"{countries[0]} government scholarship international students {level} {major}")
    if major:
        queries.append(f"{major} funded research position open 2026")
    if domains:
        queries.append(f"{domains[0]} fellowship stipend deadline 2026")
    queries.append(f"scholarships for {level} students fully funded 2026")

    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = " ".join(q.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= MAX_QUERIES_PER_RUN:
            break
    return out


# ------------------------------------------------------------- seed loader ----
def load_seed_scholarships() -> list[schemas.Scholarship]:
    """Load the local seed list. Never raises; returns [] on any failure."""
    try:
        raw = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[tools] seed list unavailable ({exc}); continuing with empty local set")
        return []
    out: list[schemas.Scholarship] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(schemas.Scholarship.from_dict(row))
        except Exception:
            continue
    return out


def _text_blob(s: schemas.Scholarship) -> str:
    tokens = [
        s.name, s.organization, s.award_amount, s.eligibility,
        s.level, s.geographic_scope, s.deadline, s.application_notes, s.source,
        *s.fields,
    ]
    return " ".join(t for t in tokens if t).lower()


def filter_by_profile(pool, profile: UserProfile) -> list[schemas.Scholarship]:
    """Deterministic pre-filter: drop only obvious country/level mismatches.

    Keyword scoring is the LLM Matchmaker's job; this just prunes clearly-wrong
    options to save tokens and rows on the final table.
    """
    out: list[schemas.Scholarship] = []
    wanted_countries = {c.strip().lower() for c in profile.target_countries if c and c.strip()}
    wanted_level = (profile.grade_level or "").strip().lower()

    for s in pool:
        s_base = s if isinstance(s, schemas.Scholarship) else _unwrap_match(
            schemas.ScholarshipMatch.from_dict(s.to_dict()))
        blob = _text_blob(s_base)
        if wanted_countries and s_base.geographic_scope.lower() not in ("international", "global"):
            if not any(c in blob or c in s_base.geographic_scope.lower() for c in wanted_countries):
                continue
        if wanted_level and s_base.level:
            lvl = s_base.level.lower()
            if not (wanted_level in lvl or lvl in wanted_level
                    or ("grad" in wanted_level or "grad" in lvl)
                    or ("phd" in wanted_level and "phd" in lvl)
                    or ("master" in wanted_level and "master" in lvl)):
                continue
        out.append(s_base)
    return out


def _unwrap_match(m: schemas.ScholarshipMatch) -> schemas.Scholarship:
    d = m.to_dict()
    for k in ("match_score", "eligible", "eligibility_reason"):
        d.pop(k, None)
    return schemas.Scholarship.from_dict(d)


def score_by_keywords(s: schemas.Scholarship, keywords: Sequence[str]) -> float:
    """Cheap local keyword score, used only when the LLM path is unavailable.

    Distinct keyword hits in title+org+fields, clamped to [0, 1].
    """
    if not keywords:
        return 0.0
    hay = " ".join([s.name, s.organization, s.award_amount, " ".join(s.fields)]).lower()
    hits = {k for k in keywords if k and k.lower() in hay}
    if not hits:
        return 0.0
    denom = max(1, min(len([k for k in keywords if k]), 5))
    return round(min(1.0, len(hits) / denom), 3)


def demo_profile() -> UserProfile:
    """Small demo profile so the UI can be tried instantly."""
    return UserProfile(
        student_name="Demo Applicant",
        gpa=3.8,
        major="Computer Science",
        grade_level="Graduate",
        target_year="2026-2027",
        financial_aid="Full funding sought",
        extracurriculars=["Open-source ML contributor", "Debate club captain"],
        target_countries=["United Kingdom", "Germany"],
        target_colleges=[],
        preferred_domains=["Machine Learning", "Computer Vision"],
        min_award="$20,000/yr",
        notes="Prefer fully-funded research with stipend.",
    )
