"""scholarhunter.crew — end-to-end scholarship pipeline.

Four loosely-coupled stages (blueprint §3/§4):

  1. Researcher   extract a structured UserProfile from free-text (LLM or local).
  2. Searcher     gather candidate scholarships (DuckDuckGo + local seed).
  3. PreFilter    cheap deterministic prune (tools.filter_by_profile).
  4. Matchmaker   per-candidate scored ranking (LLM, or keyword fallback).

Every LLM call is guarded by try/except LLMError → deterministic fallback,
so the demo path (no GROQ_API_KEY) runs end-to-end with zero exceptions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from . import llm_client, schemas, tools
from .llm_client import LLMError
from .schemas import Scholarship, ScholarshipMatch, UserProfile


# ------------------------------------------------------------------ result --
@dataclass
class PipelineResult:
    """Final, renderable state of a run_pipeline() call."""

    profile: UserProfile
    matches: list[ScholarshipMatch] = field(default_factory=list)
    llm_used: bool = False
    warnings: list[str] = field(default_factory=list)

    def ranked(self) -> list[ScholarshipMatch]:
        """Sorted eligible-first, then by descending match_score."""
        return sorted(
            self.matches,
            key=lambda m: (not m.eligible, -m.match_score, (m.deadline_iso or "")),
        )


# --------------------------------------------------------- LLM prompt bits --
_MATCH_SYSTEM_PROMPT = """You are a scholarship matchmaker.
Given a student profile and a list of candidate scholarships, score each
candidate as either eligible (True) or not (False) and assign match_score in
[0.0, 1.0].

Scoring rubric:
- 0.9+  : exact field / country / level match + strong fit to notes.
- 0.7+  : right level and field, minor country mismatch.
- 0.4+  : general fit for level and one of the domains.
- < 0.4 : likely not a great fit (mention the specific blocker).
- 0.0   : clearly not applicable (country/level mismatch hard-coded).

Return ONLY a JSON object:
{
  "matches": [
    {"index": <int>, "match_score": <float 0..1>, "eligible": <bool>,
     "eligibility_reason": "<one-sentence reason, <= 200 chars>"},
    ...
  ]
}
The "index" field must match the position of the candidate in the input
"candidates" array. Include ALL candidates (even score 0 ones).
"""


def _build_match_payload(profile: UserProfile, pool: Sequence[Scholarship]) -> list[dict]:
    slim_pool = [
        {
            "name": s.name,
            "organization": s.organization,
            "award_amount": s.award_amount,
            "level": s.level,
            "geographic_scope": s.geographic_scope,
            "deadline": s.deadline,
            "fields": s.fields,
            "eligibility": (s.eligibility or "")[:300],
        }
        for s in pool
    ]
    user_payload = {
        "profile": profile.to_dict(),
        "candidates": slim_pool,
    }
    return [
        {"role": "system", "content": _MATCH_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
# ------------------------------------------------------------- 1. profile ----
def _local_profile(text: str) -> tuple[UserProfile, list[str]]:
    """Deterministic fallback profile parser (regex over the free text).

    Reads a handful of signals (level, country, domain, major) and returns a
    partially-filled UserProfile plus warnings about what was guessed.
    """
    warnings: list[str] = []
    lowered = (text or "").lower()

    level = ""
    if re.search(r"\bphd\b|doctoral|doctorate|postdoc", lowered):
        level = "Doctoral"
    elif re.search(r"\bmaster|ms\b|m\.s\.|m\.sc|ma\b|mphil", lowered):
        level = "Master"
    elif re.search(r"\bundergrad|undergraduate|bachelor|bs\b|b\.sc|b\.a|ba\b", lowered):
        level = "Undergraduate"
    elif re.search(r"\bgrad(?:uate)?\b|postgrad", lowered):
        level = "Graduate"

    country_lex = (
        "united kingdom", "uk", "germany", "united states", "usa",
        "europe", "france", "netherlands", "canada", "australia",
        "new zealand", "india", "singapore", "japan",
    )
    countries = [c for c in country_lex if c in lowered][:3]

    domain_lex = (
        "machine learning", "computer vision", "data science",
        "artificial intelligence", "deep learning", "natural language",
        "robotics", "computer science", "biotech", "biology",
        "economics", "physics", "mathematics", "engineering",
    )
    domains = [d.title() for d in domain_lex if d in lowered][:4]

    major = domains[0] if domains else ""

    name_match = re.search(
        r"\b(?:my name is|i am called)\s+([A-Z][\w-]+(?:\s+[A-Z][\w-]+)?)", text or ""
    )
    name = name_match.group(1) if name_match else ""

    prof = UserProfile(
        student_name=name,
        major=major,
        grade_level=level,
        target_year="2026-2027",
        extracurriculars=[],
        target_countries=countries,
        target_colleges=[],
        preferred_domains=domains,
        notes=(text or "").strip()[:400],
    )
    if not level:
        warnings.append("level not detected locally; treated as unknown")
    if not countries:
        warnings.append("no target country detected locally")
    return prof, warnings


def extract_profile(text: str, force_demo: bool = False) -> tuple[UserProfile, list[str]]:
    """Stage 1 — Researcher: free text → structured UserProfile.

    Returns (profile, warnings).  If the LLM path raises LLMError or the
    environment is in demo mode (GROQ_API_KEY unset), we fall back to
    _local_profile() deterministically.
    """
    profile, warns_local = _local_profile(text)
    if force_demo or llm_client.demo_ok():
        return profile, warns_local
    try:
        messages = [
            {"role": "system", "content": schemas.PROFILE_SYSTEM_PROMPT},
            {"role": "user", "content": text or ""},
        ]
        raw = llm_client.get_client().chat(messages, max_tokens=1024, json_mode=True)
        data = schemas.extract_json(raw)
        if not isinstance(data, dict):
            raise LLMError("profile JSON missing/malformed")
        parsed = UserProfile.from_dict(data)
        # Local parse is a safety net for any field the LLM missed.
        if not parsed.grade_level and profile.grade_level:
            parsed.grade_level = profile.grade_level
        if not parsed.major and profile.major:
            parsed.major = profile.major
        if not parsed.target_countries and profile.target_countries:
            parsed.target_countries = list(profile.target_countries)
        if not parsed.preferred_domains and profile.preferred_domains:
            parsed.preferred_domains = list(profile.preferred_domains)
        if not parsed.notes and profile.notes:
            parsed.notes = profile.notes
        return parsed, warns_local
    except LLMError as e:
        msg = f"profile LLM failed: {e}; using local parse"
        print(f"[crew] {msg}")
        return profile, warns_local + [msg]


# ------------------------------------------------------------- 2. gather ----
def gather_candidates(pool: Optional[Sequence[Scholarship]] = None) -> list[Scholarship]:
    """Stage 2 — Searcher: return the raw candidate pool.

    When *pool* is provided it is used as-is; otherwise the local seed file is
    loaded.  DuckDuckGo results are logged in ``extract_profile`` but not
    normalised into Scholarship objects in this MVP.
    """
    if pool is not None:
        out = [s for s in pool if isinstance(s, Scholarship)]
        if out:
            return out
    seeds = tools.load_seed_scholarships()
    if not seeds:
        import sys
        print("[crew] WARNING: empty candidate pool — seed file may be missing", file=sys.stderr)
    return seeds


# ------------------------------------------------------------- 3. prefilter --
def prefilter(pool: Sequence[Scholarship], profile: UserProfile) -> list[Scholarship]:
    """Stage 3 — PreFilter: deterministic prune via tools.filter_by_profile."""
    return tools.filter_by_profile(list(pool), profile)


# ------------------------------------------------------------- 4. match ----
def _keywords_from(profile: UserProfile) -> list[str]:
    """Short keyword list derived from the profile for the fallback scorer."""
    kws = list(profile.preferred_domains or [])
    if profile.major:
        kws.append(profile.major)
    if profile.grade_level:
        kws.append(profile.grade_level.lower())
    kws.extend(profile.target_countries or [])
    if profile.notes:
        kws.append(profile.notes[:60])
    seen: set[str] = set()
    out: list[str] = []
    for k in kws:
        kl = k.strip().lower()
        if kl and kl not in seen:
            seen.add(kl)
            out.append(k)
    return out[:8]


def _keyword_fallback(pool: Sequence[Scholarship], profile: UserProfile) -> list[ScholarshipMatch]:
    """Deterministic match ranking via tools.score_by_keywords."""
    kws = _keywords_from(profile)
    matches: list[ScholarshipMatch] = []
    for s in pool:
        score = tools.score_by_keywords(s, kws)
        eligible = score >= 0.4
        matches.append(ScholarshipMatch.from_dict({
            "name": s.name,
            "organization": s.organization,
            "award_amount": s.award_amount,
            "level": s.level,
            "geographic_scope": s.geographic_scope,
            "deadline": s.deadline,
            "fields": list(s.fields),
            "source": s.source,
            "match_score": score,
            "eligible": eligible,
            "eligibility_reason": f"Keyword score {score:.2f} over {len(kws)} keywords.",
        }))
    return matches


def _llm_match(profile: UserProfile, pool: Sequence[Scholarship]) -> list[ScholarshipMatch]:
    """LLM Matchmaker: score every candidate, return ScholarshipMatch list."""
    messages = _build_match_payload(profile, pool)
    raw = llm_client.get_client().chat(messages, max_tokens=2048, json_mode=True)
    data = schemas.extract_json(raw)
    if not isinstance(data, dict) or "matches" not in data:
        raise LLMError("match response missing 'matches' key")
    by_index: dict[int, dict] = {}
    for row in data["matches"]:
        if isinstance(row, dict):
            idx = int(row.get("index", -1))
            if idx >= 0:
                by_index[idx] = row
    matches: list[ScholarshipMatch] = []
    for i, s in enumerate(pool):
        row = by_index.get(i, {})
        score = max(0.0, min(1.0, float(row.get("match_score", 0.0))))
        eligible = bool(row.get("eligible", score >= 0.4))
        reason = str(row.get("eligibility_reason", "no reason provided")).strip()[:200]
        matches.append(ScholarshipMatch.from_dict({
            "name": s.name,
            "organization": s.organization,
            "award_amount": s.award_amount,
            "level": s.level,
            "geographic_scope": s.geographic_scope,
            "deadline": s.deadline,
            "fields": list(s.fields),
            "source": s.source,
            "match_score": score,
            "eligible": eligible,
            "eligibility_reason": reason,
        }))
    return matches


# --------------------------------------------------------- run_pipeline -----
def run_pipeline(
    text: str,
    force_demo: bool = False,
    candidate_pool: Optional[Sequence[Scholarship]] = None,
) -> PipelineResult:
    """End-to-end: free-text profile → ranked ScholarshipMatch list.

    Parameters
    ----------
    text:
        Free-text description of the student and scholarship goals.
    force_demo:
        When True, skip ALL LLM calls (offline path).
    candidate_pool:
        Optional pre-loaded pool of Scholarship objects.

    Returns a PipelineResult with matches sorted eligible-first / by score.
    """
    warnings: list[str] = []
    llm_used = False

    # 1. Researcher
    profile, warns = extract_profile(text, force_demo=force_demo)
    warnings.extend(warns)

    # 2. Searcher
    pool = gather_candidates(candidate_pool)
    if not pool:
        warnings.append("candidate pool is empty — returning no matches")
        return PipelineResult(profile=profile, matches=[], llm_used=llm_used, warnings=warnings)

    # 3. PreFilter
    filtered = prefilter(pool, profile)
    if not filtered:
        warnings.append("all candidates removed by prefilter; "
                        "falling back to full pool so you still see results")
        filtered = list(pool)

    # 4. Matchmaker (LLM or keyword fallback)
    if force_demo or llm_client.demo_ok():
        matches = _keyword_fallback(filtered, profile)
    else:
        try:
            matches = _llm_match(profile, filtered)
            llm_used = True
        except LLMError as e:
            msg = f"match LLM failed ({e}); using keyword fallback"
            print(f"[crew] {msg}")
            warnings.append(msg)
            matches = _keyword_fallback(filtered, profile)

    return PipelineResult(profile=profile, matches=matches, llm_used=llm_used, warnings=warnings)

