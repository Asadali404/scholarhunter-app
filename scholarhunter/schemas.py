"""scholarhuner.schemas — typed data models + JSON-LLM helpers (stdlib only)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------- LLM JSON ----

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

try:  # pydantic is listed in requirements.txt but the fallback keeps us safe
    from pydantic import BaseModel  # type: ignore

    class _JSONBase(BaseModel):
        model_config = dict(extra="allow")

    def _coerce(cls: type, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls.model_validate(data)
        return data

except Exception:  # pragma: no cover - no pydantic installed (local testing etc.)
    _JSONBase = object

    def _coerce(cls: type, data: Any) -> Any:
        return data


def extract_json(text: str) -> Any:
    """Pull a JSON value out of an LLM reply (raw / fenced / embedded)."""
    if not text:
        return None
    text = text.strip()
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


# -------------------------------------------------------------- data model ---


@dataclass
class Scholarship:
    """One candidate scholarship (blueprint §3.4 Scholarship)."""

    name: str
    organization: str = "Unknown"
    url: str = ""
    award_amount: str = ""
    eligibility: str = ""
    fields: list[str] = field(default_factory=list)
    level: str = ""                 # undergraduate / graduate / ...
    geographic_scope: str = ""      # local / national / international
    deadline: str = ""             # original string as published
    deadline_iso: str = ""         # normalized YYYY-MM-DD when known
    application_notes: str = ""    # application links, process, contact
    source: str = ""               # e.g. "web search" / "cached demo data"
    source_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scholarship":
        if isinstance(d, cls):
            return d
        known = {f for f in cls.__dataclass_fields__}          # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class ScholarshipMatch(Scholarship):
    """Scholarship + match_score + eligibility_reason (§3.4)."""

    match_score: float = 0.0      # 0.0–1.0
    eligible: bool = True         # False when a hard requirement fails
    eligibility_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScholarshipMatch":
        raw = {k: (v if not isinstance(v, Scholarship) else asdict(v))
               for k, v in (d or {}).items()}
        known = {f for f in cls.__dataclass_fields__}           # type: ignore[attr-defined]
        raw.setdefault("name", "Unknown scholarship")
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class UserProfile:
    """Structured student profile produced by the Researcher (§3.2, §4.2)."""

    student_name: str = ""
    gpa: float = 0.0
    major: str = ""
    grade_level: str = ""
    target_year: str = ""
    financial_aid: str = ""
    extracurriculars: list[str] = field(default_factory=list)
    target_countries: list[str] = field(default_factory=list)
    target_colleges: list[str] = field(default_factory=list)
    preferred_domains: list[str] = field(default_factory=list)
    min_award: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        if isinstance(d, cls):
            return d
        known = {f for f in cls.__dataclass_fields__}           # type: ignore[attr-defined]
        d = d or {}
        for list_field in ("extracurriculars", "target_countries",
                           "target_colleges", "preferred_domains"):
            val = d.get(list_field)
            if isinstance(val, str):
                d[list_field] = [x.strip() for x in val.replace(";", ",").split(",") if x.strip()]
            elif not isinstance(val, list):
                d[list_field] = []
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------- prompts ----


PROFILE_SYSTEM_PROMPT = """You are a student-profile extraction agent for a scholarship search system.
Extract a structured profile from the user's free-text input.
Rules:
- Return ONLY a valid JSON object — no prose, no markdown.
- Use exactly these keys: student_name, gpa, major, grade_level, target_year,
  financial_aid, extracurriculars, target_countries, target_colleges, preferred_domains,
  min_award, notes.
- gpa is a number (0-4, or 0-10 if the scale indicates; 0 if unknown).
- Extracurriculars, country, college and domain values are arrays of short strings.
- Missing values: "" for strings, 0 for gpa, [] for arrays.
- In notes, record anything important that did not fit other fields.
"""