"""scholarhunter CLI — run the scholarship pipeline from a terminal.

Usage:
    python -m scholarhunter --demo
    python -m scholarhunter "CS grad student, ML focus, target Germany, PhD"
    python -m scholarhunter --demo --limit 3
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .crew import run_pipeline


def _print_result(result, limit: int | None) -> None:
    print("=" * 78)
    print("scholarhunter v" + __version__)
    print(f"LLM used: {result.llm_used}")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print("-" * 78)

    rows = result.ranked()
    if limit is not None and limit > 0:
        rows = rows[:limit]
    if not rows:
        print("(no scholarships matched)")
        return

    header = (
        f"{'#':>2}  {'Scholarship':<42} {'Score':>5}  {'Elig':>5}  "
        f"{'Deadline':<12}  URL"
    )
    print(header)
    print("-" * 78)
    for i, m in enumerate(rows, 1):
        name = (m.name or "")[:41]
        org = (m.organization or "")[:30]
        score = f"{m.match_score:.2f}"
        elig = "yes" if m.eligible else "no "
        dl = (m.deadline_iso or m.deadline or "")[:12]
        url = (m.url or "")[:40]
        print(f"{i:>2}  {name:<42} {score:>5}  { elig:>5}  {dl:<12}  {url}")
        print(f"    org: {org}")
        if m.eligibility_reason:
            print(f"    why: {(m.eligibility_reason or '')[:110]}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scholarhunter",
        description="Free-text scholarship matcher (stdlib-only, offline demo fallback).",
    )
    parser.add_argument(
        "text", nargs="?", default=None,
        help="Free-text description of the student (e.g. 'CS grad, ML, target Germany').",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use the bundled demo profile and force offline (no LLM) path.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Show only the top N ranked scholarships.",
    )
    args = parser.parse_args(argv)

    if args.demo:
        from .tools import demo_profile as _demo_profile_fn
        _p = _demo_profile_fn()
        text = " ".join([
            f"{_p.major} grad student",
            f"{'/'.join(_p.preferred_domains)} focus",
            f"target {', '.join(_p.target_countries)}",
        ])
        force_demo = True
    elif args.text:
        text = args.text
        force_demo = False
    else:
        parser.print_help()
        return 1

    result = run_pipeline(text, force_demo=force_demo)
    _print_result(result, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
