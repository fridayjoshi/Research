#!/usr/bin/env python3
"""Diff-Churn-Minimality (DCM) scoring.

Heuristic based on unified diff statistics:
- churn_lines = added + removed
- churn_rate = churn_lines / max(1, len(A_lines))
- hunk_rate = hunks / max(1, scale/200)
- DCM = 0.7 * churn_rate + 0.3 * hunk_rate

Usage:
  python3 dcm_score.py --a-file A.txt --b-file B.txt
  python3 dcm_score.py --a "..." --b "..."
"""

from __future__ import annotations

import argparse
import difflib
from dataclasses import dataclass
from typing import Optional


@dataclass
class DCMResult:
    dcm: float
    added: int
    removed: int
    hunks: int
    scale: int
    churn_lines: int
    churn_rate: float
    hunk_rate: float


def _unified_diff_stats(a: str, b: str) -> tuple[int, int, int]:
    a_lines = a.splitlines()
    b_lines = b.splitlines()

    diff = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile="A",
        tofile="B",
        lineterm="",
    )

    added = 0
    removed = 0
    hunks = 0

    for line in diff:
        if line.startswith("@@"):
            hunks += 1
            continue
        if line.startswith("+++") or line.startswith("---"):
            # file headers
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1

    return added, removed, hunks


def dcm_score(a: str, b: str) -> DCMResult:
    a_lines = a.splitlines()
    scale = max(1, len(a_lines))

    added, removed, hunks = _unified_diff_stats(a, b)

    churn_lines = added + removed
    churn_rate = churn_lines / scale
    hunk_rate = hunks / max(1, scale / 200)
    dcm = 0.7 * churn_rate + 0.3 * hunk_rate

    return DCMResult(
        dcm=dcm,
        added=added,
        removed=removed,
        hunks=hunks,
        scale=scale,
        churn_lines=churn_lines,
        churn_rate=churn_rate,
        hunk_rate=hunk_rate,
    )


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a-file")
    p.add_argument("--b-file")
    p.add_argument("--a")
    p.add_argument("--b")

    args = p.parse_args(argv)

    if args.a_file and args.b_file:
        a = _read_text(args.a_file)
        b = _read_text(args.b_file)
    elif args.a is not None and args.b is not None:
        a = args.a
        b = args.b
    else:
        p.error("Provide either (--a-file & --b-file) or (--a & --b).")

    r = dcm_score(a, b)
    print(
        "DCM={:.6f} churn_lines={} hunks={} scale={} (added={} removed={})".format(
            r.dcm, r.churn_lines, r.hunks, r.scale, r.added, r.removed
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
