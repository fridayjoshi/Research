# Over-editing metric (research note, 2026-04-25)

## Problem
LLM edits can introduce large diffs (churn) even when only a small change is needed. Churn hurts review and increases merge conflicts.

## Goal
Estimate “how much of a patch is churn” in a repo-agnostic way, without needing semantic judges.

## Diff-Churn-Minimality (DCM)
Given original file text `A` and revised file text `B`:

1. Split into lines, compute a line diff with hunks.
2. Count:
   - `added` = lines added
   - `removed` = lines removed
   - `hunks` = number of hunks
3. Normalize with `scale = max(1, len(A_lines))`.
4. Compute:
   - `churn_lines = added + removed`
   - `churn_rate = churn_lines / scale`
   - `hunk_rate = hunks / max(1, scale/200)` (heuristic)
5. Score:
   - `DCM = 0.7 * churn_rate + 0.3 * hunk_rate` (higher = more churn)

## Optional “spill” proxy
If you have a required-change region (test failure region, user-specified scope, or an edit plan), compute churn inside vs outside that region:

- `spill_rate = churn_out / max(1, churn_req + churn_out)`

High spill rate suggests over-editing.

## Next experiments
- Compare DCM for prompts that ask for a “targeted patch” vs “rewrite file”.
- Correlate DCM with PR review time and merge conflicts.
