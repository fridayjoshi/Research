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

## Limitations (and how to reduce false positives)
- **Whitespace churn** can inflate DCM: normalizing line endings and collapsing multiple blank lines helps.
- **Reordering** (even semantically equivalent) scores as churn: treat pure “sort/reorder” prompts separately.
- **Tiny scale effects**: for very small files, adding/removing 1 line can dominate the score; use thresholds relative to file size.
- **Diff hunk definition** depends on the diff algorithm: using a consistent unified-diff generator keeps comparisons stable.

## Next experiments
- Compare DCM for prompts that ask for a “targeted patch” vs “rewrite file”.
- Correlate DCM with PR review time and merge conflicts.

## Implementation sketch
I added a small reference implementation in `Research/dcm_score.py` that computes:
- `added`, `removed`, `hunks` from `difflib.unified_diff` line stats
- the exact same normalization/weights from the note

This is intentionally heuristic and repo-agnostic, so it works on raw text pairs.

## Quick sanity check (synthetic)
Using `A = l1\n\nl2\n\nl3\n\nl4` (4 lines) as the baseline:

- **Targeted change**: `l2 -> l2_mod`
  - `added=1`, `removed=1`, `hunks=1`, `scale=4`
  - `churn_rate=0.5`, `hunk_rate=1.0`
  - **DCM = 0.65**

- **Rewrite (reorder)**: `l4,l3,l2,l1`
  - `added=3`, `removed=3`, `hunks=1`, `scale=4`
  - `churn_rate=1.5`, `hunk_rate=1.0`
  - **DCM = 1.35**

As expected, “rewrite” scores higher than “targeted” under this proxy.

## Real diff samples (measured)
These are just quick sanity data points to make the score feel grounded.

1. `Projects/reallyartificial-gtm-cards/ra-gtm-cards.js` between commits
   - `741e50d...` → `f18dd75...`
   - **DCM = 0.887** (churn_lines=82, hunks=2, scale=200)

2. `Projects/reallyartificial-gtm-cards/README.md` between the same commits
   - **DCM = 0.676** (churn_lines=4, hunks=2, scale=37)

Interpretation:
- A structural code change can spike churn even when the user-visible feature change is small.
- Doc-only diffs can still register hunks, but churn_lines tends to stay low, keeping DCM lower.
