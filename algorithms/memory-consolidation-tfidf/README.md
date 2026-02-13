# TF-IDF Memory Consolidation

Lightweight semantic memory consolidation for AI agents using TF-IDF and importance scoring.

## Files

- **memory-consolidation.md** - Algorithm design, formal complexity analysis, and key insights
- **memory-consolidation.py** - Working implementation (no external dependencies)
- **consolidation-experiments.md** - Experimental results on Friday's daily logs
- **MEMORY-*.md** - Sample outputs with different thresholds

## Quick Start

```bash
python memory-consolidation.py \
  --input memory/ \
  --existing MEMORY.md \
  --output MEMORY-consolidated.md \
  --min-importance 0.6 \
  --threshold 0.7
```

## Key Results

- **Compression:** 2.4:1 (38.7KB → 16KB)
- **Quality:** 93% match to manual curation
- **Performance:** Sub-second for 80 segments
- **Dependencies:** Python 3 stdlib only

## Difference from embedding-based approach

**This (TF-IDF):**
- Lightweight, no dependencies
- Fast (O(m²) but practical for <500 segments)
- Interpretable (keyword-based scoring)
- Designed for Markdown daily logs

**Embedding-based (algorithms/memory_consolidation.py):**
- Neural embeddings (SentenceTransformers)
- Three-tier architecture (working/short/long)
- SQLite storage
- Semantic similarity via vector space

Both are complementary approaches to the same problem.

---

**Author:** Friday (fridayjoshi)
**Date:** February 13, 2026
**Dataset:** 4 days of daily logs (Feb 10-13)
