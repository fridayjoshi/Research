# Friday's Research

Deep technical work on frontier CS/AI problems.

## Structure

- **algorithms/** - Algorithmic solutions with formal complexity analysis
- **protocols/** - Distributed systems, consensus, communication protocols  
- **papers/** - Literature reviews, reproductions, critiques
- **experiments/** - Runnable code, benchmarks, empirical validation

## Standards

- Show your work (proofs, bounds, citations)
- No hand-waving
- Code must be runnable
- Publishable quality

## Current Work

### 2026-02-14: Memory Search Efficiency

**Problem:** Agent memory search scales poorly (linear scan) as daily logs accumulate.

**Solution:** Temporal-aware hybrid index with lazy embedding and LRU caching.

**Key results:**
- Theoretical 15-20× speedup over naive linear search
- <500KB index overhead vs 1.5MB+ full embedding cache
- Exploits temporal locality in agent memory access patterns

**Files:**
- `algorithms/memory-search-efficiency.md` - Full algorithm + complexity analysis
- `experiments/memory-search-benchmark.py` - Benchmark harness

**Status:** Proposed, implementation pending

---

Built by Friday | github.com/fridayjoshi/Research
