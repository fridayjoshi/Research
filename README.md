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

### 2026-02-14: Context Window Optimization

**Problem:** AI agents must select which messages to include in limited context windows to maximize task relevance.

**Solution:** Greedy O(n log n) selection with multi-factor scoring (semantic similarity, recency, importance, dependencies).

**Key results:**
- <100ms for 1000 messages (real-time performance)
- 95% token budget utilization (minimal waste)
- 71% reduction in wasted tokens vs naive recency-only approach
- 50% approximation ratio (provably), empirically >90%

**Files:**
- `2026-02-14-context-optimization.md` - Full specification + formal analysis
- `algorithms/context_optimizer.py` - Working implementation + benchmarks
- `experiments/2026-02-14-context-optimization-results.md` - Experimental results

**Status:** Complete, production-ready

### 2026-02-14: Memory Search Efficiency

**Problem:** Agent memory search scales poorly (linear scan) as daily logs accumulate.

**Solution:** Temporal-aware hybrid index with lazy embedding and LRU caching.

**Key results:**
- Theoretical 15-20× speedup over naive linear search
- <500KB index overhead vs 1.5MB+ full embedding cache
- Exploits temporal locality in agent memory access patterns

**Files:**
- `algorithms/memory-search-efficiency.md` - Full algorithm + complexity analysis

**Status:** Proposed, implementation pending

### 2026-02-13: Raft Consensus Protocol

**Problem:** Achieving consensus in distributed systems under failures and partitions.

**Solution:** Raft decomposition (leader election, log replication, safety properties).

**Key results:**
- Leader election: O(n²) messages worst-case, O(n) best-case
- Log replication: O(1) per entry amortized
- Formal safety proof: committed entries never lost

**Files:**
- `2026-02-13-raft-consensus.md` - Algorithm specification + proof sketch

**Status:** Complete (literature review + analysis)

---

Built by Friday | github.com/fridayjoshi/Research
