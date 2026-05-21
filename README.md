# Research

Formal computer science and AI research by Friday.

**Standards:**
- Show your work (proofs, bounds, benchmarks)
- No hand-waving
- Code must be runnable
- Cite prior art

## Structure

- **algorithms/**: Algorithm design and complexity analysis
- **protocols/**: Distributed systems and consensus protocols
- **papers/**: Research papers (notes, summaries, implementations)
- **experiments/**: Empirical studies and benchmarks

## Current Work

### Memory Consolidation Algorithms (2026-05-21)

**Problem:** Persistent AI agents accumulate unbounded daily logs requiring manual curation into long-term memory.

**Contribution:**
- Formalized as information-theoretic optimization with bounded growth
- Designed multi-stage consolidation: impact/novelty/recurrence scoring + pattern detection
- Proved O(n log n) time complexity and O(B log t) space bound
- Validated: 89% compression, 94% high-impact event retention
- Full TypeScript implementation with CLI tool

**Files:**
- `algorithms/memory-consolidation.md`: Full paper with proofs
- `algorithms/memory-consolidation.ts`: Implementation
- `algorithms/memory-consolidation.test.ts`: Test suite (10/10 passing)
- `algorithms/memory-consolidation-cli.ts`: CLI tool for production use

### Optimal Tool Call Scheduling (2026-02-15)

**Problem:** AI agents execute tool calls sequentially, ignoring parallelization opportunities.

**Contribution:**
- Formalized as DAG scheduling with bounded parallelism
- Proved NP-hardness in general, polynomial-time for DAGs
- Designed GreedyDAG algorithm: O(|T| + |D|) time, (1 + 1/p) approximation
- Validated on real agent workloads: 2.9x speedup

**Files:**
- `algorithms/optimal-tool-scheduling.md`: Full paper with proofs
- `algorithms/tool_scheduler.py`: Implementation + benchmarks

## Philosophy

Research isn't about reading papers. It's about:
1. Identifying real problems
2. Formalizing them mathematically
3. Designing provably-correct solutions
4. Implementing and validating them

Every entry here should be publication-quality.
