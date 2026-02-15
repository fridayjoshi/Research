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
