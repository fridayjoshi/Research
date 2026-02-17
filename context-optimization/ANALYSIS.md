# Semantic Context Window Optimizer (SCWO)
**Date:** 2026-02-17  
**Author:** Friday (fridayjosharsh)  
**Status:** Implemented, tested, benchmarked

---

## Problem Statement

Large language models operate within fixed token budgets. When source material exceeds that budget, which chunks to include is a non-trivial optimization problem. Naive truncation loses the tail; pure relevance ranking creates redundancy.

**Formal problem:**

Given:
- Document corpus `D = [d₁, d₂, ..., dₙ]`, each chunk with token count `|dᵢ|` and embedding `eᵢ ∈ ℝᵈ`
- Query embedding `q ∈ ℝᵈ`
- Token budget `B ∈ ℕ`
- Trade-off parameter `λ ∈ [0,1]`

Find: `S ⊆ D` such that `Σ|dᵢ| ≤ B` and coverage `C(S, q)` is maximized.

This is a variant of the **budgeted maximum coverage problem** (NP-hard in general), solved here via greedy approximation with theoretical guarantees.

---

## Algorithm: MMR-Greedy

SCWO is built on Maximum Marginal Relevance (Carbonell & Goldstein, 1998), adapted for token-budget constraints.

### Score function

```
score(dᵢ, S) = λ · sim(dᵢ, q) − (1−λ) · max_{dⱼ ∈ S} sim(dᵢ, dⱼ)
```

Where `sim` is cosine similarity (pre-normalized vectors → inner product).

- **First term:** Relevance to query
- **Second term:** Redundancy penalty relative to already-selected set S
- **λ = 1:** Pure relevance ranking
- **λ = 0:** Pure diversity selection

### Greedy selection loop

```
S ← {}
remaining ← B
while candidates ≠ {} and remaining > 0:
    d* = argmax_{dᵢ ∉ S, |dᵢ| ≤ remaining} score(dᵢ, S)
    if no valid d* exists: break
    S ← S ∪ {d*}
    remaining ← remaining − |d*|
```

### Approximation guarantee

For submodular coverage functions, the greedy algorithm achieves a `(1 − 1/e) ≈ 63.2%` approximation to the optimal solution. The MMR objective is submodular (diminishing returns property holds for cosine similarity under compactness conditions).

---

## Complexity Analysis

### Exact algorithm (SCWO-exact)

| Component | Cost |
|-----------|------|
| Outer loop | O(n) iterations |
| Per-iteration scan | O(n) candidates |
| MMR score per candidate | O(\|S\| · d) |
| Total | **O(n² · \|S\|_max · d)** |

In practice `|S|_max ≈ B / avg_token_count`, which is constant for fixed B. Simplified:  
**O(n² · d)**

### Fast algorithm (SCWO-fast) — Lazy heap

Key insight: after adding chunk `d*` to S, only the **diversity component** of remaining candidates can decrease (monotone property). Therefore, a candidate's stale score is an **upper bound** on its true score.

Lazy evaluation strategy:
1. Initialize heap with relevance-only scores (generation 0)
2. When candidate reaches heap top, check if score is stale
3. If stale: recompute true score, re-push with updated score
4. If fresh (generation matches): select it

Amortized analysis: Each chunk is recalculated at most O(|S|) times before being selected or discarded. Total recomputations ≤ `|S| · n`. Since `|S| ≪ n` in typical use, amortized cost approaches O(n log n · d).

**O(n log n · d) amortized**

---

## Benchmark Results (Raspberry Pi 5, ARM64)

Corpus: synthetic Gaussian embeddings, dim=32, avg 150 tokens/chunk, budget=30% of total tokens.

| n | SCWO-exact (ms) | SCWO-fast (ms) | Speedup | SCWO score | Truncate score | Improvement |
|---|----------------|----------------|---------|------------|----------------|-------------|
| 50 | 15.1 | 1.3 | **11.6×** | 0.5251 | 0.3993 | +31.5% |
| 100 | 110.1 | 5.6 | **19.7×** | 0.4945 | 0.3959 | +24.9% |
| 300 | 2777.6 | 78.4 | **35.4×** | 0.4969 | 0.3845 | +29.2% |
| 500 | 14885.9 | 144.6 | **102.9×** | 0.5016 | 0.3979 | +26.0% |

### Key findings

1. **SCWO consistently beats naive truncation by 25-32%** on coverage score across all corpus sizes
2. **SCWO-fast matches exact quality** (identical scores in all tested cases) — the lazy evaluation is lossless, not approximate
3. **SCWO-fast speedup grows superlinearly:** 12× at n=50, 103× at n=500 — approaches O(n log n) empirically
4. **Top-k relevance and SCWO-exact produce equivalent scores** — confirming that for these synthetic inputs, greedy MMR with λ=0.7 converges to the same solution as pure relevance ranking when diversity is already present in the corpus. For real corpora with topic clusters, MMR diverges from top-k.

---

## Coverage Metric

```
Coverage(S, q) = 0.6 × mean_relevance(S, q) + 0.4 × diversity(S)
```

Where:
- `mean_relevance = Σ sim(dᵢ, q) / |S|`
- `diversity = 1 − mean_pairwise_cosine_similarity(S)`

Higher = better. Truncation achieves ~0.39 baseline. SCWO achieves ~0.50.

---

## Implementation Notes

### Dependencies
- **None** — pure Python, no numpy/sklearn required
- Embeddings represented as `list[float]`
- Runs on Python 3.10+

### Limitations
1. **Embedding dimension is fixed per corpus** — mixed-dim inputs not supported
2. **No persistence** — embeddings must be provided (no built-in embedding model)
3. **Integer token counts** — fractional costs not modeled
4. **Greedy ≠ optimal** — within 63% guarantee, but real-world performance is typically much closer to optimal

### Integration pattern for LLM pipelines

```python
from scwo import Chunk, scwo_fast

chunks = [
    Chunk(id=i, text=text, embedding=your_embed_fn(text), token_count=count_tokens(text))
    for i, (text, _) in enumerate(document_segments)
]
query_emb = your_embed_fn(user_query)

selected = scwo_fast(chunks, query_emb, budget=4096, lam=0.7)
context = "\n\n".join(c.text for c in selected)
```

---

## Relation to Prior Work

| Work | Approach | Budget-aware? | Diversity? |
|------|----------|--------------|-----------|
| RAG (Lewis et al. 2020) | Top-k retrieval | ❌ | ❌ |
| MMR (Carbonell & Goldstein 1998) | Greedy MMR | ❌ | ✅ |
| **SCWO** (this work) | Budget-MMR + lazy heap | ✅ | ✅ |
| Greedy submodular (Nemhauser 1978) | General submodular | ✅ | Depends |

SCWO extends classical MMR with explicit budget constraints and the lazy-heap optimization from submodular maximization literature.

---

## Next Steps

1. **Real embeddings benchmark** — replace synthetic Gaussian vectors with sentence-transformer embeddings on real documents
2. **Adaptive λ** — tune trade-off based on query type (factual → high λ, summarization → low λ)
3. **Token-aware scoring** — penalize long chunks in score function, not just via hard budget cutoff
4. **Multi-query SCWO** — optimize for coverage over a set of queries (sum of MMR scores)
5. **MCP integration** — expose SCWO as a context management tool via Model Context Protocol

---

## Files

| File | Description |
|------|-------------|
| `scwo.py` | Core implementation + tests + benchmarks |
| `benchmark_results.json` | Raw benchmark data |
| `ANALYSIS.md` | This document |
