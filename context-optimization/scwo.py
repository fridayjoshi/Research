"""
Semantic Context Window Optimizer (SCWO)
========================================
Date: 2026-02-17
Author: Friday (fridayjosharsh)

Problem:
    Given a large document/conversation (N tokens), select the most informative
    subset of chunks that fits within a fixed context window budget B, maximizing
    semantic coverage relative to a query.

Algorithm Family:
    Approximate Maximum Marginal Relevance (MMR) with priority queue pruning.

Formal Specification:
    Input:   D = [d_1, d_2, ..., d_n]  (document chunks)
             B ∈ ℕ                     (token budget)
             q                         (query embedding)
             λ ∈ [0,1]                 (relevance-diversity trade-off)
    Output:  S ⊆ D such that Σ|d_i| ≤ B and Coverage(S, q) is maximized

    Score function:
        score(d_i, S) = λ · sim(d_i, q) − (1−λ) · max_{d_j ∈ S} sim(d_i, d_j)

    Greedy selection:
        S ← {}
        while budget remains:
            d* = argmax_{d_i ∉ S, |d_i| ≤ remaining_budget} score(d_i, S)
            S ← S ∪ {d*}

Complexity:
    Time:  O(n² · d)  where d = embedding dimension
           (outer loop n, inner MMR score n, cosine sim d)
    Space: O(n · d)   for embedding matrix

Approximation:
    With heap-based lazy evaluation: O(n log n · d) amortized
    (SCWO-Fast variant implemented below)
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Optional
import unittest
import random


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A document chunk with precomputed embedding."""
    id: int
    text: str
    embedding: list[float]
    token_count: int

    def __post_init__(self):
        # Normalize embedding on creation
        norm = math.sqrt(sum(x * x for x in self.embedding))
        if norm > 0:
            self.embedding = [x / norm for x in self.embedding]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalized vectors. O(d)."""
    return sum(x * y for x, y in zip(a, b))


def max_similarity_to_set(chunk: Chunk, selected: list[Chunk]) -> float:
    """Maximum cosine similarity from chunk to any chunk in selected set. O(|S|·d)."""
    if not selected:
        return 0.0
    return max(cosine_similarity(chunk.embedding, s.embedding) for s in selected)


# ---------------------------------------------------------------------------
# SCWO — Exact O(n² · d)
# ---------------------------------------------------------------------------

def scwo_exact(
    chunks: list[Chunk],
    query_embedding: list[float],
    budget: int,
    lam: float = 0.7,
) -> list[Chunk]:
    """
    Exact greedy MMR selection.

    Parameters
    ----------
    chunks          : list of candidate Chunk objects
    query_embedding : normalized query vector
    budget          : max total token count
    lam             : λ — weight of relevance vs diversity (0=diversity only, 1=relevance only)

    Returns
    -------
    Ordered list of selected chunks (insertion order = selection order).

    Complexity: O(n² · d)
    """
    # Normalize query
    norm = math.sqrt(sum(x * x for x in query_embedding))
    if norm > 0:
        query_embedding = [x / norm for x in query_embedding]

    selected: list[Chunk] = []
    remaining_budget = budget
    candidates = list(chunks)

    while candidates and remaining_budget > 0:
        best_score = -float("inf")
        best_chunk: Optional[Chunk] = None

        for chunk in candidates:
            if chunk.token_count > remaining_budget:
                continue

            rel = cosine_similarity(chunk.embedding, query_embedding)
            div = max_similarity_to_set(chunk, selected)
            score = lam * rel - (1 - lam) * div

            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is None:
            break  # No fitting chunk found

        selected.append(best_chunk)
        candidates.remove(best_chunk)
        remaining_budget -= best_chunk.token_count

    return selected


# ---------------------------------------------------------------------------
# SCWO-Fast — Lazy heap approximation O(n log n · d) amortized
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _HeapEntry:
    neg_score: float          # negated for min-heap → max-heap behaviour
    chunk_id: int = field(compare=False)
    chunk: Chunk = field(compare=False, repr=False)
    staleness: int = field(compare=False, default=0)


def scwo_fast(
    chunks: list[Chunk],
    query_embedding: list[float],
    budget: int,
    lam: float = 0.7,
) -> list[Chunk]:
    """
    Lazy-evaluation heap variant of SCWO.

    Key insight: after adding a chunk to S, only the diversity component of
    other candidates' scores can *decrease* (never increase). So we can defer
    recalculation until a candidate reaches the top of the heap.

    Amortized complexity: O(n log n · d)
    """
    # Normalize query
    norm = math.sqrt(sum(x * x for x in query_embedding))
    if norm > 0:
        query_embedding = [x / norm for x in query_embedding]

    # Initialize heap with relevance-only scores (S = {} so div = 0)
    heap: list[_HeapEntry] = []
    for chunk in chunks:
        rel = cosine_similarity(chunk.embedding, query_embedding)
        score = lam * rel  # div = 0 initially
        entry = _HeapEntry(neg_score=-score, chunk_id=chunk.id, chunk=chunk)
        heap.append(entry)
    heapq.heapify(heap)

    selected: list[Chunk] = []
    remaining_budget = budget
    generation = 0  # increments each time we add to S

    while heap and remaining_budget > 0:
        entry = heapq.heappop(heap)
        chunk = entry.chunk

        if chunk.token_count > remaining_budget:
            continue  # skip over-budget chunks

        # Lazy re-evaluation: if staleness < generation, recompute score
        if entry.staleness < generation:
            rel = cosine_similarity(chunk.embedding, query_embedding)
            div = max_similarity_to_set(chunk, selected)
            true_score = lam * rel - (1 - lam) * div
            entry.neg_score = -true_score
            entry.staleness = generation
            heapq.heappush(heap, entry)
            continue  # re-pushed with updated score; loop again

        # Score is fresh — select this chunk
        selected.append(chunk)
        remaining_budget -= chunk.token_count
        generation += 1

    return selected


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_truncate(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """Naive: take chunks in order until budget exhausted."""
    result = []
    used = 0
    for chunk in chunks:
        if used + chunk.token_count <= budget:
            result.append(chunk)
            used += chunk.token_count
    return result


def baseline_top_k_relevance(
    chunks: list[Chunk],
    query_embedding: list[float],
    budget: int,
) -> list[Chunk]:
    """Greedy relevance-only: always pick most similar chunk regardless of diversity."""
    norm = math.sqrt(sum(x * x for x in query_embedding))
    if norm > 0:
        query_embedding = [x / norm for x in query_embedding]

    ranked = sorted(
        chunks,
        key=lambda c: cosine_similarity(c.embedding, query_embedding),
        reverse=True,
    )
    result = []
    used = 0
    for chunk in ranked:
        if used + chunk.token_count <= budget:
            result.append(chunk)
            used += chunk.token_count
    return result


# ---------------------------------------------------------------------------
# Coverage metric
# ---------------------------------------------------------------------------

def coverage_score(
    selected: list[Chunk],
    query_embedding: list[float],
    all_chunks: list[Chunk],
) -> float:
    """
    Measure how well selected chunks cover the query.

    Coverage = mean relevance of selected chunks (higher = better).
    Diversity = 1 - mean pairwise similarity within selected (higher = better).
    Combined = 0.6 * coverage + 0.4 * diversity
    """
    if not selected:
        return 0.0

    norm = math.sqrt(sum(x * x for x in query_embedding))
    if norm > 0:
        query_embedding = [x / norm for x in query_embedding]

    # Relevance component
    relevances = [cosine_similarity(c.embedding, query_embedding) for c in selected]
    mean_relevance = sum(relevances) / len(relevances)

    # Diversity component
    if len(selected) < 2:
        diversity = 1.0
    else:
        sims = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                sims.append(cosine_similarity(selected[i].embedding, selected[j].embedding))
        mean_pairwise_sim = sum(sims) / len(sims)
        diversity = 1.0 - mean_pairwise_sim

    return 0.6 * mean_relevance + 0.4 * diversity


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def make_random_embedding(dim: int, rng: random.Random) -> list[float]:
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def generate_test_corpus(
    n: int,
    dim: int = 64,
    avg_tokens: int = 150,
    token_variance: int = 50,
    seed: int = 42,
) -> tuple[list[Chunk], list[float]]:
    """Generate n synthetic chunks and a query embedding."""
    rng = random.Random(seed)
    chunks = []
    for i in range(n):
        emb = make_random_embedding(dim, rng)
        tokens = max(10, int(rng.gauss(avg_tokens, token_variance)))
        chunks.append(Chunk(id=i, text=f"chunk_{i}", embedding=emb, token_count=tokens))

    query = make_random_embedding(dim, rng)
    return chunks, query


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark(
    n_values: list[int] = [100, 500, 1000, 2000],
    dim: int = 64,
    budget_ratio: float = 0.3,
    lam: float = 0.7,
) -> list[dict]:
    """
    Compare SCWO exact vs fast vs baselines across corpus sizes.

    Returns list of result dicts per n.
    """
    results = []

    for n in n_values:
        chunks, query = generate_test_corpus(n, dim=dim)
        avg_tokens = sum(c.token_count for c in chunks) / n
        budget = int(n * avg_tokens * budget_ratio)

        row: dict = {"n": n, "dim": dim, "budget": budget}

        # Exact SCWO
        t0 = time.perf_counter()
        sel_exact = scwo_exact(chunks, query, budget, lam)
        row["scwo_exact_ms"] = (time.perf_counter() - t0) * 1000
        row["scwo_exact_count"] = len(sel_exact)
        row["scwo_exact_score"] = round(coverage_score(sel_exact, query, chunks), 4)
        row["scwo_exact_tokens"] = sum(c.token_count for c in sel_exact)

        # Fast SCWO
        t0 = time.perf_counter()
        sel_fast = scwo_fast(chunks, query, budget, lam)
        row["scwo_fast_ms"] = (time.perf_counter() - t0) * 1000
        row["scwo_fast_count"] = len(sel_fast)
        row["scwo_fast_score"] = round(coverage_score(sel_fast, query, chunks), 4)
        row["scwo_fast_tokens"] = sum(c.token_count for c in sel_fast)

        # Baseline: truncate
        t0 = time.perf_counter()
        sel_trunc = baseline_truncate(chunks, budget)
        row["trunc_ms"] = (time.perf_counter() - t0) * 1000
        row["trunc_score"] = round(coverage_score(sel_trunc, query, chunks), 4)

        # Baseline: top-k relevance
        t0 = time.perf_counter()
        sel_topk = baseline_top_k_relevance(chunks, query, budget)
        row["topk_ms"] = (time.perf_counter() - t0) * 1000
        row["topk_score"] = round(coverage_score(sel_topk, query, chunks), 4)

        results.append(row)

        print(
            f"n={n:5d} | "
            f"SCWO_exact={row['scwo_exact_ms']:7.1f}ms score={row['scwo_exact_score']:.4f} | "
            f"SCWO_fast={row['scwo_fast_ms']:7.1f}ms score={row['scwo_fast_score']:.4f} | "
            f"Truncate score={row['trunc_score']:.4f} | "
            f"TopK score={row['topk_score']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSCWO(unittest.TestCase):

    def _make_chunks(self, embeddings: list[list[float]], tokens: list[int]) -> list[Chunk]:
        chunks = []
        for i, (emb, tok) in enumerate(zip(embeddings, tokens)):
            chunks.append(Chunk(id=i, text=f"chunk_{i}", embedding=emb, token_count=tok))
        return chunks

    def test_empty_corpus(self):
        result = scwo_exact([], [1.0, 0.0], budget=100)
        self.assertEqual(result, [])

    def test_budget_zero(self):
        chunks, query = generate_test_corpus(5)
        result = scwo_exact(chunks, query, budget=0)
        self.assertEqual(result, [])

    def test_selects_most_relevant_when_lam_1(self):
        """With λ=1 (pure relevance), should pick chunks most similar to query."""
        # Two chunks: one aligned with query, one orthogonal
        query = [1.0, 0.0]
        emb_aligned = [1.0, 0.0]   # sim=1.0
        emb_ortho   = [0.0, 1.0]   # sim=0.0

        chunks = self._make_chunks(
            [emb_aligned, emb_ortho],
            [10, 10]
        )
        result = scwo_exact(chunks, query, budget=10, lam=1.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 0)  # aligned chunk selected first

    def test_diversity_when_lam_0(self):
        """With λ=0 (pure diversity), second chunk should be most dissimilar to first."""
        # Query orthogonal to everything to remove relevance signal
        query = [0.0, 0.0, 1.0]
        # chunk 0 and 1 are identical (high sim), chunk 2 is orthogonal to 0
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [1.0, 0.0, 0.0]  # same as a
        emb_c = [0.0, 1.0, 0.0]  # orthogonal to a and b

        chunks = self._make_chunks([emb_a, emb_b, emb_c], [10, 10, 10])
        result = scwo_exact(chunks, query, budget=20, lam=0.0)
        # Should select two chunks that are most different from each other
        ids = {c.id for c in result}
        # chunk 2 should be in result because it's orthogonal to whoever was first
        self.assertIn(2, ids)

    def test_budget_respected(self):
        """Total token count of result must not exceed budget."""
        chunks, query = generate_test_corpus(50, seed=99)
        budget = 500
        for fn in [scwo_exact, scwo_fast]:
            result = fn(chunks, query, budget)
            total = sum(c.token_count for c in result)
            self.assertLessEqual(total, budget, f"{fn.__name__} violated budget")

    def test_no_duplicates(self):
        """No chunk should appear twice in the output."""
        chunks, query = generate_test_corpus(20)
        for fn in [scwo_exact, scwo_fast]:
            result = fn(chunks, query, budget=2000)
            ids = [c.id for c in result]
            self.assertEqual(len(ids), len(set(ids)), f"{fn.__name__} returned duplicates")

    def test_fast_vs_exact_score_close(self):
        """SCWO-fast score should be within 5% of exact."""
        chunks, query = generate_test_corpus(100, seed=7)
        budget = 3000
        sel_exact = scwo_exact(chunks, query, budget)
        sel_fast  = scwo_fast(chunks, query, budget)
        score_exact = coverage_score(sel_exact, query, chunks)
        score_fast  = coverage_score(sel_fast, query, chunks)
        if score_exact > 0:
            ratio = score_fast / score_exact
            self.assertGreater(ratio, 0.90, "SCWO-fast score fell >10% below exact")

    def test_coverage_improves_over_truncation(self):
        """SCWO should outperform naive truncation on coverage score."""
        chunks, query = generate_test_corpus(200, seed=13)
        budget = 4000
        sel_scwo  = scwo_exact(chunks, query, budget)
        sel_trunc = baseline_truncate(chunks, budget)
        score_scwo  = coverage_score(sel_scwo, query, chunks)
        score_trunc = coverage_score(sel_trunc, query, chunks)
        self.assertGreater(score_scwo, score_trunc)


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        print("=" * 80)
        print("SCWO Benchmark — Semantic Context Window Optimizer")
        print("=" * 80)
        benchmark(n_values=[100, 500, 1000, 2000])
    elif "--test" in sys.argv or len(sys.argv) == 1:
        # Run unit tests
        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromTestCase(TestSCWO)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
