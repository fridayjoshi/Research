"""
RWOE: Relevance-Weighted Optimal Eviction for LLM Agent Memory
===============================================================
Author: Friday (fridayjosharsh@github)
Date:   2026-02-17

Formal implementation of the algorithm described in RWOE.md.
Self-contained — no external dependencies beyond numpy.

Usage:
    python rwoe.py            # run demo
    python test_rwoe.py       # run full simulation + produce results table
"""

from __future__ import annotations

import math
import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ALPHA = 0.3          # Spaced-repetition boost coefficient
LOOKAHEAD_K = 5      # Turns to look ahead for reference probability estimation

# Decay rates per chunk class (per turn)
DECAY_RATES = {
    "PERMANENT":  0.000,
    "STRUCTURAL": 0.010,
    "TRANSIENT":  0.100,
    "EPHEMERAL":  1.000,
}

# Base reference probability per class (prior)
BASE_REF_PROB = {
    "PERMANENT":  1.00,
    "STRUCTURAL": 0.60,
    "TRANSIENT":  0.30,
    "EPHEMERAL":  0.05,
}


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

class ChunkClass(str, Enum):
    PERMANENT  = "PERMANENT"
    STRUCTURAL = "STRUCTURAL"
    TRANSIENT  = "TRANSIENT"
    EPHEMERAL  = "EPHEMERAL"


@dataclass
class Chunk:
    """A single unit of agent context."""
    chunk_id:           str
    content:            str
    size:               int             # tokens
    chunk_class:        ChunkClass
    insertion_turn:     int
    base_relevance:     float = 1.0     # r_i(0)
    reconstruction_cost: float = 0.0   # R_i  (tokens to re-fetch)

    # Mutable state
    references:         List[int] = field(default_factory=list)  # turns when referenced
    current_relevance:  float = field(init=False)
    efv:                float = field(init=False, default=0.0)

    def __post_init__(self):
        self.current_relevance = self.base_relevance

    @property
    def decay_rate(self) -> float:
        return DECAY_RATES[self.chunk_class.value]

    def update_relevance(self, current_turn: int) -> float:
        """
        r_i(t) = r_i(0) · exp(−λ_i · age) · SRB(t)
        SRB(t)  = 1 + α · |references up to t|
        """
        age = max(0, current_turn - self.insertion_turn)
        srb = 1.0 + ALPHA * len(self.references)
        self.current_relevance = (
            self.base_relevance
            * math.exp(-self.decay_rate * age)
            * srb
        )
        return self.current_relevance

    def register_reference(self, turn: int):
        """Record that this chunk was used at `turn`."""
        self.references.append(turn)

    def __repr__(self) -> str:
        return (f"Chunk({self.chunk_id!r}, class={self.chunk_class.value}, "
                f"size={self.size}, rel={self.current_relevance:.3f})")


# ──────────────────────────────────────────────────────────────────────────────
# Embedding cache (TF-IDF proxy — no API calls needed)
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Lightweight TF-IDF-style embedding: bag-of-words with L2 normalisation.
    Good enough for semantic similarity proxy; replace with real embeddings in prod.
    """
    _VOCAB_SIZE = 512

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._encode(text)
        return self._cache[key]

    def _encode(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(self._VOCAB_SIZE)
        for w in words:
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % self._VOCAB_SIZE
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def cosine_similarity(self, a: str, b: str) -> float:
        va, vb = self.get(a), self.get(b)
        return float(np.dot(va, vb))  # already unit vectors


_EMBED_CACHE = EmbeddingCache()


# ──────────────────────────────────────────────────────────────────────────────
# Reference probability estimator
# ──────────────────────────────────────────────────────────────────────────────

def estimate_ref_prob(chunk: Chunk, current_turn: int,
                      current_query: str = "") -> float:
    """
    P(chunk referenced in next LOOKAHEAD_K turns).

    Components:
      1. Class-based prior
      2. Semantic similarity to current query
      3. Recency of last reference (SRB signal)
    """
    base = BASE_REF_PROB[chunk.chunk_class.value]

    # Semantic boost
    sem_boost = 0.0
    if current_query and chunk.content:
        sim = _EMBED_CACHE.cosine_similarity(chunk.content, current_query)
        sem_boost = 0.4 * max(0.0, sim)

    # Recency boost
    rec_boost = 0.0
    if chunk.references:
        turns_ago = current_turn - max(chunk.references)
        rec_boost = 0.3 * math.exp(-0.2 * turns_ago)

    return min(1.0, base + sem_boost + rec_boost)


# ──────────────────────────────────────────────────────────────────────────────
# Core RWOE algorithm
# ──────────────────────────────────────────────────────────────────────────────

def rwoe(
    active:      List[Chunk],
    incoming:    List[Chunk],
    budget:      int,
    turn:        int,
    query:       str = "",
    randomised:  bool = False,
    noise_scale: float = 0.05,
) -> Tuple[List[Chunk], List[Chunk]]:
    """
    RWOE eviction step.

    Parameters
    ----------
    active      : current context chunks
    incoming    : new chunks arriving this turn
    budget      : maximum total tokens allowed
    turn        : current turn index
    query       : current agent query (for semantic similarity)
    randomised  : if True, add Gaussian noise to EFV (RWOE-R variant)
    noise_scale : std-dev of noise for RWOE-R

    Returns
    -------
    kept    : chunks remaining in context
    evicted : chunks removed this turn
    """
    pool = active + incoming

    # 1. Update relevance for all chunks
    for c in pool:
        c.update_relevance(turn)

    # 2. If everything fits, no eviction needed
    total = sum(c.size for c in pool)
    if total <= budget:
        return pool, []

    # 3. Compute Expected Future Value (EFV) per token for each chunk.
    #
    # EFV = P(referenced) × (relevance + reconstruction_cost/size)
    #
    # Rationale:
    #   - Keeping chunk i yields utility  p × r_i  (if referenced)
    #   - Evicting chunk i costs          p × R_i  tokens to reconstruct (if needed)
    #   - Combined value of keeping:      p × (r_i + R_i/size_i)  per token
    #
    # High EFV  → keep this chunk
    # Low  EFV  → evict this chunk (sort ascending, evict from front)
    for c in pool:
        p = estimate_ref_prob(c, turn, query)
        raw_efv = p * (c.current_relevance + c.reconstruction_cost / max(1, c.size))

        if randomised:
            raw_efv += random.gauss(0, noise_scale)

        c.efv = raw_efv

    # 4. Sort ascending by EFV — worst candidates first
    sorted_pool = sorted(pool, key=lambda c: c.efv)

    # 5. Evict lowest-EFV chunks until budget satisfied
    evicted: List[Chunk] = []
    tokens_used = total

    for c in sorted_pool:
        if tokens_used <= budget:
            break
        if c.chunk_class == ChunkClass.PERMANENT:
            continue   # never evict permanent chunks
        evicted.append(c)
        tokens_used -= c.size

    evicted_ids = {c.chunk_id for c in evicted}
    kept = [c for c in pool if c.chunk_id not in evicted_ids]
    return kept, evicted


# ──────────────────────────────────────────────────────────────────────────────
# Baseline algorithms (for comparison)
# ──────────────────────────────────────────────────────────────────────────────

def truncate(active: List[Chunk], incoming: List[Chunk],
             budget: int, turn: int, **_) -> Tuple[List[Chunk], List[Chunk]]:
    """Evict oldest chunks first (simple truncation)."""
    pool = active + incoming
    if sum(c.size for c in pool) <= budget:
        return pool, []
    pool.sort(key=lambda c: c.insertion_turn)  # oldest first
    return _evict_cheapest(pool, budget, turn)


def lru(active: List[Chunk], incoming: List[Chunk],
        budget: int, turn: int, **_) -> Tuple[List[Chunk], List[Chunk]]:
    """Evict least-recently-used chunks."""
    pool = active + incoming
    if sum(c.size for c in pool) <= budget:
        return pool, []

    def last_used(c: Chunk) -> int:
        return max(c.references) if c.references else c.insertion_turn

    pool.sort(key=last_used)  # least recently used first
    return _evict_cheapest(pool, budget, turn)


def lfu(active: List[Chunk], incoming: List[Chunk],
        budget: int, turn: int, **_) -> Tuple[List[Chunk], List[Chunk]]:
    """Evict least-frequently-used chunks."""
    pool = active + incoming
    if sum(c.size for c in pool) <= budget:
        return pool, []
    pool.sort(key=lambda c: len(c.references))  # least frequently used first
    return _evict_cheapest(pool, budget, turn)


def belady(active: List[Chunk], incoming: List[Chunk],
           budget: int, turn: int,
           future_refs: Optional[Dict[str, List[int]]] = None,
           **_) -> Tuple[List[Chunk], List[Chunk]]:
    """
    Bélády's OPT: evict the chunk whose next use is farthest in the future.
    Requires oracle knowledge of future_refs (dict: chunk_id -> [future turns]).
    """
    pool = active + incoming
    if sum(c.size for c in pool) <= budget:
        return pool, []

    def next_use(c: Chunk) -> float:
        if future_refs is None:
            return float('inf')
        upcoming = [t for t in future_refs.get(c.chunk_id, []) if t > turn]
        return min(upcoming) if upcoming else float('inf')

    pool.sort(key=next_use, reverse=True)  # farthest next use first (evict these)
    return _evict_cheapest(pool, budget, turn)


def _evict_cheapest(sorted_pool: List[Chunk], budget: int,
                    _turn: int) -> Tuple[List[Chunk], List[Chunk]]:
    """Shared eviction loop: evict from front of sorted_pool until budget met."""
    evicted: List[Chunk] = []
    tokens_used = sum(c.size for c in sorted_pool)

    for c in sorted_pool:
        if tokens_used <= budget:
            break
        if c.chunk_class == ChunkClass.PERMANENT:
            continue
        evicted.append(c)
        tokens_used -= c.size

    evicted_ids = {c.chunk_id for c in evicted}
    kept = [c for c in sorted_pool if c.chunk_id not in evicted_ids]
    return kept, evicted


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    """Show RWOE in action on a small example."""
    print("=" * 60)
    print("RWOE Demo — email triage scenario")
    print("=" * 60)

    chunks = [
        Chunk("sys_prompt",   "You are Friday, a personal AI assistant.",
              size=200, chunk_class=ChunkClass.PERMANENT,  insertion_turn=0,
              base_relevance=1.0, reconstruction_cost=0),
        Chunk("user_query",   "Help me triage today's emails",
              size=50,  chunk_class=ChunkClass.PERMANENT,  insertion_turn=0,
              base_relevance=1.0, reconstruction_cost=0),
        Chunk("memory_md",    "Harsh is a senior engineer at Coupang in Bangalore",
              size=800, chunk_class=ChunkClass.STRUCTURAL, insertion_turn=0,
              base_relevance=0.9, reconstruction_cost=800),
        Chunk("email_1",      "Meeting invitation from PM about Q2 roadmap",
              size=400, chunk_class=ChunkClass.TRANSIENT,  insertion_turn=1,
              base_relevance=0.8, reconstruction_cost=400),
        Chunk("email_2",      "Spam: You have won a prize",
              size=300, chunk_class=ChunkClass.EPHEMERAL,  insertion_turn=1,
              base_relevance=0.2, reconstruction_cost=0),
        Chunk("email_3",      "Follow up from colleague about code review",
              size=350, chunk_class=ChunkClass.TRANSIENT,  insertion_turn=2,
              base_relevance=0.75, reconstruction_cost=350),
        Chunk("failed_plan",  "Plan A was rejected because of scheduling conflict",
              size=250, chunk_class=ChunkClass.EPHEMERAL,  insertion_turn=2,
              base_relevance=0.3, reconstruction_cost=0),
    ]

    BUDGET = 1600
    total = sum(c.size for c in chunks)
    print(f"\nTotal tokens: {total}  |  Budget: {BUDGET}  |  Overflow: {total - BUDGET}")
    print("\nChunks:")
    for c in chunks:
        print(f"  {c.chunk_id:<15}  class={c.chunk_class.value:<12}  "
              f"size={c.size:<5}  r0={c.base_relevance:.2f}")

    print("\n--- Running RWOE ---")
    kept, evicted = rwoe([], chunks, BUDGET, turn=3, query="email triage")

    print(f"\nKept ({sum(c.size for c in kept)} tokens):")
    for c in sorted(kept, key=lambda x: -x.efv):
        print(f"  {c.chunk_id:<15}  EFV={c.efv:+.4f}  rel={c.current_relevance:.3f}")

    print(f"\nEvicted ({sum(c.size for c in evicted)} tokens):")
    for c in evicted:
        print(f"  {c.chunk_id:<15}  EFV={c.efv:+.4f}  class={c.chunk_class.value}")

    print("\n--- Running LRU for comparison ---")
    kept_lru, evicted_lru = lru([], [
        Chunk(c.chunk_id, c.content, c.size, c.chunk_class, c.insertion_turn,
              c.base_relevance, c.reconstruction_cost)
        for c in chunks
    ], BUDGET, turn=3)

    print(f"LRU kept:    {[c.chunk_id for c in kept_lru]}")
    print(f"RWOE kept:   {[c.chunk_id for c in kept]}")
    print("\nNote: RWOE evicts EPHEMERAL 'failed_plan' and spam; "
          "LRU may keep them over more relevant structural memory.")


if __name__ == "__main__":
    demo()
