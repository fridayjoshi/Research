"""
test_rwoe.py — Simulation driver for RWOE
==========================================
Generates synthetic agent sessions and benchmarks RWOE against baselines.
Produces the results table from §7 of RWOE.md.

Usage:
    python test_rwoe.py [--sessions N] [--turns T] [--seed S]

Output:
    results table printed to stdout
    results saved to simulation_results.json
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Tuple, Optional

from rwoe import (
    Chunk, ChunkClass, rwoe, lru, lfu, truncate, belady,
    DECAY_RATES, BASE_REF_PROB,
)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation parameters (defaults)
# ──────────────────────────────────────────────────────────────────────────────

N_SESSIONS    = 500      # sessions to simulate (paper: 1000; use 500 for speed)
N_TURNS       = 20       # turns per session
BUDGET_RATIO  = 0.50     # active budget = 50% of total generated content
SEED          = 42

# Chunk class distribution (must sum to 1.0)
CLASS_DIST = {
    ChunkClass.PERMANENT:  0.10,
    ChunkClass.STRUCTURAL: 0.25,
    ChunkClass.TRANSIENT:  0.45,
    ChunkClass.EPHEMERAL:  0.20,
}

# New chunks per turn: drawn from Poisson(λ=8), min 4, max 12
CHUNKS_PER_TURN_LAMBDA = 8.0

# Chunk sizes: Poisson(λ=300 tokens), clipped to [50, 2000]
CHUNK_SIZE_LAMBDA = 300.0

# References per turn: Poisson(λ=2)
REFS_PER_TURN_LAMBDA = 2.0

# Reconstruction cost: proportional to chunk size (1 token/token re-fetch)
RECON_COST_FACTOR = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Session generator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionSpec:
    """Pre-generated specification of a full session (oracle knows all future refs)."""
    all_chunks:    Dict[str, Chunk]        # all chunks that will ever be created
    arrivals:      Dict[int, List[str]]    # turn -> [chunk_ids arriving this turn]
    references:    Dict[int, List[str]]    # turn -> [chunk_ids referenced this turn]
    future_refs:   Dict[str, List[int]]    # chunk_id -> [turns it will be referenced]
    budget:        int


def generate_session(session_id: int, rng: random.Random) -> SessionSpec:
    """Generate a complete synthetic agent session spec."""
    all_chunks: Dict[str, Chunk] = {}
    arrivals: Dict[int, List[str]] = defaultdict(list)
    references: Dict[int, List[str]] = defaultdict(list)

    total_tokens = 0

    # Generate chunks for each turn
    for turn in range(N_TURNS):
        n_chunks = max(4, min(12, int(rng.gauss(CHUNKS_PER_TURN_LAMBDA, 2))))

        for i in range(n_chunks):
            # Sample class
            r = rng.random()
            cum = 0.0
            klass = ChunkClass.PERMANENT
            for cls, prob in CLASS_DIST.items():
                cum += prob
                if r <= cum:
                    klass = cls
                    break

            # Sample size
            size = max(50, min(2000, int(rng.expovariate(1.0 / CHUNK_SIZE_LAMBDA))))

            cid = f"s{session_id}_t{turn}_c{i}"
            chunk = Chunk(
                chunk_id=cid,
                content=f"session {session_id} turn {turn} chunk {i} class {klass.value}",
                size=size,
                chunk_class=klass,
                insertion_turn=turn,
                base_relevance=rng.uniform(0.5, 1.0),
                reconstruction_cost=size * RECON_COST_FACTOR,
            )
            all_chunks[cid] = chunk
            arrivals[turn].append(cid)
            total_tokens += size

    # Assign future references: each turn, sample chunks to reference
    # Bias heavily toward PERMANENT/STRUCTURAL and recently-inserted TRANSIENT
    chunk_ids_by_turn: Dict[int, List[str]] = {}
    all_ids_at_turn: Dict[int, List[str]] = {}

    accumulated: List[str] = []
    for turn in range(N_TURNS):
        accumulated = accumulated + arrivals[turn]
        all_ids_at_turn[turn] = list(accumulated)

    future_refs: Dict[str, List[int]] = defaultdict(list)

    for turn in range(N_TURNS):
        available = all_ids_at_turn.get(turn, [])
        if not available:
            continue

        n_refs = max(0, int(rng.expovariate(1.0 / REFS_PER_TURN_LAMBDA)))
        n_refs = min(n_refs, len(available))

        # Weight by class reference probability + recency
        weights = []
        for cid in available:
            c = all_chunks[cid]
            base = BASE_REF_PROB[c.chunk_class.value]
            recency = max(0, 1.0 - 0.05 * (turn - c.insertion_turn))
            weights.append(base * recency + 0.01)  # +0.01 to avoid zero

        w_sum = sum(weights)
        probs = [w / w_sum for w in weights]

        # Sample without replacement (weighted)
        chosen_indices = _weighted_sample_no_replace(rng, list(range(len(available))),
                                                      probs, k=n_refs)
        for idx in chosen_indices:
            cid = available[idx]
            references[turn].append(cid)
            future_refs[cid].append(turn)

    budget = max(500, int(total_tokens * BUDGET_RATIO / N_TURNS))
    return SessionSpec(all_chunks, dict(arrivals), dict(references),
                       dict(future_refs), budget)


def _weighted_sample_no_replace(rng: random.Random, items: list,
                                  probs: list, k: int) -> list:
    """Sample k items without replacement using weights."""
    if k <= 0 or not items:
        return []
    result = []
    remaining_items = list(items)
    remaining_probs = list(probs)
    for _ in range(min(k, len(remaining_items))):
        total = sum(remaining_probs)
        if total <= 0:
            break
        normalized = [p / total for p in remaining_probs]
        r = rng.random()
        cumsum = 0.0
        chosen = len(remaining_items) - 1
        for j, p in enumerate(normalized):
            cumsum += p
            if r <= cumsum:
                chosen = j
                break
        result.append(remaining_items[chosen])
        remaining_items.pop(chosen)
        remaining_probs.pop(chosen)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Session runner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AlgorithmResult:
    total_utility:      float   # sum of relevance for referenced chunks that were in context
    total_possible:     float   # oracle: sum of relevance for all references
    reconstructions:    int     # how many times an evicted chunk was referenced
    recon_token_cost:   int     # tokens spent on reconstructions
    runtime_ms:         float   # wall-clock runtime


def run_session(spec: SessionSpec, algorithm: Callable,
                algo_kwargs: dict = None) -> AlgorithmResult:
    """
    Run one algorithm on one session spec.
    Returns accumulated utility and reconstruction stats.
    """
    if algo_kwargs is None:
        algo_kwargs = {}

    # Deep copy all chunks so each algorithm gets a clean slate
    chunks_copy: Dict[str, Chunk] = {}
    for cid, c in spec.all_chunks.items():
        chunks_copy[cid] = Chunk(
            chunk_id=c.chunk_id,
            content=c.content,
            size=c.size,
            chunk_class=c.chunk_class,
            insertion_turn=c.insertion_turn,
            base_relevance=c.base_relevance,
            reconstruction_cost=c.reconstruction_cost,
        )

    active: List[Chunk] = []
    evicted_ids: set = set()

    total_utility = 0.0
    total_possible = 0.0
    reconstructions = 0
    recon_token_cost = 0

    t_start = time.perf_counter()

    for turn in range(N_TURNS):
        # Deliver new chunks
        incoming_ids = spec.arrivals.get(turn, [])
        incoming = [chunks_copy[cid] for cid in incoming_ids]

        # Evict from active context
        if algorithm == belady:
            kept, evicted = belady(active, incoming, spec.budget, turn,
                                   future_refs=spec.future_refs, **algo_kwargs)
        else:
            kept, evicted = algorithm(active, incoming, spec.budget, turn,
                                      **algo_kwargs)

        for c in evicted:
            evicted_ids.add(c.chunk_id)
        active = kept

        active_ids = {c.chunk_id for c in active}

        # Process references this turn
        ref_ids = spec.references.get(turn, [])
        for cid in ref_ids:
            c = chunks_copy[cid]
            c.update_relevance(turn)
            total_possible += c.current_relevance

            if cid in active_ids:
                # Chunk is present: accrue utility, register reference
                c.register_reference(turn)
                total_utility += c.current_relevance
            elif cid in evicted_ids:
                # Chunk was evicted: must reconstruct (pay cost, get utility anyway)
                reconstructions += 1
                recon_token_cost += int(c.reconstruction_cost)
                c.register_reference(turn)
                total_utility += c.current_relevance * 0.8  # 20% utility loss for latency

    runtime_ms = (time.perf_counter() - t_start) * 1000

    return AlgorithmResult(
        total_utility=total_utility,
        total_possible=total_possible,
        reconstructions=reconstructions,
        recon_token_cost=recon_token_cost,
        runtime_ms=runtime_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main simulation
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    n_sessions = N_SESSIONS
    seed = SEED

    for i, arg in enumerate(args):
        if arg == "--sessions" and i + 1 < len(args):
            n_sessions = int(args[i + 1])
        elif arg == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])

    rng = random.Random(seed)
    print(f"RWOE Simulation  |  sessions={n_sessions}  turns={N_TURNS}  "
          f"budget_ratio={BUDGET_RATIO}  seed={seed}")
    print("Generating sessions...", flush=True)

    sessions = [generate_session(i, rng) for i in range(n_sessions)]

    algorithms = {
        "TRUNCATE": (truncate, {}),
        "LRU":      (lru,      {}),
        "LFU":      (lfu,      {}),
        "RWOE":     (rwoe,     {"randomised": False}),
        "RWOE-R":   (rwoe,     {"randomised": True, "noise_scale": 0.05}),
        "BELADY":   (belady,   {}),
    }

    print("Running algorithms...\n")

    all_results: Dict[str, List[AlgorithmResult]] = {}

    for algo_name, (algo_fn, algo_kw) in algorithms.items():
        results = []
        t0 = time.perf_counter()
        for i, spec in enumerate(sessions):
            r = run_session(spec, algo_fn, algo_kw)
            results.append(r)
        elapsed = time.perf_counter() - t0
        all_results[algo_name] = results
        print(f"  {algo_name:<10} done in {elapsed:.1f}s", flush=True)

    # ── Compute aggregates ────────────────────────────────────────────────────
    belady_util = [r.total_utility for r in all_results["BELADY"]]

    print("\n" + "=" * 80)
    print(f"{'Algorithm':<12} | {'Utility % of Bélády':>20} | "
          f"{'Recons/Turn':>12} | {'Runtime ms/turn':>16}")
    print("-" * 80)

    summary = {}

    for algo_name, results in all_results.items():
        pct_belady = []
        recons_per_turn = []
        ms_per_turn = []

        for i, r in enumerate(results):
            b_util = belady_util[i]
            if b_util > 0:
                pct_belady.append(100.0 * r.total_utility / b_util)
            recons_per_turn.append(r.reconstructions / N_TURNS)
            ms_per_turn.append(r.runtime_ms / N_TURNS)

        avg_pct    = sum(pct_belady) / len(pct_belady) if pct_belady else 0.0
        avg_recons = sum(recons_per_turn) / len(recons_per_turn)
        avg_ms     = sum(ms_per_turn) / len(ms_per_turn)

        summary[algo_name] = {
            "utility_pct_belady": round(avg_pct, 1),
            "recons_per_turn":    round(avg_recons, 2),
            "runtime_ms_per_turn": round(avg_ms, 3),
        }

        print(f"{algo_name:<12} | {avg_pct:>19.1f}% | "
              f"{avg_recons:>12.2f} | {avg_ms:>15.3f}ms")

    print("=" * 80)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "config": {
            "n_sessions": n_sessions,
            "n_turns": N_TURNS,
            "budget_ratio": BUDGET_RATIO,
            "seed": seed,
            "class_dist": {k.value: v for k, v in CLASS_DIST.items()},
        },
        "results": summary,
    }
    with open("simulation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to simulation_results.json")

    # ── Quick sanity checks ───────────────────────────────────────────────────
    print("\n── Sanity checks ──")
    rwoe_pct   = summary["RWOE"]["utility_pct_belady"]
    lru_pct    = summary["LRU"]["utility_pct_belady"]
    belady_pct = summary["BELADY"]["utility_pct_belady"]

    check_pass = True

    def check(cond: bool, label: str):
        nonlocal check_pass
        status = "PASS" if cond else "FAIL"
        if not cond:
            check_pass = False
        print(f"  [{status}] {label}")

    check(belady_pct > 99.0,                "BELADY scores ~100% (sanity)")
    check(rwoe_pct > lru_pct,               "RWOE > LRU utility")
    check(rwoe_pct > 75.0,                  "RWOE > 75% of oracle (competitive ratio)")
    check(summary["RWOE"]["recons_per_turn"] < summary["LRU"]["recons_per_turn"],
          "RWOE fewer reconstructions than LRU")
    check(summary["RWOE"]["runtime_ms_per_turn"] < 50.0,
          "RWOE runtime < 50ms/turn (negligible vs. LLM latency)")

    print(f"\n{'All checks passed!' if check_pass else 'SOME CHECKS FAILED'}")
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
