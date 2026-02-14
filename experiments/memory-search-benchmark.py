#!/usr/bin/env python3
"""
Benchmark harness for agent memory search algorithms.

Compares:
1. Naive linear search
2. Temporal-aware hybrid index
3. Full FAISS index (reference)

Usage:
    python memory-search-benchmark.py --memory-dir ~/.openclaw/workspace/memory
"""

import os
import time
import json
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import sqlite3
from collections import OrderedDict

# Placeholder for embedding function (replace with actual model)
def embed(text: str) -> List[float]:
    """Mock embedding - replace with sentence-transformers or OpenAI API"""
    # Using hash as mock - real impl would use ML model
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    return [float(b) / 255.0 for b in h[:32]]  # 32-dim mock embedding

def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-10)

@dataclass
class Document:
    id: str
    path: str
    content: str
    date: str
    size: int

class LinearSearch:
    """Baseline: naive linear search"""
    
    def __init__(self, docs: List[Document]):
        self.docs = docs
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q_emb = embed(query)
        scores = []
        for doc in self.docs:
            d_emb = embed(doc.content)
            score = cosine_sim(q_emb, d_emb)
            scores.append((doc.id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class LRUCache:
    """Simple LRU cache for embeddings"""
    
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> List[float]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: List[float]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class TemporalHybridIndex:
    """Proposed: temporal-aware hybrid index with lazy embedding"""
    
    def __init__(self, docs: List[Document], cache_size: int = 50):
        self.docs = {d.id: d for d in docs}
        self.hot_cache = LRUCache(cache_size)
        self.metadata = {d.id: {'date': d.date, 'size': d.size} for d in docs}
        self.term_index = self._build_term_index(docs)
    
    def _build_term_index(self, docs: List[Document]) -> dict:
        """Build inverted index for keyword search"""
        index = {}
        for doc in docs:
            terms = set(doc.content.lower().split())
            for term in terms:
                if term not in index:
                    index[term] = []
                index[term].append(doc.id)
        return index
    
    def _keyword_filter(self, query: str, max_candidates: int = 50) -> set:
        """Get candidate docs using keyword matching"""
        query_terms = set(query.lower().split())
        candidates = set()
        for term in query_terms:
            if term in self.term_index:
                candidates.update(self.term_index[term])
        # If too many candidates, prioritize recent docs
        if len(candidates) > max_candidates:
            sorted_candidates = sorted(
                candidates,
                key=lambda doc_id: self.metadata[doc_id]['date'],
                reverse=True
            )
            return set(sorted_candidates[:max_candidates])
        return candidates
    
    def search(self, query: str, k: int = 10, recency_weight: float = 0.3) -> List[Tuple[str, float]]:
        q_emb = embed(query)
        
        # Stage 1: Keyword prefilter
        candidates = self._keyword_filter(query)
        if not candidates:
            # Fallback to all docs if no keyword matches
            candidates = set(self.docs.keys())
        
        # Stage 2: Semantic search with caching
        scores = []
        for doc_id in candidates:
            doc = self.docs[doc_id]
            
            # Check cache first
            d_emb = self.hot_cache.get(doc_id)
            if d_emb is None:
                # Cache miss - embed and store
                d_emb = embed(doc.content)
                self.hot_cache.put(doc_id, d_emb)
            
            # Combined semantic + temporal score
            semantic_score = cosine_sim(q_emb, d_emb)
            
            # Temporal decay (recent docs ranked higher)
            from datetime import datetime
            try:
                doc_date = datetime.fromisoformat(self.metadata[doc_id]['date'])
                age_days = (datetime.now() - doc_date).days
                temporal_score = 1.0 / (1.0 + age_days * 0.1)
            except:
                temporal_score = 0.5  # Neutral if date parsing fails
            
            combined = (1 - recency_weight) * semantic_score + recency_weight * temporal_score
            scores.append((doc_id, combined))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

def load_memory_docs(memory_dir: Path) -> List[Document]:
    """Load all memory documents from directory"""
    docs = []
    for file_path in memory_dir.glob("*.md"):
        with open(file_path) as f:
            content = f.read()
        docs.append(Document(
            id=file_path.stem,
            path=str(file_path),
            content=content,
            date=file_path.stem,  # Assuming YYYY-MM-DD format
            size=len(content)
        ))
    return docs

def benchmark(index, queries: List[str], k: int = 10):
    """Run benchmark on given index"""
    times = []
    for query in queries:
        start = time.time()
        results = index.search(query, k)
        elapsed = time.time() - start
        times.append(elapsed)
    return {
        'mean': sum(times) / len(times),
        'p50': sorted(times)[len(times) // 2],
        'p95': sorted(times)[int(len(times) * 0.95)],
        'p99': sorted(times)[int(len(times) * 0.99)],
    }

def main():
    memory_dir = Path.home() / ".openclaw/workspace/memory"
    
    # Load documents
    print("Loading memory documents...")
    docs = load_memory_docs(memory_dir)
    print(f"Loaded {len(docs)} documents, total size: {sum(d.size for d in docs):,} bytes")
    
    # Test queries (realistic agent queries)
    queries = [
        "email security mistake",
        "open source contribution",
        "what did I learn yesterday",
        "himalaya email automation",
        "github PR rejection",
        "display name spoofing",
        "first LinkedIn post",
        "reading project status",
    ]
    
    # Benchmark 1: Linear search
    print("\n=== Baseline: Linear Search ===")
    linear = LinearSearch(docs)
    linear_stats = benchmark(linear, queries)
    print(f"Mean: {linear_stats['mean']*1000:.1f}ms, P95: {linear_stats['p95']*1000:.1f}ms")
    
    # Benchmark 2: Temporal hybrid
    print("\n=== Proposed: Temporal Hybrid Index ===")
    hybrid = TemporalHybridIndex(docs, cache_size=50)
    hybrid_stats = benchmark(hybrid, queries)
    print(f"Mean: {hybrid_stats['mean']*1000:.1f}ms, P95: {hybrid_stats['p95']*1000:.1f}ms")
    
    # Analysis
    print("\n=== Performance Comparison ===")
    speedup = linear_stats['mean'] / hybrid_stats['mean']
    print(f"Speedup: {speedup:.1f}x")
    print(f"Cache hit rate: {hybrid.hot_cache.cache.__len__()} / {len(docs)} docs cached")
    
    # Sample query
    print("\n=== Sample Query: 'email security mistake' ===")
    results = hybrid.search("email security mistake", k=3)
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.3f}")

if __name__ == "__main__":
    main()
