#!/usr/bin/env python3
"""
Test suite for memory consolidation system.
Tests correctness, performance, and edge cases.

Author: Friday
Date: February 12, 2026
"""

import os
import time
import numpy as np
from datetime import datetime, timedelta
from memory_consolidation import MemoryConsolidator, Memory


def test_basic_add_retrieve():
    """Test basic memory addition and retrieval"""
    print("\n=== Test: Basic Add & Retrieve ===")
    
    consolidator = MemoryConsolidator("test_basic.db")
    
    # Add memories
    content = "Test memory about Python programming"
    mem_id = consolidator.add_memory(content, tags=["test"], metadata={"source": "test"})
    
    # Retrieve
    results = consolidator.retrieve_memories("Python programming", max_results=5)
    
    assert len(results) > 0, "Should retrieve at least one memory"
    assert content in results[0][0], "Should find exact match"
    
    print(f"✓ Added memory {mem_id}, retrieved {len(results)} results")
    
    consolidator.close()
    os.remove("test_basic.db")
    return True


def test_importance_scoring():
    """Test importance computation"""
    print("\n=== Test: Importance Scoring ===")
    
    consolidator = MemoryConsolidator("test_importance.db")
    
    # Create test memories with different characteristics
    old_memory = Memory(
        id=1,
        content="Old memory",
        timestamp=datetime.now() - timedelta(days=10),
        embedding=np.random.randn(384),
        access_count=0,
        tags=[]
    )
    
    recent_memory = Memory(
        id=2,
        content="Recent memory",
        timestamp=datetime.now(),
        embedding=np.random.randn(384),
        access_count=0,
        tags=[]
    )
    
    important_memory = Memory(
        id=3,
        content="Important memory",
        timestamp=datetime.now() - timedelta(days=10),
        embedding=np.random.randn(384),
        access_count=0,
        tags=["important"]
    )
    
    frequently_accessed = Memory(
        id=4,
        content="Frequently accessed",
        timestamp=datetime.now() - timedelta(days=10),
        embedding=np.random.randn(384),
        access_count=50,
        tags=[]
    )
    
    # Compute scores
    score_old = consolidator.compute_importance(old_memory)
    score_recent = consolidator.compute_importance(recent_memory)
    score_important = consolidator.compute_importance(important_memory)
    score_frequent = consolidator.compute_importance(frequently_accessed)
    
    print(f"Old memory: {score_old:.3f}")
    print(f"Recent memory: {score_recent:.3f}")
    print(f"Important memory: {score_important:.3f}")
    print(f"Frequently accessed: {score_frequent:.3f}")
    
    # Assertions
    assert score_recent > score_old, "Recent should score higher than old"
    assert score_important > score_old, "Tagged important should score higher"
    assert score_frequent > score_old, "Frequently accessed should score higher"
    
    print("✓ Importance scoring works correctly")
    
    consolidator.close()
    os.remove("test_importance.db")
    return True


def test_clustering():
    """Test semantic clustering"""
    print("\n=== Test: Semantic Clustering ===")
    
    consolidator = MemoryConsolidator("test_clustering.db")
    
    # Add semantically similar memories
    similar_memories = [
        "Python is a programming language",
        "I love coding in Python",
        "Python development is fun",
    ]
    
    different_memories = [
        "I ate pizza for dinner",
        "The weather is nice today",
    ]
    
    for content in similar_memories + different_memories:
        consolidator.add_memory(content)
    
    # Get all memories and cluster
    all_memories = consolidator.get_old_memories(threshold_days=0)
    clusters = consolidator.temporal_semantic_clustering(all_memories, eps=0.4, min_samples=2)
    
    print(f"Formed {len(clusters)} clusters from {len(all_memories)} memories")
    
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {len(cluster)} memories")
        for mem in cluster:
            print(f"  - {mem.content[:50]}")
    
    # Should form at least one cluster for similar Python memories
    assert len(clusters) > 0, "Should form at least one cluster"
    
    # Check coherence
    if clusters:
        coherence = consolidator.cluster_coherence(clusters[0])
        print(f"Cluster 0 coherence: {coherence:.3f}")
        assert coherence > 0.0, "Coherence should be positive"
    
    print("✓ Clustering works correctly")
    
    consolidator.close()
    os.remove("test_clustering.db")
    return True


def test_consolidation_pipeline():
    """Test end-to-end consolidation"""
    print("\n=== Test: Consolidation Pipeline ===")
    
    consolidator = MemoryConsolidator("test_consolidation.db")
    
    # Add memories (simulate old timestamps)
    test_memories = [
        "Implemented feature A",
        "Implemented feature B",
        "Implemented feature C",
        "Had lunch with team",
        "Attended meeting",
        "Fixed critical bug",
        "Fixed minor bug",
        "Updated documentation",
    ]
    
    for content in test_memories:
        consolidator.add_memory(content)
    
    # Check initial state
    initial_stats = consolidator.get_stats()
    print(f"Initial state: {initial_stats['short_term_count']} in short-term")
    
    # Run consolidation (threshold = 0 to consolidate immediately)
    consolidation_stats = consolidator.consolidate_memories(threshold_days=0)
    
    print(f"Consolidation stats:")
    print(f"  Memories processed: {consolidation_stats['memories_processed']}")
    print(f"  Clusters formed: {consolidation_stats['clusters_formed']}")
    print(f"  Summaries created: {consolidation_stats['summaries_created']}")
    
    # Check final state
    final_stats = consolidator.get_stats()
    print(f"Final state:")
    print(f"  Short-term: {final_stats['short_term_count']}")
    print(f"  Long-term: {final_stats['long_term_count']}")
    
    assert final_stats['short_term_count'] == 0, "All memories should be consolidated"
    assert final_stats['long_term_count'] > 0, "Should have long-term memories"
    
    print("✓ Consolidation pipeline works correctly")
    
    consolidator.close()
    os.remove("test_consolidation.db")
    return True


def benchmark_retrieval():
    """Benchmark retrieval performance"""
    print("\n=== Benchmark: Retrieval Performance ===")
    
    consolidator = MemoryConsolidator("test_benchmark.db")
    
    # Add 1000 memories
    n_memories = 1000
    print(f"Adding {n_memories} memories...")
    
    start = time.time()
    for i in range(n_memories):
        content = f"Memory {i}: " + " ".join([f"word{j}" for j in range(10)])
        consolidator.add_memory(content)
    add_time = time.time() - start
    
    print(f"✓ Added {n_memories} memories in {add_time:.2f}s ({n_memories/add_time:.0f} memories/s)")
    
    # Benchmark retrieval
    queries = [
        "memory architecture",
        "implementation details",
        "word5 word7",
        "Memory 500",
    ]
    
    retrieval_times = []
    for query in queries:
        start = time.time()
        results = consolidator.retrieve_memories(query, max_results=10)
        elapsed = time.time() - start
        retrieval_times.append(elapsed * 1000)  # Convert to ms
        print(f"Query '{query}': {elapsed*1000:.2f}ms, {len(results)} results")
    
    avg_retrieval = np.mean(retrieval_times)
    p95_retrieval = np.percentile(retrieval_times, 95)
    
    print(f"\nRetrieval performance:")
    print(f"  Average: {avg_retrieval:.2f}ms")
    print(f"  P95: {p95_retrieval:.2f}ms")
    
    # Target: <50ms p95
    if p95_retrieval < 50:
        print(f"✓ Meets target (<50ms p95)")
    else:
        print(f"⚠ Slower than target (got {p95_retrieval:.2f}ms, target <50ms)")
    
    consolidator.close()
    os.remove("test_benchmark.db")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Memory Consolidation System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Add & Retrieve", test_basic_add_retrieve),
        ("Importance Scoring", test_importance_scoring),
        ("Semantic Clustering", test_clustering),
        ("Consolidation Pipeline", test_consolidation_pipeline),
        ("Retrieval Performance", benchmark_retrieval),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {name}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
