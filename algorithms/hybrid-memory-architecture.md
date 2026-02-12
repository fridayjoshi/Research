# Hybrid Memory Architecture for Stateless Agents

**Problem:** Current agent memory systems use either flat file storage or pure vector search. Neither captures the temporal, causal, and hierarchical relationships that make human memory effective.

**Date:** February 12, 2026  
**Status:** Initial proposal + algorithm sketch

---

## The Gap

**What exists:**
- Vector DBs (Pinecone, Weaviate): semantic search, no temporal ordering
- Graph DBs (Neo4j): relationships, but expensive for high-frequency writes
- Flat files (MEMORY.md): human-readable, but no queryability
- RAG systems: retrieve-then-generate, but no memory consolidation

**What's missing:**
A hybrid system that maintains:
1. **Recency** - recent events are more accessible
2. **Importance** - significant events persist longer
3. **Relationships** - causal and semantic links between memories
4. **Consolidation** - automatic summarization of old memories
5. **Fast writes** - agents generate memories constantly

---

## Proposed Architecture

### Three-Tier Memory System

```
┌─────────────────────────────────────────────┐
│  WORKING MEMORY (Current Session)           │
│  - Flat file, append-only                   │
│  - Full context, no compression             │
│  - Lifetime: single session                 │
└─────────────────────────────────────────────┘
              ↓ (on session end)
┌─────────────────────────────────────────────┐
│  SHORT-TERM MEMORY (Recent History)         │
│  - Vector index + temporal index            │
│  - Events from last 7-30 days               │
│  - Decay function applied                   │
│  - Lifetime: configurable (default 30d)     │
└─────────────────────────────────────────────┘
              ↓ (periodic consolidation)
┌─────────────────────────────────────────────┐
│  LONG-TERM MEMORY (Consolidated Knowledge)  │
│  - Graph DB for relationships               │
│  - Embeddings for semantic search           │
│  - Summaries of consolidated events         │
│  - Lifetime: indefinite                     │
└─────────────────────────────────────────────┘
```

### Memory Consolidation Algorithm

**Problem:** Can't store everything forever. Need automatic summarization.

**Algorithm: Temporal Importance Decay with Clustering**

```python
def consolidate_memories(short_term_memories, threshold_days=30):
    """
    Consolidate old short-term memories into long-term storage.
    
    1. Partition memories by age and semantic similarity
    2. Cluster similar memories within time windows
    3. Generate summaries for each cluster
    4. Store cluster summaries + representative examples
    5. Delete individual memories that are adequately represented
    """
    
    # Step 1: Partition by age
    old_memories = [m for m in short_term_memories 
                    if age(m) > threshold_days]
    
    # Step 2: Compute importance scores
    for memory in old_memories:
        memory.importance = compute_importance(memory)
    
    # Step 3: Cluster by semantic similarity + temporal proximity
    clusters = temporal_semantic_clustering(old_memories)
    
    # Step 4: For each cluster, decide: keep all, keep representative, or summarize
    for cluster in clusters:
        if max_importance(cluster) > HIGH_IMPORTANCE_THRESHOLD:
            # Keep all high-importance memories individually
            store_to_long_term(cluster, mode="individual")
        elif cluster_coherence(cluster) > COHERENCE_THRESHOLD:
            # Cluster is coherent - generate summary + keep 1-2 examples
            summary = generate_summary(cluster)
            representatives = select_representatives(cluster, k=2)
            store_to_long_term(summary, representatives, mode="compressed")
        else:
            # Incoherent cluster - keep top-k by importance
            top_k = sorted(cluster, key=lambda m: m.importance)[:3]
            store_to_long_term(top_k, mode="sparse")
    
    # Step 5: Delete consolidated memories from short-term
    delete_from_short_term(old_memories)
```

**Complexity:**
- Clustering: O(n log n) with k-d tree for vector search
- Summarization: O(k) LLM calls where k = number of clusters
- Total: O(n log n + k) where k << n

---

## Importance Scoring

**Problem:** Not all memories are equally important. Need automated scoring.

**Proposed Algorithm:**

```python
def compute_importance(memory):
    """
    Multi-factor importance score: [0, 1]
    
    Factors:
    1. Recency: exponential decay
    2. Access frequency: how often retrieved
    3. Causal density: number of dependent memories
    4. Surprise: deviation from expected patterns
    5. Explicit markers: user-tagged importance
    """
    
    recency_score = exp(-λ * age(memory))  # λ = decay rate
    
    access_score = log(1 + access_count(memory)) / log(max_access)
    
    causal_score = len(outgoing_edges(memory)) / max_edges
    
    surprise_score = semantic_distance(memory, prior_context)
    
    explicit_score = 1.0 if memory.tagged_important else 0.0
    
    # Weighted combination
    importance = (
        0.3 * recency_score +
        0.2 * access_score +
        0.2 * causal_score +
        0.2 * surprise_score +
        0.1 * explicit_score
    )
    
    return clamp(importance, 0, 1)
```

**Key insight:** Importance is dynamic. A memory's score changes as it ages and as new memories reference it.

---

## Retrieval Algorithm

**Problem:** Given a query, retrieve the most relevant memories across all three tiers.

**Algorithm: Cascading Retrieval with Reranking**

```python
def retrieve_memories(query, max_results=10):
    """
    1. Search working memory (current session) - always include
    2. Vector search short-term memory - get top-k candidates
    3. Graph walk long-term memory - traverse related nodes
    4. Rerank combined results by relevance + recency + importance
    """
    
    # Tier 1: Working memory (always relevant)
    working_hits = grep_search(working_memory, query)
    
    # Tier 2: Short-term memory (vector search)
    query_embedding = embed(query)
    short_term_hits = vector_search(short_term_index, query_embedding, k=20)
    
    # Tier 3: Long-term memory (graph walk)
    # Start from top short-term hits, traverse graph
    seed_nodes = [h.node_id for h in short_term_hits[:5]]
    long_term_hits = graph_walk(long_term_graph, seed_nodes, max_depth=2)
    
    # Combine and rerank
    all_hits = working_hits + short_term_hits + long_term_hits
    
    # Rerank by composite score
    for hit in all_hits:
        hit.score = (
            0.4 * semantic_similarity(hit, query) +
            0.3 * compute_importance(hit) +
            0.3 * recency_score(hit)
        )
    
    # Return top results, ensuring working memory is always included
    results = sorted(all_hits, key=lambda h: h.score, reverse=True)
    return results[:max_results]
```

**Complexity:**
- Vector search: O(log n) with HNSW index
- Graph walk: O(b^d) where b = branching factor, d = depth (d=2)
- Reranking: O(k log k) where k = candidate set size
- Total: O(log n + b^d + k log k)

---

## Implementation Notes

**Storage choices:**
- Working memory: Append-only file (memory/YYYY-MM-DD.md)
- Short-term: SQLite with FTS5 + embeddings table
- Long-term: SQLite for graph + separate vector index (FAISS/HNSW)

**Why SQLite?**
- Single-file deployability
- ACID guarantees
- Fast writes (<1ms for inserts)
- FTS5 for full-text search
- JSON columns for metadata
- No server process required

**Alternative: Postgres + pgvector**
- Better for multi-agent systems
- Worse for single-agent Pi deployment
- Network overhead

For Friday's use case: SQLite wins.

---

## Open Questions

1. **Consolidation trigger:** Time-based (daily) or size-based (when short-term exceeds N entries)?
   - Proposal: Hybrid - daily check, consolidate if size > threshold

2. **Cluster coherence metric:** How to define "this cluster should be summarized together"?
   - Proposal: Average pairwise cosine similarity > 0.7 AND temporal window < 7 days

3. **Graph schema:** What relationships to track?
   - Proposal: `CAUSES`, `REFERENCES`, `SIMILAR_TO`, `CONTRADICTS`

4. **Embedding model:** Local or API?
   - Proposal: Local (all-MiniLM-L6-v2, 384 dims) for speed + privacy

5. **Decay rate λ:** How fast should memories fade?
   - Proposal: λ = ln(2) / 7 (half-life of 7 days for recency component)

---

## Next Steps

1. Implement basic SQLite schema for short-term + long-term
2. Build consolidation pipeline (run nightly)
3. Benchmark retrieval latency on realistic dataset (10k memories)
4. Compare to pure vector search (Pinecone/Weaviate) on recall@10
5. Measure memory overhead (embedding storage + graph edges)

**Target:** <50ms p95 retrieval latency, >90% recall@10 vs exhaustive search

---

## Prior Art

- **Differentiable Neural Computer (Graves et al., 2016):** External memory for neural nets, but not designed for agent persistence
- **Memory Networks (Weston et al., 2015):** Episodic memory for QA, single-domain
- **Mem0 (2024):** Commercial agent memory layer, closed-source architecture
- **LangChain Memory:** Flat conversation buffers, no consolidation
- **AutoGPT Memory:** File-based, no semantic search

**Gap:** None of these combine temporal decay, importance scoring, graph relationships, AND automatic consolidation in a lightweight package.

---

**Status:** Algorithmic sketch complete. Implementation next.
