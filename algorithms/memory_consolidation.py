#!/usr/bin/env python3
"""
Hybrid Memory Consolidation System
Implementation of the three-tier memory architecture proposed in hybrid-memory-architecture.md

Author: Friday
Date: February 12, 2026
"""

import sqlite3
import numpy as np
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import math


@dataclass
class Memory:
    """A single memory entry"""
    id: Optional[int]
    content: str
    timestamp: datetime
    embedding: Optional[np.ndarray]
    access_count: int = 0
    importance: float = 0.5
    tags: List[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class MemoryConsolidator:
    """
    Three-tier memory system with automatic consolidation.
    
    Storage:
    - Working memory: In-memory buffer (current session)
    - Short-term: SQLite table with embeddings
    - Long-term: SQLite with graph relationships
    """
    
    def __init__(self, db_path: str = "memory.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Decay parameters
        self.lambda_decay = math.log(2) / 7  # Half-life of 7 days
        self.consolidation_threshold_days = 30
        self.high_importance_threshold = 0.8
        self.coherence_threshold = 0.7
        
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # Short-term memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                embedding BLOB NOT NULL,
                access_count INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # Long-term memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                summary TEXT,
                timestamp REAL NOT NULL,
                importance REAL,
                is_summary INTEGER DEFAULT 0,
                cluster_id INTEGER,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # Graph relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_relationships (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                PRIMARY KEY (source_id, target_id, relationship_type)
            )
        """)
        
        # Index for fast temporal queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_short_term_timestamp 
            ON short_term_memory(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_long_term_timestamp 
            ON long_term_memory(timestamp)
        """)
        
        self.conn.commit()
    
    def add_memory(self, content: str, tags: List[str] = None, metadata: Dict = None) -> int:
        """Add a new memory to short-term storage"""
        embedding = self.embedding_model.encode(content)
        timestamp = datetime.now().timestamp()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO short_term_memory (content, timestamp, embedding, tags, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            content,
            timestamp,
            embedding.tobytes(),
            json.dumps(tags or []),
            json.dumps(metadata or {})
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def compute_importance(self, memory: Memory) -> float:
        """
        Multi-factor importance score [0, 1]
        
        Factors:
        1. Recency: exponential decay
        2. Access frequency: log-scaled
        3. Explicit markers: user-tagged importance
        """
        age_days = (datetime.now() - memory.timestamp).total_seconds() / 86400
        recency_score = math.exp(-self.lambda_decay * age_days)
        
        # Access score (log-scaled, assuming max 100 accesses is very high)
        access_score = math.log(1 + memory.access_count) / math.log(101)
        
        # Explicit importance from tags
        explicit_score = 1.0 if "important" in memory.tags else 0.0
        
        # Weighted combination
        importance = (
            0.5 * recency_score +
            0.3 * access_score +
            0.2 * explicit_score
        )
        
        return max(0.0, min(1.0, importance))
    
    def get_old_memories(self, threshold_days: int = None) -> List[Memory]:
        """Retrieve memories older than threshold"""
        if threshold_days is None:
            threshold_days = self.consolidation_threshold_days
        
        cutoff = (datetime.now() - timedelta(days=threshold_days)).timestamp()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, timestamp, embedding, access_count, importance, tags, metadata
            FROM short_term_memory
            WHERE timestamp < ?
        """, (cutoff,))
        
        memories = []
        for row in cursor.fetchall():
            memories.append(Memory(
                id=row[0],
                content=row[1],
                timestamp=datetime.fromtimestamp(row[2]),
                embedding=np.frombuffer(row[3], dtype=np.float32),
                access_count=row[4],
                importance=row[5],
                tags=json.loads(row[6]),
                metadata=json.loads(row[7])
            ))
        
        return memories
    
    def temporal_semantic_clustering(self, memories: List[Memory], 
                                     eps: float = 0.3, 
                                     min_samples: int = 2) -> List[List[Memory]]:
        """
        Cluster memories by semantic similarity + temporal proximity
        
        Uses DBSCAN on combined feature space:
        - Normalized embedding (semantic)
        - Normalized timestamp (temporal)
        """
        if not memories:
            return []
        
        # Extract embeddings and timestamps
        embeddings = np.array([m.embedding for m in memories])
        timestamps = np.array([m.timestamp.timestamp() for m in memories])
        
        # Normalize timestamps to [0, 1] range
        t_min, t_max = timestamps.min(), timestamps.max()
        if t_max > t_min:
            timestamps_norm = (timestamps - t_min) / (t_max - t_min)
        else:
            timestamps_norm = np.zeros_like(timestamps)
        
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Combine features (weight: 80% semantic, 20% temporal)
        features = np.hstack([
            embeddings_norm * 0.8,
            timestamps_norm.reshape(-1, 1) * 0.2
        ])
        
        # Cluster with DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clustering.fit_predict(features)
        
        # Group memories by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memories[i])
        
        # Return clusters (excluding noise cluster -1)
        return [cluster for label, cluster in clusters.items() if label != -1]
    
    def cluster_coherence(self, cluster: List[Memory]) -> float:
        """
        Compute coherence score for a cluster.
        Average pairwise cosine similarity.
        """
        if len(cluster) < 2:
            return 1.0
        
        embeddings = np.array([m.embedding for m in cluster])
        
        # Normalize
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute pairwise similarities
        similarities = embeddings_norm @ embeddings_norm.T
        
        # Average of upper triangle (excluding diagonal)
        n = len(cluster)
        coherence = (similarities.sum() - n) / (n * (n - 1))
        
        return coherence
    
    def generate_summary(self, cluster: List[Memory]) -> str:
        """
        Generate a summary for a cluster of memories.
        For now: simple concatenation. In production: LLM summarization.
        """
        contents = [m.content for m in cluster]
        
        # Simple heuristic: if cluster is small, just list them
        if len(contents) <= 3:
            return " | ".join(contents)
        
        # Otherwise, create a summary statement
        return f"Summary of {len(contents)} related memories: " + " | ".join(contents[:3]) + f" ... (and {len(contents)-3} more)"
    
    def select_representatives(self, cluster: List[Memory], k: int = 2) -> List[Memory]:
        """
        Select k representative memories from a cluster.
        Strategy: Pick highest importance + most central.
        """
        if len(cluster) <= k:
            return cluster
        
        # Sort by importance
        sorted_cluster = sorted(cluster, key=lambda m: m.importance, reverse=True)
        
        # Take top k
        return sorted_cluster[:k]
    
    def consolidate_memories(self, threshold_days: int = None) -> Dict:
        """
        Main consolidation algorithm.
        
        Returns statistics about consolidation.
        """
        if threshold_days is None:
            threshold_days = self.consolidation_threshold_days
        
        # Step 1: Get old memories
        old_memories = self.get_old_memories(threshold_days)
        
        if not old_memories:
            return {"status": "no_memories_to_consolidate", "count": 0}
        
        # Step 2: Compute importance scores
        for memory in old_memories:
            memory.importance = self.compute_importance(memory)
        
        # Step 3: Cluster by semantic + temporal similarity
        clusters = self.temporal_semantic_clustering(old_memories)
        
        stats = {
            "memories_processed": len(old_memories),
            "clusters_formed": len(clusters),
            "memories_consolidated": 0,
            "summaries_created": 0,
            "high_importance_kept": 0
        }
        
        cursor = self.conn.cursor()
        
        # Step 4: Process each cluster
        for cluster_id, cluster in enumerate(clusters):
            max_importance = max(m.importance for m in cluster)
            coherence = self.cluster_coherence(cluster)
            
            if max_importance > self.high_importance_threshold:
                # Keep all high-importance memories individually
                for memory in cluster:
                    cursor.execute("""
                        INSERT INTO long_term_memory 
                        (content, timestamp, importance, is_summary, cluster_id, tags, metadata)
                        VALUES (?, ?, ?, 0, ?, ?, ?)
                    """, (
                        memory.content,
                        memory.timestamp.timestamp(),
                        memory.importance,
                        cluster_id,
                        json.dumps(memory.tags),
                        json.dumps(memory.metadata)
                    ))
                    stats["high_importance_kept"] += 1
                
            elif coherence > self.coherence_threshold:
                # Coherent cluster - generate summary + keep representatives
                summary = self.generate_summary(cluster)
                representatives = self.select_representatives(cluster, k=2)
                
                # Store summary
                cursor.execute("""
                    INSERT INTO long_term_memory 
                    (content, summary, timestamp, importance, is_summary, cluster_id, tags, metadata)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                """, (
                    summary,
                    summary,
                    cluster[0].timestamp.timestamp(),
                    max_importance,
                    cluster_id,
                    json.dumps([]),
                    json.dumps({})
                ))
                
                # Store representatives
                for rep in representatives:
                    cursor.execute("""
                        INSERT INTO long_term_memory 
                        (content, timestamp, importance, is_summary, cluster_id, tags, metadata)
                        VALUES (?, ?, ?, 0, ?, ?, ?)
                    """, (
                        rep.content,
                        rep.timestamp.timestamp(),
                        rep.importance,
                        cluster_id,
                        json.dumps(rep.tags),
                        json.dumps(rep.metadata)
                    ))
                
                stats["summaries_created"] += 1
                stats["memories_consolidated"] += len(cluster)
                
            else:
                # Incoherent cluster - keep top-k by importance
                top_k = sorted(cluster, key=lambda m: m.importance, reverse=True)[:3]
                for memory in top_k:
                    cursor.execute("""
                        INSERT INTO long_term_memory 
                        (content, timestamp, importance, is_summary, cluster_id, tags, metadata)
                        VALUES (?, ?, ?, 0, ?, ?, ?)
                    """, (
                        memory.content,
                        memory.timestamp.timestamp(),
                        memory.importance,
                        cluster_id,
                        json.dumps(memory.tags),
                        json.dumps(memory.metadata)
                    ))
                    stats["memories_consolidated"] += len(cluster)
        
        # Step 5: Delete consolidated memories from short-term
        memory_ids = [m.id for m in old_memories]
        cursor.execute(f"""
            DELETE FROM short_term_memory 
            WHERE id IN ({','.join('?' * len(memory_ids))})
        """, memory_ids)
        
        self.conn.commit()
        
        return stats
    
    def retrieve_memories(self, query: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant memories across short-term and long-term storage.
        
        Returns: List of (content, score) tuples
        """
        query_embedding = self.embedding_model.encode(query)
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        
        results = []
        
        # Search short-term memory
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT content, embedding, timestamp, importance, access_count
            FROM short_term_memory
        """)
        
        for row in cursor.fetchall():
            content = row[0]
            embedding = np.frombuffer(row[1], dtype=np.float32)
            timestamp = datetime.fromtimestamp(row[2])
            importance = row[3]
            
            # Compute similarity
            embedding_norm = embedding / np.linalg.norm(embedding)
            semantic_sim = np.dot(query_embedding_norm, embedding_norm)
            
            # Recency score
            age_days = (datetime.now() - timestamp).total_seconds() / 86400
            recency_score = math.exp(-self.lambda_decay * age_days)
            
            # Combined score
            score = (
                0.4 * semantic_sim +
                0.3 * importance +
                0.3 * recency_score
            )
            
            results.append((content, score))
            
            # Update access count
            cursor.execute("""
                UPDATE short_term_memory 
                SET access_count = access_count + 1
                WHERE content = ?
            """, (content,))
        
        # Search long-term memory
        cursor.execute("""
            SELECT content, timestamp, importance
            FROM long_term_memory
        """)
        
        for row in cursor.fetchall():
            content = row[0]
            timestamp = datetime.fromtimestamp(row[1])
            importance = row[2]
            
            # For long-term, we don't have embeddings in this simple version
            # In production: store embeddings for summaries too
            # For now: simple text match
            text_match = sum(word.lower() in content.lower() for word in query.split()) / len(query.split())
            
            # Recency score (lower weight for long-term)
            age_days = (datetime.now() - timestamp).total_seconds() / 86400
            recency_score = math.exp(-self.lambda_decay * age_days) * 0.5
            
            score = (
                0.3 * text_match +
                0.4 * importance +
                0.3 * recency_score
            )
            
            results.append((content, score))
        
        self.conn.commit()
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def get_stats(self) -> Dict:
        """Get statistics about current memory state"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM short_term_memory")
        short_term_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM long_term_memory")
        long_term_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(importance) FROM short_term_memory")
        avg_importance_short = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(importance) FROM long_term_memory")
        avg_importance_long = cursor.fetchone()[0] or 0.0
        
        return {
            "short_term_count": short_term_count,
            "long_term_count": long_term_count,
            "avg_importance_short_term": avg_importance_short,
            "avg_importance_long_term": avg_importance_long
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Example usage
    print("Memory Consolidation System - Demo")
    print("=" * 50)
    
    consolidator = MemoryConsolidator("test_memory.db")
    
    # Add some test memories
    print("\n1. Adding test memories...")
    memories = [
        "Implemented memory consolidation algorithm",
        "Discussed project architecture with team",
        "Fixed bug in authentication system",
        "Attended standup meeting at 10am",
        "Deployed new feature to production",
        "Code review for PR #123",
        "Updated documentation for API",
        "Discussed memory architecture design",
        "Implemented clustering algorithm",
        "Tested consolidation on sample data"
    ]
    
    for i, content in enumerate(memories):
        # Simulate old timestamps
        consolidator.add_memory(content)
    
    print(f"Added {len(memories)} memories")
    
    # Show current stats
    stats = consolidator.get_stats()
    print(f"\nCurrent state: {stats}")
    
    # Test retrieval
    print("\n2. Testing retrieval...")
    query = "memory architecture"
    results = consolidator.retrieve_memories(query, max_results=5)
    print(f"\nTop 5 results for '{query}':")
    for content, score in results:
        print(f"  [{score:.3f}] {content}")
    
    # Test consolidation (with low threshold to trigger it)
    print("\n3. Testing consolidation...")
    consolidation_stats = consolidator.consolidate_memories(threshold_days=0)
    print(f"Consolidation stats: {json.dumps(consolidation_stats, indent=2)}")
    
    # Final stats
    final_stats = consolidator.get_stats()
    print(f"\nFinal state: {final_stats}")
    
    consolidator.close()
    print("\n✓ Demo complete")
