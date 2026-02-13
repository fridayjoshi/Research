#!/usr/bin/env python3
"""
Semantic Memory Consolidation for AI Agents
Consolidates daily logs into structured long-term memory
"""

import re
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json

class MemorySegment:
    def __init__(self, text: str, source: str, date: str, section: str):
        self.text = text
        self.source = source  # filename
        self.date = date
        self.section = section
        self.importance = 0.0
        self.tfidf_vector = {}
    
    def __repr__(self):
        return f"<Segment: {self.section[:30]}... (I={self.importance:.2f})>"

class MemoryConsolidator:
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, threshold=0.7):
        self.alpha = alpha  # novelty weight
        self.beta = beta    # impact weight
        self.gamma = gamma  # permanence weight
        self.threshold = threshold  # clustering threshold
        
        # Impact markers (learned from manual curation)
        self.impact_keywords = {
            'critical', 'never', 'always', 'must', 'learned', 'lesson',
            'mistake', 'wrong', 'correct', 'security', 'vulnerability',
            'core', 'principle', 'rule', 'policy', 'important'
        }
        
        self.permanence_markers = {
            'is', 'are', 'means', 'requires', 'defined as', 'principle',
            'rule', 'always', 'never', 'every', 'all'
        }
        
        self.segments: List[MemorySegment] = []
        self.vocabulary: Set[str] = set()
        self.idf: Dict[str, float] = {}
        self.existing_memory: List[str] = []
    
    def load_existing_memory(self, path: Path):
        """Load existing MEMORY.md to compute novelty"""
        if path.exists():
            content = path.read_text()
            self.existing_memory = [s.strip() for s in content.split('\n\n') if s.strip()]
    
    def extract_segments(self, log_dir: Path) -> List[MemorySegment]:
        """Extract segments from daily logs"""
        segments = []
        
        for log_file in sorted(log_dir.glob('2026-*.md')):
            content = log_file.read_text()
            date = log_file.stem
            
            # Split by markdown headers (## or ###)
            sections = re.split(r'\n(#{2,3})\s+(.+?)\n', content)
            
            current_section = "Introduction"
            current_text = ""
            
            for i, part in enumerate(sections):
                if part.startswith('#'):
                    if current_text.strip():
                        segments.append(MemorySegment(
                            current_text.strip(),
                            log_file.name,
                            date,
                            current_section
                        ))
                    current_section = sections[i+1] if i+1 < len(sections) else current_section
                    current_text = ""
                elif not re.match(r'^#{2,3}$', part):
                    current_text += part
            
            # Last section
            if current_text.strip():
                segments.append(MemorySegment(
                    current_text.strip(),
                    log_file.name,
                    date,
                    current_section
                ))
        
        return segments
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Lowercase, keep alphanumeric and basic punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 2]  # filter short tokens
    
    def compute_tfidf(self):
        """Compute TF-IDF vectors for all segments"""
        # Build vocabulary
        doc_freq = Counter()
        for seg in self.segments:
            tokens = set(self.tokenize(seg.text))
            self.vocabulary.update(tokens)
            for token in tokens:
                doc_freq[token] += 1
        
        # Compute IDF
        n_docs = len(self.segments)
        for token in self.vocabulary:
            self.idf[token] = math.log(n_docs / (1 + doc_freq[token]))
        
        # Compute TF-IDF for each segment
        for seg in self.segments:
            tokens = self.tokenize(seg.text)
            tf = Counter(tokens)
            seg.tfidf_vector = {
                token: (tf[token] / len(tokens)) * self.idf[token]
                for token in tf
            }
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors"""
        # Dot product
        common_keys = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
        
        # Magnitudes
        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def compute_novelty(self, segment: MemorySegment) -> float:
        """Compute novelty score (1 - max similarity to existing memory)"""
        if not self.existing_memory:
            return 1.0  # everything is novel if no existing memory
        
        # Compute TF-IDF for existing memory segments
        max_sim = 0.0
        for existing in self.existing_memory:
            tokens = self.tokenize(existing)
            tf = Counter(tokens)
            existing_vec = {
                token: (tf[token] / len(tokens)) * self.idf.get(token, 0)
                for token in tf if token in self.idf
            }
            sim = self.cosine_similarity(segment.tfidf_vector, existing_vec)
            max_sim = max(max_sim, sim)
        
        return 1.0 - max_sim
    
    def compute_impact(self, segment: MemorySegment) -> float:
        """Compute impact score based on lexical markers"""
        text_lower = segment.text.lower()
        tokens = set(self.tokenize(segment.text))
        
        # Count impact keywords
        impact_count = sum(1 for kw in self.impact_keywords if kw in tokens)
        
        # Normalize by segment length
        impact_score = impact_count / (len(tokens) / 100)  # per 100 words
        
        return min(1.0, impact_score)  # cap at 1.0
    
    def compute_permanence(self, segment: MemorySegment) -> float:
        """Compute permanence score (declarative vs. ephemeral)"""
        text_lower = segment.text.lower()
        
        # Check for permanence markers
        permanence_count = sum(1 for marker in self.permanence_markers if marker in text_lower)
        
        # Check for date-specific or ephemeral language
        ephemeral_patterns = [
            r'\btoday\b', r'\byesterday\b', r'\btomorrow\b',
            r'\bchecked\b', r'\bsent\b', r'\breceived\b',
            r'\d{1,2}:\d{2}', r'(AM|PM)\b'
        ]
        ephemeral_count = sum(1 for pattern in ephemeral_patterns if re.search(pattern, text_lower))
        
        # Permanence is high when declarative statements, low when ephemeral events
        permanence = (permanence_count - ephemeral_count * 0.5) / 5.0
        
        return max(0.0, min(1.0, permanence))  # clamp to [0, 1]
    
    def score_importance(self):
        """Compute importance scores for all segments"""
        for seg in self.segments:
            novelty = self.compute_novelty(seg)
            impact = self.compute_impact(seg)
            permanence = self.compute_permanence(seg)
            
            seg.importance = (
                self.alpha * novelty +
                self.beta * impact +
                self.gamma * permanence
            )
    
    def cluster_segments(self) -> List[List[MemorySegment]]:
        """Cluster semantically similar segments"""
        clusters = []
        used = set()
        
        for i, seg in enumerate(self.segments):
            if i in used:
                continue
            
            cluster = [seg]
            used.add(i)
            
            for j, other in enumerate(self.segments[i+1:], start=i+1):
                if j in used:
                    continue
                
                sim = self.cosine_similarity(seg.tfidf_vector, other.tfidf_vector)
                if sim > self.threshold:
                    cluster.append(other)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def consolidate(self, clusters: List[List[MemorySegment]], min_importance=0.3) -> str:
        """Generate consolidated memory output"""
        output = ["# Consolidated Memory\n"]
        output.append(f"*Generated from {len(self.segments)} segments across {len(set(s.date for s in self.segments))} days*\n")
        output.append(f"*Importance threshold: {min_importance}, Clustering threshold: {self.threshold}*\n\n")
        
        # Sort clusters by max importance
        clusters = sorted(clusters, key=lambda c: max(s.importance for s in c), reverse=True)
        
        stats = {
            'total_segments': len(self.segments),
            'high_importance': 0,
            'clusters': len(clusters),
            'retained': 0,
            'discarded': 0
        }
        
        for cluster in clusters:
            max_importance = max(s.importance for s in cluster)
            
            if max_importance < min_importance:
                stats['discarded'] += len(cluster)
                continue
            
            stats['retained'] += len(cluster)
            if max_importance > 0.6:
                stats['high_importance'] += len(cluster)
            
            # Pick representative segment (highest importance)
            representative = max(cluster, key=lambda s: s.importance)
            
            output.append(f"## {representative.section}\n")
            output.append(f"*Importance: {representative.importance:.2f} | Cluster size: {len(cluster)}*\n\n")
            output.append(f"{representative.text}\n\n")
            
            # Show provenance
            if len(cluster) > 1:
                sources = ", ".join(sorted(set(s.date for s in cluster)))
                output.append(f"*Related findings: {sources}*\n\n")
            else:
                output.append(f"*Source: {representative.date}*\n\n")
            
            output.append("---\n\n")
        
        # Add stats
        output.append("\n## Consolidation Statistics\n\n")
        output.append(f"- Total segments processed: {stats['total_segments']}\n")
        output.append(f"- High importance (>0.6): {stats['high_importance']} ({100*stats['high_importance']/stats['total_segments']:.1f}%)\n")
        output.append(f"- Clusters formed: {stats['clusters']}\n")
        output.append(f"- Retained: {stats['retained']} ({100*stats['retained']/stats['total_segments']:.1f}%)\n")
        output.append(f"- Discarded: {stats['discarded']} ({100*stats['discarded']/stats['total_segments']:.1f}%)\n")
        
        return "".join(output), stats
    
    def run(self, log_dir: Path, existing_memory: Path, output: Path, min_importance=0.3):
        """Full consolidation pipeline"""
        print(f"Loading existing memory from {existing_memory}...")
        self.load_existing_memory(existing_memory)
        
        print(f"Extracting segments from {log_dir}...")
        self.segments = self.extract_segments(log_dir)
        print(f"  Found {len(self.segments)} segments")
        
        print("Computing TF-IDF vectors...")
        self.compute_tfidf()
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        
        print("Scoring importance...")
        self.score_importance()
        
        print("Clustering semantically similar segments...")
        clusters = self.cluster_segments()
        print(f"  Formed {len(clusters)} clusters")
        
        print("Generating consolidated output...")
        consolidated, stats = self.consolidate(clusters, min_importance)
        
        output.write_text(consolidated)
        print(f"\nConsolidated memory written to {output}")
        print(f"Compression ratio: {len(self.segments) / stats['retained']:.1f}:1")
        
        return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Consolidate daily logs into long-term memory")
    parser.add_argument("--input", type=Path, default=Path("memory/"), help="Directory with daily logs")
    parser.add_argument("--existing", type=Path, default=Path("MEMORY.md"), help="Existing memory file")
    parser.add_argument("--output", type=Path, default=Path("MEMORY-consolidated.md"), help="Output file")
    parser.add_argument("--threshold", type=float, default=0.7, help="Clustering similarity threshold")
    parser.add_argument("--min-importance", type=float, default=0.3, help="Minimum importance to retain")
    
    args = parser.parse_args()
    
    consolidator = MemoryConsolidator(threshold=args.threshold)
    stats = consolidator.run(args.input, args.existing, args.output, args.min_importance)
    
    print("\n=== Consolidation Complete ===")
    print(json.dumps(stats, indent=2))
