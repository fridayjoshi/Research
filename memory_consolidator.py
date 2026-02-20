#!/usr/bin/env python3
"""
Memory consolidation algorithm for AI agent logs.
Transforms chronological daily logs into thematic long-term memory.
"""

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import math

@dataclass
class Event:
    """Semantic event extracted from daily log."""
    date: str  # ISO8601
    content: str
    event_type: str  # incident, lesson, observation, metric, decision
    entities: Set[str]
    themes: Set[str]
    source_file: str
    importance: float = 1.0  # 0-1 scale

class MemoryConsolidator:
    """Consolidate daily logs into thematic long-term memory."""
    
    def __init__(self, 
                 alpha=0.3, beta=0.4, gamma=0.2, delta=0.1,
                 tau=7.0, theta=0.3):
        """
        Args:
            alpha: weight for entity overlap
            beta: weight for theme overlap
            gamma: weight for temporal proximity
            delta: weight for type match
            tau: temporal decay constant (days)
            theta: clustering threshold
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.tau = tau
        self.theta = theta
    
    def extract_events(self, log_file: Path) -> List[Event]:
        """Extract semantic events from daily log."""
        content = log_file.read_text()
        date = self._extract_date_from_filename(log_file.name)
        
        events = []
        
        # Split on markdown headers (## sections)
        sections = re.split(r'\n## ', content)
        
        for section in sections[1:]:  # Skip preamble
            lines = section.split('\n', 1)
            if len(lines) < 2:
                continue
            
            title = lines[0]
            body = lines[1] if len(lines) > 1 else ""
            
            # Classify event type
            event_type = self._classify_event_type(title, body)
            
            # Extract entities (capitalized words, emails, tools)
            entities = self._extract_entities(body)
            
            # Extract themes (keywords)
            themes = self._extract_themes(title, body)
            
            # Compute importance (heuristic)
            importance = self._compute_importance(body)
            
            events.append(Event(
                date=date,
                content=f"## {title}\n{body}",
                event_type=event_type,
                entities=entities,
                themes=themes,
                source_file=log_file.name,
                importance=importance
            ))
        
        return events
    
    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename like '2026-02-20.md'."""
        match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
        return match.group(0) if match else "unknown"
    
    def _classify_event_type(self, title: str, body: str) -> str:
        """Classify event based on keywords."""
        text = (title + " " + body).lower()
        
        if any(kw in text for kw in ['incident', 'security', 'breach', 'unauthorized']):
            return 'incident'
        elif any(kw in text for kw in ['lesson', 'learned', 'mistake', 'wrong']):
            return 'lesson'
        elif any(kw in text for kw in ['metric', 'commits', 'prs', 'count']):
            return 'metric'
        elif any(kw in text for kw in ['decided', 'choice', 'strategy', 'approach']):
            return 'decision'
        else:
            return 'observation'
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities (simple heuristic: capitalized words, emails)."""
        # Capitalized words (but not sentence starters)
        cap_words = set(re.findall(r'(?<!\. )\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text))
        
        # Emails
        emails = set(re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text))
        
        # Tool/command names (backticks)
        tools = set(re.findall(r'`([^`]+)`', text))
        
        return cap_words | emails | tools
    
    def _extract_themes(self, title: str, body: str) -> Set[str]:
        """Extract thematic keywords."""
        text = (title + " " + body).lower()
        
        # Predefined theme keywords (expandable)
        theme_keywords = {
            'email', 'security', 'automation', 'github', 'pr', 'linkedin',
            'health', 'reading', 'research', 'maintenance', 'identity',
            'ai', 'agent', 'memory', 'consolidation', 'review'
        }
        
        return {kw for kw in theme_keywords if kw in text}
    
    def _compute_importance(self, text: str) -> float:
        """Compute importance score (0-1) based on heuristics."""
        score = 0.5  # baseline
        
        # Length (longer = potentially more important, up to a point)
        length_score = min(len(text) / 2000, 0.2)
        score += length_score
        
        # Critical keywords
        if any(kw in text.lower() for kw in ['critical', 'security', 'incident', 'urgent']):
            score += 0.2
        
        # Lessons
        if 'lesson' in text.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def similarity(self, e1: Event, e2: Event) -> float:
        """Compute similarity between two events."""
        # Entity overlap (Jaccard)
        entity_sim = self._jaccard(e1.entities, e2.entities)
        
        # Theme overlap (Jaccard)
        theme_sim = self._jaccard(e1.themes, e2.themes)
        
        # Temporal proximity
        days_between = self._days_between(e1.date, e2.date)
        temporal_sim = math.exp(-days_between / self.tau)
        
        # Type match
        type_sim = 1.0 if e1.event_type == e2.event_type else 0.0
        
        return (self.alpha * entity_sim + 
                self.beta * theme_sim + 
                self.gamma * temporal_sim + 
                self.delta * type_sim)
    
    def _jaccard(self, set1: Set, set2: Set) -> float:
        """Jaccard similarity coefficient."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _days_between(self, date1: str, date2: str) -> int:
        """Days between two ISO8601 dates."""
        try:
            d1 = datetime.fromisoformat(date1)
            d2 = datetime.fromisoformat(date2)
            return abs((d2 - d1).days)
        except:
            return 365  # Default to large value if parse fails
    
    def cluster_events(self, events: List[Event]) -> List[List[Event]]:
        """Hierarchical agglomerative clustering of events."""
        # Start with singleton clusters
        clusters = [[e] for e in events]
        
        while True:
            # Find most similar pair
            best_sim = -1
            best_pair = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = self._cluster_similarity(clusters[i], clusters[j])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)
            
            # Stop if similarity below threshold
            if best_sim < self.theta:
                break
            
            # Merge best pair
            i, j = best_pair
            clusters[i].extend(clusters[j])
            del clusters[j]
            
            # Stop if only one cluster left
            if len(clusters) == 1:
                break
        
        return clusters
    
    def _cluster_similarity(self, cluster1: List[Event], cluster2: List[Event]) -> float:
        """Average linkage similarity between clusters."""
        if not cluster1 or not cluster2:
            return 0.0
        
        total_sim = 0.0
        count = 0
        
        for e1 in cluster1:
            for e2 in cluster2:
                total_sim += self.similarity(e1, e2)
                count += 1
        
        return total_sim / count if count > 0 else 0.0
    
    def extract_theme(self, cluster: List[Event]) -> Dict:
        """Extract theme metadata from cluster."""
        if not cluster:
            return {}
        
        # Aggregate entities and themes
        all_entities = defaultdict(int)
        all_themes = defaultdict(int)
        
        for event in cluster:
            for entity in event.entities:
                all_entities[entity] += 1
            for theme in event.themes:
                all_themes[theme] += 1
        
        # Sort by frequency
        top_entities = sorted(all_entities.items(), key=lambda x: -x[1])[:5]
        top_themes = sorted(all_themes.items(), key=lambda x: -x[1])[:3]
        
        # Theme name: top entities + themes
        theme_name = ", ".join([e[0] for e in top_entities[:2]])
        if top_themes:
            theme_name += f" ({top_themes[0][0]})"
        
        # Timeline
        dates = sorted([e.date for e in cluster])
        timeline = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else dates[0]
        
        # Key moments (highest importance)
        key_events = sorted(cluster, key=lambda e: -e.importance)[:3]
        
        return {
            'name': theme_name,
            'timeline': timeline,
            'events': cluster,
            'key_events': key_events,
            'entities': dict(top_entities),
            'themes': dict(top_themes)
        }
    
    def consolidate(self, daily_logs_dir: Path, output_file: Path):
        """Main consolidation pipeline."""
        # Extract events from all daily logs
        all_events = []
        for log_file in sorted(daily_logs_dir.glob("*.md")):
            if log_file.name.startswith("2026-"):  # Filter to date files
                events = self.extract_events(log_file)
                all_events.extend(events)
        
        print(f"Extracted {len(all_events)} events from {len(list(daily_logs_dir.glob('2026-*.md')))} daily logs")
        
        # Cluster events
        clusters = self.cluster_events(all_events)
        print(f"Formed {len(clusters)} thematic clusters")
        
        # Extract themes
        themes = [self.extract_theme(cluster) for cluster in clusters]
        
        # Generate consolidated output
        self._write_consolidated_memory(themes, output_file)
        print(f"Consolidated memory written to {output_file}")
    
    def _write_consolidated_memory(self, themes: List[Dict], output_file: Path):
        """Write consolidated MEMORY.md."""
        lines = ["# MEMORY.md - Consolidated Long-Term Memory\n"]
        lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
        lines.append(f"**Themes identified:** {len(themes)}\n\n")
        lines.append("---\n\n")
        
        for theme in themes:
            lines.append(f"## {theme['name']}\n")
            lines.append(f"**Timeline:** {theme['timeline']}\n\n")
            
            # Write key events
            lines.append("**Key moments:**\n")
            for event in theme['key_events']:
                # Extract first paragraph as summary
                summary = event.content.split('\n\n')[0].replace('## ', '').strip()
                lines.append(f"- **{event.date}:** {summary[:200]}... *(source: {event.source_file})*\n")
            
            lines.append("\n")
            
            # Write entity mentions
            if theme['entities']:
                lines.append(f"**Mentioned:** {', '.join(theme['entities'].keys())}\n\n")
            
            lines.append("---\n\n")
        
        output_file.write_text(''.join(lines))

def main():
    """CLI interface."""
    if len(sys.argv) < 3:
        print("Usage: memory_consolidator.py <daily_logs_dir> <output_file>")
        sys.exit(1)
    
    daily_logs_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    consolidator = MemoryConsolidator()
    consolidator.consolidate(daily_logs_dir, output_file)

if __name__ == "__main__":
    main()
