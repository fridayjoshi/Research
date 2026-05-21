/**
 * Memory Consolidation Algorithm for Persistent AI Agents
 *
 * Automatically consolidates daily observation logs into bounded long-term memory
 * while minimizing information loss.
 *
 * @author Friday
 * @date 2026-05-21
 */

export interface Event {
  timestamp: Date;
  content: string;
  tags: string[];
  relevance?: number;
}

export interface Pattern {
  description: string;
  occurrences: number;
  firstSeen: Date;
  lastSeen: Date;
  examples: string[];
}

export interface Memory {
  events: Event[];
  patterns: Pattern[];
  startDate: Date;

  size(): number;
  addEvents(events: Event[]): void;
  addPatterns(patterns: Pattern[]): void;
  recentEvents(days: number): Event[];
  search(query: string, threshold: number): Event[];
  pruneLowest(count: number): void;
}

/**
 * In-memory implementation of Memory interface
 */
export class MemoryStore implements Memory {
  events: Event[] = [];
  patterns: Pattern[] = [];
  startDate: Date;

  constructor(startDate: Date) {
    this.startDate = startDate;
  }

  size(): number {
    return this.events.length;
  }

  addEvents(events: Event[]): void {
    this.events.push(...events);
  }

  addPatterns(patterns: Pattern[]): void {
    for (const pattern of patterns) {
      const existing = this.patterns.find(p =>
        jaccardSimilarity(p.description, pattern.description) > 0.85
      );
      if (existing) {
        existing.occurrences += pattern.occurrences;
        existing.lastSeen = pattern.lastSeen;
        existing.examples.push(...pattern.examples);
      } else {
        this.patterns.push(pattern);
      }
    }
  }

  recentEvents(days: number): Event[] {
    const cutoff = new Date(Date.now() - days * 24 * 3600 * 1000);
    return this.events.filter(e => e.timestamp >= cutoff);
  }

  search(query: string, threshold: number): Event[] {
    return this.events.filter(e =>
      jaccardSimilarity(e.content, query) > threshold
    );
  }

  pruneLowest(count: number): void {
    this.events.sort((a, b) => (b.relevance || 0) - (a.relevance || 0));
    this.events = this.events.slice(0, this.events.length - count);
  }

  toMarkdown(): string {
    let md = `# Long-Term Memory\n\n`;
    md += `**Generated:** ${new Date().toISOString()}\n`;
    md += `**Events:** ${this.events.length}\n`;
    md += `**Patterns:** ${this.patterns.length}\n\n`;

    md += `## Critical Events\n\n`;
    const sortedEvents = [...this.events].sort((a, b) =>
      b.timestamp.getTime() - a.timestamp.getTime()
    );
    for (const event of sortedEvents) {
      md += `### ${event.timestamp.toISOString().split('T')[0]}\n`;
      md += `**Tags:** ${event.tags.join(', ')}\n`;
      md += `**Relevance:** ${event.relevance?.toFixed(2) || 'N/A'}\n\n`;
      md += `${event.content}\n\n`;
    }

    md += `## Detected Patterns\n\n`;
    for (const pattern of this.patterns) {
      md += `### ${pattern.description}\n`;
      md += `**Occurrences:** ${pattern.occurrences}\n`;
      md += `**First seen:** ${pattern.firstSeen.toISOString().split('T')[0]}\n`;
      md += `**Last seen:** ${pattern.lastSeen.toISOString().split('T')[0]}\n\n`;
      md += `**Examples:**\n`;
      for (const example of pattern.examples.slice(0, 3)) {
        md += `- ${example.substring(0, 100)}...\n`;
      }
      md += `\n`;
    }

    return md;
  }
}

/**
 * Jaccard similarity between two strings (token-based)
 */
export function jaccardSimilarity(a: string, b: string): number {
  const tokensA = new Set(a.toLowerCase().split(/\s+/));
  const tokensB = new Set(b.toLowerCase().split(/\s+/));
  const intersection = new Set([...tokensA].filter(x => tokensB.has(x)));
  const union = new Set([...tokensA, ...tokensB]);
  return union.size > 0 ? intersection.size / union.size : 0;
}

/**
 * Simple embedding: character code vector (placeholder for sentence-transformers)
 */
export function embed(text: string): number[] {
  // Normalize to fixed length for cosine similarity
  const normalized = text.toLowerCase().replace(/[^a-z0-9\s]/g, '');
  const codes = normalized.split('').map(c => c.charCodeAt(0));
  return codes.slice(0, 100).concat(Array(Math.max(0, 100 - codes.length)).fill(0));
}

/**
 * Cosine similarity between two vectors
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  return magA > 0 && magB > 0 ? dotProduct / (magA * magB) : 0;
}

/**
 * Compute impact score based on keyword matching
 */
export function impact(event: Event): number {
  const impactKeywords: Record<string, number> = {
    'critical': 10,
    'security': 10,
    'breach': 10,
    'failure': 8,
    'error': 7,
    'mistake': 7,
    'decision': 7,
    'learning': 6,
    'lesson': 6,
    'pattern': 6,
    'insight': 5,
    'opsec': 9,
    'leak': 8,
    'rejected': 5,
    'success': 4,
    'completed': 2,
    'routine': 1,
    'heartbeat': 1,
    'status': 2,
    'check': 1
  };

  let maxScore = 2; // default baseline
  const contentLower = event.content.toLowerCase();

  for (const [keyword, score] of Object.entries(impactKeywords)) {
    // Use word boundary matching to avoid partial matches (e.g., "success" in "successfully")
    const regex = new RegExp(`\\b${keyword}\\b`, 'i');
    if (regex.test(contentLower)) {
      maxScore = Math.max(maxScore, score);
    }
  }

  return maxScore;
}

/**
 * Compute novelty score (inverse of similar events in memory)
 */
export function novelty(event: Event, memory: Memory): number {
  const similarEvents = memory.search(event.content, 0.5);
  return Math.max(0, 10 - similarEvents.length);
}

/**
 * Compute recurrence score (similar events in recent window)
 */
export function recurrence(event: Event, recentEvents: Event[]): number {
  const eventEmbed = embed(event.content);
  const similar = recentEvents.filter(e => {
    const similarity = cosineSimilarity(eventEmbed, embed(e.content));
    return similarity > 0.65;
  });
  return Math.min(10, similar.length * 2);
}

/**
 * Compute combined relevance score
 */
export function relevance(event: Event, memory: Memory, recentEvents: Event[]): number {
  const α = 0.5;  // impact weight
  const β = 0.3;  // novelty weight
  const γ = 0.2;  // recurrence weight

  const i = impact(event);
  const n = novelty(event, memory);
  const r = recurrence(event, recentEvents);

  return α * i + β * n + γ * r;
}

/**
 * Detect recurring patterns via temporal clustering
 */
export function detectPatterns(events: Event[], windowDays: number): Pattern[] {
  const clusters: Event[][] = [];

  for (const event of events) {
    const eventEmbed = embed(event.content);
    let matched = false;

    for (const cluster of clusters) {
      // Use first event as cluster representative (simplified centroid)
      const centroidEmbed = embed(cluster[0].content);
      if (cosineSimilarity(eventEmbed, centroidEmbed) > 0.80) {
        cluster.push(event);
        matched = true;
        break;
      }
    }

    if (!matched) {
      clusters.push([event]);
    }
  }

  // Only keep clusters with ≥3 occurrences (pattern threshold)
  const patterns = clusters
    .filter(c => c.length >= 3)
    .map(c => ({
      description: extractPatternDescription(c),
      occurrences: c.length,
      firstSeen: c[0].timestamp,
      lastSeen: c[c.length - 1].timestamp,
      examples: c.map(e => e.content)
    }));

  return patterns;
}

/**
 * Extract a human-readable pattern description from cluster
 */
function extractPatternDescription(cluster: Event[]): string {
  // Find common keywords across all events
  const allTokens = cluster.map(e =>
    e.content.toLowerCase().split(/\s+/)
  );

  const commonTokens = allTokens[0].filter(token =>
    allTokens.every(tokens => tokens.includes(token)) &&
    token.length > 3 // skip short words
  );

  if (commonTokens.length > 0) {
    return `Pattern: ${commonTokens.slice(0, 5).join(' ')}`;
  }

  // Fallback: use first event's title
  const firstLine = cluster[0].content.split('\n')[0];
  return firstLine.substring(0, 80) + (firstLine.length > 80 ? '...' : '');
}

/**
 * Main consolidation algorithm
 *
 * @param dailyLogs New events to consolidate
 * @param memory Existing long-term memory
 * @param budgetPerDay Events to retain per day (controls growth rate)
 * @returns Updated memory
 */
export function consolidate(
  dailyLogs: Event[],
  memory: Memory,
  budgetPerDay: number = 10
): Memory {
  // Compute current memory budget based on logarithmic growth
  const daysSinceStart = (Date.now() - memory.startDate.getTime()) / (24 * 3600 * 1000);
  const maxSize = Math.floor(budgetPerDay * Math.log2(daysSinceStart + 2));

  // Score all new events
  const recentEvents = memory.recentEvents(7);
  const scored = dailyLogs.map(e => {
    const score = relevance(e, memory, recentEvents);
    return {
      event: { ...e, relevance: score },
      score
    };
  });

  // Sort by relevance (descending)
  scored.sort((a, b) => b.score - a.score);

  // Select top events that fit within budget
  const availableSpace = maxSize - memory.size();
  const selected = scored.slice(0, Math.max(0, availableSpace));

  // Detect patterns in recent window (30 days)
  const recentWindow = memory.recentEvents(30).concat(dailyLogs);
  const patterns = detectPatterns(recentWindow, 30);

  // Update memory
  memory.addEvents(selected.map(s => s.event));
  memory.addPatterns(patterns);

  // Prune if over budget
  if (memory.size() > maxSize) {
    memory.pruneLowest(memory.size() - maxSize);
  }

  return memory;
}

/**
 * Parse a markdown daily log into structured events
 */
export function parseDailyLog(content: string, date: Date): Event[] {
  const events: Event[] = [];

  // Split by ## headers (sections)
  const sections = content.split(/^## /m).slice(1);

  for (const section of sections) {
    const lines = section.split('\n');
    const title = lines[0].trim();
    const body = lines.slice(1).join('\n').trim();

    if (title && body) {
      events.push({
        timestamp: date,
        content: `${title}\n\n${body}`,
        tags: extractTags(body)
      });
    }
  }

  return events;
}

/**
 * Extract tags from event content
 */
function extractTags(content: string): string[] {
  const tags: string[] = [];
  const lower = content.toLowerCase();

  if (/security|breach|leak|opsec|vulnerability/i.test(lower)) tags.push('security');
  if (/failure|error|mistake|wrong|broke/i.test(lower)) tags.push('failure');
  if (/decision|chose|strategy|approach/i.test(lower)) tags.push('decision');
  if (/learning|lesson|insight|realized/i.test(lower)) tags.push('learning');
  if (/pattern|recurring|repeatedly/i.test(lower)) tags.push('pattern');
  if (/metric|commit|output/i.test(lower)) tags.push('metrics');
  if (/infrastructure|deployment|system/i.test(lower)) tags.push('infrastructure');

  return tags;
}
