/**
 * Adaptive Context Window Manager (ACWM)
 *
 * A meta-algorithm that dynamically manages context window utilization
 * for long-running LLM agents by predicting compression needs and
 * selecting optimal strategies.
 *
 * @author Friday
 * @date 2026-05-26
 */

// ============================================================================
// Types
// ============================================================================

export interface Message {
  id: string;
  content: string;
  tokens: number;
  timestamp: number;
  role: 'user' | 'assistant' | 'system';
}

export interface CompressionStrategy {
  name: string;
  compress(messages: Message[], targetRatio: number): Message[];
  estimateQuality(messages: Message[]): number;
  estimateCost(messages: Message[]): number;
}

export interface ACWMConfig {
  capacity: number;              // Max tokens
  triggerRatio: number;          // When to compress (0.85 = 85% full)
  lookahead: number;             // Messages to predict ahead
  qualityTarget: number;         // Desired quality [0,1]
  costBudget: number;            // Max cost per 1K tokens
  qualityWeight: number;         // Alpha in scoring (0-1)
}

export interface ConversationFeatures {
  avgMessageLength: number;
  hasCode: boolean;
  dependencyCount: number;
  conversationType: 'technical' | 'casual' | 'instruction' | 'mixed';
  recencyImportance: number;
}

export interface CompressionEvent {
  timestamp: number;
  strategy: string;
  originalMessages: number;
  compressedMessages: number;
  originalTokens: number;
  compressedTokens: number;
  duration: number;
  features: ConversationFeatures;
}

export interface ACWMMetrics {
  currentTokens: number;
  capacity: number;
  utilization: number;
  growthRate: number;
  compressionCount: number;
  strategyScores: Record<string, number>;
  avgCompressionRatio: number;
}

// ============================================================================
// Main Manager
// ============================================================================

export class AdaptiveContextManager {
  private messages: Message[] = [];
  private currentTokens = 0;
  private growthRate = 0;
  private strategyScores = new Map<string, number>();
  private compressionHistory: CompressionEvent[] = [];

  constructor(
    private config: ACWMConfig,
    private strategies: CompressionStrategy[]
  ) {
    if (strategies.length === 0) {
      throw new Error('At least one compression strategy required');
    }
  }

  onMessageReceived(msg: Message): Message[] {
    this.messages.push(msg);
    this.currentTokens += msg.tokens;
    this.updateGrowthRate(msg.tokens);

    if (this.shouldCompress()) {
      const strategy = this.selectStrategy();
      const compressed = this.executeCompression(strategy);
      this.messages = compressed;
      this.currentTokens = compressed.reduce((sum, m) => sum + m.tokens, 0);
    }

    return this.messages;
  }

  private shouldCompress(): boolean {
    const predicted = this.currentTokens + (this.growthRate * this.config.lookahead);
    const threshold = this.config.capacity * this.config.triggerRatio;
    return predicted > threshold;
  }

  private selectStrategy(): CompressionStrategy {
    const features = this.extractFeatures();
    const scores = new Map<CompressionStrategy, number>();

    for (const strategy of this.strategies) {
      const qualityScore = this.estimateQuality(strategy, features);
      const costScore = this.estimateCost(strategy, features);

      const alpha = this.config.qualityWeight;
      const score = alpha * qualityScore - (1 - alpha) * costScore;

      scores.set(strategy, score);
    }

    return Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])[0][0];
  }

  private executeCompression(strategy: CompressionStrategy): Message[] {
    const targetRatio = this.config.triggerRatio * 0.7; // Compress to 70% of trigger
    const startTime = Date.now();

    const compressed = strategy.compress(this.messages, targetRatio);

    const event: CompressionEvent = {
      timestamp: Date.now(),
      strategy: strategy.name,
      originalMessages: this.messages.length,
      compressedMessages: compressed.length,
      originalTokens: this.currentTokens,
      compressedTokens: compressed.reduce((sum, m) => sum + m.tokens, 0),
      duration: Date.now() - startTime,
      features: this.extractFeatures()
    };

    this.compressionHistory.push(event);
    this.updateStrategyScores(event);

    return compressed;
  }

  private updateGrowthRate(tokens: number): void {
    const decay = 0.1;
    this.growthRate = decay * tokens + (1 - decay) * this.growthRate;
  }

  private extractFeatures(): ConversationFeatures {
    const recentMessages = this.messages.slice(-20);

    return {
      avgMessageLength: recentMessages.reduce((sum, m) => sum + m.tokens, 0) / recentMessages.length,
      hasCode: recentMessages.some(m => /```/.test(m.content)),
      dependencyCount: this.countDependencies(recentMessages),
      conversationType: this.classifyConversation(recentMessages),
      recencyImportance: this.computeRecencyScore(recentMessages)
    };
  }

  private countDependencies(messages: Message[]): number {
    let count = 0;
    const keywords = ['as mentioned', 'like i said', 'from earlier', 'previously', 'that code', 'this function', 'above', 'earlier'];

    for (const msg of messages) {
      const lower = msg.content.toLowerCase();
      for (const keyword of keywords) {
        if (lower.includes(keyword)) {
          count++;
          break;
        }
      }
    }

    return count;
  }

  private classifyConversation(messages: Message[]): ConversationFeatures['conversationType'] {
    const text = messages.map(m => m.content).join(' ').toLowerCase();

    const technicalTerms = ['function', 'class', 'algorithm', 'implement', 'debug', 'error', 'code', 'variable'];
    const casualTerms = ['hey', 'thanks', 'cool', 'nice', 'lol', 'btw', 'yeah', 'okay'];
    const instructionTerms = ['please', 'can you', 'i need', 'help me', 'create', 'show me', 'explain'];

    const technicalScore = technicalTerms.filter(term => text.includes(term)).length;
    const casualScore = casualTerms.filter(term => text.includes(term)).length;
    const instructionScore = instructionTerms.filter(term => text.includes(term)).length;

    const max = Math.max(technicalScore, casualScore, instructionScore);

    if (technicalScore === max && technicalScore > 3) return 'technical';
    if (casualScore === max && casualScore > 3) return 'casual';
    if (instructionScore === max && instructionScore > 3) return 'instruction';
    return 'mixed';
  }

  private computeRecencyScore(messages: Message[]): number {
    if (messages.length < 2) return 0.5;

    const timespan = messages[messages.length - 1].timestamp - messages[0].timestamp;
    const avgGap = timespan / messages.length;

    // Shorter gaps = more continuous conversation = higher recency importance
    // 60000ms = 1 minute
    return Math.exp(-avgGap / 60000);
  }

  private estimateQuality(strategy: CompressionStrategy, features: ConversationFeatures): number {
    // Historical performance
    const history = this.compressionHistory.filter(e => e.strategy === strategy.name);
    const historicalScore = history.length > 0
      ? history.reduce((sum, e) => sum + (e.compressedMessages / e.originalMessages), 0) / history.length
      : 0.5;

    // Feature match score
    const matchScore = this.computeFeatureMatch(strategy.name, features);

    return 0.6 * historicalScore + 0.4 * matchScore;
  }

  private computeFeatureMatch(strategyName: string, features: ConversationFeatures): number {
    const matches: Record<string, Record<string, number>> = {
      semacomp: {
        technical: 0.9,
        casual: 0.5,
        instruction: 0.7,
        mixed: 0.8
      },
      sliding_window: {
        technical: 0.5,
        casual: 0.9,
        instruction: 0.6,
        mixed: 0.6
      },
      importance: {
        technical: 0.7,
        casual: 0.4,
        instruction: 0.9,
        mixed: 0.7
      }
    };

    const strategyKey = strategyName.toLowerCase().replace(/-/g, '_');
    return matches[strategyKey]?.[features.conversationType] ?? 0.5;
  }

  private estimateCost(strategy: CompressionStrategy, features: ConversationFeatures): number {
    // Normalize cost by computational complexity
    const complexities: Record<string, number> = {
      semacomp: 3.0,        // O(n² log n)
      sliding_window: 1.0,   // O(n)
      importance: 1.5        // O(n log n)
    };

    const strategyKey = strategy.name.toLowerCase().replace(/-/g, '_');
    return complexities[strategyKey] ?? 2.0;
  }

  private updateStrategyScores(event: CompressionEvent): void {
    const quality = event.compressedMessages / event.originalMessages;
    const cost = event.duration / event.originalMessages;

    const score = this.config.qualityWeight * quality - (1 - this.config.qualityWeight) * cost;
    this.strategyScores.set(event.strategy, score);
  }

  getMetrics(): ACWMMetrics {
    return {
      currentTokens: this.currentTokens,
      capacity: this.config.capacity,
      utilization: this.currentTokens / this.config.capacity,
      growthRate: this.growthRate,
      compressionCount: this.compressionHistory.length,
      strategyScores: Object.fromEntries(this.strategyScores),
      avgCompressionRatio: this.compressionHistory.length > 0
        ? this.compressionHistory.reduce((sum, e) =>
            sum + (e.compressedTokens / e.originalTokens), 0) / this.compressionHistory.length
        : 1.0
    };
  }

  getCompressionHistory(): CompressionEvent[] {
    return [...this.compressionHistory];
  }

  reset(): void {
    this.messages = [];
    this.currentTokens = 0;
    this.growthRate = 0;
    this.compressionHistory = [];
  }
}

// ============================================================================
// Compression Strategies
// ============================================================================

export class SlidingWindowStrategy implements CompressionStrategy {
  name = 'sliding-window';

  compress(messages: Message[], targetRatio: number): Message[] {
    const targetCount = Math.max(1, Math.floor(messages.length * targetRatio));
    return messages.slice(-targetCount);
  }

  estimateQuality(messages: Message[]): number {
    const recentTokens = messages.slice(-20).reduce((sum, m) => sum + m.tokens, 0);
    const totalTokens = messages.reduce((sum, m) => sum + m.tokens, 0);
    return totalTokens > 0 ? recentTokens / totalTokens : 0.5;
  }

  estimateCost(messages: Message[]): number {
    return messages.length * 0.001;
  }
}

export class ImportanceSamplingStrategy implements CompressionStrategy {
  name = 'importance';

  compress(messages: Message[], targetRatio: number): Message[] {
    const scored = messages.map(msg => ({
      message: msg,
      score: this.scoreMessage(msg, messages)
    }));

    scored.sort((a, b) => b.score - a.score);

    const targetCount = Math.max(1, Math.floor(messages.length * targetRatio));
    return scored.slice(0, targetCount)
      .sort((a, b) => a.message.timestamp - b.message.timestamp)
      .map(item => item.message);
  }

  private scoreMessage(msg: Message, context: Message[]): number {
    let score = 0;

    // User messages more important
    if (msg.role === 'user') score += 2;

    // System messages critical
    if (msg.role === 'system') score += 3;

    // Questions more important
    if (msg.content.includes('?')) score += 1.5;

    // Commands/instructions important
    const commands = ['please', 'can you', 'i need', 'help me', 'create', 'implement'];
    if (commands.some(cmd => msg.content.toLowerCase().includes(cmd))) {
      score += 1.8;
    }

    // Code blocks important
    if (/```/.test(msg.content)) score += 1.3;

    // Long messages often more substantive
    if (msg.tokens > 200) score += 0.5;

    // Recency bonus (decay exponentially)
    const position = context.indexOf(msg);
    const recencyBonus = Math.exp((position - context.length) / 50);
    score += recencyBonus;

    return score;
  }

  estimateQuality(messages: Message[]): number {
    return 0.75;
  }

  estimateCost(messages: Message[]): number {
    return messages.length * Math.log(messages.length) * 0.002;
  }
}

export class SemaCompStrategy implements CompressionStrategy {
  name = 'semacomp';

  compress(messages: Message[], targetRatio: number): Message[] {
    // Simplified SemaComp: dependency-aware compression
    // Build dependency graph
    const deps = new Map<number, Set<number>>();

    for (let i = 0; i < messages.length; i++) {
      deps.set(i, new Set());

      const content = messages[i].content.toLowerCase();
      const keywords = ['that', 'this', 'above', 'earlier', 'previously', 'mentioned'];

      if (keywords.some(kw => content.includes(kw))) {
        // Assume dependency on recent messages
        for (let j = Math.max(0, i - 5); j < i; j++) {
          deps.get(i)!.add(j);
        }
      }
    }

    // Score messages with dependency expansion
    const scored = messages.map((msg, idx) => ({
      message: msg,
      index: idx,
      score: this.scoreWithDeps(idx, messages, deps)
    }));

    scored.sort((a, b) => b.score - a.score);

    // Select top messages
    const targetCount = Math.max(1, Math.floor(messages.length * targetRatio));
    const selected = new Set<number>();

    for (let i = 0; i < targetCount && i < scored.length; i++) {
      const idx = scored[i].index;
      selected.add(idx);

      // Include dependencies
      deps.get(idx)?.forEach(dep => selected.add(dep));
    }

    // Return in original order
    return Array.from(selected)
      .sort((a, b) => a - b)
      .map(idx => messages[idx]);
  }

  private scoreWithDeps(
    idx: number,
    messages: Message[],
    deps: Map<number, Set<number>>
  ): number {
    const msg = messages[idx];
    let score = 0;

    // Base importance
    if (msg.role === 'user') score += 2;
    if (msg.role === 'system') score += 3;
    if (/```/.test(msg.content)) score += 2;

    // Dependency coverage
    const depCount = deps.get(idx)?.size ?? 0;
    score += depCount * 0.5;

    // Recency
    const recency = (idx / messages.length);
    score += recency;

    // Technical content
    const technical = ['function', 'class', 'error', 'bug', 'fix'];
    if (technical.some(term => msg.content.toLowerCase().includes(term))) {
      score += 1.5;
    }

    return score;
  }

  estimateQuality(messages: Message[]): number {
    return 0.9;
  }

  estimateCost(messages: Message[]): number {
    const n = messages.length;
    return n * n * Math.log(n + 1) * 0.005;
  }
}

// ============================================================================
// Utilities
// ============================================================================

export function countTokens(text: string): number {
  // Simplified token counting: ~4 chars per token
  return Math.ceil(text.length / 4);
}

export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// ============================================================================
// Example Usage
// ============================================================================

export function createDefaultManager(capacity: number = 128000): AdaptiveContextManager {
  const config: ACWMConfig = {
    capacity,
    triggerRatio: 0.85,
    lookahead: 10,
    qualityTarget: 0.85,
    costBudget: 0.05,
    qualityWeight: 0.7
  };

  const strategies = [
    new SlidingWindowStrategy(),
    new ImportanceSamplingStrategy(),
    new SemaCompStrategy()
  ];

  return new AdaptiveContextManager(config, strategies);
}
