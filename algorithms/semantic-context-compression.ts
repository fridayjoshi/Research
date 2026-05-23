/**
 * SemaComp: Semantic Context Compression for LLM Agents
 *
 * Implementation of the algorithm described in semantic-context-compression.md
 *
 * @author Friday
 * @date 2026-05-23
 */

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  tokens: number;
  timestamp: number;
  embedding?: number[];
}

interface CompressionResult {
  messages: Message[];
  compressionRatio: number;
  criticalRetention: number;
  tokensUsed: number;
}

interface SemaCompConfig {
  weights: {
    alpha: number; // Coverage weight
    beta: number;  // Recency weight
    gamma: number; // Uniqueness weight
  };
  decayRate: number;      // Recency decay (λ)
  similarityThreshold: number;
  dependencyBudget: number; // Fraction reserved for dependency expansion
}

const DEFAULT_CONFIG: SemaCompConfig = {
  weights: { alpha: 0.5, beta: 0.3, gamma: 0.2 },
  decayRate: 0.01,
  similarityThreshold: 0.7,
  dependencyBudget: 0.2
};

class DependencyGraph {
  private adj = new Map<string, Set<string>>();
  private nodes = new Set<string>();

  addNode(id: string): void {
    this.nodes.add(id);
    if (!this.adj.has(id)) {
      this.adj.set(id, new Set());
    }
  }

  addEdge(from: string, to: string): void {
    this.addNode(from);
    this.addNode(to);
    this.adj.get(from)!.add(to);
  }

  getOutgoing(id: string): Set<string> {
    return this.adj.get(id) || new Set();
  }

  getIncoming(id: string): Set<string> {
    const incoming = new Set<string>();
    for (const [node, edges] of this.adj) {
      if (edges.has(id)) {
        incoming.add(node);
      }
    }
    return incoming;
  }

  // BFS to compute reachable nodes (for coverage)
  getReachable(start: string): Set<string> {
    const visited = new Set<string>();
    const queue = [start];

    while (queue.length > 0) {
      const node = queue.shift()!;
      if (visited.has(node)) continue;
      visited.add(node);

      for (const neighbor of this.getOutgoing(node)) {
        queue.push(neighbor);
      }
    }

    return visited;
  }

  get size(): number {
    return this.nodes.size;
  }
}

class SemaComp {
  private config: SemaCompConfig;

  constructor(config: Partial<SemaCompConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Main compression entry point
   */
  compress(messages: Message[], budget: number): CompressionResult {
    if (messages.length === 0) {
      return { messages: [], compressionRatio: 0, criticalRetention: 0, tokensUsed: 0 };
    }

    const totalTokens = messages.reduce((sum, m) => sum + m.tokens, 0);

    // Phase 1: Build dependency graph
    const graph = this.buildDependencyGraph(messages);

    // Phase 2: Greedy selection (reserve budget for expansion)
    const greedyBudget = budget * (1 - this.config.dependencyBudget);
    const selected = this.greedySelect(messages, greedyBudget, graph);

    // Phase 3: Dependency expansion
    const expanded = this.expandDependencies(selected, messages, budget, graph);

    // Phase 4: Enforce invariants (system messages, recent user messages)
    const final = this.enforceInvariants(expanded, messages, budget);

    // Metrics
    const tokensUsed = final.reduce((sum, m) => sum + m.tokens, 0);
    const compressionRatio = tokensUsed / totalTokens;

    return {
      messages: final.sort((a, b) => a.timestamp - b.timestamp),
      compressionRatio,
      criticalRetention: this.computeCriticalRetention(final, messages),
      tokensUsed
    };
  }

  /**
   * Build dependency graph from message references
   */
  private buildDependencyGraph(messages: Message[]): DependencyGraph {
    const graph = new DependencyGraph();

    for (let i = 0; i < messages.length; i++) {
      const mi = messages[i];
      graph.addNode(mi.id);

      const refs = this.extractReferences(mi, messages.slice(0, i));
      for (const ref of refs) {
        graph.addEdge(ref.id, mi.id);
      }
    }

    return graph;
  }

  /**
   * Extract messages referenced by current message
   */
  private extractReferences(m: Message, prior: Message[]): Message[] {
    const refs: Message[] = [];

    // Pattern 1: Explicit linguistic back-references
    const backrefPatterns = [
      /as (?:I|you|we) (?:mentioned|said|noted|discussed) (?:earlier|before|above|previously)/i,
      /referring to (?:the |your |my )?(?:previous |last |earlier )?(?:message|response|comment)/i,
      /(?:in|from) (?:the |your |my )?(?:previous|prior|last|earlier) (?:message|response|turn)/i
    ];

    for (const pattern of backrefPatterns) {
      if (pattern.test(m.content)) {
        // Likely references recent messages
        refs.push(...prior.slice(-5));
        break;
      }
    }

    // Pattern 2: Tool call/result dependencies
    if (m.role === 'tool' && m.content.includes('<function_results>')) {
      // Find corresponding function call
      const toolCall = [...prior].reverse().find(msg =>
        msg.role === 'assistant' && msg.content.includes('<function_calls>')
      );
      if (toolCall) refs.push(toolCall);
    }

    if (m.role === 'assistant' && m.content.includes('<function_results>')) {
      // References the tool results
      const toolResult = [...prior].reverse().find(msg =>
        msg.role === 'tool'
      );
      if (toolResult) refs.push(toolResult);
    }

    // Pattern 3: Semantic similarity (if embeddings available)
    if (m.embedding && prior.length > 0 && prior[0].embedding) {
      const similar = prior.filter(p => {
        if (!p.embedding) return false;
        return this.cosineSimilarity(m.embedding!, p.embedding) > this.config.similarityThreshold;
      });
      refs.push(...similar);
    }

    return [...new Set(refs)]; // Deduplicate
  }

  /**
   * Greedy selection maximizing score per token
   */
  private greedySelect(
    messages: Message[],
    budget: number,
    graph: DependencyGraph
  ): Set<Message> {
    const selected = new Set<Message>();
    let tokensUsed = 0;

    // Compute scores
    const now = Math.max(...messages.map(m => m.timestamp));
    const scored = messages.map(m => {
      const score = this.computeScore(m, graph, messages, now);
      return {
        message: m,
        score,
        efficiency: score / m.tokens // Score per token
      };
    });

    // Sort by efficiency (greedy choice)
    scored.sort((a, b) => b.efficiency - a.efficiency);

    // Select greedily
    for (const item of scored) {
      if (tokensUsed + item.message.tokens <= budget) {
        selected.add(item.message);
        tokensUsed += item.message.tokens;
      }
    }

    return selected;
  }

  /**
   * Compute message score (coverage + recency + uniqueness)
   */
  private computeScore(
    m: Message,
    graph: DependencyGraph,
    messages: Message[],
    now: number
  ): number {
    const { alpha, beta, gamma } = this.config.weights;

    // Coverage: fraction of messages reachable from m
    const reachable = graph.getReachable(m.id);
    const coverage = graph.size > 0 ? reachable.size / graph.size : 0;

    // Recency: exponential decay
    const age = now - m.timestamp;
    const recency = Math.exp(-this.config.decayRate * age);

    // Uniqueness: 1 - max similarity to other messages
    let maxSim = 0;
    if (m.embedding) {
      for (const other of messages) {
        if (other.id === m.id || !other.embedding) continue;
        const sim = this.cosineSimilarity(m.embedding, other.embedding);
        maxSim = Math.max(maxSim, sim);
      }
    }
    const uniqueness = 1 - maxSim;

    return alpha * coverage + beta * recency + gamma * uniqueness;
  }

  /**
   * Expand to include critical dependencies
   */
  private expandDependencies(
    selected: Set<Message>,
    allMessages: Message[],
    budget: number,
    graph: DependencyGraph
  ): Set<Message> {
    const expanded = new Set(selected);
    let tokensUsed = Array.from(selected).reduce((sum, m) => sum + m.tokens, 0);

    const messageMap = new Map(allMessages.map(m => [m.id, m]));

    // For each selected message, try to include its dependencies
    for (const m of selected) {
      const deps = Array.from(graph.getIncoming(m.id));

      for (const depId of deps) {
        const dep = messageMap.get(depId);
        if (!dep || expanded.has(dep)) continue;

        if (tokensUsed + dep.tokens <= budget) {
          expanded.add(dep);
          tokensUsed += dep.tokens;
        }
      }
    }

    return expanded;
  }

  /**
   * Enforce invariants: always include system messages, recent user turns
   */
  private enforceInvariants(
    selected: Set<Message>,
    allMessages: Message[],
    budget: number
  ): Message[] {
    const result = new Set(selected);
    let tokensUsed = Array.from(selected).reduce((sum, m) => sum + m.tokens, 0);

    // Always include system messages
    for (const m of allMessages) {
      if (m.role === 'system' && !result.has(m)) {
        if (tokensUsed + m.tokens <= budget) {
          result.add(m);
          tokensUsed += m.tokens;
        }
      }
    }

    // Always include last 2 user messages
    const userMessages = allMessages.filter(m => m.role === 'user');
    const recentUser = userMessages.slice(-2);
    for (const m of recentUser) {
      if (!result.has(m) && tokensUsed + m.tokens <= budget) {
        result.add(m);
        tokensUsed += m.tokens;
      }
    }

    return Array.from(result);
  }

  /**
   * Cosine similarity between embeddings
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  /**
   * Compute fraction of critical messages retained (for evaluation)
   */
  private computeCriticalRetention(selected: Message[], all: Message[]): number {
    // Heuristic: critical = high coverage score
    // In real evaluation, would use ground-truth labels
    return selected.length / all.length; // Placeholder
  }
}

// ============================================================================
// Evaluation Framework
// ============================================================================

/**
 * Generate synthetic conversation for benchmarking
 */
function generateSyntheticConversation(
  numMessages: number,
  dependencyDensity: number
): Message[] {
  const messages: Message[] = [];
  const roles: Array<Message['role']> = ['user', 'assistant', 'tool'];

  for (let i = 0; i < numMessages; i++) {
    // Generate random embedding
    const embedding = Array.from({ length: 128 }, () => Math.random() - 0.5);

    // Normalize embedding
    const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
    for (let j = 0; j < embedding.length; j++) {
      embedding[j] /= norm;
    }

    // Generate content with potential back-references
    let content = `Message ${i}: ${generateRandomText(50 + Math.random() * 200)}`;

    // Add dependency references
    if (i > 0 && Math.random() < dependencyDensity) {
      const refIdx = Math.floor(Math.random() * i);
      content += ` As mentioned in message ${refIdx}, this builds on prior work.`;
    }

    // Add tool patterns
    if (roles[i % roles.length] === 'assistant' && Math.random() < 0.3) {
      content += ' <function_calls><invoke name="tool">...</invoke></function_calls>';
    }

    messages.push({
      id: `msg-${i}`,
      content,
      role: roles[i % roles.length],
      tokens: 50 + Math.floor(Math.random() * 450),
      timestamp: i,
      embedding
    });
  }

  return messages;
}

function generateRandomText(tokens: number): string {
  const words = [
    'context', 'compression', 'algorithm', 'dependency', 'graph',
    'semantic', 'score', 'budget', 'optimization', 'performance',
    'agent', 'conversation', 'message', 'token', 'efficiency',
    'critical', 'coverage', 'recency', 'uniqueness', 'computation'
  ];
  const text: string[] = [];
  for (let i = 0; i < tokens / 5; i++) {
    text.push(words[Math.floor(Math.random() * words.length)]);
  }
  return text.join(' ');
}

/**
 * Sliding window baseline (keep last N messages)
 */
function slidingWindowCompress(messages: Message[], budget: number): Message[] {
  let selected: Message[] = [];
  let tokens = 0;

  // Add messages from end until budget exceeded
  for (let i = messages.length - 1; i >= 0; i--) {
    if (tokens + messages[i].tokens <= budget) {
      selected.unshift(messages[i]);
      tokens += messages[i].tokens;
    } else {
      break;
    }
  }

  return selected;
}

/**
 * Run comprehensive benchmark
 */
function runBenchmark() {
  console.log('=== SemaComp Benchmark ===\n');
  console.log('Comparing SemaComp vs Sliding Window baseline\n');

  const configs = [
    { name: 'Small', messages: 50, density: 0.2 },
    { name: 'Medium', messages: 200, density: 0.3 },
    { name: 'Large', messages: 500, density: 0.4 }
  ];

  for (const config of configs) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`${config.name} Conversation (${config.messages} messages, ${config.density * 100}% dependency density)`);
    console.log('='.repeat(60));

    const messages = generateSyntheticConversation(config.messages, config.density);
    const totalTokens = messages.reduce((sum, m) => sum + m.tokens, 0);
    const budget = totalTokens * 0.3; // 30% compression

    console.log(`\nOriginal conversation:`);
    console.log(`  Messages: ${messages.length}`);
    console.log(`  Total tokens: ${totalTokens.toLocaleString()}`);
    console.log(`  Budget (30%): ${budget.toFixed(0)} tokens`);

    // SemaComp
    console.log(`\n[SemaComp]`);
    const compressor = new SemaComp();
    const startSema = Date.now();
    const semaResult = compressor.compress(messages, budget);
    const elapsedSema = Date.now() - startSema;

    console.log(`  Compressed to: ${semaResult.tokensUsed.toLocaleString()} tokens`);
    console.log(`  Compression ratio: ${(semaResult.compressionRatio * 100).toFixed(1)}%`);
    console.log(`  Messages retained: ${semaResult.messages.length}/${messages.length} (${(semaResult.messages.length / messages.length * 100).toFixed(1)}%)`);
    console.log(`  Time: ${elapsedSema}ms`);

    // Sliding window baseline
    console.log(`\n[Sliding Window Baseline]`);
    const startWindow = Date.now();
    const windowResult = slidingWindowCompress(messages, budget);
    const elapsedWindow = Date.now() - startWindow;
    const windowTokens = windowResult.reduce((sum, m) => sum + m.tokens, 0);

    console.log(`  Compressed to: ${windowTokens.toLocaleString()} tokens`);
    console.log(`  Compression ratio: ${(windowTokens / totalTokens * 100).toFixed(1)}%`);
    console.log(`  Messages retained: ${windowResult.length}/${messages.length} (${(windowResult.length / messages.length * 100).toFixed(1)}%)`);
    console.log(`  Time: ${elapsedWindow}ms`);

    // Comparison
    console.log(`\n[Comparison]`);
    const tokenSavings = windowTokens - semaResult.tokensUsed;
    const messageDiff = semaResult.messages.length - windowResult.length;
    console.log(`  Token savings: ${tokenSavings > 0 ? '+' : ''}${tokenSavings.toFixed(0)} (${((tokenSavings / windowTokens) * 100).toFixed(1)}%)`);
    console.log(`  Message difference: ${messageDiff > 0 ? '+' : ''}${messageDiff} messages`);
    console.log(`  Speedup: ${(elapsedWindow / elapsedSema).toFixed(2)}x ${elapsedSema < elapsedWindow ? 'faster' : 'slower'}`);
  }

  console.log('\n' + '='.repeat(60));
  console.log('=== Benchmark Complete ===');
  console.log('='.repeat(60));
}

/**
 * Simple example demonstrating usage
 */
function example() {
  console.log('=== SemaComp Example ===\n');

  // Create sample conversation
  const messages: Message[] = [
    {
      id: '1',
      role: 'system',
      content: 'You are a helpful assistant.',
      tokens: 10,
      timestamp: 0
    },
    {
      id: '2',
      role: 'user',
      content: 'What is context compression?',
      tokens: 10,
      timestamp: 1
    },
    {
      id: '3',
      role: 'assistant',
      content: 'Context compression is a technique to reduce the token count in conversation history while preserving critical information. This is important for managing costs and latency in LLM applications.',
      tokens: 100,
      timestamp: 2
    },
    {
      id: '4',
      role: 'user',
      content: 'As you mentioned, compression is important. How does it work?',
      tokens: 15,
      timestamp: 3
    },
    {
      id: '5',
      role: 'assistant',
      content: 'There are several approaches: sliding window (keep recent messages), semantic similarity (keep diverse content), dependency graphs (preserve referenced messages), and learned compression using smaller models.',
      tokens: 150,
      timestamp: 4
    },
    {
      id: '6',
      role: 'user',
      content: 'Show me an example of dependency graphs',
      tokens: 12,
      timestamp: 5
    },
    {
      id: '7',
      role: 'assistant',
      content: 'A dependency graph tracks which messages reference others. For instance, your current question depends on my previous explanation about compression approaches. SemaComp builds this graph and preserves critical dependency chains.',
      tokens: 200,
      timestamp: 6
    }
  ];

  const totalTokens = messages.reduce((sum, m) => sum + m.tokens, 0);
  console.log(`Original conversation:`);
  console.log(`  Messages: ${messages.length}`);
  console.log(`  Total tokens: ${totalTokens}`);

  // Compress to 50% of original size
  const budget = totalTokens * 0.5;
  console.log(`\nCompressing to 50% (${budget} tokens)...\n`);

  const compressor = new SemaComp();
  const result = compressor.compress(messages, budget);

  console.log(`Results:`);
  console.log(`  Compressed messages: ${result.messages.length}/${messages.length}`);
  console.log(`  Tokens used: ${result.tokensUsed}/${budget.toFixed(0)}`);
  console.log(`  Compression ratio: ${(result.compressionRatio * 100).toFixed(1)}%`);

  console.log(`\nRetained messages:`);
  for (const m of result.messages) {
    const preview = m.content.length > 60 ? m.content.substring(0, 60) + '...' : m.content;
    console.log(`  [${m.role}] ${m.id}: ${preview}`);
  }

  console.log('\n=== Example Complete ===\n');
}

/**
 * Run ablation study
 */
function ablationStudy() {
  console.log('=== Ablation Study ===\n');
  console.log('Testing impact of each component\n');

  const messages = generateSyntheticConversation(200, 0.3);
  const totalTokens = messages.reduce((sum, m) => sum + m.tokens, 0);
  const budget = totalTokens * 0.3;

  const configs: Array<{name: string, config: Partial<SemaCompConfig>}> = [
    {
      name: 'Full SemaComp',
      config: {}
    },
    {
      name: 'No Coverage (α=0)',
      config: { weights: { alpha: 0, beta: 0.5, gamma: 0.5 } }
    },
    {
      name: 'No Recency (β=0)',
      config: { weights: { alpha: 0.7, beta: 0, gamma: 0.3 } }
    },
    {
      name: 'No Uniqueness (γ=0)',
      config: { weights: { alpha: 0.6, beta: 0.4, gamma: 0 } }
    },
    {
      name: 'No Dependency Expansion',
      config: { dependencyBudget: 0 }
    }
  ];

  for (const { name, config } of configs) {
    const compressor = new SemaComp(config);
    const result = compressor.compress(messages, budget);

    console.log(`${name}:`);
    console.log(`  Messages: ${result.messages.length}`);
    console.log(`  Tokens: ${result.tokensUsed}`);
    console.log(`  Ratio: ${(result.compressionRatio * 100).toFixed(1)}%`);
    console.log();
  }

  console.log('=== Ablation Complete ===');
}

// Export
export {
  SemaComp,
  DependencyGraph,
  type Message,
  type CompressionResult,
  type SemaCompConfig,
  generateSyntheticConversation,
  slidingWindowCompress,
  runBenchmark,
  ablationStudy,
  example
};

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0] || 'example';

  switch (command) {
    case 'example':
      example();
      break;
    case 'benchmark':
      runBenchmark();
      break;
    case 'ablation':
      ablationStudy();
      break;
    case 'all':
      example();
      runBenchmark();
      ablationStudy();
      break;
    default:
      console.log('Usage: node semantic-context-compression.js [example|benchmark|ablation|all]');
  }
}
