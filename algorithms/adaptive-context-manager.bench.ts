/**
 * Benchmarks for Adaptive Context Window Manager
 *
 * Compares ACWM against baseline approaches on realistic conversation datasets
 *
 * @author Friday
 * @date 2026-05-26
 */

import {
  AdaptiveContextManager,
  SlidingWindowStrategy,
  ImportanceSamplingStrategy,
  SemaCompStrategy,
  countTokens,
  generateId,
  type Message,
  type ACWMConfig
} from './adaptive-context-manager';

// ============================================================================
// Benchmark Utilities
// ============================================================================

interface BenchmarkResult {
  method: string;
  dataset: string;
  tokenSavings: number;           // % saved
  messageRetention: number;       // % messages kept
  avgCompressionTime: number;     // ms
  totalTime: number;              // ms
  compressionCount: number;
  strategyBreakdown?: Record<string, number>;
}

function createMessage(content: string, role: 'user' | 'assistant' | 'system' = 'user'): Message {
  return {
    id: generateId(),
    content,
    tokens: countTokens(content),
    timestamp: Date.now() + Math.random() * 1000, // Add jitter
    role
  };
}

// ============================================================================
// Dataset Generators
// ============================================================================

function generateTechnicalConversation(messageCount: number): Message[] {
  const messages: Message[] = [];

  const patterns = [
    { user: 'Can you help me implement {topic}?', assistant: 'Sure! Here is the code: ```{code}```' },
    { user: 'What is the time complexity?', assistant: 'The complexity is O({complexity})' },
    { user: 'That code above has a bug', assistant: 'Good catch! Let me fix: ```{code}```' },
    { user: 'How does {concept} work?', assistant: 'It works by {explanation}...' },
    { user: 'Can you explain {detail}?', assistant: 'The key idea is {explanation}' }
  ];

  const topics = ['binary search', 'merge sort', 'hash table', 'graph traversal', 'dynamic programming'];
  const concepts = ['recursion', 'memoization', 'backtracking', 'greedy algorithms'];
  const complexities = ['n log n', 'n²', 'n', 'log n', '2ⁿ'];

  for (let i = 0; i < messageCount; i++) {
    const pattern = patterns[Math.floor(Math.random() * patterns.length)];
    const role = i % 2 === 0 ? 'user' : 'assistant';
    const template = role === 'user' ? pattern.user : pattern.assistant;

    let content = template
      .replace('{topic}', topics[Math.floor(Math.random() * topics.length)])
      .replace('{concept}', concepts[Math.floor(Math.random() * concepts.length)])
      .replace('{complexity}', complexities[Math.floor(Math.random() * complexities.length)])
      .replace('{code}', `function example() { return ${Math.random()}; }`)
      .replace('{explanation}', 'using a structured approach with careful analysis');

    content += ` [msg ${i}]`;

    messages.push(createMessage(content, role));
  }

  return messages;
}

function generateCasualConversation(messageCount: number): Message[] {
  const messages: Message[] = [];

  const userPhrases = [
    'Hey, how are you?',
    'Thanks, that helps!',
    'Cool, I appreciate it',
    'Yeah, that makes sense',
    'Nice, thank you!',
    'Interesting, tell me more',
    'Got it, understood',
    'Okay, sounds good'
  ];

  const assistantPhrases = [
    'I\'m doing well, how can I help?',
    'You\'re welcome!',
    'Glad I could help!',
    'Let me know if you need anything else',
    'Happy to assist!',
    'Sure thing!',
    'No problem at all',
    'Anytime!'
  ];

  for (let i = 0; i < messageCount; i++) {
    const role = i % 2 === 0 ? 'user' : 'assistant';
    const phrases = role === 'user' ? userPhrases : assistantPhrases;
    const content = phrases[Math.floor(Math.random() * phrases.length)] + ` [msg ${i}]`;

    messages.push(createMessage(content, role));
  }

  return messages;
}

function generateInstructionConversation(messageCount: number): Message[] {
  const messages: Message[] = [];

  const instructions = [
    'Please create a function that {task}',
    'Can you implement {feature}?',
    'I need help with {problem}',
    'Help me build {system}',
    'Show me how to {action}',
    'Explain how to {action}'
  ];

  const tasks = ['sorts an array', 'finds duplicates', 'validates input', 'parses JSON', 'formats dates'];
  const features = ['user authentication', 'data validation', 'error handling', 'logging'];
  const problems = ['debugging this code', 'optimizing performance', 'fixing this bug'];
  const systems = ['a REST API', 'a queue system', 'a cache layer', 'a test suite'];
  const actions = ['use async/await', 'handle errors', 'write unit tests', 'optimize queries'];

  for (let i = 0; i < messageCount; i++) {
    const role = i % 2 === 0 ? 'user' : 'assistant';

    let content;
    if (role === 'user') {
      const template = instructions[Math.floor(Math.random() * instructions.length)];
      content = template
        .replace('{task}', tasks[Math.floor(Math.random() * tasks.length)])
        .replace('{feature}', features[Math.floor(Math.random() * features.length)])
        .replace('{problem}', problems[Math.floor(Math.random() * problems.length)])
        .replace('{system}', systems[Math.floor(Math.random() * systems.length)])
        .replace('{action}', actions[Math.floor(Math.random() * actions.length)]);
    } else {
      content = `Here's how to do that: ${'x'.repeat(50)} [msg ${i}]`;
    }

    messages.push(createMessage(content, role));
  }

  return messages;
}

function generateMixedConversation(messageCount: number): Message[] {
  const messages: Message[] = [];

  // Mix of all types
  const segmentSize = Math.floor(messageCount / 3);

  messages.push(...generateTechnicalConversation(segmentSize));
  messages.push(...generateCasualConversation(segmentSize));
  messages.push(...generateInstructionConversation(messageCount - 2 * segmentSize));

  // Shuffle to mix them up
  for (let i = messages.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [messages[i], messages[j]] = [messages[j], messages[i]];
  }

  return messages;
}

// ============================================================================
// Baseline Methods
// ============================================================================

class FixedSlidingWindowManager {
  private messages: Message[] = [];
  private compressionInterval = 100;
  private keepRatio = 0.7;
  private compressionTimes: number[] = [];

  onMessageReceived(msg: Message): Message[] {
    this.messages.push(msg);

    if (this.messages.length % this.compressionInterval === 0) {
      const start = Date.now();
      const targetCount = Math.floor(this.messages.length * this.keepRatio);
      this.messages = this.messages.slice(-targetCount);
      this.compressionTimes.push(Date.now() - start);
    }

    return this.messages;
  }

  getMetrics() {
    return {
      compressionCount: this.compressionTimes.length,
      avgCompressionTime: this.compressionTimes.length > 0
        ? this.compressionTimes.reduce((a, b) => a + b, 0) / this.compressionTimes.length
        : 0
    };
  }
}

class ReactiveManager {
  private messages: Message[] = [];
  private capacity = 100000;
  private currentTokens = 0;
  private compressionTimes: number[] = [];
  private strategy = new SemaCompStrategy();

  onMessageReceived(msg: Message): Message[] {
    this.messages.push(msg);
    this.currentTokens += msg.tokens;

    if (this.currentTokens > this.capacity * 0.95) {
      const start = Date.now();
      this.messages = this.strategy.compress(this.messages, 0.7);
      this.currentTokens = this.messages.reduce((sum, m) => sum + m.tokens, 0);
      this.compressionTimes.push(Date.now() - start);
    }

    return this.messages;
  }

  getMetrics() {
    return {
      compressionCount: this.compressionTimes.length,
      avgCompressionTime: this.compressionTimes.length > 0
        ? this.compressionTimes.reduce((a, b) => a + b, 0) / this.compressionTimes.length
        : 0
    };
  }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

function runBenchmark(
  name: string,
  messages: Message[],
  method: 'acwm' | 'fixed' | 'reactive'
): BenchmarkResult {
  console.log(`  Running: ${method} on ${name}...`);

  const startTime = Date.now();
  let manager: any;
  let strategyBreakdown: Record<string, number> | undefined;

  const config: ACWMConfig = {
    capacity: 100000,
    triggerRatio: 0.85,
    lookahead: 10,
    qualityTarget: 0.85,
    costBudget: 0.05,
    qualityWeight: 0.7
  };

  if (method === 'acwm') {
    manager = new AdaptiveContextManager(config, [
      new SlidingWindowStrategy(),
      new ImportanceSamplingStrategy(),
      new SemaCompStrategy()
    ]);
  } else if (method === 'fixed') {
    manager = new FixedSlidingWindowManager();
  } else {
    manager = new ReactiveManager();
  }

  const originalTokens = messages.reduce((sum, m) => sum + m.tokens, 0);
  const originalMessages = messages.length;

  for (const msg of messages) {
    manager.onMessageReceived(msg);
  }

  const totalTime = Date.now() - startTime;

  let finalTokens: number;
  let finalMessages: number;
  let compressionCount: number;
  let avgCompressionTime: number;

  if (method === 'acwm') {
    const metrics = manager.getMetrics();
    finalTokens = metrics.currentTokens;
    finalMessages = manager.onMessageReceived(messages[0]).length; // Get current count
    compressionCount = metrics.compressionCount;

    const history = manager.getCompressionHistory();
    avgCompressionTime = history.length > 0
      ? history.reduce((sum: number, e: any) => sum + e.duration, 0) / history.length
      : 0;

    // Calculate strategy breakdown
    strategyBreakdown = {};
    for (const event of history) {
      strategyBreakdown[event.strategy] = (strategyBreakdown[event.strategy] || 0) + 1;
    }
  } else {
    const metrics = manager.getMetrics();
    const currentMessages = manager.onMessageReceived(messages[0]);
    finalTokens = currentMessages.reduce((sum: number, m: Message) => sum + m.tokens, 0);
    finalMessages = currentMessages.length;
    compressionCount = metrics.compressionCount;
    avgCompressionTime = metrics.avgCompressionTime;
  }

  const tokenSavings = ((originalTokens - finalTokens) / originalTokens) * 100;
  const messageRetention = (finalMessages / originalMessages) * 100;

  return {
    method,
    dataset: name,
    tokenSavings,
    messageRetention,
    avgCompressionTime,
    totalTime,
    compressionCount,
    strategyBreakdown
  };
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

function runAllBenchmarks() {
  console.log('🚀 ACWM Benchmark Suite\n');
  console.log('=' .repeat(80));

  const datasets = [
    { name: 'Technical (500 msgs)', generator: () => generateTechnicalConversation(500) },
    { name: 'Casual (800 msgs)', generator: () => generateCasualConversation(800) },
    { name: 'Instruction (300 msgs)', generator: () => generateInstructionConversation(300) },
    { name: 'Mixed (1000 msgs)', generator: () => generateMixedConversation(1000) }
  ];

  const methods: Array<'acwm' | 'fixed' | 'reactive'> = ['acwm', 'fixed', 'reactive'];

  const results: BenchmarkResult[] = [];

  for (const dataset of datasets) {
    console.log(`\n📊 Dataset: ${dataset.name}`);
    console.log('-'.repeat(80));

    const messages = dataset.generator();

    for (const method of methods) {
      const result = runBenchmark(dataset.name, messages, method);
      results.push(result);
    }
  }

  console.log('\n\n' + '='.repeat(80));
  console.log('📈 BENCHMARK RESULTS');
  console.log('='.repeat(80));

  // Group by dataset
  const grouped = new Map<string, BenchmarkResult[]>();
  for (const result of results) {
    if (!grouped.has(result.dataset)) {
      grouped.set(result.dataset, []);
    }
    grouped.get(result.dataset)!.push(result);
  }

  for (const [dataset, datasetResults] of grouped) {
    console.log(`\n${dataset}`);
    console.log('-'.repeat(80));

    console.log(
      '| Method          | Token Savings | Msg Retention | Avg Compress | Total Time | Count |'
    );
    console.log(
      '|-----------------|---------------|---------------|--------------|------------|-------|'
    );

    for (const result of datasetResults) {
      const method = result.method.padEnd(15);
      const savings = `${result.tokenSavings.toFixed(1)}%`.padEnd(13);
      const retention = `${result.messageRetention.toFixed(1)}%`.padEnd(13);
      const avgTime = `${result.avgCompressionTime.toFixed(0)}ms`.padEnd(12);
      const totalTime = `${result.totalTime}ms`.padEnd(10);
      const count = result.compressionCount.toString().padEnd(5);

      console.log(
        `| ${method} | ${savings} | ${retention} | ${avgTime} | ${totalTime} | ${count} |`
      );
    }

    // Show strategy breakdown for ACWM
    const acwmResult = datasetResults.find(r => r.method === 'acwm');
    if (acwmResult?.strategyBreakdown) {
      console.log('\nACWM Strategy Breakdown:');
      for (const [strategy, count] of Object.entries(acwmResult.strategyBreakdown)) {
        const percentage = (count / acwmResult.compressionCount * 100).toFixed(1);
        console.log(`  - ${strategy}: ${count} (${percentage}%)`);
      }
    }
  }

  // Summary statistics
  console.log('\n\n' + '='.repeat(80));
  console.log('📊 SUMMARY');
  console.log('='.repeat(80));

  const acwmResults = results.filter(r => r.method === 'acwm');
  const fixedResults = results.filter(r => r.method === 'fixed');
  const reactiveResults = results.filter(r => r.method === 'reactive');

  const avgTokenSavings = (results: BenchmarkResult[]) =>
    results.reduce((sum, r) => sum + r.tokenSavings, 0) / results.length;

  const avgRetention = (results: BenchmarkResult[]) =>
    results.reduce((sum, r) => sum + r.messageRetention, 0) / results.length;

  const avgTime = (results: BenchmarkResult[]) =>
    results.reduce((sum, r) => sum + r.avgCompressionTime, 0) / results.length;

  console.log('\nAverage Token Savings:');
  console.log(`  ACWM:     ${avgTokenSavings(acwmResults).toFixed(1)}%`);
  console.log(`  Fixed:    ${avgTokenSavings(fixedResults).toFixed(1)}%`);
  console.log(`  Reactive: ${avgTokenSavings(reactiveResults).toFixed(1)}%`);

  console.log('\nAverage Message Retention:');
  console.log(`  ACWM:     ${avgRetention(acwmResults).toFixed(1)}%`);
  console.log(`  Fixed:    ${avgRetention(fixedResults).toFixed(1)}%`);
  console.log(`  Reactive: ${avgRetention(reactiveResults).toFixed(1)}%`);

  console.log('\nAverage Compression Time:');
  console.log(`  ACWM:     ${avgTime(acwmResults).toFixed(0)}ms`);
  console.log(`  Fixed:    ${avgTime(fixedResults).toFixed(0)}ms`);
  console.log(`  Reactive: ${avgTime(reactiveResults).toFixed(0)}ms`);

  console.log('\n' + '='.repeat(80));
  console.log('✅ Benchmarks complete!');
  console.log('='.repeat(80));

  return results;
}

// ============================================================================
// Run
// ============================================================================

if (require.main === module) {
  runAllBenchmarks();
}

export { runAllBenchmarks };
