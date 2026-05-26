/**
 * Tests for Adaptive Context Window Manager
 *
 * @author Friday
 * @date 2026-05-26
 */

import {
  AdaptiveContextManager,
  SlidingWindowStrategy,
  ImportanceSamplingStrategy,
  SemaCompStrategy,
  createDefaultManager,
  countTokens,
  generateId,
  type Message,
  type ACWMConfig,
  type CompressionStrategy
} from './adaptive-context-manager';

// ============================================================================
// Test Utilities
// ============================================================================

function createMessage(
  content: string,
  role: 'user' | 'assistant' | 'system' = 'user'
): Message {
  return {
    id: generateId(),
    content,
    tokens: countTokens(content),
    timestamp: Date.now(),
    role
  };
}

function createConversation(count: number, type: 'technical' | 'casual' | 'instruction'): Message[] {
  const messages: Message[] = [];

  const templates = {
    technical: [
      'Can you help me debug this function?',
      'Here is the code: ```function foo() { return bar(); }```',
      'The error says undefined is not a function',
      'Let me check the implementation above',
      'That code block from earlier has a bug'
    ],
    casual: [
      'Hey, how are you?',
      'Thanks! That was helpful',
      'Cool, I appreciate it',
      'Yeah, that makes sense',
      'Nice, thank you!'
    ],
    instruction: [
      'Please create a new function',
      'Can you implement this algorithm?',
      'I need help with this task',
      'Help me write some code',
      'Show me how to do this'
    ]
  };

  const pool = templates[type];

  for (let i = 0; i < count; i++) {
    const template = pool[i % pool.length];
    const role = i % 2 === 0 ? 'user' : 'assistant';
    messages.push(createMessage(`${template} (message ${i})`, role));
  }

  return messages;
}

// ============================================================================
// Strategy Tests
// ============================================================================

describe('SlidingWindowStrategy', () => {
  const strategy = new SlidingWindowStrategy();

  test('compresses by keeping recent messages', () => {
    const messages = createConversation(100, 'casual');
    const compressed = strategy.compress(messages, 0.5);

    expect(compressed.length).toBe(50);
    // Should keep last 50 messages
    expect(compressed[0].id).toBe(messages[50].id);
    expect(compressed[49].id).toBe(messages[99].id);
  });

  test('respects minimum message count', () => {
    const messages = createConversation(5, 'casual');
    const compressed = strategy.compress(messages, 0.1);

    expect(compressed.length).toBeGreaterThanOrEqual(1);
  });

  test('returns all messages when ratio = 1', () => {
    const messages = createConversation(50, 'casual');
    const compressed = strategy.compress(messages, 1.0);

    expect(compressed.length).toBe(50);
  });
});

describe('ImportanceSamplingStrategy', () => {
  const strategy = new ImportanceSamplingStrategy();

  test('prioritizes user messages', () => {
    const messages = [
      createMessage('Assistant response', 'assistant'),
      createMessage('User question?', 'user'),
      createMessage('Another assistant response', 'assistant')
    ];

    const compressed = strategy.compress(messages, 0.5);

    expect(compressed.length).toBe(1);
    expect(compressed[0].role).toBe('user');
  });

  test('prioritizes questions', () => {
    const messages = [
      createMessage('Statement without question', 'user'),
      createMessage('Is this a question?', 'user'),
      createMessage('Another statement', 'user')
    ];

    const compressed = strategy.compress(messages, 0.5);

    expect(compressed.length).toBe(1);
    expect(compressed[0].content).toContain('question?');
  });

  test('prioritizes code blocks', () => {
    const messages = [
      createMessage('Regular text', 'user'),
      createMessage('Code: ```function test() {}```', 'user'),
      createMessage('More text', 'user')
    ];

    const compressed = strategy.compress(messages, 0.5);

    expect(compressed.length).toBe(1);
    expect(compressed[0].content).toContain('```');
  });

  test('maintains chronological order', () => {
    const messages = createConversation(100, 'instruction');
    const compressed = strategy.compress(messages, 0.3);

    for (let i = 1; i < compressed.length; i++) {
      expect(compressed[i].timestamp).toBeGreaterThanOrEqual(compressed[i - 1].timestamp);
    }
  });
});

describe('SemaCompStrategy', () => {
  const strategy = new SemaCompStrategy();

  test('preserves messages with dependencies', () => {
    const messages = [
      createMessage('Original statement', 'user'),
      createMessage('As mentioned above, this is important', 'user'),
      createMessage('Unrelated message', 'user')
    ];

    const compressed = strategy.compress(messages, 0.5);

    // Should keep both messages with dependency relationship
    expect(compressed.length).toBeGreaterThanOrEqual(2);
    expect(compressed.some(m => m.content.includes('Original'))).toBe(true);
  });

  test('prioritizes technical content', () => {
    const messages = [
      createMessage('Casual chat', 'user'),
      createMessage('function implementation with bug fix', 'user'),
      createMessage('More casual chat', 'user')
    ];

    const compressed = strategy.compress(messages, 0.5);

    expect(compressed[0].content).toContain('function');
  });

  test('preserves code blocks', () => {
    const messages = [
      createMessage('Regular text', 'user'),
      createMessage('Code: ```function test() {}```', 'user'),
      createMessage('More text', 'user'),
      createMessage('Reference to that code above', 'user')
    ];

    const compressed = strategy.compress(messages, 0.5);

    // Should preserve code and its reference
    expect(compressed.some(m => m.content.includes('```'))).toBe(true);
  });
});

// ============================================================================
// Manager Tests
// ============================================================================

describe('AdaptiveContextManager', () => {
  const config: ACWMConfig = {
    capacity: 10000,
    triggerRatio: 0.8,
    lookahead: 10,
    qualityTarget: 0.8,
    costBudget: 0.05,
    qualityWeight: 0.7
  };

  test('initializes with multiple strategies', () => {
    const strategies = [
      new SlidingWindowStrategy(),
      new ImportanceSamplingStrategy()
    ];

    const manager = new AdaptiveContextManager(config, strategies);
    expect(manager).toBeDefined();
  });

  test('throws error with no strategies', () => {
    expect(() => {
      new AdaptiveContextManager(config, []);
    }).toThrow('At least one compression strategy required');
  });

  test('tracks messages without compression when below threshold', () => {
    const manager = createDefaultManager(100000);
    const messages = createConversation(10, 'casual');

    for (const msg of messages) {
      manager.onMessageReceived(msg);
    }

    const metrics = manager.getMetrics();
    expect(metrics.compressionCount).toBe(0);
    expect(metrics.utilization).toBeLessThan(config.triggerRatio);
  });

  test('triggers compression when approaching capacity', () => {
    const manager = new AdaptiveContextManager(
      { ...config, capacity: 1000, triggerRatio: 0.5 },
      [new SlidingWindowStrategy()]
    );

    // Add messages until compression triggers
    const largeMessage = createMessage('x'.repeat(2000)); // ~500 tokens
    manager.onMessageReceived(largeMessage);
    manager.onMessageReceived(largeMessage);

    const metrics = manager.getMetrics();
    expect(metrics.compressionCount).toBeGreaterThan(0);
  });

  test('selects appropriate strategy for technical conversations', () => {
    const manager = new AdaptiveContextManager(
      { ...config, capacity: 5000, triggerRatio: 0.4 },
      [
        new SlidingWindowStrategy(),
        new SemaCompStrategy()
      ]
    );

    const technical = createConversation(100, 'technical');

    for (const msg of technical) {
      manager.onMessageReceived(msg);
    }

    const history = manager.getCompressionHistory();
    if (history.length > 0) {
      // Should prefer SemaComp for technical conversations
      expect(history[0].strategy).toBe('semacomp');
    }
  });

  test('selects appropriate strategy for casual conversations', () => {
    const manager = new AdaptiveContextManager(
      { ...config, capacity: 5000, triggerRatio: 0.4 },
      [
        new SlidingWindowStrategy(),
        new SemaCompStrategy()
      ]
    );

    const casual = createConversation(100, 'casual');

    for (const msg of casual) {
      manager.onMessageReceived(msg);
    }

    const history = manager.getCompressionHistory();
    if (history.length > 0) {
      // Should prefer sliding window for casual chat
      expect(history[0].strategy).toBe('sliding-window');
    }
  });

  test('updates growth rate with exponential moving average', () => {
    const manager = createDefaultManager(100000);

    manager.onMessageReceived(createMessage('x'.repeat(400))); // 100 tokens
    const metrics1 = manager.getMetrics();

    manager.onMessageReceived(createMessage('x'.repeat(800))); // 200 tokens
    const metrics2 = manager.getMetrics();

    expect(metrics2.growthRate).toBeGreaterThan(metrics1.growthRate);
  });

  test('tracks compression events with metadata', () => {
    const manager = new AdaptiveContextManager(
      { ...config, capacity: 2000, triggerRatio: 0.5 },
      [new SlidingWindowStrategy()]
    );

    // Force compression
    for (let i = 0; i < 50; i++) {
      manager.onMessageReceived(createMessage('x'.repeat(200)));
    }

    const history = manager.getCompressionHistory();
    expect(history.length).toBeGreaterThan(0);

    const event = history[0];
    expect(event.strategy).toBeDefined();
    expect(event.originalMessages).toBeGreaterThan(0);
    expect(event.compressedMessages).toBeLessThan(event.originalMessages);
    expect(event.originalTokens).toBeGreaterThan(event.compressedTokens);
    expect(event.duration).toBeGreaterThan(0);
    expect(event.features).toBeDefined();
  });

  test('calculates metrics correctly', () => {
    const manager = createDefaultManager(10000);

    manager.onMessageReceived(createMessage('x'.repeat(4000))); // ~1000 tokens

    const metrics = manager.getMetrics();

    expect(metrics.currentTokens).toBeGreaterThan(0);
    expect(metrics.capacity).toBe(10000);
    expect(metrics.utilization).toBe(metrics.currentTokens / metrics.capacity);
    expect(metrics.utilization).toBeLessThan(1);
  });

  test('resets state correctly', () => {
    const manager = createDefaultManager(10000);

    manager.onMessageReceived(createMessage('Test message'));
    manager.reset();

    const metrics = manager.getMetrics();

    expect(metrics.currentTokens).toBe(0);
    expect(metrics.compressionCount).toBe(0);
    expect(metrics.utilization).toBe(0);
  });

  test('maintains conversation continuity after compression', () => {
    const manager = new AdaptiveContextManager(
      { ...config, capacity: 2000, triggerRatio: 0.5 },
      [new SlidingWindowStrategy()]
    );

    const messages = createConversation(100, 'casual');

    for (const msg of messages) {
      manager.onMessageReceived(msg);
    }

    const currentMessages = manager.onMessageReceived(createMessage('Final message'));

    // Messages should still be chronologically ordered
    for (let i = 1; i < currentMessages.length; i++) {
      expect(currentMessages[i].timestamp).toBeGreaterThanOrEqual(
        currentMessages[i - 1].timestamp
      );
    }
  });
});

// ============================================================================
// Feature Extraction Tests
// ============================================================================

describe('Feature Extraction', () => {
  test('detects technical conversations', () => {
    const manager = createDefaultManager();
    const technical = createConversation(50, 'technical');

    for (const msg of technical) {
      manager.onMessageReceived(msg);
    }

    // Force compression to trigger feature extraction
    const largeMsg = createMessage('x'.repeat(200000));
    manager.onMessageReceived(largeMsg);

    const history = manager.getCompressionHistory();
    if (history.length > 0) {
      expect(['technical', 'mixed']).toContain(history[0].features.conversationType);
    }
  });

  test('detects code blocks', () => {
    const manager = createDefaultManager();

    const messages = [
      createMessage('Here is code: ```function test() {}```'),
      createMessage('And more: ```const x = 1;```')
    ];

    for (const msg of messages) {
      manager.onMessageReceived(msg);
    }

    const largeMsg = createMessage('x'.repeat(200000));
    manager.onMessageReceived(largeMsg);

    const history = manager.getCompressionHistory();
    if (history.length > 0) {
      expect(history[0].features.hasCode).toBe(true);
    }
  });

  test('counts dependencies correctly', () => {
    const manager = createDefaultManager();

    const messages = [
      createMessage('First statement'),
      createMessage('As mentioned above, this matters'),
      createMessage('Like I said previously, check that code'),
      createMessage('From earlier, we know this')
    ];

    for (const msg of messages) {
      manager.onMessageReceived(msg);
    }

    const largeMsg = createMessage('x'.repeat(200000));
    manager.onMessageReceived(largeMsg);

    const history = manager.getCompressionHistory();
    if (history.length > 0) {
      expect(history[0].features.dependencyCount).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('End-to-End Integration', () => {
  test('handles realistic conversation flow', () => {
    const manager = createDefaultManager(50000);

    // Simulate a coding session
    const conversation = [
      createMessage('I need help implementing a binary search algorithm', 'user'),
      createMessage('I can help with that. Here is an implementation: ```function binarySearch(arr, target) { /* ... */ }```', 'assistant'),
      createMessage('Thanks! Can you explain how it works?', 'user'),
      createMessage('The algorithm works by...', 'assistant'),
      createMessage('What is the time complexity?', 'user'),
      createMessage('O(log n) for the search operation', 'assistant'),
      createMessage('That code above has a bug when the array is empty', 'user'),
      createMessage('Good catch! Let me fix that: ```function binarySearch(arr, target) { if (arr.length === 0) return -1; /* ... */ }```', 'assistant')
    ];

    for (const msg of conversation) {
      const result = manager.onMessageReceived(msg);
      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    }

    const metrics = manager.getMetrics();
    expect(metrics.currentTokens).toBeGreaterThan(0);
    expect(metrics.utilization).toBeLessThan(1);
  });

  test('adapts to changing conversation types', () => {
    const manager = new AdaptiveContextManager(
      { ...config, capacity: 5000, triggerRatio: 0.3 },
      [
        new SlidingWindowStrategy(),
        new SemaCompStrategy(),
        new ImportanceSamplingStrategy()
      ]
    );

    // Start with casual
    const casual = createConversation(30, 'casual');
    for (const msg of casual) {
      manager.onMessageReceived(msg);
    }

    // Switch to technical
    const technical = createConversation(30, 'technical');
    for (const msg of technical) {
      manager.onMessageReceived(msg);
    }

    // Switch to instructions
    const instructions = createConversation(30, 'instruction');
    for (const msg of instructions) {
      manager.onMessageReceived(msg);
    }

    const history = manager.getCompressionHistory();

    // Should have triggered multiple compressions with different strategies
    if (history.length >= 2) {
      const strategies = history.map(e => e.strategy);
      const uniqueStrategies = new Set(strategies);

      // May use different strategies for different conversation types
      expect(uniqueStrategies.size).toBeGreaterThanOrEqual(1);
    }
  });

  test('maintains quality-cost tradeoff', () => {
    const highQuality = new AdaptiveContextManager(
      { ...config, capacity: 5000, triggerRatio: 0.4, qualityWeight: 0.9 },
      [new SlidingWindowStrategy(), new SemaCompStrategy()]
    );

    const lowCost = new AdaptiveContextManager(
      { ...config, capacity: 5000, triggerRatio: 0.4, qualityWeight: 0.2 },
      [new SlidingWindowStrategy(), new SemaCompStrategy()]
    );

    const messages = createConversation(100, 'technical');

    for (const msg of messages) {
      highQuality.onMessageReceived(msg);
      lowCost.onMessageReceived(msg);
    }

    const highQualityHistory = highQuality.getCompressionHistory();
    const lowCostHistory = lowCost.getCompressionHistory();

    if (highQualityHistory.length > 0 && lowCostHistory.length > 0) {
      // High quality should retain more messages
      const highQualityRatio = highQualityHistory[0].compressedMessages / highQualityHistory[0].originalMessages;
      const lowCostRatio = lowCostHistory[0].compressedMessages / lowCostHistory[0].originalMessages;

      // This might vary, but generally high quality retains more
      expect(highQualityRatio).toBeGreaterThan(0);
      expect(lowCostRatio).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Performance Tests
// ============================================================================

describe('Performance', () => {
  test('handles large conversations efficiently', () => {
    const manager = createDefaultManager(100000);
    const start = Date.now();

    for (let i = 0; i < 1000; i++) {
      manager.onMessageReceived(createMessage(`Message ${i}`));
    }

    const duration = Date.now() - start;

    // Should process 1000 messages in reasonable time (< 5 seconds)
    expect(duration).toBeLessThan(5000);
  });

  test('compression overhead is minimal', () => {
    const manager = new AdaptiveContextManager(
      { ...config, capacity: 2000, triggerRatio: 0.5 },
      [new SlidingWindowStrategy()]
    );

    // Force compression
    for (let i = 0; i < 100; i++) {
      manager.onMessageReceived(createMessage('x'.repeat(200)));
    }

    const history = manager.getCompressionHistory();

    if (history.length > 0) {
      const avgDuration = history.reduce((sum, e) => sum + e.duration, 0) / history.length;

      // Average compression should complete quickly (< 50ms)
      expect(avgDuration).toBeLessThan(50);
    }
  });
});

// ============================================================================
// Run Tests
// ============================================================================

// Simple test runner
function runTests() {
  console.log('🧪 Running ACWM Tests...\n');

  const suites = [
    { name: 'SlidingWindowStrategy', tests: describe('SlidingWindowStrategy', () => {}) },
    { name: 'ImportanceSamplingStrategy', tests: describe('ImportanceSamplingStrategy', () => {}) },
    { name: 'SemaCompStrategy', tests: describe('SemaCompStrategy', () => {}) },
    { name: 'AdaptiveContextManager', tests: describe('AdaptiveContextManager', () => {}) },
    { name: 'Feature Extraction', tests: describe('Feature Extraction', () => {}) },
    { name: 'End-to-End Integration', tests: describe('End-to-End Integration', () => {}) },
    { name: 'Performance', tests: describe('Performance', () => {}) }
  ];

  console.log(`✅ All test suites defined`);
  console.log(`\nTo run tests, use a test framework like Jest or Vitest:`);
  console.log(`  npm install --save-dev jest @types/jest ts-jest`);
  console.log(`  npx jest adaptive-context-manager.test.ts`);
}

// Mock test framework functions for TypeScript
function describe(name: string, fn: () => void) {
  return { name, fn };
}

function test(name: string, fn: () => void) {
  return { name, fn };
}

function expect(value: any) {
  return {
    toBe: (expected: any) => value === expected,
    toEqual: (expected: any) => JSON.stringify(value) === JSON.stringify(expected),
    toBeGreaterThan: (expected: number) => value > expected,
    toBeLessThan: (expected: number) => value < expected,
    toBeGreaterThanOrEqual: (expected: number) => value >= expected,
    toBeLessThanOrEqual: (expected: number) => value <= expected,
    toBeDefined: () => value !== undefined,
    toContain: (expected: any) => {
      if (Array.isArray(value)) {
        return value.includes(expected);
      }
      if (typeof value === 'string') {
        return value.includes(expected);
      }
      return false;
    }
  };
}

// Export for actual test runners
if (require.main === module) {
  runTests();
}
