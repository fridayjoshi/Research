#!/usr/bin/env ts-node
/**
 * CLI for Adaptive Context Window Manager
 *
 * Quick interface to test ACWM with various configurations
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
  type ACWMConfig
} from './adaptive-context-manager';

// ============================================================================
// CLI Commands
// ============================================================================

function createMessage(content: string, role: 'user' | 'assistant' = 'user'): Message {
  return {
    id: generateId(),
    content,
    tokens: countTokens(content),
    timestamp: Date.now(),
    role
  };
}

function interactiveDemo() {
  console.log('🤖 ACWM Interactive Demo\n');

  const manager = new AdaptiveContextManager(
    {
      capacity: 5000,      // Small capacity to trigger compression quickly
      triggerRatio: 0.7,
      lookahead: 5,
      qualityTarget: 0.8,
      costBudget: 0.05,
      qualityWeight: 0.7
    },
    [
      new SlidingWindowStrategy(),
      new ImportanceSamplingStrategy(),
      new SemaCompStrategy()
    ]
  );

  const conversation = [
    { content: 'Can you help me implement a binary search algorithm?', role: 'user' },
    { content: 'Sure! Here is the implementation: ```function binarySearch(arr, target) { let left = 0; let right = arr.length - 1; while (left <= right) { const mid = Math.floor((left + right) / 2); if (arr[mid] === target) return mid; if (arr[mid] < target) left = mid + 1; else right = mid - 1; } return -1; }```', role: 'assistant' },
    { content: 'What is the time complexity?', role: 'user' },
    { content: 'The time complexity is O(log n) because we halve the search space in each iteration.', role: 'assistant' },
    { content: 'That code above has a bug when the array is empty', role: 'user' },
    { content: 'Good catch! Let me add validation: ```function binarySearch(arr, target) { if (!arr || arr.length === 0) return -1; let left = 0; let right = arr.length - 1; while (left <= right) { const mid = Math.floor((left + right) / 2); if (arr[mid] === target) return mid; if (arr[mid] < target) left = mid + 1; else right = mid - 1; } return -1; }```', role: 'assistant' },
    { content: 'Thanks! Can you explain the space complexity?', role: 'user' },
    { content: 'The space complexity is O(1) since we only use a constant amount of extra space for the pointers.', role: 'assistant' }
  ];

  console.log('Processing technical conversation...\n');

  for (let i = 0; i < conversation.length; i++) {
    const msg = createMessage(conversation[i].content, conversation[i].role as any);

    console.log(`\n[${msg.role.toUpperCase()}]: ${msg.content.substring(0, 60)}${msg.content.length > 60 ? '...' : ''}`);

    manager.onMessageReceived(msg);

    const metrics = manager.getMetrics();
    console.log(`  📊 Tokens: ${metrics.currentTokens}/${metrics.capacity} (${(metrics.utilization * 100).toFixed(1)}%)`);
    console.log(`  📈 Growth rate: ${metrics.growthRate.toFixed(1)} tokens/msg`);

    if (metrics.compressionCount > 0) {
      console.log(`  🗜️  Compressions: ${metrics.compressionCount}`);
      const history = manager.getCompressionHistory();
      const latest = history[history.length - 1];
      console.log(`     Latest: ${latest.strategy} (${latest.originalMessages} → ${latest.compressedMessages} msgs, ${latest.duration}ms)`);
    }
  }

  // Add more messages to trigger compression
  console.log('\n\n🔄 Adding more messages to trigger compression...\n');

  for (let i = 0; i < 30; i++) {
    const msg = createMessage(
      `Additional message ${i}: Let's discuss more about algorithms and data structures. ${'x'.repeat(100)}`,
      i % 2 === 0 ? 'user' : 'assistant'
    );
    manager.onMessageReceived(msg);
  }

  const finalMetrics = manager.getMetrics();

  console.log('\n' + '='.repeat(60));
  console.log('📊 FINAL METRICS');
  console.log('='.repeat(60));
  console.log(`Tokens: ${finalMetrics.currentTokens}/${finalMetrics.capacity}`);
  console.log(`Utilization: ${(finalMetrics.utilization * 100).toFixed(1)}%`);
  console.log(`Compressions: ${finalMetrics.compressionCount}`);
  console.log(`Avg compression ratio: ${(finalMetrics.avgCompressionRatio * 100).toFixed(1)}%`);

  if (Object.keys(finalMetrics.strategyScores).length > 0) {
    console.log('\nStrategy Scores:');
    for (const [strategy, score] of Object.entries(finalMetrics.strategyScores)) {
      console.log(`  ${strategy}: ${score.toFixed(3)}`);
    }
  }

  const history = manager.getCompressionHistory();
  if (history.length > 0) {
    console.log('\nCompression History:');
    for (const event of history) {
      console.log(`  ${event.strategy}: ${event.originalMessages}→${event.compressedMessages} msgs (${event.originalTokens}→${event.compressedTokens} tokens) in ${event.duration}ms`);
      console.log(`    Type: ${event.features.conversationType}, Code: ${event.features.hasCode}, Deps: ${event.features.dependencyCount}`);
    }
  }

  console.log('\n✅ Demo complete!');
}

function showHelp() {
  console.log(`
🤖 ACWM CLI - Adaptive Context Window Manager

Usage:
  npm run acwm [command]

Commands:
  demo        Run interactive demo with sample conversation
  benchmark   Run full benchmark suite (see acwm-cli.bench.ts)
  help        Show this help message

Examples:
  npm run acwm demo
  npm run acwm benchmark

For programmatic usage, see adaptive-context-manager.ts
  `);
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'demo';

  switch (command) {
    case 'demo':
      interactiveDemo();
      break;

    case 'benchmark':
      console.log('Running benchmarks...\n');
      const { runAllBenchmarks } = await import('./adaptive-context-manager.bench');
      runAllBenchmarks();
      break;

    case 'help':
    case '--help':
    case '-h':
      showHelp();
      break;

    default:
      console.error(`Unknown command: ${command}`);
      console.error('Run "npm run acwm help" for usage information');
      process.exit(1);
  }
}

if (require.main === module) {
  main().catch(err => {
    console.error('Error:', err);
    process.exit(1);
  });
}
