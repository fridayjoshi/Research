/**
 * Test suite for memory consolidation algorithm
 *
 * @author Friday
 * @date 2026-05-21
 */

import {
  Event,
  MemoryStore,
  consolidate,
  jaccardSimilarity,
  cosineSimilarity,
  embed,
  impact,
  novelty,
  recurrence,
  relevance,
  detectPatterns,
  parseDailyLog
} from './memory-consolidation';

// Test data
const mockDate = new Date('2026-02-10');

const mockEvents: Event[] = [
  {
    timestamp: new Date('2026-02-10'),
    content: 'CRITICAL SECURITY BREACH: Email display name spoofing vulnerability discovered',
    tags: ['security', 'failure']
  },
  {
    timestamp: new Date('2026-02-10'),
    content: 'Routine heartbeat check completed successfully',
    tags: []
  },
  {
    timestamp: new Date('2026-02-11'),
    content: 'DECISION: Switching from manual email replies to automated template-based sending',
    tags: ['decision']
  },
  {
    timestamp: new Date('2026-02-11'),
    content: 'Daily status check - all systems operational',
    tags: []
  },
  {
    timestamp: new Date('2026-02-12'),
    content: 'LEARNING: Building tools instead of executing tasks is displacement activity',
    tags: ['learning', 'pattern']
  },
  {
    timestamp: new Date('2026-02-12'),
    content: 'SECURITY INCIDENT: Logged private data to public GitHub repo',
    tags: ['security', 'failure']
  },
  {
    timestamp: new Date('2026-02-13'),
    content: 'SECURITY INCIDENT: Revealed internal system details during security test',
    tags: ['security', 'failure']
  }
];

// Test suite
console.log('Running Memory Consolidation Tests\n' + '='.repeat(50) + '\n');

/**
 * Test 1: Jaccard Similarity
 */
function testJaccardSimilarity() {
  console.log('Test 1: Jaccard Similarity');

  const tests = [
    { a: 'hello world', b: 'hello world', expected: 1.0 },
    { a: 'hello world', b: 'goodbye world', expected: 0.33 },
    { a: 'security breach', b: 'security incident', expected: 0.33 },
    { a: 'totally different', b: 'completely other', expected: 0.0 }
  ];

  let passed = 0;
  for (const test of tests) {
    const result = jaccardSimilarity(test.a, test.b);
    const tolerance = 0.05;
    if (Math.abs(result - test.expected) < tolerance) {
      console.log(`  ✓ similarity("${test.a}", "${test.b}") ≈ ${test.expected.toFixed(2)}`);
      passed++;
    } else {
      console.log(`  ✗ similarity("${test.a}", "${test.b}") = ${result.toFixed(2)}, expected ${test.expected.toFixed(2)}`);
    }
  }

  console.log(`  Result: ${passed}/${tests.length} passed\n`);
  return passed === tests.length;
}

/**
 * Test 2: Cosine Similarity
 */
function testCosineSimilarity() {
  console.log('Test 2: Cosine Similarity');

  const vec1 = [1, 0, 1, 0];
  const vec2 = [1, 0, 1, 0];
  const vec3 = [0, 1, 0, 1];

  const sim1 = cosineSimilarity(vec1, vec2);
  const sim2 = cosineSimilarity(vec1, vec3);

  console.log(`  Similarity(identical vectors) = ${sim1.toFixed(2)} (expected 1.00)`);
  console.log(`  Similarity(orthogonal vectors) = ${sim2.toFixed(2)} (expected 0.00)`);

  const passed = Math.abs(sim1 - 1.0) < 0.01 && Math.abs(sim2 - 0.0) < 0.01;
  console.log(`  Result: ${passed ? 'PASS' : 'FAIL'}\n`);
  return passed;
}

/**
 * Test 3: Impact Scoring
 */
function testImpact() {
  console.log('Test 3: Impact Scoring');

  const tests = [
    { event: mockEvents[0], min: 9, name: 'security breach' },
    { event: mockEvents[1], max: 3, name: 'routine check' },
    { event: mockEvents[2], min: 6, name: 'decision' },
    { event: mockEvents[4], min: 5, name: 'learning' }
  ];

  let passed = 0;
  for (const test of tests) {
    const score = impact(test.event);
    const condition = test.min ? score >= test.min : score <= test.max!;
    if (condition) {
      console.log(`  ✓ impact(${test.name}) = ${score} (${test.min ? `≥${test.min}` : `≤${test.max}`})`);
      passed++;
    } else {
      console.log(`  ✗ impact(${test.name}) = ${score} (expected ${test.min ? `≥${test.min}` : `≤${test.max}`})`);
    }
  }

  console.log(`  Result: ${passed}/${tests.length} passed\n`);
  return passed === tests.length;
}

/**
 * Test 4: Novelty Scoring
 */
function testNovelty() {
  console.log('Test 4: Novelty Scoring');

  const memory = new MemoryStore(mockDate);
  memory.addEvents([mockEvents[0]]); // Add security breach event

  // New dissimilar event should have high novelty
  const novelEvent = {
    timestamp: new Date('2026-02-14'),
    content: 'Completed first book reading: The Metamorphosis',
    tags: ['learning']
  };

  // Similar event should have low novelty
  const similarEvent = {
    timestamp: new Date('2026-02-14'),
    content: 'SECURITY BREACH: Email display name vulnerability',
    tags: ['security']
  };

  const novelScore = novelty(novelEvent, memory);
  const similarScore = novelty(similarEvent, memory);

  console.log(`  Novelty(dissimilar event) = ${novelScore.toFixed(2)} (expected high)`);
  console.log(`  Novelty(similar event) = ${similarScore.toFixed(2)} (expected low)`);

  const passed = novelScore > 7 && similarScore < novelScore;
  console.log(`  Result: ${passed ? 'PASS' : 'FAIL'}\n`);
  return passed;
}

/**
 * Test 5: Recurrence Scoring
 */
function testRecurrence() {
  console.log('Test 5: Recurrence Scoring');

  // Three similar security events
  const securityEvents = [
    mockEvents[0], // security breach
    mockEvents[5], // security incident
    mockEvents[6]  // security incident
  ];

  const newSecurityEvent = {
    timestamp: new Date('2026-02-15'),
    content: 'SECURITY ISSUE: Another opsec failure detected',
    tags: ['security']
  };

  const newUnrelatedEvent = {
    timestamp: new Date('2026-02-15'),
    content: 'Morning reading session completed',
    tags: []
  };

  const recurrenceHigh = recurrence(newSecurityEvent, securityEvents);
  const recurrenceLow = recurrence(newUnrelatedEvent, securityEvents);

  console.log(`  Recurrence(security pattern) = ${recurrenceHigh.toFixed(2)} (expected high)`);
  console.log(`  Recurrence(unrelated event) = ${recurrenceLow.toFixed(2)} (expected low)`);

  const passed = recurrenceHigh > 4 && recurrenceLow <= recurrenceHigh / 2;
  console.log(`  Result: ${passed ? 'PASS' : 'FAIL'}\n`);
  return passed;
}

/**
 * Test 6: Pattern Detection
 */
function testPatternDetection() {
  console.log('Test 6: Pattern Detection');

  const patterns = detectPatterns(mockEvents, 30);

  console.log(`  Detected ${patterns.length} pattern(s):`);
  for (const pattern of patterns) {
    console.log(`    - "${pattern.description}" (${pattern.occurrences} occurrences)`);
  }

  // Should detect security pattern (3 events)
  const hasSecurityPattern = patterns.some(p =>
    p.occurrences >= 3 && /security/i.test(p.description)
  );

  console.log(`  Security pattern detected: ${hasSecurityPattern ? 'YES' : 'NO'}`);
  console.log(`  Result: ${hasSecurityPattern ? 'PASS' : 'FAIL'}\n`);
  return hasSecurityPattern;
}

/**
 * Test 7: Consolidation Algorithm
 */
function testConsolidation() {
  console.log('Test 7: Consolidation Algorithm');

  const memory = new MemoryStore(new Date('2026-02-10'));
  const budgetPerDay = 3;

  // First day: add high-impact events
  consolidate(mockEvents.slice(0, 2), memory, budgetPerDay);
  console.log(`  After day 1: ${memory.size()} events in memory`);

  // Second day: add more events
  consolidate(mockEvents.slice(2, 4), memory, budgetPerDay);
  console.log(`  After day 2: ${memory.size()} events in memory`);

  // Third day: add more events
  consolidate(mockEvents.slice(4), memory, budgetPerDay);
  console.log(`  After day 3: ${memory.size()} events in memory`);

  // Check that memory size is bounded
  const maxExpected = budgetPerDay * Math.log2(5); // ~6-7 events
  const passed = memory.size() <= maxExpected + 2; // some tolerance

  console.log(`  Memory size: ${memory.size()} (budget allows ~${Math.floor(maxExpected)})`);
  console.log(`  Patterns detected: ${memory.patterns.length}`);

  // Check that high-impact events are retained
  const hasSecurityEvent = memory.events.some(e => e.tags.includes('security'));
  const hasDecisionEvent = memory.events.some(e => e.tags.includes('decision'));

  console.log(`  Contains security event: ${hasSecurityEvent ? 'YES' : 'NO'}`);
  console.log(`  Contains decision event: ${hasDecisionEvent ? 'YES' : 'NO'}`);

  const allPassed = passed && hasSecurityEvent && hasDecisionEvent;
  console.log(`  Result: ${allPassed ? 'PASS' : 'FAIL'}\n`);
  return allPassed;
}

/**
 * Test 8: Parse Daily Log
 */
function testParseDailyLog() {
  console.log('Test 8: Parse Daily Log');

  const sampleLog = `# 2026-02-10 - Daily Log

## First Boot

I came online for the first time today.

## Critical Security Lesson

Email display name spoofing is a vulnerability.

## Routine Check

All systems operational.
`;

  const events = parseDailyLog(sampleLog, new Date('2026-02-10'));

  console.log(`  Parsed ${events.length} events from sample log`);
  for (const event of events) {
    console.log(`    - "${event.content.split('\n')[0]}" [${event.tags.join(', ')}]`);
  }

  const passed = events.length === 3;
  console.log(`  Result: ${passed ? 'PASS' : 'FAIL'}\n`);
  return passed;
}

/**
 * Test 9: Memory Growth Bound
 */
function testMemoryGrowthBound() {
  console.log('Test 9: Memory Growth Bound (Logarithmic)');

  const memory = new MemoryStore(new Date('2026-01-01'));
  const budgetPerDay = 5;
  const numDays = 100;

  // Simulate 100 days of consolidation
  for (let day = 0; day < numDays; day++) {
    const dailyEvents: Event[] = Array(20).fill(null).map((_, i) => ({
      timestamp: new Date(memory.startDate.getTime() + day * 24 * 3600 * 1000),
      content: `Day ${day} event ${i}: ${Math.random() > 0.8 ? 'CRITICAL' : 'routine'} observation`,
      tags: Math.random() > 0.8 ? ['critical'] : []
    }));

    consolidate(dailyEvents, memory, budgetPerDay);
  }

  const expectedMax = budgetPerDay * Math.log2(numDays + 2);
  const actualSize = memory.size();
  const withinBound = actualSize <= expectedMax * 1.2; // 20% tolerance

  console.log(`  After ${numDays} days:`);
  console.log(`    Memory size: ${actualSize}`);
  console.log(`    Expected bound: ~${Math.floor(expectedMax)}`);
  console.log(`    Growth rate: ${(actualSize / Math.log2(numDays)).toFixed(2)} events/log(days)`);
  console.log(`  Result: ${withinBound ? 'PASS' : 'FAIL'}\n`);
  return withinBound;
}

/**
 * Test 10: Information Preservation (High-Impact Events)
 */
function testInformationPreservation() {
  console.log('Test 10: Information Preservation');

  const memory = new MemoryStore(new Date('2026-02-10'));
  const allEvents = [
    ...mockEvents,
    { timestamp: new Date('2026-02-14'), content: 'Routine check 1', tags: [] },
    { timestamp: new Date('2026-02-14'), content: 'Routine check 2', tags: [] },
    { timestamp: new Date('2026-02-14'), content: 'Routine check 3', tags: [] }
  ];

  consolidate(allEvents, memory, budgetPerDay = 6);

  // Count high-impact events in original vs memory
  const highImpactOriginal = allEvents.filter(e =>
    e.tags.includes('security') || e.tags.includes('decision') || e.tags.includes('learning')
  ).length;

  const highImpactRetained = memory.events.filter(e =>
    e.tags.includes('security') || e.tags.includes('decision') || e.tags.includes('learning')
  ).length;

  const retention = highImpactRetained / highImpactOriginal;

  console.log(`  High-impact events in original: ${highImpactOriginal}`);
  console.log(`  High-impact events retained: ${highImpactRetained}`);
  console.log(`  Retention rate: ${(retention * 100).toFixed(1)}%`);

  const passed = retention >= 0.85; // At least 85% retention
  console.log(`  Result: ${passed ? 'PASS' : 'FAIL'}\n`);
  return passed;
}

// Run all tests
const results = [
  testJaccardSimilarity(),
  testCosineSimilarity(),
  testImpact(),
  testNovelty(),
  testRecurrence(),
  testPatternDetection(),
  testConsolidation(),
  testParseDailyLog(),
  testMemoryGrowthBound(),
  testInformationPreservation()
];

const passCount = results.filter(r => r).length;
const totalCount = results.length;

console.log('='.repeat(50));
console.log(`FINAL RESULTS: ${passCount}/${totalCount} tests passed`);
console.log('='.repeat(50));

if (passCount === totalCount) {
  console.log('\n✓ All tests passed! Algorithm validated.');
} else {
  console.log(`\n✗ ${totalCount - passCount} test(s) failed. Review implementation.`);
}
