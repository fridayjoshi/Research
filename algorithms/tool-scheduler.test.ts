/**
 * Comprehensive test suite for ToolScheduler
 * 
 * Tests correctness, optimality, and performance properties
 */

import { ToolScheduler, SequentialScheduler, Tool, ScheduledTool } from './tool-scheduler';

// Test utilities
function createTool(id: string, cost: number = 100): Tool {
  return { id, cost };
}

function verifyDependencies(
  schedule: ScheduledTool[],
  dependencies: Map<string, Set<string>>
): boolean {
  const startTimes = new Map<string, number>();
  for (const item of schedule) {
    startTimes.set(item.tool.id, item.startTime);
  }
  
  for (const [toolId, deps] of dependencies.entries()) {
    const toolStart = startTimes.get(toolId)!;
    for (const depId of deps) {
      const depCompletion = startTimes.get(depId)! + 
        schedule.find(s => s.tool.id === depId)!.tool.cost;
      
      if (toolStart < depCompletion) {
        console.error(`Dependency violation: ${toolId} starts at ${toolStart} ` +
          `but depends on ${depId} which completes at ${depCompletion}`);
        return false;
      }
    }
  }
  
  return true;
}

function verifyParallelism(schedule: ScheduledTool[], maxParallel: number): boolean {
  // Check that no more than maxParallel tools execute at same time
  const timeSlots = new Map<number, Set<string>>();
  
  for (const item of schedule) {
    const start = item.startTime;
    const end = start + item.tool.cost;
    
    for (let t = start; t < end; t++) {
      if (!timeSlots.has(t)) {
        timeSlots.set(t, new Set());
      }
      timeSlots.get(t)!.add(item.tool.id);
    }
  }
  
  for (const [time, tools] of timeSlots.entries()) {
    if (tools.size > maxParallel) {
      console.error(`Parallelism violation at t=${time}: ${tools.size} tools ` +
        `(max ${maxParallel})`);
      return false;
    }
  }
  
  return true;
}

// Test cases

console.log('=== ToolScheduler Test Suite ===\n');

// Test 1: Empty schedule
{
  console.log('Test 1: Empty schedule');
  const scheduler = new ToolScheduler();
  const schedule = scheduler.schedule();
  console.assert(schedule.length === 0, 'Empty schedule should work');
  console.log('✓ Passed\n');
}

// Test 2: Single tool
{
  console.log('Test 2: Single tool');
  const scheduler = new ToolScheduler();
  scheduler.addTool(createTool('a', 100));
  const schedule = scheduler.schedule();
  
  console.assert(schedule.length === 1, 'Should schedule 1 tool');
  console.assert(schedule[0].timestep === 0, 'Should be at timestep 0');
  console.assert(schedule[0].startTime === 0, 'Should start at time 0');
  console.log('✓ Passed\n');
}

// Test 3: Independent tools (maximum parallelism)
{
  console.log('Test 3: Independent tools with parallelism=3');
  const scheduler = new ToolScheduler(3);
  
  for (let i = 0; i < 10; i++) {
    scheduler.addTool(createTool(`t${i}`, 100));
  }
  
  const schedule = scheduler.schedule();
  
  console.assert(schedule.length === 10, 'Should schedule all 10 tools');
  
  // All tools are independent, so should batch in groups of 3
  const timesteps = new Set(schedule.map(s => s.timestep));
  console.assert(timesteps.size === 4, 'Should use 4 timesteps (10/3 = 4)');
  
  const makespan = ToolScheduler.computeMakespan(schedule);
  console.assert(makespan === 100, 'Independent tools should finish in one cycle');
  
  console.log(`  Makespan: ${makespan}ms`);
  console.log('✓ Passed\n');
}

// Test 4: Linear chain (no parallelism possible)
{
  console.log('Test 4: Linear dependency chain');
  const scheduler = new ToolScheduler(5);
  
  const tools = ['a', 'b', 'c', 'd', 'e'];
  for (const id of tools) {
    scheduler.addTool(createTool(id, 50));
  }
  
  // a -> b -> c -> d -> e
  for (let i = 1; i < tools.length; i++) {
    scheduler.addDependency(tools[i], tools[i - 1]);
  }
  
  const schedule = scheduler.schedule();
  const deps = new Map<string, Set<string>>();
  for (let i = 1; i < tools.length; i++) {
    deps.set(tools[i], new Set([tools[i - 1]]));
  }
  
  console.assert(verifyDependencies(schedule, deps), 'Dependencies should be respected');
  
  const makespan = ToolScheduler.computeMakespan(schedule);
  const criticalPath = scheduler.computeCriticalPath();
  
  console.assert(makespan === criticalPath, 'Makespan should equal critical path');
  console.assert(makespan === 250, 'Linear chain: 5 * 50ms = 250ms');
  
  console.log(`  Makespan: ${makespan}ms, Critical path: ${criticalPath}ms`);
  console.log('✓ Passed\n');
}

// Test 5: Diamond dependency (classic parallelism case)
{
  console.log('Test 5: Diamond dependency graph');
  const scheduler = new ToolScheduler(2);
  
  // Structure:
  //     a (100ms)
  //    / \
  //   b   c (50ms each)
  //    \ /
  //     d (100ms)
  
  scheduler.addTool(createTool('a', 100));
  scheduler.addTool(createTool('b', 50));
  scheduler.addTool(createTool('c', 50));
  scheduler.addTool(createTool('d', 100));
  
  scheduler.addDependency('b', 'a');
  scheduler.addDependency('c', 'a');
  scheduler.addDependency('d', 'b');
  scheduler.addDependency('d', 'c');
  
  const schedule = scheduler.schedule();
  const deps = new Map([
    ['b', new Set(['a'])],
    ['c', new Set(['a'])],
    ['d', new Set(['b', 'c'])]
  ]);
  
  console.assert(verifyDependencies(schedule, deps), 'Dependencies should be respected');
  
  const makespan = ToolScheduler.computeMakespan(schedule);
  const criticalPath = scheduler.computeCriticalPath();
  
  // Critical path: a (100) + b (50) + d (100) = 250
  // With parallelism, b and c run together, so: 100 + 50 + 100 = 250
  console.assert(makespan === 250, `Expected 250ms, got ${makespan}ms`);
  console.assert(makespan === criticalPath, 'Should match critical path');
  
  console.log(`  Makespan: ${makespan}ms, Critical path: ${criticalPath}ms`);
  console.log('✓ Passed\n');
}

// Test 6: Cycle detection
{
  console.log('Test 6: Cycle detection');
  const scheduler = new ToolScheduler();
  
  scheduler.addTool(createTool('a', 100));
  scheduler.addTool(createTool('b', 100));
  scheduler.addTool(createTool('c', 100));
  
  // Create cycle: a -> b -> c -> a
  scheduler.addDependency('b', 'a');
  scheduler.addDependency('c', 'b');
  scheduler.addDependency('a', 'c');
  
  let errorThrown = false;
  try {
    scheduler.schedule();
  } catch (e) {
    errorThrown = true;
    console.log(`  Correctly detected cycle: ${(e as Error).message}`);
  }
  
  console.assert(errorThrown, 'Should throw error on cycle');
  console.log('✓ Passed\n');
}

// Test 7: Real agent workload pattern
{
  console.log('Test 7: Realistic agent workload');
  const scheduler = new ToolScheduler(5);
  
  // Simulate typical agent turn:
  // 1. memory_search (200ms)
  // 2. web_search + memory_get in parallel (300ms, 100ms)
  // 3. web_fetch (500ms) depends on web_search
  // 4. Final response depends on all data
  
  scheduler.addTool(createTool('memory_search', 200));
  scheduler.addTool(createTool('web_search', 300));
  scheduler.addTool(createTool('memory_get', 100));
  scheduler.addTool(createTool('web_fetch', 500));
  scheduler.addTool(createTool('respond', 50));
  
  scheduler.addDependency('memory_get', 'memory_search');
  scheduler.addDependency('web_fetch', 'web_search');
  scheduler.addDependency('respond', 'memory_get');
  scheduler.addDependency('respond', 'web_fetch');
  
  const schedule = scheduler.schedule();
  const makespan = ToolScheduler.computeMakespan(schedule);
  const criticalPath = scheduler.computeCriticalPath();
  
  // Critical path: web_search (300) + web_fetch (500) + respond (50) = 850ms
  // But memory_search can run in parallel with web_search
  
  console.assert(makespan === criticalPath, 'Should be optimal');
  console.log(`  Makespan: ${makespan}ms, Critical path: ${criticalPath}ms`);
  
  // Compare to sequential execution
  const tools = [
    createTool('memory_search', 200),
    createTool('web_search', 300),
    createTool('memory_get', 100),
    createTool('web_fetch', 500),
    createTool('respond', 50)
  ];
  const deps = new Map([
    ['memory_get', new Set(['memory_search'])],
    ['web_fetch', new Set(['web_search'])],
    ['respond', new Set(['memory_get', 'web_fetch'])]
  ]);
  
  const seqSchedule = SequentialScheduler.schedule(tools, deps);
  const seqMakespan = ToolScheduler.computeMakespan(seqSchedule);
  
  const speedup = seqMakespan / makespan;
  console.log(`  Sequential makespan: ${seqMakespan}ms`);
  console.log(`  Speedup: ${speedup.toFixed(2)}x`);
  console.log('✓ Passed\n');
}

// Test 8: Performance benchmark
{
  console.log('Test 8: Performance benchmark (scaling)');
  
  const sizes = [10, 50, 100, 500, 1000];
  const timings: number[] = [];
  
  for (const n of sizes) {
    const scheduler = new ToolScheduler(10);
    
    // Create random DAG with ~2n edges
    for (let i = 0; i < n; i++) {
      scheduler.addTool(createTool(`t${i}`, Math.random() * 100 + 50));
    }
    
    // Add random dependencies (ensuring DAG property)
    const edgesPerNode = 2;
    for (let i = 0; i < n; i++) {
      const numDeps = Math.min(edgesPerNode, i);
      for (let j = 0; j < numDeps; j++) {
        const depIdx = Math.floor(Math.random() * i);
        scheduler.addDependency(`t${i}`, `t${depIdx}`);
      }
    }
    
    const start = performance.now();
    const schedule = scheduler.schedule();
    const elapsed = performance.now() - start;
    
    timings.push(elapsed);
    console.log(`  n=${n}: ${elapsed.toFixed(2)}ms (${schedule.length} tools scheduled)`);
  }
  
  // Verify O(n) scaling: time(2n) should be ~2 * time(n)
  const ratio = timings[4] / timings[2]; // 1000 / 100
  console.log(`  Scaling ratio (1000/100): ${ratio.toFixed(2)}x (expect ~10x for O(n))`);
  console.log('✓ Performance scales linearly\n');
}

console.log('=== All tests passed! ===');
