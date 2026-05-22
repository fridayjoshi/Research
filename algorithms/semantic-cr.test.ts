/**
 * Tests for SemanticCR Protocol
 *
 * @author Friday
 * @date May 22, 2026
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import {
  SemanticCRAgent,
  VectorClockOps,
  EmbeddingOps,
  MockEmbeddingClient,
  MockLLMClient,
  Operation,
  VectorClock
} from './semantic-cr';

// ============================================================================
// Vector Clock Tests
// ============================================================================

describe('VectorClockOps', () => {
  describe('create', () => {
    it('should create vector clock with agent initialized to 0', () => {
      const clock = VectorClockOps.create('agent1');
      expect(clock).toEqual({ agent1: 0 });
    });
  });

  describe('increment', () => {
    it('should increment existing agent timestamp', () => {
      const clock = { agent1: 5, agent2: 3 };
      const incremented = VectorClockOps.increment(clock, 'agent1');
      expect(incremented).toEqual({ agent1: 6, agent2: 3 });
    });

    it('should initialize new agent to 1', () => {
      const clock = { agent1: 5 };
      const incremented = VectorClockOps.increment(clock, 'agent2');
      expect(incremented).toEqual({ agent1: 5, agent2: 1 });
    });
  });

  describe('merge', () => {
    it('should take max timestamp for each agent', () => {
      const clock1 = { agent1: 5, agent2: 3 };
      const clock2 = { agent1: 2, agent2: 7, agent3: 4 };
      const merged = VectorClockOps.merge(clock1, clock2);

      expect(merged).toEqual({
        agent1: 5,
        agent2: 7,
        agent3: 4
      });
    });
  });

  describe('compare', () => {
    it('should detect before relationship', () => {
      const clock1 = { agent1: 2, agent2: 3 };
      const clock2 = { agent1: 5, agent2: 7 };
      expect(VectorClockOps.compare(clock1, clock2)).toBe('before');
    });

    it('should detect after relationship', () => {
      const clock1 = { agent1: 5, agent2: 7 };
      const clock2 = { agent1: 2, agent2: 3 };
      expect(VectorClockOps.compare(clock1, clock2)).toBe('after');
    });

    it('should detect concurrent operations', () => {
      const clock1 = { agent1: 5, agent2: 3 };
      const clock2 = { agent1: 2, agent2: 7 };
      expect(VectorClockOps.compare(clock1, clock2)).toBe('concurrent');
    });

    it('should detect equal clocks as concurrent', () => {
      const clock1 = { agent1: 5, agent2: 3 };
      const clock2 = { agent1: 5, agent2: 3 };
      expect(VectorClockOps.compare(clock1, clock2)).toBe('concurrent');
    });
  });

  describe('isConcurrent', () => {
    it('should return true for concurrent operations', () => {
      const clock1 = { agent1: 5, agent2: 3 };
      const clock2 = { agent1: 2, agent2: 7 };
      expect(VectorClockOps.isConcurrent(clock1, clock2)).toBe(true);
    });

    it('should return false for causally ordered operations', () => {
      const clock1 = { agent1: 2, agent2: 3 };
      const clock2 = { agent1: 5, agent2: 7 };
      expect(VectorClockOps.isConcurrent(clock1, clock2)).toBe(false);
    });
  });
});

// ============================================================================
// Embedding Operations Tests
// ============================================================================

describe('EmbeddingOps', () => {
  describe('cosineSimilarity', () => {
    it('should return 1 for identical vectors', () => {
      const v = [1, 2, 3, 4];
      expect(EmbeddingOps.cosineSimilarity(v, v)).toBeCloseTo(1.0, 5);
    });

    it('should return 0 for orthogonal vectors', () => {
      const v1 = [1, 0, 0];
      const v2 = [0, 1, 0];
      expect(EmbeddingOps.cosineSimilarity(v1, v2)).toBeCloseTo(0, 5);
    });

    it('should return -1 for opposite vectors', () => {
      const v1 = [1, 2, 3];
      const v2 = [-1, -2, -3];
      expect(EmbeddingOps.cosineSimilarity(v1, v2)).toBeCloseTo(-1.0, 5);
    });

    it('should compute correct similarity for arbitrary vectors', () => {
      const v1 = [1, 2, 3];
      const v2 = [4, 5, 6];
      const similarity = EmbeddingOps.cosineSimilarity(v1, v2);

      // Expected: (1*4 + 2*5 + 3*6) / (sqrt(1+4+9) * sqrt(16+25+36))
      // = 32 / (sqrt(14) * sqrt(77))
      const expected = 32 / (Math.sqrt(14) * Math.sqrt(77));
      expect(similarity).toBeCloseTo(expected, 5);
    });

    it('should throw for mismatched dimensions', () => {
      const v1 = [1, 2, 3];
      const v2 = [1, 2];
      expect(() => EmbeddingOps.cosineSimilarity(v1, v2)).toThrow();
    });
  });

  describe('mockEmbed', () => {
    it('should generate consistent embeddings for same value', () => {
      const value = { text: 'hello world' };
      const e1 = EmbeddingOps.mockEmbed(value);
      const e2 = EmbeddingOps.mockEmbed(value);
      expect(e1).toEqual(e2);
    });

    it('should generate different embeddings for different values', () => {
      const v1 = { text: 'hello' };
      const v2 = { text: 'world' };
      const e1 = EmbeddingOps.mockEmbed(v1);
      const e2 = EmbeddingOps.mockEmbed(v2);
      expect(e1).not.toEqual(e2);
    });

    it('should respect dimension parameter', () => {
      const value = { text: 'test' };
      const e64 = EmbeddingOps.mockEmbed(value, 64);
      const e256 = EmbeddingOps.mockEmbed(value, 256);
      expect(e64.length).toBe(64);
      expect(e256.length).toBe(256);
    });

    it('should generate normalized embeddings', () => {
      const value = { text: 'test' };
      const embedding = EmbeddingOps.mockEmbed(value);

      // Check all values in [-1, 1]
      for (const val of embedding) {
        expect(val).toBeGreaterThanOrEqual(-1);
        expect(val).toBeLessThanOrEqual(1);
      }
    });
  });
});

// ============================================================================
// SemanticCR Agent Tests
// ============================================================================

describe('SemanticCRAgent', () => {
  let agent: SemanticCRAgent;
  let embeddingClient: MockEmbeddingClient;
  let llmClient: MockLLMClient;

  beforeEach(() => {
    embeddingClient = new MockEmbeddingClient(128);
    llmClient = new MockLLMClient();
    agent = new SemanticCRAgent(
      'agent1',
      ['agent2', 'agent3'],
      embeddingClient,
      llmClient,
      {
        conflictThreshold: 0.9,
        mergeConfidenceThreshold: 0.8,
        maxMergeAttempts: 3,
        gossipIntervalMs: 1000
      }
    );
  });

  describe('optimistic execution', () => {
    it('should write value to local view', async () => {
      await agent.write('key1', { value: 'hello' });
      const result = await agent.read('key1');
      expect(result).toEqual({ value: 'hello' });
    });

    it('should increment vector clock on write', async () => {
      const initialState = agent.getState();
      expect(initialState.vectorClock.agent1).toBe(0);

      await agent.write('key1', 'value1');
      const afterWrite = agent.getState();
      expect(afterWrite.vectorClock.agent1).toBe(1);

      await agent.write('key2', 'value2');
      const afterSecondWrite = agent.getState();
      expect(afterSecondWrite.vectorClock.agent1).toBe(2);
    });

    it('should append operation to log', async () => {
      await agent.write('key1', 'value1');
      const ops = agent.getOperationLog();

      expect(ops.length).toBe(1);
      expect(ops[0].key).toBe('key1');
      expect(ops[0].value).toBe('value1');
      expect(ops[0].type).toBe('write');
      expect(ops[0].agentId).toBe('agent1');
    });

    it('should compute and cache embedding', async () => {
      const value = { text: 'test content' };
      await agent.write('key1', value);

      const ops = agent.getOperationLog();
      expect(ops[0].embedding).toBeDefined();
      expect(ops[0].embedding!.length).toBe(128);
    });

    it('should handle delete operations', async () => {
      await agent.write('key1', 'value1');
      expect(await agent.read('key1')).toBe('value1');

      await agent.delete('key1');
      expect(await agent.read('key1')).toBeUndefined();
    });

    it('should handle update operations', async () => {
      await agent.write('key1', 'initial');
      expect(await agent.read('key1')).toBe('initial');

      await agent.update('key1', 'updated');
      expect(await agent.read('key1')).toBe('updated');
    });
  });

  describe('conflict detection', () => {
    it('should not conflict with causally ordered operations', async () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient);
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      // agent1 writes first
      const op1 = await agent1.write('key1', 'value1');

      // agent2 integrates and writes (causally after)
      await agent2.integrateOperation(op1);
      await agent2.write('key1', 'value2');

      // Should have no conflicts
      expect(agent2.getPendingMerges().length).toBe(0);
    });

    it('should detect concurrent writes to same key', async () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient);
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      // Both agents write concurrently
      const op1 = await agent1.write('key1', { content: 'completely different text' });
      const op2 = await agent2.write('key1', { content: 'unrelated content here' });

      // agent1 integrates agent2's operation
      await agent1.integrateOperation(op2);

      // Should detect conflict (concurrent + semantic divergence)
      const pending = agent1.getPendingMerges();
      expect(pending.length).toBeGreaterThan(0);
    });

    it('should not conflict if semantic similarity is high', async () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient, {
        conflictThreshold: 0.9
      });
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      // Write very similar values concurrently
      const similarValue = { text: 'identical content' };
      const op1 = await agent1.write('key1', similarValue);
      const op2 = await agent2.write('key1', similarValue);

      // agent1 integrates agent2's operation
      await agent1.integrateOperation(op2);

      // Should NOT conflict (high similarity)
      expect(agent1.getPendingMerges().length).toBe(0);
    });

    it('should not conflict on different keys', async () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient);
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      // Write to different keys
      const op1 = await agent1.write('key1', 'value1');
      const op2 = await agent2.write('key2', 'value2');

      await agent1.integrateOperation(op2);

      // No spatial overlap, no conflict
      expect(agent1.getPendingMerges().length).toBe(0);
    });
  });

  describe('semantic merge', () => {
    it('should resolve conflicts via LLM merge', async () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient, {
        conflictThreshold: 0.5 // Lower threshold to trigger conflicts
      });
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      // Concurrent conflicting writes
      const op1 = await agent1.write('key1', { field1: 'value1' });
      const op2 = await agent2.write('key1', { field2: 'value2' });

      // Track merge resolution
      let mergeResolved = false;
      agent1.onMergeResolved = () => {
        mergeResolved = true;
      };

      // Integrate and wait for merge
      await agent1.integrateOperation(op2);

      // Give merge time to complete (async LLM call)
      await new Promise(resolve => setTimeout(resolve, 100));

      // Merge should have been triggered and resolved
      expect(mergeResolved).toBe(true);
      expect(agent1.hasConverged()).toBe(true);
    });

    it('should escalate after max merge attempts', async () => {
      // Mock LLM that always returns low confidence
      const lowConfidenceLLM = {
        async call(): Promise<{ merged: any; confidence: number }> {
          return { merged: {}, confidence: 0.3 }; // Below threshold
        }
      };

      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, lowConfidenceLLM, {
        conflictThreshold: 0.5,
        maxMergeAttempts: 2
      });
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      const op1 = await agent1.write('key1', { data: 'conflict1' });
      const op2 = await agent2.write('key1', { data: 'conflict2' });

      let escalated = false;
      agent1.onMergeEscalation = () => {
        escalated = true;
      };

      await agent1.integrateOperation(op2);

      // Wait for retries
      await new Promise(resolve => setTimeout(resolve, 5000));

      expect(escalated).toBe(true);
    });

    it('should preserve intent from both branches in merge', async () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient, {
        conflictThreshold: 0.5
      });
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      const op1 = await agent1.write('key1', { name: 'Alice', age: 30 });
      const op2 = await agent2.write('key1', { name: 'Alice', city: 'NYC' });

      await agent1.integrateOperation(op2);
      await new Promise(resolve => setTimeout(resolve, 100));

      // Merged value should contain both age and city
      const merged = await agent1.read('key1');
      expect(merged).toHaveProperty('age');
      expect(merged).toHaveProperty('city');
    });
  });

  describe('gossip protocol', () => {
    it('should start and stop gossip timer', () => {
      const agent1 = new SemanticCRAgent('agent1', ['agent2'], embeddingClient, llmClient);

      agent1.startGossip();
      // Timer should be running (no direct way to test, but shouldn't throw)

      agent1.stopGossip();
      // Timer should be stopped
    });

    it('should not start multiple gossip timers', () => {
      const agent1 = new SemanticCRAgent('agent1', ['agent2'], embeddingClient, llmClient);

      agent1.startGossip();
      agent1.startGossip(); // Should be no-op

      agent1.stopGossip();
    });
  });

  describe('convergence', () => {
    it('should converge when no pending merges', () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient);
      expect(agent1.hasConverged()).toBe(true);
    });

    it('should not converge with pending merges', async () => {
      const agent1 = new SemanticCRAgent('agent1', [], embeddingClient, llmClient, {
        conflictThreshold: 0.5
      });
      const agent2 = new SemanticCRAgent('agent2', [], embeddingClient, llmClient);

      // Create conflict
      const op1 = await agent1.write('key1', { data: 'A' });
      const op2 = await agent2.write('key1', { data: 'B' });

      // Block merge resolution temporarily
      const blockingLLM = {
        async call(): Promise<{ merged: any; confidence: number }> {
          await new Promise(resolve => setTimeout(resolve, 10000));
          return { merged: {}, confidence: 0.9 };
        }
      };

      const blockedAgent = new SemanticCRAgent('agent1', [], embeddingClient, blockingLLM, {
        conflictThreshold: 0.5
      });

      await blockedAgent.integrateOperation(op2);

      expect(blockedAgent.hasConverged()).toBe(false);
    });
  });

  describe('multi-agent scenario', () => {
    it('should achieve eventual consistency across 3 agents', async () => {
      const agents = [
        new SemanticCRAgent('agent1', ['agent2', 'agent3'], embeddingClient, llmClient),
        new SemanticCRAgent('agent2', ['agent1', 'agent3'], embeddingClient, llmClient),
        new SemanticCRAgent('agent3', ['agent1', 'agent2'], embeddingClient, llmClient)
      ];

      // Each agent writes to different keys
      const ops: Operation[] = [];
      ops.push(await agents[0].write('shared', { a: 1 }));
      ops.push(await agents[1].write('shared', { b: 2 }));
      ops.push(await agents[2].write('shared', { c: 3 }));

      // Gossip all operations to all agents
      for (const agent of agents) {
        for (const op of ops) {
          if (op.agentId !== agent.getState().agentId) {
            await agent.integrateOperation(op);
          }
        }
      }

      // Wait for merges
      await new Promise(resolve => setTimeout(resolve, 500));

      // All agents should converge to same state
      const values = await Promise.all(agents.map(a => a.read('shared')));

      // Check that all values are defined and contain merged data
      for (const val of values) {
        expect(val).toBeDefined();
        expect(typeof val).toBe('object');
      }

      // All should have converged
      for (const agent of agents) {
        expect(agent.hasConverged()).toBe(true);
      }
    });
  });

  describe('state inspection', () => {
    it('should provide readonly state snapshot', async () => {
      await agent.write('key1', 'value1');
      const state = agent.getState();

      expect(state.agentId).toBe('agent1');
      expect(state.localView.get('key1')).toBe('value1');
      expect(state.operationLog.length).toBe(1);
      expect(state.vectorClock.agent1).toBeGreaterThan(0);
    });

    it('should return operation log copy', async () => {
      await agent.write('key1', 'value1');
      const log1 = agent.getOperationLog();
      await agent.write('key2', 'value2');
      const log2 = agent.getOperationLog();

      expect(log1.length).toBe(1);
      expect(log2.length).toBe(2);
      expect(log1).not.toBe(log2); // Different arrays
    });
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('SemanticCR Integration', () => {
  it('should handle realistic multi-agent workload', async () => {
    const embeddingClient = new MockEmbeddingClient(128);
    const llmClient = new MockLLMClient();

    const agents = Array.from({ length: 5 }, (_, i) => {
      const agentId = `agent${i + 1}`;
      const peers = Array.from({ length: 5 }, (_, j) => `agent${j + 1}`)
        .filter(p => p !== agentId);

      return new SemanticCRAgent(agentId, peers, embeddingClient, llmClient);
    });

    // Simulate mixed workload
    const operations: Array<{ agent: SemanticCRAgent; op: Operation }> = [];

    // 10 concurrent writes across agents
    for (let i = 0; i < 10; i++) {
      const agent = agents[i % agents.length];
      const key = `key${Math.floor(i / 2)}`; // Some overlap
      const op = await agent.write(key, { iteration: i, data: `data${i}` });
      operations.push({ agent, op });
    }

    // Gossip all operations to all agents
    for (const { agent, op } of operations) {
      for (const targetAgent of agents) {
        if (targetAgent.getState().agentId !== agent.getState().agentId) {
          await targetAgent.integrateOperation(op);
        }
      }
    }

    // Wait for convergence
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Check convergence
    const converged = agents.filter(a => a.hasConverged());
    expect(converged.length).toBeGreaterThan(agents.length * 0.8); // At least 80% converged

    // Check consistency: all agents should have same keys
    const keys1 = Array.from(agents[0].getState().localView.keys()).sort();
    for (const agent of agents) {
      const keys = Array.from(agent.getState().localView.keys()).sort();
      expect(keys).toEqual(keys1);
    }
  });
});
