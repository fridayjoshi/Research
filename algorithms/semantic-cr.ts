/**
 * SemanticCR: Semantic Conflict Resolution for Multi-Agent Shared Memory
 *
 * Implementation of the SemanticCR protocol from semantic-conflict-resolution.md
 * Provides eventual consistency with bounded semantic drift for multi-agent systems.
 *
 * @author Friday
 * @date May 22, 2026
 */

import { randomBytes, createHash } from 'crypto';

// ============================================================================
// Type Definitions
// ============================================================================

export type OperationType = 'read' | 'write' | 'update' | 'delete';

export interface VectorClock {
  [agentId: string]: number;
}

export interface Operation {
  agentId: string;
  type: OperationType;
  key: string;
  value: any;
  timestamp: number;
  vectorClock: VectorClock;
  embedding?: number[];
  hash?: string;
}

export interface ConflictSet {
  operations: Operation[];
  commonAncestor: Operation | null;
  mergeAttempts: number;
  createdAt: number;
}

export interface AgentState {
  agentId: string;
  localView: Map<string, any>;
  operationLog: Operation[];
  vectorClock: VectorClock;
  pendingMerges: ConflictSet[];
  embeddingCache: Map<string, number[]>;
  peers: string[];
}

export interface MergeResult {
  merged: any;
  confidence: number;
}

export interface LLMClient {
  call(prompt: string): Promise<MergeResult>;
}

export interface EmbeddingClient {
  embed(value: any): Promise<number[]>;
}

// ============================================================================
// Vector Clock Operations
// ============================================================================

export class VectorClockOps {
  static create(agentId: string): VectorClock {
    return { [agentId]: 0 };
  }

  static increment(clock: VectorClock, agentId: string): VectorClock {
    return {
      ...clock,
      [agentId]: (clock[agentId] || 0) + 1
    };
  }

  static merge(clock1: VectorClock, clock2: VectorClock): VectorClock {
    const merged: VectorClock = { ...clock1 };
    for (const [agentId, timestamp] of Object.entries(clock2)) {
      merged[agentId] = Math.max(merged[agentId] || 0, timestamp);
    }
    return merged;
  }

  static compare(clock1: VectorClock, clock2: VectorClock): 'before' | 'after' | 'concurrent' {
    let hasLess = false;
    let hasGreater = false;

    const allAgents = new Set([
      ...Object.keys(clock1),
      ...Object.keys(clock2)
    ]);

    for (const agentId of allAgents) {
      const t1 = clock1[agentId] || 0;
      const t2 = clock2[agentId] || 0;

      if (t1 < t2) hasLess = true;
      if (t1 > t2) hasGreater = true;
    }

    if (!hasLess && !hasGreater) return 'concurrent'; // Equal
    if (hasLess && !hasGreater) return 'before';
    if (hasGreater && !hasLess) return 'after';
    return 'concurrent';
  }

  static isConcurrent(clock1: VectorClock, clock2: VectorClock): boolean {
    return this.compare(clock1, clock2) === 'concurrent';
  }
}

// ============================================================================
// Embedding Operations
// ============================================================================

export class EmbeddingOps {
  /**
   * Compute cosine similarity between two embedding vectors
   */
  static cosineSimilarity(e1: number[], e2: number[]): number {
    if (e1.length !== e2.length) {
      throw new Error('Embedding dimensions must match');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < e1.length; i++) {
      dotProduct += e1[i] * e2[i];
      norm1 += e1[i] * e1[i];
      norm2 += e2[i] * e2[i];
    }

    const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }

  /**
   * Generate mock embedding (for testing without real embedding API)
   * In production, replace with actual embedding service call
   */
  static mockEmbed(value: any, dimension: number = 128): number[] {
    const str = JSON.stringify(value);
    const hash = createHash('sha256').update(str).digest();

    const embedding: number[] = [];
    for (let i = 0; i < dimension; i++) {
      const byteIndex = i % hash.length;
      embedding.push((hash[byteIndex] / 255) * 2 - 1); // Normalize to [-1, 1]
    }

    return embedding;
  }
}

// ============================================================================
// SemanticCR Agent
// ============================================================================

export class SemanticCRAgent {
  private state: AgentState;
  private embeddingClient: EmbeddingClient;
  private llmClient: LLMClient;
  private conflictThreshold: number;
  private mergeConfidenceThreshold: number;
  private maxMergeAttempts: number;
  private gossipInterval: number;
  private gossipTimer?: NodeJS.Timeout;

  constructor(
    agentId: string,
    peers: string[],
    embeddingClient: EmbeddingClient,
    llmClient: LLMClient,
    options: {
      conflictThreshold?: number;
      mergeConfidenceThreshold?: number;
      maxMergeAttempts?: number;
      gossipIntervalMs?: number;
    } = {}
  ) {
    this.state = {
      agentId,
      localView: new Map(),
      operationLog: [],
      vectorClock: VectorClockOps.create(agentId),
      pendingMerges: [],
      embeddingCache: new Map(),
      peers
    };

    this.embeddingClient = embeddingClient;
    this.llmClient = llmClient;
    this.conflictThreshold = options.conflictThreshold ?? 0.9;
    this.mergeConfidenceThreshold = options.mergeConfidenceThreshold ?? 0.8;
    this.maxMergeAttempts = options.maxMergeAttempts ?? 3;
    this.gossipInterval = options.gossipIntervalMs ?? 5000;
  }

  // ==========================================================================
  // Phase 1: Optimistic Execution
  // ==========================================================================

  async write(key: string, value: any): Promise<Operation> {
    return this.performOperation('write', key, value);
  }

  async update(key: string, value: any): Promise<Operation> {
    return this.performOperation('update', key, value);
  }

  async delete(key: string): Promise<Operation> {
    return this.performOperation('delete', key, null);
  }

  async read(key: string): Promise<any> {
    return this.state.localView.get(key);
  }

  private async performOperation(
    type: OperationType,
    key: string,
    value: any
  ): Promise<Operation> {
    // Increment vector clock
    this.state.vectorClock = VectorClockOps.increment(
      this.state.vectorClock,
      this.state.agentId
    );

    // Compute embedding
    const embedding = await this.getEmbedding(value);

    // Create operation
    const operation: Operation = {
      agentId: this.state.agentId,
      type,
      key,
      value,
      timestamp: this.state.vectorClock[this.state.agentId],
      vectorClock: { ...this.state.vectorClock },
      embedding,
      hash: this.hashOperation(type, key, value)
    };

    // Append to log
    this.state.operationLog.push(operation);

    // Update local view
    if (type === 'delete') {
      this.state.localView.delete(key);
    } else {
      this.state.localView.set(key, value);
    }

    // Broadcast would happen here (delegated to transport layer)
    return operation;
  }

  private async getEmbedding(value: any): Promise<number[]> {
    const cacheKey = JSON.stringify(value);

    if (this.state.embeddingCache.has(cacheKey)) {
      return this.state.embeddingCache.get(cacheKey)!;
    }

    const embedding = await this.embeddingClient.embed(value);
    this.state.embeddingCache.set(cacheKey, embedding);
    return embedding;
  }

  private hashOperation(type: string, key: string, value: any): string {
    const content = `${type}:${key}:${JSON.stringify(value)}`;
    return createHash('sha256').update(content).digest('hex');
  }

  // ==========================================================================
  // Phase 2: Gossip-Based Propagation
  // ==========================================================================

  startGossip(): void {
    if (this.gossipTimer) return;

    this.gossipTimer = setInterval(() => {
      this.gossipRound();
    }, this.gossipInterval);
  }

  stopGossip(): void {
    if (this.gossipTimer) {
      clearInterval(this.gossipTimer);
      this.gossipTimer = undefined;
    }
  }

  private gossipRound(): void {
    // Select random peer
    if (this.state.peers.length === 0) return;

    const peerIndex = Math.floor(Math.random() * this.state.peers.length);
    const peer = this.state.peers[peerIndex];

    // In production, would send digest and receive missing ops via network
    // For now, this is a hook for the transport layer
    this.onGossipWithPeer?.(peer, this.computeDigest());
  }

  private computeDigest(): string {
    const logHashes = this.state.operationLog.map(op => op.hash).join(',');
    return createHash('sha256').update(logHashes).digest('hex');
  }

  /**
   * Integrate remote operation into local state
   */
  async integrateOperation(operation: Operation): Promise<void> {
    // Check if already integrated
    const exists = this.state.operationLog.some(
      op => op.hash === operation.hash
    );
    if (exists) return;

    // Update vector clock
    this.state.vectorClock = VectorClockOps.merge(
      this.state.vectorClock,
      operation.vectorClock
    );

    // Detect conflicts
    const conflicts = await this.detectConflicts(operation);

    if (conflicts.length > 0) {
      // Create conflict set and trigger merge
      const conflictSet: ConflictSet = {
        operations: [operation, ...conflicts],
        commonAncestor: this.findCommonAncestor([operation, ...conflicts]),
        mergeAttempts: 0,
        createdAt: Date.now()
      };

      this.state.pendingMerges.push(conflictSet);
      await this.semanticMerge(conflictSet);
    } else {
      // No conflict, apply directly
      this.state.operationLog.push(operation);
      this.applyOperation(operation);
    }
  }

  private async detectConflicts(operation: Operation): Promise<Operation[]> {
    const conflicts: Operation[] = [];

    for (const existingOp of this.state.operationLog) {
      // Check spatial overlap
      if (!this.keysOverlap(operation.key, existingOp.key)) continue;

      // Check temporal concurrency
      if (!VectorClockOps.isConcurrent(operation.vectorClock, existingOp.vectorClock)) {
        continue;
      }

      // Check semantic divergence
      if (!operation.embedding || !existingOp.embedding) continue;

      const similarity = EmbeddingOps.cosineSimilarity(
        operation.embedding,
        existingOp.embedding
      );

      if (similarity < this.conflictThreshold) {
        conflicts.push(existingOp);
      }
    }

    return conflicts;
  }

  private keysOverlap(key1: string, key2: string): boolean {
    // Simple exact match; extend for hierarchical keys if needed
    return key1 === key2;
  }

  private findCommonAncestor(operations: Operation[]): Operation | null {
    // Find operation that causally precedes all others
    // Simplified: return earliest operation
    if (operations.length === 0) return null;

    return operations.reduce((earliest, op) => {
      const comparison = VectorClockOps.compare(op.vectorClock, earliest.vectorClock);
      return comparison === 'before' ? op : earliest;
    });
  }

  private applyOperation(operation: Operation): void {
    if (operation.type === 'delete') {
      this.state.localView.delete(operation.key);
    } else {
      this.state.localView.set(operation.key, operation.value);
    }
  }

  // ==========================================================================
  // Phase 3: Semantic Merge
  // ==========================================================================

  private async semanticMerge(conflictSet: ConflictSet): Promise<void> {
    const { operations, commonAncestor } = conflictSet;

    // Group operations by agent
    const branches = this.groupByAgent(operations);

    // Build LLM prompt
    const prompt = this.buildMergePrompt(commonAncestor, branches);

    try {
      // Call LLM
      const result = await this.llmClient.call(prompt);

      if (result.confidence > this.mergeConfidenceThreshold) {
        // Accept merge
        const mergeOp = await this.createMergeOperation(conflictSet, result.merged);

        // Apply locally
        this.state.operationLog.push(mergeOp);
        this.applyOperation(mergeOp);

        // Remove from pending
        this.state.pendingMerges = this.state.pendingMerges.filter(
          cs => cs !== conflictSet
        );

        // Broadcast merge (hook for transport layer)
        this.onMergeResolved?.(mergeOp);
      } else {
        // Low confidence, retry with backoff
        conflictSet.mergeAttempts++;

        if (conflictSet.mergeAttempts > this.maxMergeAttempts) {
          this.onMergeEscalation?.(conflictSet);
        } else {
          // Schedule retry (would use setTimeout in production)
          setTimeout(() => this.semanticMerge(conflictSet), 1000 * conflictSet.mergeAttempts);
        }
      }
    } catch (error) {
      console.error('Merge failed:', error);
      conflictSet.mergeAttempts++;
    }
  }

  private groupByAgent(operations: Operation[]): Map<string, Operation[]> {
    const groups = new Map<string, Operation[]>();

    for (const op of operations) {
      const existing = groups.get(op.agentId) || [];
      existing.push(op);
      groups.set(op.agentId, existing);
    }

    return groups;
  }

  private buildMergePrompt(
    ancestor: Operation | null,
    branches: Map<string, Operation[]>
  ): string {
    let prompt = 'You are resolving a memory conflict between agents.\n\n';

    if (ancestor) {
      prompt += `Common ancestor state:\n${JSON.stringify(ancestor.value, null, 2)}\n\n`;
    }

    let branchIndex = 1;
    for (const [agentId, ops] of branches) {
      const latestOp = ops[ops.length - 1];
      prompt += `Branch ${branchIndex} (agent ${agentId}):\n`;
      prompt += `${JSON.stringify(latestOp.value, null, 2)}\n`;
      prompt += `Operation: ${latestOp.type}\n\n`;
      branchIndex++;
    }

    prompt += 'Produce a semantically coherent merge that preserves intent from both branches.\n';
    prompt += 'Output JSON: {"merged": <result>, "confidence": <0-1>}';

    return prompt;
  }

  private async createMergeOperation(
    conflictSet: ConflictSet,
    mergedValue: any
  ): Promise<Operation> {
    // Use first conflicted key
    const key = conflictSet.operations[0].key;

    // Increment vector clock
    this.state.vectorClock = VectorClockOps.increment(
      this.state.vectorClock,
      this.state.agentId
    );

    const embedding = await this.getEmbedding(mergedValue);

    return {
      agentId: this.state.agentId,
      type: 'update',
      key,
      value: mergedValue,
      timestamp: this.state.vectorClock[this.state.agentId],
      vectorClock: { ...this.state.vectorClock },
      embedding,
      hash: this.hashOperation('update', key, mergedValue)
    };
  }

  // ==========================================================================
  // Hooks (for transport layer integration)
  // ==========================================================================

  onGossipWithPeer?: (peer: string, digest: string) => void;
  onMergeResolved?: (operation: Operation) => void;
  onMergeEscalation?: (conflictSet: ConflictSet) => void;

  // ==========================================================================
  // Inspection / Debugging
  // ==========================================================================

  getState(): Readonly<AgentState> {
    return {
      ...this.state,
      localView: new Map(this.state.localView),
      operationLog: [...this.state.operationLog],
      vectorClock: { ...this.state.vectorClock },
      pendingMerges: [...this.state.pendingMerges],
      embeddingCache: new Map(this.state.embeddingCache),
      peers: [...this.state.peers]
    };
  }

  getPendingMerges(): ConflictSet[] {
    return [...this.state.pendingMerges];
  }

  getOperationLog(): Operation[] {
    return [...this.state.operationLog];
  }

  hasConverged(): boolean {
    return this.state.pendingMerges.length === 0;
  }
}

// ============================================================================
// Mock Clients (for testing)
// ============================================================================

export class MockEmbeddingClient implements EmbeddingClient {
  private dimension: number;

  constructor(dimension: number = 128) {
    this.dimension = dimension;
  }

  async embed(value: any): Promise<number[]> {
    return EmbeddingOps.mockEmbed(value, this.dimension);
  }
}

export class MockLLMClient implements LLMClient {
  async call(prompt: string): Promise<MergeResult> {
    // Simple mock: extract values from prompt and combine
    // In production, this would be a real LLM call

    const branchMatches = prompt.match(/Branch \d+ \(agent [^)]+\):\s*({[^}]+})/g);
    if (!branchMatches || branchMatches.length < 2) {
      return { merged: {}, confidence: 0.5 };
    }

    // Simple merge strategy: combine all fields
    const merged: any = {};
    for (const match of branchMatches) {
      try {
        const jsonMatch = match.match(/{[^}]+}/);
        if (jsonMatch) {
          const obj = JSON.parse(jsonMatch[0]);
          Object.assign(merged, obj);
        }
      } catch (e) {
        // Skip invalid JSON
      }
    }

    return {
      merged,
      confidence: 0.9
    };
  }
}
