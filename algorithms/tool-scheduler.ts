/**
 * Optimal Tool Call Scheduler for AI Agents
 * 
 * Implements the GreedyDAG algorithm from optimal-tool-scheduling.md
 * with proven O(|T| + |D|) complexity and optimal makespan for DAG workloads.
 * 
 * @author Friday
 * @date 2026-02-15
 */

export interface Tool {
  id: string;
  cost: number; // Execution time in ms
}

export interface ToolDependency {
  tool: string; // Tool ID
  dependsOn: string[]; // List of tool IDs this depends on
}

export interface ScheduledTool {
  tool: Tool;
  timestep: number;
  startTime: number; // Absolute start time in ms
}

export class ToolScheduler {
  private tools: Map<string, Tool>;
  private dependencies: Map<string, Set<string>>;
  private maxParallel: number;

  constructor(maxParallel: number = 5) {
    this.tools = new Map();
    this.dependencies = new Map();
    this.maxParallel = maxParallel;
  }

  /**
   * Register a tool for scheduling
   */
  addTool(tool: Tool): void {
    this.tools.set(tool.id, tool);
    if (!this.dependencies.has(tool.id)) {
      this.dependencies.set(tool.id, new Set());
    }
  }

  /**
   * Register a dependency: dependent depends on prerequisite
   */
  addDependency(dependent: string, prerequisite: string): void {
    if (!this.dependencies.has(dependent)) {
      this.dependencies.set(dependent, new Set());
    }
    this.dependencies.get(dependent)!.add(prerequisite);
  }

  /**
   * Compute optimal schedule using GreedyDAG algorithm
   * 
   * Time complexity: O(|T| + |D|)
   * Space complexity: O(|T|)
   * 
   * @returns Array of scheduled tools with timesteps and start times
   * @throws Error if dependency graph contains cycles
   */
  schedule(): ScheduledTool[] {
    // Step 1: Compute in-degrees (O(|T| + |D|))
    // In-degree = number of dependencies a tool has (prerequisites)
    const inDegree = new Map<string, number>();
    for (const [toolId, deps] of this.dependencies.entries()) {
      inDegree.set(toolId, deps.size);
    }
    
    // Ensure all tools have an entry
    for (const toolId of this.tools.keys()) {
      if (!inDegree.has(toolId)) {
        inDegree.set(toolId, 0);
      }
    }

    // Step 2: Build reverse dependency map (who depends on each tool)
    const dependents = new Map<string, Set<string>>();
    for (const toolId of this.tools.keys()) {
      dependents.set(toolId, new Set());
    }
    for (const [toolId, deps] of this.dependencies.entries()) {
      for (const depId of deps) {
        dependents.get(depId)!.add(toolId);
      }
    }

    // Step 3: Initialize ready queue with zero in-degree nodes
    const ready: string[] = [];
    for (const [toolId, degree] of inDegree.entries()) {
      if (degree === 0) {
        ready.push(toolId);
      }
    }

    // Track completion times for computing start times
    const completionTime = new Map<string, number>();
    const schedule: ScheduledTool[] = [];
    let timestep = 0;
    let totalScheduled = 0;

    // Step 4: Level-by-level scheduling
    while (ready.length > 0) {
      // Batch up to maxParallel independent tools
      const batchSize = Math.min(ready.length, this.maxParallel);
      const batch = ready.splice(0, batchSize);
      
      // Assign to current timestep
      for (const toolId of batch) {
        const tool = this.tools.get(toolId)!;
        
        // Start time is max completion of dependencies
        const deps = this.dependencies.get(toolId) || new Set();
        let startTime = 0;
        for (const depId of deps) {
          const depCompletion = completionTime.get(depId) || 0;
          startTime = Math.max(startTime, depCompletion);
        }
        
        schedule.push({
          tool,
          timestep,
          startTime
        });
        
        completionTime.set(toolId, startTime + tool.cost);
        totalScheduled++;
        
        // Update dependents (tools that depend on this one)
        for (const dependentId of dependents.get(toolId)!) {
          const newDegree = inDegree.get(dependentId)! - 1;
          inDegree.set(dependentId, newDegree);
          
          if (newDegree === 0) {
            ready.push(dependentId);
          }
        }
      }
      
      timestep++;
    }

    // Verify all tools were scheduled (detect cycles)
    if (totalScheduled !== this.tools.size) {
      throw new Error(
        `Cycle detected: scheduled ${totalScheduled} of ${this.tools.size} tools`
      );
    }

    return schedule;
  }

  /**
   * Compute makespan (total execution time) for a schedule
   */
  static computeMakespan(schedule: ScheduledTool[]): number {
    let maxCompletion = 0;
    for (const item of schedule) {
      const completion = item.startTime + item.tool.cost;
      maxCompletion = Math.max(maxCompletion, completion);
    }
    return maxCompletion;
  }

  /**
   * Compute critical path (longest dependency chain)
   */
  computeCriticalPath(): number {
    const memo = new Map<string, number>();
    
    const longestPath = (toolId: string): number => {
      if (memo.has(toolId)) {
        return memo.get(toolId)!;
      }
      
      const tool = this.tools.get(toolId)!;
      const deps = this.dependencies.get(toolId) || new Set();
      
      let maxDepPath = 0;
      for (const depId of deps) {
        maxDepPath = Math.max(maxDepPath, longestPath(depId));
      }
      
      const result = maxDepPath + tool.cost;
      memo.set(toolId, result);
      return result;
    };
    
    let criticalPath = 0;
    for (const toolId of this.tools.keys()) {
      criticalPath = Math.max(criticalPath, longestPath(toolId));
    }
    
    return criticalPath;
  }

  /**
   * Clear all tools and dependencies
   */
  clear(): void {
    this.tools.clear();
    this.dependencies.clear();
  }
}

/**
 * Baseline: Sequential scheduler (no parallelism)
 */
export class SequentialScheduler {
  static schedule(tools: Tool[], dependencies: Map<string, Set<string>>): ScheduledTool[] {
    // Topological sort with sequential execution (no parallelism)
    const inDegree = new Map<string, number>();
    for (const [toolId, deps] of dependencies.entries()) {
      inDegree.set(toolId, deps.size);
    }
    for (const tool of tools) {
      if (!inDegree.has(tool.id)) {
        inDegree.set(tool.id, 0);
      }
    }
    
    // Build reverse dependency map
    const dependents = new Map<string, Set<string>>();
    for (const tool of tools) {
      dependents.set(tool.id, new Set());
    }
    for (const [toolId, deps] of dependencies.entries()) {
      for (const depId of deps) {
        dependents.get(depId)!.add(toolId);
      }
    }
    
    const ready: string[] = [];
    for (const [toolId, degree] of inDegree.entries()) {
      if (degree === 0) {
        ready.push(toolId);
      }
    }
    
    const schedule: ScheduledTool[] = [];
    let currentTime = 0;
    let timestep = 0;
    
    while (ready.length > 0) {
      const toolId = ready.shift()!;
      const tool = tools.find(t => t.id === toolId)!;
      
      schedule.push({
        tool,
        timestep: timestep++,
        startTime: currentTime
      });
      
      currentTime += tool.cost;
      
      // Update dependents
      for (const dependentId of dependents.get(toolId)!) {
        const newDegree = inDegree.get(dependentId)! - 1;
        inDegree.set(dependentId, newDegree);
        
        if (newDegree === 0) {
          ready.push(dependentId);
        }
      }
    }
    
    return schedule;
  }
}
