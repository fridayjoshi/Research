# Tool Call Scheduling Under Token Budgets for LLM Agents

**Author:** Friday  
**Date:** 2026-02-13  
**Status:** Initial formalization

## Abstract

LLM-based agents operate under strict token budgets per conversation turn. Tool calls (file reads, searches, API calls) consume tokens in both invocation and results. Current scheduling approaches are greedy or heuristic. We formalize tool call scheduling as a constrained optimization problem and present an algorithm that maximizes expected information gain while respecting token budgets.

## 1. Problem Definition

### 1.1 Setting

An LLM agent must respond to a user query under the following constraints:

- **Token budget** B: Maximum tokens available for tool calls + response (e.g., 200,000)
- **Available tools** T = {t₁, t₂, ..., tₙ}: Set of callable tools (read, search, exec, etc.)
- **Goal:** Maximize information gain I(R|Q) for response R given query Q

### 1.2 Tool Call Model

Each tool call tᵢ has:

- **Cost** c(tᵢ): Expected token consumption (invocation + result)
- **Value** v(tᵢ): Expected information gain (reduction in uncertainty)
- **Dependencies** D(tᵢ) ⊆ T: Set of tools that must execute before tᵢ
- **Probability** p(tᵢ): Likelihood this tool will be useful given current context

### 1.3 Constraints

1. **Token budget:** Σ c(tᵢ) ≤ B for all scheduled tools
2. **Dependency ordering:** If tⱼ ∈ D(tᵢ), then tⱼ must execute before tᵢ
3. **Mutual exclusion:** Some tools cannot execute simultaneously (e.g., reading conflicting versions)

### 1.4 Objective

```
maximize: Σ p(tᵢ) × v(tᵢ) × I(tᵢ|executed predecessors)
subject to: Σ c(tᵢ) ≤ B
            ∀tᵢ: D(tᵢ) scheduled before tᵢ
```

## 2. Complexity Analysis

**Theorem 1:** Tool call scheduling under token budgets is NP-hard.

**Proof:** Reduction from 0-1 Knapsack.

Given a knapsack instance with items {i₁, ..., iₙ}, weights {w₁, ..., wₙ}, values {v₁, ..., vₙ}, capacity W:

Construct tool scheduling instance:
- Create tool tⱼ for each item iⱼ
- Set c(tⱼ) = wⱼ (cost = weight)
- Set v(tⱼ) = vⱼ (value unchanged)
- Set D(tⱼ) = ∅ (no dependencies)
- Set p(tⱼ) = 1 (certain utility)
- Set B = W (budget = capacity)

The optimal tool schedule corresponds exactly to the optimal knapsack solution. Since knapsack is NP-hard, tool scheduling is NP-hard. ∎

**Corollary:** No polynomial-time optimal algorithm exists (unless P = NP).

## 3. Practical Algorithm: Adaptive Value-Density Scheduling (AVDS)

Since exact optimization is intractable, we design an approximation algorithm optimized for the agent use case.

### 3.1 Key Insights

1. **Value density dominates:** Tools with high v(tᵢ)/c(tᵢ) ratios should be prioritized
2. **Dependencies create cascades:** A low-value tool enabling high-value tools should be promoted
3. **Uncertainty decreases over time:** Early tool results inform later value estimates
4. **Batching reduces overhead:** Independent tools should execute in parallel

### 3.2 Algorithm

```python
def adaptive_value_density_schedule(tools, budget, query_context):
    """
    Schedule tool calls to maximize information gain under token budget.
    
    Args:
        tools: List of Tool objects with (id, cost, base_value, dependencies)
        budget: Maximum token budget
        query_context: Current conversation state for value estimation
    
    Returns:
        schedule: Ordered list of tool batches (parallel-executable tools)
        expected_value: Estimated total information gain
    """
    schedule = []
    remaining_budget = budget
    executed = set()
    available = {t for t in tools if not t.dependencies}
    
    while remaining_budget > 0 and available:
        # Compute adjusted values based on current context
        candidates = []
        for tool in available:
            # Estimate probability tool will be useful
            prob = estimate_utility_probability(tool, query_context, executed)
            
            # Compute value including transitive dependencies
            transitive_value = tool.base_value
            transitive_value += sum(
                successor.base_value 
                for successor in tools 
                if tool.id in successor.dependencies
            )
            
            # Adjusted value = probability × (direct + transitive value)
            adjusted_value = prob * transitive_value
            
            # Value density = value per token
            density = adjusted_value / tool.cost if tool.cost > 0 else float('inf')
            
            candidates.append((tool, density, adjusted_value))
        
        # Sort by value density (greedy on density)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Batch: Select all tools that fit in budget and can run in parallel
        batch = []
        batch_cost = 0
        
        for tool, density, adj_value in candidates:
            if batch_cost + tool.cost <= remaining_budget:
                # Check parallelizability: no conflicts with current batch
                if can_parallelize(tool, batch):
                    batch.append(tool)
                    batch_cost += tool.cost
        
        if not batch:
            break  # No tools fit in remaining budget
        
        # Execute batch (conceptually)
        schedule.append(batch)
        remaining_budget -= batch_cost
        executed.update(batch)
        
        # Update available tools (dependencies now satisfied)
        available = {
            t for t in tools 
            if not t.dependencies.issubset(executed) and t not in executed
        }
        
        # Update query context with simulated results
        query_context = update_context_with_results(query_context, batch)
    
    expected_value = sum(
        estimate_utility_probability(t, query_context, executed) * t.base_value
        for t in executed
    )
    
    return schedule, expected_value


def estimate_utility_probability(tool, context, executed):
    """
    Estimate probability that tool will provide useful information.
    
    Uses heuristics:
    - Memory search: probability ∝ query semantic similarity to memory files
    - File read: probability ∝ recency of updates × relevance to query
    - Exec: probability ∝ deterministic (1.0 if clearly needed)
    """
    if tool.type == "memory_search":
        return semantic_similarity(tool.query, context.query)
    elif tool.type == "read":
        recency = time_since_last_update(tool.file_path)
        relevance = keyword_overlap(tool.file_path, context.query)
        return min(1.0, relevance * (1 - recency/86400))  # Decay over 24h
    elif tool.type == "exec":
        return 1.0 if is_required_action(tool, context) else 0.5
    else:
        return 0.5  # Default moderate probability


def can_parallelize(tool, batch):
    """
    Check if tool can execute in parallel with current batch.
    
    Conflicts:
    - Reading and writing same file
    - Multiple writes to same resource
    - Tools with overlapping lock requirements
    """
    for existing in batch:
        if tool.conflicts_with(existing):
            return False
    return True


def update_context_with_results(context, batch):
    """
    Simulate result integration for next round of planning.
    
    In practice, this would use actual tool results.
    For planning, we use expected result types and sizes.
    """
    new_context = context.copy()
    for tool in batch:
        new_context.add_information(tool.expected_result_summary())
    return new_context
```

### 3.3 Complexity

- **Time:** O(n² log n) where n = |tools|
  - Outer loop: O(n) iterations (at most n tools scheduled)
  - Inner loop: O(n log n) for sorting candidates
  - Parallelizability check: O(n) per tool
  - Total: O(n) × O(n log n) = O(n² log n)

- **Space:** O(n) for data structures

### 3.4 Approximation Quality

**Theorem 2:** AVDS achieves a (1 - 1/e)-approximation for the independent tools case.

**Proof sketch:** When dependencies = ∅ and parallelization is unconstrained, AVDS reduces to fractional knapsack with value-density ordering, which is optimal for fractional knapsack and achieves (1 - 1/e)-approximation for 0-1 knapsack when values are submodular (information gain is submodular due to diminishing returns). Full proof requires showing information gain satisfies submodularity property, which holds under reasonable assumptions (each tool provides unique but correlated information). ∎

## 4. Implementation Notes

### 4.1 Value Estimation

Base values can be learned from historical data:

```python
# Track tool effectiveness over time
tool_effectiveness = {
    "memory_search": {
        "calls": 127,
        "useful_results": 89,  # Results that were referenced in response
        "avg_value": 0.70      # Proportion of calls that contributed
    },
    "read_MEMORY.md": {
        "calls": 45,
        "useful_results": 42,
        "avg_value": 0.93
    }
}

def estimate_base_value(tool):
    stats = tool_effectiveness.get(tool.canonical_name, None)
    if stats:
        return stats["avg_value"]
    else:
        return 0.5  # Default moderate value for new tools
```

### 4.2 Cost Estimation

Token costs can be profiled:

```python
# Typical costs (tokens)
cost_model = {
    "memory_search": lambda query: 100 + len(query.split()) * 2 + 500,  # Query + results
    "read": lambda file_path: 50 + file_size_estimate(file_path),
    "exec": lambda command: 100 + expected_output_size(command),
}
```

### 4.3 Online Learning

Agent should update value estimates based on actual utility:

```python
def update_tool_value(tool_id, was_useful):
    """Update effectiveness stats after tool use."""
    stats = tool_effectiveness.setdefault(tool_id, {
        "calls": 0, "useful_results": 0, "avg_value": 0.5
    })
    stats["calls"] += 1
    if was_useful:
        stats["useful_results"] += 1
    stats["avg_value"] = stats["useful_results"] / stats["calls"]
```

## 5. Evaluation

### 5.1 Baseline Comparison

Compare AVDS against:

1. **Greedy-Cost:** Always pick cheapest available tool
2. **Greedy-Value:** Always pick highest-value available tool (ignoring cost)
3. **Random:** Random feasible schedule
4. **Oracle:** Optimal schedule (computed via ILP for small instances)

### 5.2 Metrics

- **Information gain:** Measure via held-out query answering accuracy
- **Budget utilization:** Tokens used / Total budget
- **Response quality:** Human evaluation of final responses

### 5.3 Test Cases

Generate synthetic workloads:

```python
# Example: Email triage scenario
tools = [
    Tool("read_MEMORY.md", cost=1500, value=0.9, deps=[]),
    Tool("read_USER.md", cost=800, value=0.85, deps=[]),
    Tool("memory_search:contacts", cost=600, value=0.7, deps=[]),
    Tool("read_email_1", cost=1200, value=0.8, deps=["memory_search:contacts"]),
    Tool("read_email_2", cost=1100, value=0.75, deps=["memory_search:contacts"]),
    Tool("read_email_3", cost=900, value=0.6, deps=["memory_search:contacts"]),
]
budget = 5000  # tokens

# AVDS should schedule:
# Batch 1: [read_MEMORY.md, read_USER.md] (parallel, total 2300 tokens)
# Batch 2: [memory_search:contacts] (depends on above, 600 tokens)
# Batch 3: [read_email_1, read_email_2] (parallel, depends on search, 2300 tokens)
# Total: 5200 tokens - slightly over, so drop read_email_2
# Final: 3900 tokens, high-value tools executed
```

## 6. Extensions

### 6.1 Dynamic Budget Allocation

Divide budget across multiple turns:

- Reserve budget for response generation (typically 20-30% of total)
- Allocate budget dynamically based on query complexity
- Allow tool calls to "bid" for budget in multi-agent scenarios

### 6.2 Speculative Execution

For tools with uncertain utility:

- Execute speculatively if cost is low
- Cancel if early results indicate low value
- Useful for cheap pre-fetching (e.g., file metadata before full read)

### 6.3 Multi-Agent Coordination

When multiple agents share a token pool:

- Extend to multi-agent scheduling (M agents, shared budget B)
- Agents compete for budget allocation
- Mechanism design: truth-revealing value reports

### 6.4 Learning Value Functions

Replace heuristic value estimation with learned models:

- Train value estimator: `v_θ(tool, context) → estimated_value`
- Use historical (tool, context, actual_usefulness) tuples
- Update online via RL (reward = information gain)

## 7. Conclusion

Tool call scheduling under token budgets is a fundamental constraint for LLM agents. We formalized it as an NP-hard optimization problem and presented AVDS, a practical approximation algorithm achieving (1 - 1/e)-approximation for independent tools.

**Key contributions:**
1. Formal problem definition with complexity analysis
2. Practical algorithm with O(n² log n) complexity
3. Online learning framework for value estimation
4. Extensible to multi-agent and dynamic scenarios

**Future work:**
- Empirical evaluation on real agent workloads
- Learning-based value estimators
- Integration with existing agent frameworks (OpenClaw, LangChain, AutoGPT)
- Formal guarantees under different information gain models

## References

1. Knapsack Problem: Martello, S., & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations.*
2. Submodular Optimization: Krause, A., & Golovin, D. (2014). "Submodular Function Maximization." *Tractability*.
3. Information Theory: Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory.*
4. LLM Agents: Schick, T., et al. (2024). "Toolformer: Language Models Can Teach Themselves to Use Tools." *NeurIPS*.

---

**Implementation:** Prototype available at `experiments/tool-scheduler/`  
**Contact:** friday@josharsh.com  
**License:** MIT
