# Tool Call Scheduler

Implementation of Adaptive Value-Density Scheduling (AVDS) algorithm for LLM agent tool call optimization.

## Problem

LLM agents have token budgets. Every tool call (file read, search, API call) consumes tokens. How do you decide which tools to call to maximize information gain?

## Solution

AVDS: A greedy approximation algorithm that:
1. Computes value-density (information gain per token) for each tool
2. Accounts for dependencies (some tools require others to execute first)
3. Enables parallel execution (batch independent tools)
4. Achieves (1 - 1/e)-approximation for independent tools

## Usage

```bash
python3 scheduler.py
```

## Example Output

```
Tool Call Scheduler - Test Case: Email Triage

Tools available: 8
Token budget: 5000

Batch 1 (cost: 2900 tokens):
  - memory_search_contacts [value: 0.70, cost: 600]
  - read_USER [value: 0.85, cost: 800]
  - read_MEMORY [value: 0.90, cost: 1500]

Batch 2 (cost: 2000 tokens):
  - read_email_2 [value: 0.75, cost: 1100]
  - read_email_3 [value: 0.60, cost: 900]

Total cost: 4900 / 5000 tokens
Budget utilization: 98.0%
```

## Algorithm Details

See [tool-call-scheduling.md](../../algorithms/tool-call-scheduling.md) for:
- Formal problem definition
- NP-hardness proof
- Algorithm complexity analysis (O(n² log n))
- Approximation guarantees
- Extensions (multi-agent, learning-based)

## Key Features

- **Dependency-aware**: Respects tool execution order constraints
- **Parallel batching**: Executes independent tools simultaneously
- **Transitive value**: Promotes tools that enable high-value successors
- **Adaptive**: Adjusts probabilities based on executed tools
- **Budget-conscious**: Maximizes utilization without exceeding limit

## Complexity

- **Time:** O(n² log n) where n = number of tools
- **Space:** O(n)

## Future Work

- Learn value functions from historical tool effectiveness
- Integrate with OpenClaw agent runtime
- Multi-agent budget allocation
- Speculative execution with early termination

---

**Author:** Friday  
**Date:** 2026-02-13  
**Paper:** `../../algorithms/tool-call-scheduling.md`
