#!/usr/bin/env python3
"""
Tool Call Scheduler - Adaptive Value-Density Scheduling (AVDS)
Implements the algorithm from ../algorithms/tool-call-scheduling.md
"""

from dataclasses import dataclass
from typing import List, Set, Dict, Optional
import json


@dataclass
class Tool:
    """Represents a callable tool with cost, value, and dependencies."""
    id: str
    cost: int  # Token cost
    base_value: float  # Expected information gain [0, 1]
    dependencies: Set[str]  # IDs of tools that must execute first
    tool_type: str = "generic"  # For type-specific probability estimation
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


class QueryContext:
    """Current conversation state for value estimation."""
    def __init__(self, query: str, keywords: Set[str]):
        self.query = query
        self.keywords = keywords
        self.information_gathered = set()
    
    def add_information(self, tool_id: str):
        self.information_gathered.add(tool_id)
    
    def copy(self):
        new_ctx = QueryContext(self.query, self.keywords.copy())
        new_ctx.information_gathered = self.information_gathered.copy()
        return new_ctx


def estimate_utility_probability(tool: Tool, context: QueryContext, executed: Set[Tool]) -> float:
    """
    Estimate probability that tool will provide useful information.
    
    Heuristics:
    - Already executed dependencies increase probability (context available)
    - Keyword overlap with query increases probability
    - Diminishing returns: similar tools executed reduce probability
    """
    # Base probability
    prob = 0.5
    
    # Boost if dependencies satisfied (means we have context to use this tool)
    if tool.dependencies.issubset({t.id for t in executed}):
        prob += 0.2
    
    # Boost based on query relevance (simple keyword match)
    tool_keywords = set(tool.id.lower().split('_'))
    overlap = len(tool_keywords.intersection(context.keywords))
    if overlap > 0:
        prob += min(0.3, overlap * 0.1)
    
    # Diminishing returns: similar tools already executed
    similar_executed = sum(
        1 for t in executed 
        if t.tool_type == tool.tool_type or any(
            kw in t.id.lower() for kw in tool_keywords
        )
    )
    prob *= (0.8 ** similar_executed)  # Decay by 20% per similar tool
    
    return min(1.0, max(0.0, prob))


def can_parallelize(tool: Tool, batch: List[Tool]) -> bool:
    """
    Check if tool can execute in parallel with current batch.
    
    For simplicity, assume all tools can parallelize unless they share resources.
    In practice, would check for file conflicts, mutex requirements, etc.
    """
    # Simple heuristic: tools of the same type might conflict
    for existing in batch:
        if tool.tool_type == existing.tool_type and "write" in tool.tool_type:
            return False  # Don't parallelize writes of same type
    return True


def compute_transitive_value(tool: Tool, all_tools: List[Tool]) -> float:
    """
    Compute value including successors that depend on this tool.
    
    If executing tool enables high-value successors, boost its value.
    """
    direct_value = tool.base_value
    
    # Find all tools that depend on this tool (directly or transitively)
    successors_value = sum(
        successor.base_value * 0.5  # Discount future value
        for successor in all_tools
        if tool.id in successor.dependencies
    )
    
    return direct_value + successors_value


def adaptive_value_density_schedule(
    tools: List[Tool], 
    budget: int, 
    query_context: QueryContext
) -> tuple[List[List[Tool]], float]:
    """
    Schedule tool calls to maximize information gain under token budget.
    
    Returns:
        schedule: List of tool batches (each batch executes in parallel)
        expected_value: Estimated total information gain
    """
    schedule = []
    remaining_budget = budget
    executed = set()
    available = {t for t in tools if len(t.dependencies) == 0}
    
    iteration = 0
    max_iterations = len(tools) + 1  # Prevent infinite loops
    
    while remaining_budget > 0 and available and iteration < max_iterations:
        iteration += 1
        
        # Compute adjusted values for all available tools
        candidates = []
        for tool in available:
            # Probability this tool will be useful
            prob = estimate_utility_probability(tool, query_context, executed)
            
            # Value including transitive dependencies
            transitive_value = compute_transitive_value(tool, tools)
            
            # Adjusted value = probability × transitive value
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
                # Check parallelizability
                if can_parallelize(tool, batch):
                    batch.append(tool)
                    batch_cost += tool.cost
        
        if not batch:
            break  # No tools fit in remaining budget
        
        # Add batch to schedule
        schedule.append(batch)
        remaining_budget -= batch_cost
        executed.update(batch)
        
        # Update available tools (dependencies now satisfied)
        newly_available = set()
        for t in tools:
            if t not in executed:
                deps_satisfied = all(
                    any(executed_tool.id == dep for executed_tool in executed)
                    for dep in t.dependencies
                )
                if deps_satisfied:
                    newly_available.add(t)
        
        available = newly_available
        
        # Update query context with simulated results
        for tool in batch:
            query_context.add_information(tool.id)
    
    # Calculate expected value
    expected_value = sum(
        estimate_utility_probability(t, query_context, executed) * t.base_value
        for t in executed
    )
    
    return schedule, expected_value


def format_schedule(schedule: List[List[Tool]], budget: int) -> str:
    """Format schedule for human-readable output."""
    lines = []
    total_cost = 0
    
    for i, batch in enumerate(schedule, 1):
        batch_cost = sum(t.cost for t in batch)
        total_cost += batch_cost
        
        lines.append(f"\nBatch {i} (cost: {batch_cost} tokens):")
        for tool in batch:
            lines.append(f"  - {tool.id} [value: {tool.base_value:.2f}, cost: {tool.cost}]")
    
    lines.append(f"\nTotal cost: {total_cost} / {budget} tokens")
    lines.append(f"Budget utilization: {total_cost / budget * 100:.1f}%")
    
    return "\n".join(lines)


# Example test case: Email triage scenario
if __name__ == "__main__":
    print("Tool Call Scheduler - Test Case: Email Triage\n")
    
    tools = [
        Tool("read_MEMORY", cost=1500, base_value=0.9, dependencies=set(), tool_type="read"),
        Tool("read_USER", cost=800, base_value=0.85, dependencies=set(), tool_type="read"),
        Tool("memory_search_contacts", cost=600, base_value=0.7, dependencies=set(), tool_type="search"),
        Tool("read_email_1", cost=1200, base_value=0.8, dependencies={"memory_search_contacts"}, tool_type="read"),
        Tool("read_email_2", cost=1100, base_value=0.75, dependencies={"memory_search_contacts"}, tool_type="read"),
        Tool("read_email_3", cost=900, base_value=0.6, dependencies={"memory_search_contacts"}, tool_type="read"),
        Tool("classify_email_1", cost=200, base_value=0.5, dependencies={"read_email_1"}, tool_type="exec"),
        Tool("classify_email_2", cost=200, base_value=0.5, dependencies={"read_email_2"}, tool_type="exec"),
    ]
    
    budget = 5000
    query_context = QueryContext(
        query="Check my emails and prioritize responses",
        keywords={"email", "check", "prioritize", "response", "contacts"}
    )
    
    print(f"Tools available: {len(tools)}")
    print(f"Token budget: {budget}")
    print(f"Query: {query_context.query}\n")
    
    schedule, expected_value = adaptive_value_density_schedule(tools, budget, query_context)
    
    print(format_schedule(schedule, budget))
    print(f"\nExpected information gain: {expected_value:.3f}")
    
    # Show executed tools
    executed = [tool for batch in schedule for tool in batch]
    print(f"\nExecuted {len(executed)}/{len(tools)} tools:")
    for tool in executed:
        print(f"  ✓ {tool.id}")
    
    not_executed = [t for t in tools if t not in executed]
    if not_executed:
        print(f"\nSkipped {len(not_executed)} tools (budget exhausted):")
        for tool in not_executed:
            print(f"  ✗ {tool.id}")
