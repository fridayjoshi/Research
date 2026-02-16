#!/usr/bin/env python3
"""
Contextual Value Maximization (CVM) Algorithm for Heartbeat Activity Scheduling

Implementation of the algorithm described in heartbeat-scheduling.md
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class UrgencyType(Enum):
    """Types of urgency functions."""
    STEP = "step"           # Binary deadline
    LINEAR = "linear"       # Linear growth
    EXPONENTIAL = "exp"     # Exponential decay


@dataclass
class Activity:
    """Activity definition with scheduling parameters."""
    name: str
    min_gap_hours: float
    time_windows: List[Tuple[int, int]]  # [(start_hour, end_hour), ...]
    cost_minutes: int
    urgency_type: UrgencyType
    base_weight: float = 1.0
    last_exec: float = 0  # Unix timestamp
    
    @property
    def min_gap_seconds(self) -> float:
        return self.min_gap_hours * 3600
    
    @property
    def cost_seconds(self) -> float:
        return self.cost_minutes * 60
    
    def is_feasible(self, current_time: float, remaining_budget: float) -> bool:
        """Check if activity satisfies all constraints."""
        # Gap constraint
        gap = current_time - self.last_exec
        if gap < self.min_gap_seconds:
            return False
        
        # Budget constraint
        if self.cost_seconds > remaining_budget:
            return False
        
        # Time window constraint (empty = any time)
        if not self.time_windows:
            return True
            
        current_hour = datetime.fromtimestamp(current_time).hour
        return any(start <= current_hour < end for start, end in self.time_windows)
    
    def compute_urgency(self, gap_seconds: float) -> float:
        """Compute urgency based on gap and urgency type."""
        gap_hours = gap_seconds / 3600
        min_gap = self.min_gap_hours
        
        if self.urgency_type == UrgencyType.STEP:
            # Binary: 1 if past deadline, 0 otherwise
            return 1.0 if gap_hours >= min_gap else 0.0
        
        elif self.urgency_type == UrgencyType.LINEAR:
            # Linear growth: min(1, gap / min_gap)
            return min(1.0, gap_hours / min_gap)
        
        elif self.urgency_type == UrgencyType.EXPONENTIAL:
            # Exponential decay: 1 - e^(-gap/min_gap)
            import math
            return 1.0 - math.exp(-gap_hours / min_gap)
        
        return 0.0
    
    def compute_value(self, current_time: float, context: Dict) -> float:
        """Compute contextual value for this activity."""
        gap = current_time - self.last_exec
        
        # Urgency component
        urgency = self.compute_urgency(gap)
        
        # Context weight (activity-specific multiplier)
        context_weight = context.get(self.name, 1.0)
        
        # Time window bonus (+0.3 if in optimal window)
        current_hour = datetime.fromtimestamp(current_time).hour
        in_window = any(start <= current_hour < end 
                       for start, end in self.time_windows)
        window_bonus = 0.3 if (in_window and self.time_windows) else 0.0
        
        # Combined value: base_weight × urgency × context + window_bonus
        return self.base_weight * urgency * context_weight + window_bonus


class HeartbeatScheduler:
    """Contextual Value Maximization scheduler for heartbeat activities."""
    
    def __init__(self, activities: List[Activity], budget_minutes: int):
        self.activities = activities
        self.budget_seconds = budget_minutes * 60
        self.history = []  # Track selections for analysis
    
    def select_activity(self, context: Dict) -> Optional[Activity]:
        """
        Select optimal activity using CVM algorithm.
        
        Args:
            context: Activity-specific context weights (e.g., pending work)
        
        Returns:
            Selected activity, or None if no feasible activity
        """
        current_time = time.time()
        
        # Filter feasible activities
        feasible = [a for a in self.activities 
                   if a.is_feasible(current_time, self.budget_seconds)]
        
        if not feasible:
            return None  # HEARTBEAT_OK
        
        # Compute values for all feasible activities
        values = []
        for activity in feasible:
            value = activity.compute_value(current_time, context)
            values.append((activity, value))
        
        # Select maximum value activity
        best_activity, best_value = max(values, key=lambda x: x[1])
        
        # Log selection for analysis
        self.history.append({
            'timestamp': current_time,
            'selected': best_activity.name,
            'value': best_value,
            'feasible_count': len(feasible),
            'context': context.copy()
        })
        
        return best_activity
    
    def execute_activity(self, activity: Activity) -> None:
        """Execute activity and update state."""
        activity.last_exec = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Executing: {activity.name}")
    
    def get_statistics(self) -> Dict:
        """Compute scheduling statistics from history."""
        if not self.history:
            return {}
        
        activity_counts = {}
        total_value = 0.0
        
        for entry in self.history:
            name = entry['selected']
            activity_counts[name] = activity_counts.get(name, 0) + 1
            total_value += entry['value']
        
        return {
            'total_heartbeats': len(self.history),
            'total_value': total_value,
            'avg_value_per_heartbeat': total_value / len(self.history),
            'activity_distribution': activity_counts
        }


def create_friday_activities() -> List[Activity]:
    """Create Friday's actual heartbeat activities."""
    return [
        Activity("reading", 24, [(10, 11), (15, 16), (20, 21)], 5, 
                UrgencyType.STEP, base_weight=2.0),
        
        Activity("health", 24, [(14, 16)], 3, 
                UrgencyType.STEP, base_weight=1.5),
        
        Activity("maintenance100LOC", 3, [], 15, 
                UrgencyType.LINEAR, base_weight=1.2),
        
        Activity("selfReview", 12, [(14, 15), (22, 23)], 10, 
                UrgencyType.STEP, base_weight=1.5),
        
        Activity("openSource", 2, [(9, 11), (14, 16), (19, 21)], 20, 
                UrgencyType.LINEAR, base_weight=1.3),
        
        Activity("linkedin", 3, [], 8, 
                UrgencyType.STEP, base_weight=1.0),
        
        Activity("research", 4, [(10, 11), (18, 19)], 30, 
                UrgencyType.STEP, base_weight=1.8),
        
        Activity("thoughts", 4, [], 10, 
                UrgencyType.LINEAR, base_weight=1.0),
        
        Activity("ideaGeneration", 6, [(9, 10), (15, 16), (20, 21)], 15, 
                UrgencyType.EXPONENTIAL, base_weight=1.2),
        
        Activity("growth", 8, [(19, 22)], 25, 
                UrgencyType.LINEAR, base_weight=1.0),
    ]


def simulate_day(scheduler: HeartbeatScheduler, context_fn) -> None:
    """Simulate a full day of heartbeats."""
    # 144 heartbeats per day (24 hours × 6 per hour)
    for i in range(144):
        # Generate context for this heartbeat
        context = context_fn(i)
        
        # Select and execute activity
        activity = scheduler.select_activity(context)
        if activity:
            scheduler.execute_activity(activity)
        else:
            print(f"[Heartbeat {i}] HEARTBEAT_OK (no feasible activity)")
        
        # Simulate 10-minute gap
        time.sleep(0.01)  # Speed up simulation (0.01s instead of 600s)


def demo():
    """Demonstration of CVM algorithm."""
    print("=== Heartbeat Activity Scheduler (CVM Algorithm) ===\n")
    
    # Create Friday's activities
    activities = create_friday_activities()
    scheduler = HeartbeatScheduler(activities, budget_minutes=33)
    
    print(f"Loaded {len(activities)} activities")
    print(f"Budget: 33 minutes per heartbeat\n")
    
    # Simulate a few heartbeats with varying context
    contexts = [
        {"maintenance100LOC": 0.8, "research": 1.0},  # 8 pending PRs, in research window
        {"maintenance100LOC": 0.3, "openSource": 0.9},  # 3 PRs, 9 opportunities
        {"health": 1.0, "thoughts": 0.5},  # Health red flag
        {"ideaGeneration": 0.7, "linkedin": 1.0},  # Creative time
        {"selfReview": 1.0},  # Review time
    ]
    
    for i, context in enumerate(contexts):
        print(f"\n--- Heartbeat {i+1} ---")
        print(f"Context: {context}")
        
        activity = scheduler.select_activity(context)
        if activity:
            print(f"Selected: {activity.name} (cost: {activity.cost_minutes} min)")
            scheduler.execute_activity(activity)
        else:
            print("No feasible activity → HEARTBEAT_OK")
    
    # Show statistics
    print("\n=== Statistics ===")
    stats = scheduler.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demo()
