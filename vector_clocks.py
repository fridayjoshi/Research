from typing import List, Tuple
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class VectorClock:
    """Vector clock implementation for n processes."""
    n_processes: int
    clock: List[int]
    process_id: int

    def __init__(self, n_processes: int, process_id: int):
        self.n_processes = n_processes
        self.process_id = process_id
        self.clock = [0] * n_processes

    def local_event(self) -> None:
        """Increment clock for local event."""
        self.clock[self.process_id] += 1

    def send_message(self) -> List[int]:
        """Prepare message with current vector clock."""
        self.local_event()
        return deepcopy(self.clock)

    def receive_message(self, sender_clock: List[int]) -> None:
        """Merge sender's clock and increment own counter."""
        # Take maximum of each component
        for i in range(self.n_processes):
            self.clock[i] = max(self.clock[i], sender_clock[i])
        # Increment own counter
        self.clock[self.process_id] += 1

    def happens_before(self, other_clock: List[int]) -> bool:
        """Check if self → other (self happens before other)."""
        less_equal_all = all(self.clock[i] <= other_clock[i]
                            for i in range(self.n_processes))
        strictly_less_one = any(self.clock[i] < other_clock[i]
                               for i in range(self.n_processes))
        return less_equal_all and strictly_less_one

    def concurrent(self, other_clock: List[int]) -> bool:
        """Check if self || other (concurrent)."""
        return not self.happens_before(other_clock) and \
               not VectorClock.static_happens_before(other_clock, self.clock)

    @staticmethod
    def static_happens_before(clock1: List[int], clock2: List[int]) -> bool:
        """Static version of happens_before for two clocks."""
        n = len(clock1)
        less_equal_all = all(clock1[i] <= clock2[i] for i in range(n))
        strictly_less_one = any(clock1[i] < clock2[i] for i in range(n))
        return less_equal_all and strictly_less_one

    def __repr__(self) -> str:
        return f"VC_P{self.process_id}{self.clock}"

# Example: Distributed system with 3 processes
def simulate_distributed_system():
    """Simulate message passing between 3 processes."""
    print("=== Vector Clock Simulation ===\n")

    # Initialize 3 processes
    p0 = VectorClock(n_processes=3, process_id=0)
    p1 = VectorClock(n_processes=3, process_id=1)
    p2 = VectorClock(n_processes=3, process_id=2)

    print(f"Initial: P0={p0}, P1={p1}, P2={p2}\n")

    # Event 1: P0 local event
    p0.local_event()
    print(f"Event 1 (P0 local): {p0}")
    e1 = deepcopy(p0.clock)

    # Event 2: P1 local event
    p1.local_event()
    print(f"Event 2 (P1 local): {p1}")
    e2 = deepcopy(p1.clock)

    # Event 3: P0 sends to P2
    msg_0_to_2 = p0.send_message()
    print(f"Event 3 (P0 sends to P2): {p0}, msg={msg_0_to_2}")
    e3 = deepcopy(p0.clock)

    # Event 4: P2 receives from P0
    p2.receive_message(msg_0_to_2)
    print(f"Event 4 (P2 receives from P0): {p2}")
    e4 = deepcopy(p2.clock)

    # Event 5: P1 sends to P2
    msg_1_to_2 = p1.send_message()
    print(f"Event 5 (P1 sends to P2): {p1}, msg={msg_1_to_2}")
    e5 = deepcopy(p1.clock)

    # Event 6: P2 receives from P1
    p2.receive_message(msg_1_to_2)
    print(f"Event 6 (P2 receives from P1): {p2}\n")
    e6 = deepcopy(p2.clock)

    # Test causality
    print("=== Causality Analysis ===")
    print(f"e1 → e3? {VectorClock.static_happens_before(e1, e3)} (Expected: True)")
    print(f"e1 → e4? {VectorClock.static_happens_before(e1, e4)} (Expected: True)")
    print(f"e3 → e4? {VectorClock.static_happens_before(e3, e4)} (Expected: True)")
    print(f"e1 → e2? {VectorClock.static_happens_before(e1, e2)} (Expected: False)")
    print(f"e2 → e1? {VectorClock.static_happens_before(e2, e1)} (Expected: False)")
    print(f"e1 || e2? {not VectorClock.static_happens_before(e1, e2) and not VectorClock.static_happens_before(e2, e1)} (Expected: True - concurrent)")
    print(f"e2 → e6? {VectorClock.static_happens_before(e2, e6)} (Expected: True)")
    print(f"e5 → e6? {VectorClock.static_happens_before(e5, e6)} (Expected: True)")

if __name__ == "__main__":
    simulate_distributed_system()
