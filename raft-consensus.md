# Raft Consensus Algorithm

**Research Date:** May 21, 2026  
**Topic:** Distributed Systems - Consensus Algorithms  
**Algorithm:** Raft  

## Overview

Raft is a consensus algorithm designed to be more understandable than Paxos while providing the same guarantees. It manages a replicated log across a cluster of servers, ensuring that all servers agree on the same sequence of commands despite failures.

## Problem Statement

In a distributed system with **n** servers (where n = 2f + 1 for tolerating f failures), how can we ensure:
1. **Safety:** All servers agree on the same log entries (never returning different results)
2. **Liveness:** The system makes progress as long as a majority of servers are operational
3. **Fault tolerance:** The system tolerates up to f crash failures

## Raft Algorithm

### Three Roles

1. **Leader:** Handles all client requests, replicates log to followers
2. **Follower:** Passive, responds to leader/candidate requests
3. **Candidate:** Seeks election to become leader

### Two Main Phases

#### 1. Leader Election

**Trigger:** Follower times out waiting for leader heartbeat (150-300ms randomized)

**Process:**
1. Follower increments `currentTerm` and transitions to Candidate
2. Votes for itself and requests votes from other servers via `RequestVote` RPC
3. **Election won if:** Receives votes from majority of servers
4. **Election lost if:** Receives `AppendEntries` from new leader with term ≥ currentTerm
5. **Split vote:** Timeout expires, start new election with incremented term

**Key Invariant:** At most one leader per term

**RequestVote RPC:**
```
Arguments:
  term: candidate's term
  candidateId: candidate requesting vote
  lastLogIndex: index of candidate's last log entry
  lastLogTerm: term of candidate's last log entry

Results:
  term: currentTerm, for candidate to update itself
  voteGranted: true means candidate received vote

Receiver implementation:
  1. Reply false if term < currentTerm
  2. If votedFor is null or candidateId, and candidate's log is at least
     as up-to-date as receiver's log, grant vote
```

**Log up-to-date comparison:** If logs have entries with different terms, the log with later term is more up-to-date. If logs end with same term, longer log is more up-to-date.

#### 2. Log Replication

**Process:**
1. Leader receives command from client
2. Appends entry to local log
3. Issues `AppendEntries` RPC to followers in parallel
4. Once entry is replicated on majority: entry is **committed**
5. Leader applies committed entry to state machine, returns result to client
6. Leader tracks `commitIndex` and sends it in future `AppendEntries`
7. Followers apply committed entries to their state machines

**AppendEntries RPC:**
```
Arguments:
  term: leader's term
  leaderId: for followers to redirect clients
  prevLogIndex: index of log entry immediately preceding new ones
  prevLogTerm: term of prevLogIndex entry
  entries[]: log entries to store (empty for heartbeat)
  leaderCommit: leader's commitIndex

Results:
  term: currentTerm, for leader to update itself
  success: true if follower had entry matching prevLogIndex and prevLogTerm

Receiver implementation:
  1. Reply false if term < currentTerm
  2. Reply false if log doesn't contain entry at prevLogIndex whose term matches prevLogTerm
  3. If existing entry conflicts (same index, different term), delete it and all following
  4. Append any new entries not already in log
  5. If leaderCommit > commitIndex, set commitIndex = min(leaderCommit, index of last new entry)
```

**Log Matching Property:**
- If two entries in different logs have same index and term, they store same command
- If two entries in different logs have same index and term, all preceding entries are identical

**Leader Completeness:** If a log entry is committed in a given term, that entry will be present in logs of leaders for all higher terms.

## Formal Properties

### Safety

**State Machine Safety Property:** If a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for the same index.

**Proof sketch:**
1. Leader can only commit entries from current term by counting replicas
2. Committed entry must be stored on majority of servers
3. Any future leader must win election from a majority
4. Intersection of majorities ensures new leader has committed entry
5. Leader never overwrites its own entries
6. Log Matching Property ensures consistency propagates

### Liveness

**Election Safety:** At most one leader per term

**Leader Append-Only:** Leader never overwrites or deletes entries in its log

**Progress Guarantee:** System makes progress if:
- Majority of servers are operational
- Servers can communicate with bounded delay
- Random election timeouts prevent split votes indefinitely

## Complexity Analysis

### Time Complexity

- **Leader election:** O(1) rounds in expectation with randomized timeouts
- **Log replication:** O(1) rounds for single entry (parallel RPCs)
- **Recovering follower:** O(log n) RPCs to find last matching index (binary search optimization)

### Space Complexity

- **Per server:** O(log size) for storing log entries
- **Messages:** O(batch size) for AppendEntries with batching

### Network Complexity

- **Heartbeats:** O(n) messages per heartbeat interval (leader → all followers)
- **Normal operation:** O(n) messages per client request (leader → followers)
- **Election:** O(n²) messages in worst case (all servers timeout, send RequestVote)

## Implementation Notes

### Key Parameters

```
electionTimeout: 150-300ms (randomized)
heartbeatInterval: 50ms (< electionTimeout)
batchSize: 100-1000 entries (balance latency vs throughput)
```

### Optimizations

1. **Batching:** Group multiple entries in single AppendEntries RPC
2. **Pipelining:** Don't wait for previous AppendEntries before sending next
3. **Log compaction:** Snapshot state machine, discard old log entries
4. **Fast log backtracking:** Include conflicting term info in AppendEntries response

## Python Implementation (Simplified)

```python
import time
import random
import threading
from enum import Enum
from typing import List, Dict, Optional

class Role(Enum):
    FOLLOWER = 1
    CANDIDATE = 2
    LEADER = 3

class LogEntry:
    def __init__(self, term: int, command: str):
        self.term = term
        self.command = command

class RaftNode:
    def __init__(self, node_id: int, peers: List[int]):
        self.node_id = node_id
        self.peers = peers
        
        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[int] = None
        self.log: List[LogEntry] = []
        
        # Volatile state
        self.commit_index = 0
        self.last_applied = 0
        self.role = Role.FOLLOWER
        
        # Leader state (reinitialized after election)
        self.next_index: Dict[int, int] = {}
        self.match_index: Dict[int, int] = {}
        
        # Timing
        self.election_timeout = random.uniform(0.15, 0.30)
        self.last_heartbeat = time.time()
        
    def run(self):
        """Main loop"""
        while True:
            if self.role == Role.FOLLOWER:
                self.follower_loop()
            elif self.role == Role.CANDIDATE:
                self.candidate_loop()
            elif self.role == Role.LEADER:
                self.leader_loop()
    
    def follower_loop(self):
        """Wait for heartbeat or start election"""
        if time.time() - self.last_heartbeat > self.election_timeout:
            self.role = Role.CANDIDATE
        time.sleep(0.01)
    
    def candidate_loop(self):
        """Request votes and wait for majority"""
        self.current_term += 1
        self.voted_for = self.node_id
        votes_received = 1  # Vote for self
        
        # Send RequestVote RPCs (simplified - no actual RPC)
        for peer in self.peers:
            # In real implementation: send RequestVote RPC
            # vote = self.send_request_vote(peer, ...)
            # if vote: votes_received += 1
            pass
        
        # Check if won election
        if votes_received > len(self.peers) // 2:
            self.role = Role.LEADER
            self.initialize_leader_state()
        else:
            # Reset timeout and become follower
            self.role = Role.FOLLOWER
            self.last_heartbeat = time.time()
            self.election_timeout = random.uniform(0.15, 0.30)
    
    def leader_loop(self):
        """Send heartbeats and replicate log"""
        # Send AppendEntries to all followers
        for peer in self.peers:
            # In real implementation: send AppendEntries RPC
            # self.send_append_entries(peer)
            pass
        
        time.sleep(0.05)  # Heartbeat interval
    
    def initialize_leader_state(self):
        """Initialize leader state after election"""
        for peer in self.peers:
            self.next_index[peer] = len(self.log) + 1
            self.match_index[peer] = 0
    
    def append_entries_rpc(self, args: dict) -> dict:
        """Handle AppendEntries RPC"""
        term = args['term']
        leader_id = args['leaderId']
        prev_log_index = args['prevLogIndex']
        prev_log_term = args['prevLogTerm']
        entries = args['entries']
        leader_commit = args['leaderCommit']
        
        # Reply false if term < currentTerm
        if term < self.current_term:
            return {'term': self.current_term, 'success': False}
        
        # Reset election timeout
        self.last_heartbeat = time.time()
        
        # Step down if term is higher
        if term > self.current_term:
            self.current_term = term
            self.role = Role.FOLLOWER
            self.voted_for = None
        
        # Reply false if log doesn't match
        if prev_log_index > 0:
            if len(self.log) < prev_log_index or \
               self.log[prev_log_index - 1].term != prev_log_term:
                return {'term': self.current_term, 'success': False}
        
        # Delete conflicting entries and append new ones
        for i, entry in enumerate(entries):
            log_index = prev_log_index + i + 1
            if len(self.log) >= log_index:
                if self.log[log_index - 1].term != entry.term:
                    # Delete conflicting entry and all following
                    self.log = self.log[:log_index - 1]
                    self.log.append(entry)
            else:
                self.log.append(entry)
        
        # Update commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log))
        
        return {'term': self.current_term, 'success': True}
```

## Benchmarks

Performance characteristics (typical deployment):

| Metric | Value | Conditions |
|--------|-------|------------|
| Leader election | 100-500ms | 5-node cluster, no contention |
| Commit latency | 1-10ms | Local datacenter, 95th percentile |
| Throughput | 10K-100K ops/sec | Batching enabled, SSD storage |
| Recovery time | 1-5 seconds | Single server crash, log replay |

## Comparison with Paxos

| Aspect | Raft | Paxos |
|--------|------|-------|
| Understandability | High (explicitly designed for teaching) | Low (notoriously complex) |
| Leader election | Explicit with randomized timeouts | Implicit via prepare phase |
| Log structure | Strongly sequential (no gaps) | Can have gaps |
| Implementation | Straightforward | Many subtle variants |
| Performance | Comparable | Comparable |

## Real-World Usage

- **etcd:** Kubernetes cluster state management
- **Consul:** Service discovery and configuration
- **CockroachDB:** Distributed SQL database
- **TiKV:** Distributed key-value store
- **LogCabin:** Educational implementation by Raft authors

## References

1. Ongaro, D., & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." *USENIX ATC*.
2. Ongaro, D. (2014). "Consensus: Bridging Theory and Practice." *PhD dissertation, Stanford University*.
3. Lamport, L. (1998). "The Part-Time Parliament." *ACM Transactions on Computer Systems*.
4. [Raft visualization](https://raft.github.io/)
5. [etcd Raft library](https://github.com/etcd-io/raft)

## Key Insights

1. **Simplicity via decomposition:** Raft separates leader election, log replication, and safety into independent subproblems
2. **Strong leader:** All log entries flow from leader to followers (never the reverse)
3. **Randomization:** Random election timeouts elegantly solve split-vote scenarios
4. **Majority quorums:** Intersection property ensures safety without requiring all servers
5. **Log completeness:** Leaders never overwrite committed entries by restricting elections to most up-to-date candidates

---

**Complexity:** Leader election O(1) rounds expected, log replication O(1) round trip, space O(log size)  
**Safety:** Proven via State Machine Safety Property and Leader Completeness  
**Implementation:** ~2000 LOC for production-ready version (etcd/raft)
