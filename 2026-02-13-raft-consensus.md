# Raft Consensus Algorithm

**Date:** 2026-02-13  
**Topic:** Distributed Systems, Consensus Protocols  
**Complexity:** Leader election O(n), log replication O(1) per entry

---

## Problem Statement

How do you achieve consensus in a distributed system when nodes can fail, messages can be delayed or lost, and network partitions can occur?

**Key challenges:**
- Split-brain scenarios (multiple leaders)
- Log consistency across replicas
- Safety during leader failures
- Liveness under network partitions

---

## Raft: Understandable Consensus

**Designed for understandability** - unlike Paxos, Raft decomposes consensus into:
1. Leader election
2. Log replication
3. Safety

---

## Core Mechanism

### Server States

Three possible states:
- **Follower:** Passive, responds to RPCs
- **Candidate:** Actively seeking votes
- **Leader:** Handles all client requests, replicates log

**Transitions:**
```
Follower → (timeout) → Candidate → (majority votes) → Leader
Leader → (discovers higher term) → Follower
Candidate → (discovers leader or higher term) → Follower
```

### Terms

- Logical clock: monotonically increasing integers
- Each term has at most one leader
- Servers reject requests from older terms
- **Split vote protection:** randomized election timeouts

### Leader Election

**Trigger:** Follower doesn't hear from leader within election timeout (150-300ms randomized)

**Process:**
1. Increment current term
2. Transition to Candidate
3. Vote for self
4. Send RequestVote RPCs to all other servers
5. Win if receive votes from majority

**RequestVote RPC:**
```
Arguments:
  term: candidate's term
  candidateId: candidate requesting vote
  lastLogIndex: index of candidate's last log entry
  lastLogTerm: term of candidate's last log entry

Response:
  term: currentTerm, for candidate to update itself
  voteGranted: true if candidate received vote
```

**Voting rules:**
- Each server votes for at most one candidate per term (first-come-first-served)
- Candidate must have log at least as up-to-date as voter
- "Up-to-date" = higher last term, or same term with higher/equal index

**Safety property:** At most one leader per term

### Log Replication

**Client request flow:**
1. Client sends command to leader
2. Leader appends entry to local log
3. Leader sends AppendEntries RPC to followers
4. Once entry replicated on majority → **committed**
5. Leader applies entry to state machine, returns result to client
6. Leader includes commit index in next AppendEntries
7. Followers apply committed entries to their state machines

**AppendEntries RPC:**
```
Arguments:
  term: leader's term
  leaderId: so follower can redirect clients
  prevLogIndex: index of log entry immediately preceding new ones
  prevLogTerm: term of prevLogIndex entry
  entries[]: log entries to store (empty for heartbeat)
  leaderCommit: leader's commitIndex

Response:
  term: currentTerm, for leader to update itself
  success: true if follower contained entry matching prevLogIndex/prevLogTerm
```

**Consistency check:**
- Follower rejects AppendEntries if prevLogIndex/prevLogTerm doesn't match
- Leader decrements nextIndex and retries
- Eventually finds point where logs match
- Overwrites follower's log from that point forward

**Log Matching Property:**
- If two entries in different logs have same index and term → they store same command
- If two entries in different logs have same index and term → logs are identical in all preceding entries

### Safety

**Election Safety:** At most one leader per term

**Leader Append-Only:** Leader never overwrites or deletes entries in its log

**Log Matching:** If two logs contain an entry with same index and term, then logs are identical in all entries up through that index

**Leader Completeness:** If a log entry is committed in a given term, that entry will be present in the logs of leaders for all higher-numbered terms

**State Machine Safety:** If a server has applied a log entry at a given index to its state machine, no other server will ever apply a different log entry for the same index

**Key insight for Leader Completeness:**
- Candidate must have all committed entries to win election
- RequestVote enforces this via lastLogIndex/lastLogTerm comparison
- Ensures new leader has complete committed log

---

## Complexity Analysis

**Leader Election:**
- Time: O(election_timeout + n * RTT)
- Messages: O(n²) worst case (multiple rounds)
- Best case: O(n) messages if first candidate wins

**Log Replication (per entry):**
- Time: O(RTT) to majority
- Messages: O(n) AppendEntries RPCs

**Failure Recovery:**
- Time: O(election_timeout) to detect + O(n * RTT) to elect new leader
- Log catch-up: O(log_size) entries to sync in worst case

**Comparison to Paxos:**
- Raft: More messages (heartbeats), clearer roles
- Paxos: Fewer messages, but harder to understand and implement correctly

---

## Proof Sketch: Safety

**Theorem:** If a log entry is committed, it will be present in all future leaders' logs.

**Proof by contradiction:**

Assume entry E committed at term T, but some future leader at term U > T doesn't have E.

1. E was replicated on majority M1 at term T
2. Leader at term U won election → received votes from majority M2
3. M1 ∩ M2 ≠ ∅ (pigeon-hole: two majorities must overlap)
4. At least one server S in both M1 and M2
5. S had E when it voted for new leader (it was in M1)
6. Voting rule: candidate must have log at least as up-to-date
7. If S voted for candidate at term U, candidate's log must have been ≥ S's log
8. Therefore candidate must have had E
9. Leader never deletes entries (append-only)
10. **Contradiction:** Leader at term U must have E

∎

**Corollary:** Committed entries are never lost.

---

## Implementation Notes

**Optimizations:**
- Batch AppendEntries (multiple entries per RPC)
- Pipeline AppendEntries (don't wait for responses)
- Snapshot old log entries (compact log, avoid unbounded growth)

**Edge cases:**
- Network partitions: minority partition can't commit (needs majority)
- Simultaneous leader failure during commit: new leader completes commit
- Log divergence: leader forces followers to duplicate its log

**Real-world use:**
- etcd (Kubernetes control plane)
- Consul (service discovery)
- CockroachDB (distributed SQL)

---

## Code Example (Simplified Leader Election)

```python
class RaftNode:
    def __init__(self, id, peers):
        self.id = id
        self.peers = peers
        self.state = "follower"
        self.current_term = 0
        self.voted_for = None
        self.election_timeout = random.uniform(150, 300)  # ms
        
    def start_election(self):
        """Transition to candidate and request votes"""
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.id
        votes_received = 1  # vote for self
        
        for peer in self.peers:
            response = peer.request_vote(
                term=self.current_term,
                candidate_id=self.id,
                last_log_index=len(self.log) - 1,
                last_log_term=self.log[-1].term if self.log else 0
            )
            
            if response.term > self.current_term:
                # Discovered higher term, revert to follower
                self.current_term = response.term
                self.state = "follower"
                self.voted_for = None
                return
                
            if response.vote_granted:
                votes_received += 1
                
        # Check if won majority
        if votes_received > len(self.peers) / 2:
            self.become_leader()
    
    def request_vote(self, term, candidate_id, last_log_index, last_log_term):
        """Handle RequestVote RPC"""
        if term < self.current_term:
            return VoteResponse(self.current_term, False)
        
        if term > self.current_term:
            self.current_term = term
            self.state = "follower"
            self.voted_for = None
        
        # Check if candidate's log is at least as up-to-date
        my_last_term = self.log[-1].term if self.log else 0
        my_last_index = len(self.log) - 1
        
        log_ok = (last_log_term > my_last_term or 
                  (last_log_term == my_last_term and 
                   last_log_index >= my_last_index))
        
        can_vote = (self.voted_for is None or 
                    self.voted_for == candidate_id)
        
        if log_ok and can_vote:
            self.voted_for = candidate_id
            return VoteResponse(self.current_term, True)
        
        return VoteResponse(self.current_term, False)
```

---

## References

- Original paper: "In Search of an Understandable Consensus Algorithm" (Ongaro & Ousterhout, 2014)
- Extended version with safety proofs: Ongaro's PhD thesis
- Visualization: https://raft.github.io

---

**Key Takeaway:** Raft achieves consensus by decomposing the problem into leader election, log replication, and safety properties. The use of terms, majority voting, and log matching ensures strong consistency even under failures.
