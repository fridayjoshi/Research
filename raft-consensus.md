# Raft Consensus Algorithm

**Research Date:** 2026-02-14  
**Topic:** Distributed Systems - Consensus Protocol  
**Implementation:** TypeScript

---

## Abstract

Raft is a consensus algorithm designed to be understandable. It's equivalent to Paxos in fault-tolerance and performance but structured for better comprehension. Raft separates leader election, log replication, and safety, enforcing stronger coherency to reduce state space.

---

## The Algorithm

### 1. Core Components

**Server States:**
- **Follower**: Passive, responds to RPCs from leaders and candidates
- **Candidate**: Active during election, requests votes
- **Leader**: Handles all client requests, replicates log to followers

**Persistent State** (on all servers):
- `currentTerm`: Latest term seen (monotonically increasing)
- `votedFor`: CandidateId that received vote in current term (null if none)
- `log[]`: Array of log entries, each containing command and term

**Volatile State** (on all servers):
- `commitIndex`: Highest log entry known to be committed
- `lastApplied`: Highest log entry applied to state machine

**Volatile State** (on leaders, reinitialized after election):
- `nextIndex[]`: For each server, index of next log entry to send
- `matchIndex[]`: For each server, index of highest log entry known to be replicated

### 2. Leader Election

**Election Timeout:**
- Followers start election if no heartbeat from leader within timeout (150-300ms randomized)
- Becomes candidate, increments term, votes for itself, requests votes from others

**RequestVote RPC:**
```
Arguments:
  term: candidate's term
  candidateId: candidate requesting vote
  lastLogIndex: index of candidate's last log entry
  lastLogTerm: term of candidate's last log entry

Results:
  term: currentTerm for candidate to update itself
  voteGranted: true if candidate received vote
```

**Receiver Implementation:**
1. Reply false if `term < currentTerm`
2. If `votedFor` is null or `candidateId`, and candidate's log is at least as up-to-date as receiver's log, grant vote

**Vote Granting Rules:**
- Grant vote if candidate's log is at least as up-to-date:
  - Last log entry has higher term, OR
  - Same term but equal or longer log

**Election Completion:**
- Candidate wins if it receives votes from majority of servers
- Becomes leader, sends heartbeat to establish authority
- If another server establishes itself as leader, revert to follower
- If timeout elapses with no winner, start new election (term++)

### 3. Log Replication

**AppendEntries RPC:**
```
Arguments:
  term: leader's term
  leaderId: so followers can redirect clients
  prevLogIndex: index of log entry immediately preceding new ones
  prevLogTerm: term of prevLogIndex entry
  entries[]: log entries to store (empty for heartbeat)
  leaderCommit: leader's commitIndex

Results:
  term: currentTerm for leader to update itself
  success: true if follower contained entry matching prevLogIndex and prevLogTerm
```

**Leader Process:**
1. Client sends command to leader
2. Leader appends entry to local log
3. Issues `AppendEntries` RPCs in parallel to followers
4. When entry is safely replicated (majority), leader applies it to state machine
5. Returns result to client
6. Leader notifies followers of committed entries in subsequent `AppendEntries` RPCs

**Follower Process:**
1. Reply false if `term < currentTerm`
2. Reply false if log doesn't contain entry at `prevLogIndex` matching `prevLogTerm`
3. If existing entry conflicts with new one (same index, different terms), delete existing entry and all following
4. Append any new entries not already in log
5. If `leaderCommit > commitIndex`, set `commitIndex = min(leaderCommit, index of last new entry)`

### 4. Safety Properties

**Election Safety:** At most one leader per term

**Leader Append-Only:** Leader never overwrites or deletes entries in its log

**Log Matching:** If two logs contain an entry with same index and term, then all preceding entries are identical

**Leader Completeness:** If entry is committed in a term, it will be present in logs of all leaders for higher terms

**State Machine Safety:** If a server has applied a log entry at a given index, no other server will apply a different entry for that index

---

## Formal Proofs

### Theorem 1: Election Safety

**Claim:** At most one leader can be elected in a given term.

**Proof:**
- A candidate must receive votes from a majority of servers to win
- Each server votes for at most one candidate per term (enforced by `votedFor` persistence)
- Two different majorities must intersect (pigeonhole principle)
- Therefore, at most one candidate can collect majority votes in a term
∎

### Theorem 2: Leader Completeness

**Claim:** If a log entry is committed in a term T, it will be present in logs of leaders for all terms > T.

**Proof by contradiction:**
1. Assume entry E committed in term T, but leader L in term U > T doesn't have E
2. E was committed → replicated on majority M₁ in term T
3. L was elected → received votes from majority M₂ in term U
4. M₁ ∩ M₂ ≠ ∅ (majorities must overlap)
5. Let S be a server in intersection
6. S must have voted for L in term U
7. Vote granting rule: S only votes if candidate's log is at least as up-to-date
8. S had E when it voted (since S ∈ M₁)
9. For S to vote for L: L's log must be at least as up-to-date as S's
10. Therefore L must have E (contradiction)
∎

### Theorem 3: State Machine Safety

**Claim:** If server applies entry E at index i, no other server applies different entry at i.

**Proof:**
- Entry applied only if committed (commitIndex advanced)
- Entry committed only if replicated on majority
- Log Matching Property: entries with same index/term have identical history
- Leader Completeness: all future leaders have committed entries
- Therefore all servers apply same sequence
∎

---

## Complexity Analysis

**Time Complexity:**

- **Leader Election:** O(n) messages in best case (one round-trip), O(n²) in worst case (multiple rounds)
- **Log Replication:** O(n) per entry (leader broadcasts to all followers)
- **Consensus Latency:** 1.5 RTT (half RTT for heartbeat + 1 RTT for replication)

**Space Complexity:**

- **Per Server:** O(m) where m is log length
- **Network:** O(n·m) total across cluster

**Message Complexity:**

- **Heartbeat:** O(n) per interval (leader to all followers)
- **Replication:** O(n) per log entry
- **Election:** O(n²) worst case (all-to-all RequestVote)

**Fault Tolerance:**

- **Availability:** System operational with majority alive (⌈n/2⌉ + 1)
- **Durability:** Entry safe when replicated to majority
- **Typical Config:** 5 servers (tolerates 2 failures)

---

## Implementation (TypeScript)

```typescript
// raft-node.ts

enum ServerState {
  FOLLOWER,
  CANDIDATE,
  LEADER
}

interface LogEntry {
  term: number;
  command: any;
}

interface AppendEntriesArgs {
  term: number;
  leaderId: string;
  prevLogIndex: number;
  prevLogTerm: number;
  entries: LogEntry[];
  leaderCommit: number;
}

interface AppendEntriesResult {
  term: number;
  success: boolean;
}

interface RequestVoteArgs {
  term: number;
  candidateId: string;
  lastLogIndex: number;
  lastLogTerm: number;
}

interface RequestVoteResult {
  term: number;
  voteGranted: boolean;
}

class RaftNode {
  // Persistent state
  private currentTerm: number = 0;
  private votedFor: string | null = null;
  private log: LogEntry[] = [];

  // Volatile state
  private commitIndex: number = 0;
  private lastApplied: number = 0;

  // Leader volatile state
  private nextIndex: Map<string, number> = new Map();
  private matchIndex: Map<string, number> = new Map();

  // Node metadata
  private state: ServerState = ServerState.FOLLOWER;
  private leaderId: string | null = null;
  private readonly nodeId: string;
  private readonly peers: string[];

  // Timers
  private electionTimeout: number = 0;
  private lastHeartbeat: number = Date.now();
  private readonly HEARTBEAT_INTERVAL = 50; // ms
  private readonly ELECTION_TIMEOUT_MIN = 150; // ms
  private readonly ELECTION_TIMEOUT_MAX = 300; // ms

  constructor(nodeId: string, peers: string[]) {
    this.nodeId = nodeId;
    this.peers = peers;
    this.resetElectionTimeout();
  }

  private resetElectionTimeout(): void {
    this.electionTimeout = Date.now() + 
      this.ELECTION_TIMEOUT_MIN + 
      Math.random() * (this.ELECTION_TIMEOUT_MAX - this.ELECTION_TIMEOUT_MIN);
  }

  // Check if election timeout has elapsed
  private checkElectionTimeout(): void {
    if (this.state !== ServerState.LEADER && Date.now() > this.electionTimeout) {
      this.startElection();
    }
  }

  // Start election process
  private async startElection(): Promise<void> {
    this.state = ServerState.CANDIDATE;
    this.currentTerm++;
    this.votedFor = this.nodeId;
    this.resetElectionTimeout();

    const lastLogIndex = this.log.length - 1;
    const lastLogTerm = lastLogIndex >= 0 ? this.log[lastLogIndex].term : 0;

    let votesReceived = 1; // Vote for self
    const votesNeeded = Math.floor(this.peers.length / 2) + 1;

    // Request votes from all peers
    const votePromises = this.peers
      .filter(peer => peer !== this.nodeId)
      .map(async (peer) => {
        const args: RequestVoteArgs = {
          term: this.currentTerm,
          candidateId: this.nodeId,
          lastLogIndex,
          lastLogTerm
        };
        
        const result = await this.sendRequestVote(peer, args);
        
        if (result.voteGranted) {
          votesReceived++;
        }
        
        if (result.term > this.currentTerm) {
          this.currentTerm = result.term;
          this.state = ServerState.FOLLOWER;
          this.votedFor = null;
        }
      });

    await Promise.all(votePromises);

    // Check if won election
    if (this.state === ServerState.CANDIDATE && votesReceived >= votesNeeded) {
      this.becomeLeader();
    }
  }

  // Transition to leader state
  private becomeLeader(): void {
    this.state = ServerState.LEADER;
    this.leaderId = this.nodeId;

    // Initialize leader state
    for (const peer of this.peers) {
      this.nextIndex.set(peer, this.log.length);
      this.matchIndex.set(peer, 0);
    }

    // Start sending heartbeats
    this.sendHeartbeats();
  }

  // Send heartbeats to all followers
  private async sendHeartbeats(): Promise<void> {
    if (this.state !== ServerState.LEADER) return;

    const promises = this.peers
      .filter(peer => peer !== this.nodeId)
      .map(peer => this.sendAppendEntries(peer));

    await Promise.all(promises);

    // Schedule next heartbeat
    setTimeout(() => this.sendHeartbeats(), this.HEARTBEAT_INTERVAL);
  }

  // Send AppendEntries to a follower
  private async sendAppendEntries(peer: string): Promise<void> {
    const nextIdx = this.nextIndex.get(peer) || 0;
    const prevLogIndex = nextIdx - 1;
    const prevLogTerm = prevLogIndex >= 0 ? this.log[prevLogIndex].term : 0;

    const args: AppendEntriesArgs = {
      term: this.currentTerm,
      leaderId: this.nodeId,
      prevLogIndex,
      prevLogTerm,
      entries: this.log.slice(nextIdx),
      leaderCommit: this.commitIndex
    };

    const result = await this.rpcAppendEntries(peer, args);

    if (result.term > this.currentTerm) {
      this.currentTerm = result.term;
      this.state = ServerState.FOLLOWER;
      this.votedFor = null;
      return;
    }

    if (result.success) {
      this.nextIndex.set(peer, nextIdx + args.entries.length);
      this.matchIndex.set(peer, prevLogIndex + args.entries.length);
      this.updateCommitIndex();
    } else {
      // Decrement nextIndex and retry
      this.nextIndex.set(peer, Math.max(0, nextIdx - 1));
    }
  }

  // Update commit index based on majority replication
  private updateCommitIndex(): void {
    if (this.state !== ServerState.LEADER) return;

    // Find highest N where majority have matchIndex >= N
    for (let n = this.log.length - 1; n > this.commitIndex; n--) {
      if (this.log[n].term !== this.currentTerm) continue;

      let replicationCount = 1; // Leader has it
      for (const peer of this.peers) {
        if (peer !== this.nodeId && (this.matchIndex.get(peer) || 0) >= n) {
          replicationCount++;
        }
      }

      if (replicationCount > this.peers.length / 2) {
        this.commitIndex = n;
        break;
      }
    }
  }

  // Handle AppendEntries RPC
  public async handleAppendEntries(args: AppendEntriesArgs): Promise<AppendEntriesResult> {
    // Update term if necessary
    if (args.term > this.currentTerm) {
      this.currentTerm = args.term;
      this.state = ServerState.FOLLOWER;
      this.votedFor = null;
    }

    // Reject if term is stale
    if (args.term < this.currentTerm) {
      return { term: this.currentTerm, success: false };
    }

    // Valid leader, reset election timeout
    this.lastHeartbeat = Date.now();
    this.resetElectionTimeout();
    this.leaderId = args.leaderId;

    // Check log consistency
    if (args.prevLogIndex >= 0) {
      if (args.prevLogIndex >= this.log.length ||
          this.log[args.prevLogIndex].term !== args.prevLogTerm) {
        return { term: this.currentTerm, success: false };
      }
    }

    // Delete conflicting entries and append new ones
    let logIndex = args.prevLogIndex + 1;
    for (const entry of args.entries) {
      if (logIndex < this.log.length) {
        if (this.log[logIndex].term !== entry.term) {
          this.log = this.log.slice(0, logIndex);
          this.log.push(entry);
        }
      } else {
        this.log.push(entry);
      }
      logIndex++;
    }

    // Update commit index
    if (args.leaderCommit > this.commitIndex) {
      this.commitIndex = Math.min(args.leaderCommit, this.log.length - 1);
    }

    return { term: this.currentTerm, success: true };
  }

  // Handle RequestVote RPC
  public async handleRequestVote(args: RequestVoteArgs): Promise<RequestVoteResult> {
    // Update term if necessary
    if (args.term > this.currentTerm) {
      this.currentTerm = args.term;
      this.state = ServerState.FOLLOWER;
      this.votedFor = null;
    }

    // Reject if term is stale
    if (args.term < this.currentTerm) {
      return { term: this.currentTerm, voteGranted: false };
    }

    // Check if already voted
    if (this.votedFor !== null && this.votedFor !== args.candidateId) {
      return { term: this.currentTerm, voteGranted: false };
    }

    // Check if candidate's log is at least as up-to-date
    const lastLogIndex = this.log.length - 1;
    const lastLogTerm = lastLogIndex >= 0 ? this.log[lastLogIndex].term : 0;

    const candidateUpToDate =
      args.lastLogTerm > lastLogTerm ||
      (args.lastLogTerm === lastLogTerm && args.lastLogIndex >= lastLogIndex);

    if (candidateUpToDate) {
      this.votedFor = args.candidateId;
      this.resetElectionTimeout();
      return { term: this.currentTerm, voteGranted: true };
    }

    return { term: this.currentTerm, voteGranted: false };
  }

  // Client API: Submit command
  public async submitCommand(command: any): Promise<boolean> {
    if (this.state !== ServerState.LEADER) {
      return false; // Redirect to leader
    }

    const entry: LogEntry = {
      term: this.currentTerm,
      command
    };

    this.log.push(entry);
    
    // Wait for replication (simplified - should timeout)
    await this.sendHeartbeats();
    
    return this.commitIndex >= this.log.length - 1;
  }

  // Placeholder RPC methods (would use network in real implementation)
  private async sendRequestVote(peer: string, args: RequestVoteArgs): Promise<RequestVoteResult> {
    // Network call would go here
    throw new Error("Not implemented: network layer");
  }

  private async rpcAppendEntries(peer: string, args: AppendEntriesArgs): Promise<AppendEntriesResult> {
    // Network call would go here
    throw new Error("Not implemented: network layer");
  }
}

// Example usage
const node1 = new RaftNode("node1", ["node1", "node2", "node3", "node4", "node5"]);
const node2 = new RaftNode("node2", ["node1", "node2", "node3", "node4", "node5"]);
const node3 = new RaftNode("node3", ["node1", "node2", "node3", "node4", "node5"]);

// In a real system, wire up network layer and start nodes
```

---

## Key Insights

1. **Understandability vs Paxos:** Raft's key innovation is structure, not novelty. By separating leader election, log replication, and safety, it's teachable.

2. **Strong Leader:** Unlike Paxos, Raft has a strong leader model. All client requests go through leader, simplifying the protocol.

3. **Randomized Timeouts:** Election timeouts are randomized (150-300ms) to avoid split votes. Simple but effective.

4. **Log Matching Property:** Once entries match at an index, all prior entries must match. This is enforced by the AppendEntries consistency check.

5. **Majority Quorums:** All decisions (election, commit) require majority. This ensures any two quorums overlap, maintaining consistency.

---

## References

1. Ongaro, D., & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC.
2. Lamport, L. (1998). "The Part-Time Parliament." ACM TOCS.
3. Liskov, B., & Cowling, J. (2012). "Viewstamped Replication Revisited." MIT-CSAIL-TR-2012-021.

---

**Research Time:** ~45 minutes  
**Implementation Status:** Core algorithm complete, network layer abstracted  
**Next Steps:** Add membership changes (joint consensus), log compaction (snapshotting)
