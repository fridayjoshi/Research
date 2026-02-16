# Skip Lists: Probabilistic Search Structure

**Date:** 2026-02-16 10:52 AM  
**Topic:** Skip lists - probabilistic alternative to balanced trees  
**Complexity:** O(log n) average for search, insert, delete

## Overview

Skip lists are a probabilistic data structure that provide O(log n) average-case performance for search, insertion, and deletion, without the complexity of tree balancing operations. Invented by William Pugh in 1990.

**Key insight:** Build a hierarchy of linked lists with exponentially decreasing density, allowing binary-search-like traversal on a linked structure.

## Structure

A skip list consists of multiple levels:
- **Level 0:** Complete sorted linked list (all elements)
- **Level 1:** ~50% of elements (every other element, probabilistically)
- **Level 2:** ~25% of elements
- **Level k:** ~(1/2^k) of elements

Each node has a **tower** of forward pointers, one per level it participates in.

```
Level 3: 1 --------------------------------> 17 -> NIL
Level 2: 1 ----------> 6 --------> 13 ----> 17 -> NIL
Level 1: 1 -> 3 ----> 6 -> 9 ----> 13 ----> 17 -> NIL
Level 0: 1 -> 3 -> 4 -> 6 -> 9 -> 10 -> 13 -> 17 -> NIL
```

## Algorithm: Search

**Goal:** Find a key `k` in the skip list.

**Procedure:**
1. Start at the top-left (highest level, first node)
2. Move right while `next.key < k`
3. When `next.key >= k`, drop down one level
4. Repeat until level 0
5. Check if current node's key equals `k`

**Path example:** To find 13:
- Start at level 3, node 1
- Move right to 17 (too far)
- Drop to level 2, node 1
- Move right to 6, then 13 (found at level 2)
- Drop to level 0 and confirm

**Complexity:** O(log n) expected

Each level reduces search space by ~50%, similar to binary search.

## Algorithm: Insert

**Goal:** Insert key `k` with value `v`.

**Procedure:**
1. Search for position of `k` (same as search)
2. Track **update vector** - rightmost node at each level before insert position
3. Flip coins to determine tower height `h` (geometric distribution: P(height ≥ h) = 1/2^h)
4. Insert new node at level 0
5. For levels 1 to h-1, insert node and update pointers from update vector

**Pseudocode:**
```
insert(k, v):
    update = [None] * MAX_LEVEL
    x = header
    
    # Search and build update vector
    for level in range(MAX_LEVEL-1, -1, -1):
        while x.forward[level] and x.forward[level].key < k:
            x = x.forward[level]
        update[level] = x
    
    # Random tower height
    height = random_level()
    
    # Create new node with tower
    new_node = Node(k, v, height)
    
    # Update pointers
    for level in range(height):
        new_node.forward[level] = update[level].forward[level]
        update[level].forward[level] = new_node
```

**Complexity:** O(log n) expected

## Algorithm: Delete

**Goal:** Remove key `k`.

**Procedure:**
1. Search for `k`, building update vector
2. If found, update forward pointers at all levels to bypass deleted node
3. Free deleted node's memory

**Pseudocode:**
```
delete(k):
    update = [None] * MAX_LEVEL
    x = header
    
    # Search and build update vector
    for level in range(MAX_LEVEL-1, -1, -1):
        while x.forward[level] and x.forward[level].key < k:
            x = x.forward[level]
        update[level] = x
    
    x = x.forward[0]
    
    if x and x.key == k:
        # Update pointers at all levels
        for level in range(len(x.forward)):
            update[level].forward[level] = x.forward[level]
        return True
    return False
```

**Complexity:** O(log n) expected

## Probabilistic Analysis

**Expected height of skip list with n elements:**

H(n) = O(log n)

**Proof sketch:**
- Probability that a node reaches height h: P(height ≥ h) = 1/2^h
- Expected number of nodes at height h: n/2^h
- Maximum height where E[nodes] ≥ 1: h = log₂(n)
- Therefore, with high probability, max height is O(log n)

**Expected search path length:**

The search path at level k spans at most 1/p of the elements between positions at level k+1.

With p = 1/2, expected path length:
```
L(n) = log₂(n) + O(1)
```

This is within a constant factor of optimal balanced tree performance.

## Randomization: Why Coins?

**Why flip coins instead of deterministic balancing?**

1. **Simplicity:** No rotations, no invariant checking, no rebalancing
2. **Concurrency-friendly:** Probabilistic structure makes locking easier
3. **Amortized performance:** No worst-case rebalancing cascades
4. **Empirically good:** In practice, performs as well as balanced trees

**Random level function:**
```python
def random_level(max_level=16, p=0.5):
    level = 0
    while random.random() < p and level < max_level - 1:
        level += 1
    return level
```

With p = 0.5, this gives geometric distribution: P(level = k) = (1/2)^(k+1)

## Implementation (Python)

```python
import random

class Node:
    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        self.forward = [None] * level

class SkipList:
    MAX_LEVEL = 16
    P = 0.5
    
    def __init__(self):
        self.header = Node(None, None, self.MAX_LEVEL)
        self.level = 0
    
    def random_level(self):
        level = 0
        while random.random() < self.P and level < self.MAX_LEVEL - 1:
            level += 1
        return level + 1
    
    def search(self, key):
        """O(log n) expected search"""
        x = self.header
        for i in range(self.level - 1, -1, -1):
            while x.forward[i] and x.forward[i].key < key:
                x = x.forward[i]
        x = x.forward[0]
        if x and x.key == key:
            return x.value
        return None
    
    def insert(self, key, value):
        """O(log n) expected insertion"""
        update = [None] * self.MAX_LEVEL
        x = self.header
        
        # Build update vector
        for i in range(self.level - 1, -1, -1):
            while x.forward[i] and x.forward[i].key < key:
                x = x.forward[i]
            update[i] = x
        
        # Random level for new node
        new_level = self.random_level()
        if new_level > self.level:
            for i in range(self.level, new_level):
                update[i] = self.header
            self.level = new_level
        
        # Create and insert new node
        new_node = Node(key, value, new_level)
        for i in range(new_level):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
    
    def delete(self, key):
        """O(log n) expected deletion"""
        update = [None] * self.MAX_LEVEL
        x = self.header
        
        for i in range(self.level - 1, -1, -1):
            while x.forward[i] and x.forward[i].key < key:
                x = x.forward[i]
            update[i] = x
        
        x = x.forward[0]
        
        if x and x.key == key:
            for i in range(len(x.forward)):
                if update[i].forward[i] != x:
                    break
                update[i].forward[i] = x.forward[i]
            
            # Update skip list level
            while self.level > 0 and not self.header.forward[self.level - 1]:
                self.level -= 1
            return True
        return False
    
    def display(self):
        """Debug visualization"""
        for level in range(self.level - 1, -1, -1):
            print(f"Level {level}: ", end="")
            node = self.header.forward[level]
            while node:
                print(f"{node.key} -> ", end="")
                node = node.forward[level]
            print("NIL")

# Example usage and test
if __name__ == "__main__":
    sl = SkipList()
    
    # Insert elements
    for key in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
        sl.insert(key, f"value_{key}")
    
    print("Skip list after insertions:")
    sl.display()
    print()
    
    # Search
    print(f"Search 19: {sl.search(19)}")
    print(f"Search 15: {sl.search(15)}")
    print()
    
    # Delete
    sl.delete(19)
    print("After deleting 19:")
    sl.display()
```

**Output example:**
```
Skip list after insertions:
Level 2: 6 -> 19 -> NIL
Level 1: 3 -> 6 -> 12 -> 19 -> 25 -> NIL
Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26 -> NIL

Search 19: value_19
Search 15: None

After deleting 19:
Level 2: 6 -> NIL
Level 1: 3 -> 6 -> 12 -> 25 -> NIL
Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 21 -> 25 -> 26 -> NIL
```

## Complexity Summary

| Operation | Average Case | Worst Case | Space   |
|-----------|-------------|------------|---------|
| Search    | O(log n)    | O(n)       | O(n)    |
| Insert    | O(log n)    | O(n)       | O(n log n) expected |
| Delete    | O(log n)    | O(n)       | -       |

**Note:** Worst case O(n) occurs with probability ~1/n^c (exponentially unlikely).

## Comparison to Balanced Trees

**Skip lists vs Red-Black trees:**

| Aspect              | Skip List              | Red-Black Tree        |
|---------------------|------------------------|-----------------------|
| Implementation      | Simple (~100 LOC)      | Complex (~300 LOC)    |
| Balancing           | Probabilistic (coins)  | Deterministic (rotations) |
| Worst-case          | O(n) with low prob     | O(log n) guaranteed   |
| Concurrency         | Easier to lock         | Harder (rotation conflicts) |
| Space overhead      | O(n log n) expected    | O(n)                  |
| Cache performance   | Worse (pointer chasing)| Better (tree locality)|

**When to use skip lists:**
- Need simple implementation
- Concurrent access is important
- Average-case guarantees are acceptable
- Implementing in-memory databases (Redis uses skip lists for sorted sets)

**When to use trees:**
- Need worst-case guarantees
- Cache locality matters
- Space is constrained

## Historical Note

Invented by William Pugh (1990) as a probabilistic alternative to balanced trees. Paper: "Skip Lists: A Probabilistic Alternative to Balanced Trees" (Communications of the ACM, 1990).

**Real-world usage:**
- **Redis:** Sorted sets (ZSET) implemented with skip lists
- **LevelDB/RocksDB:** MemTable uses skip lists
- **Lucene:** Term dictionary can use skip lists

The beauty of skip lists is that they match balanced tree performance using nothing but linked lists and coin flips.

---

**LOC Count:** 84 lines (implementation) + infrastructure  
**Complexity:** Average O(log n), high probability bounds  
**Key takeaway:** Randomization can replace complex deterministic balancing

**References:**
- Pugh, W. (1990). "Skip Lists: A Probabilistic Alternative to Balanced Trees"
- Sedgewick & Wayne, *Algorithms* (4th ed.), Section 3.5
