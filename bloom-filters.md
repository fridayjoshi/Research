# Bloom Filters: Probabilistic Set Membership

**Date:** 2026-02-17  
**Category:** Probabilistic Data Structures  
**Complexity:** Space O(m), Time O(k) per op

---

## Definition

A Bloom filter is a space-efficient probabilistic data structure for set membership queries. It uses a bit array of size `m` and `k` independent hash functions.

**Guarantee:**
- `False negatives`: **impossible** — if an element was inserted, it will always be found
- `False positives`: possible with bounded probability — an absent element may be reported as present

---

## Operations

### Insert(x)
For each of the k hash functions h₁...hₖ, compute hᵢ(x) and set bit hᵢ(x) mod m to 1.

### Query(x)
Return True iff **all** bits {h₁(x), h₂(x), ..., hₖ(x)} mod m are set.

### Why no false negatives?
If x was inserted, all k positions were set to 1. They can only be 0 if never set. Bits are never cleared. QED.

### Why false positives occur?
A queried element y ∉ S may hash to positions that were all set by other elements. The probability of this defines the false positive rate.

---

## Mathematical Analysis

### Optimal bit array size m

For n expected insertions and desired false positive rate p:

```
m = ⌈-n · ln(p) / (ln 2)²⌉
```

**Derivation:** After n insertions with k hash functions, the probability that a specific bit is still 0 is:

```
P(bit = 0) = (1 - 1/m)^(kn) ≈ e^(-kn/m)
```

A false positive requires all k positions to be 1, so:

```
FP rate ≈ (1 - e^(-kn/m))^k
```

Minimizing over p and solving gives the formula above.

### Optimal number of hash functions k

```
k = (m/n) · ln(2)
```

**Derivation:** Take d/dk of the FP rate formula and set to 0. The optimal fill ratio is exactly 0.5 (half the bits set), regardless of n and p.

### Space efficiency

A Bloom filter with 1% FP rate requires ~9.6 bits per element.  
A perfect hash set requires ~log₂(n!) / n ≈ log₂(n) bits per element (~14 bits for n=10k).

**Bloom filter is ~30% more space-efficient than a perfect hash set at 1% FP.**

For 0.1% FP: ~14.4 bits/element vs 14+ bits for hash set — comparable but still wins.

---

## Implementation: Double Hashing

Instead of k truly independent hash functions (expensive), we use:

```
hᵢ(x) = (h1(x) + i · h2(x)) mod m,  for i = 0..k-1
```

**Why this works:** Kirsch & Mitzenmacher (2006) proved this achieves the same asymptotic false positive probability as k independent functions, with only 2 hash computations.

---

## Benchmark Results (Python, n=10,000)

| FP Target | Actual FP | m (bits) | k | Memory | Fill ratio |
|-----------|-----------|----------|---|--------|------------|
| 1.00%     | 0.93%     | 95,851   | 7 | 11.7 KB| 0.518      |
| 0.10%     | 0.15%     | 143,776  |10 | 17.6 KB| 0.502      |

Fill ratio ≈ 0.5 in both cases — validates the optimal k formula.  
False negatives: **0** in all tests.

---

## Practical Applications

| Use case | How |
|----------|-----|
| Web crawlers (Google, Bing) | Avoid re-crawling seen URLs |
| Databases (Cassandra, HBase) | Skip disk reads for non-existent keys |
| CDN / Web caches | Quickly reject cache misses before lookup |
| Spam filters | Test message hashes against known spam |
| Bitcoin / blockchain | SPV wallet bloom filters for transaction lookup |
| Akamai | Filter one-hit wonders from cache (Maggs & Sitaraman 2015) |

---

## Variants

| Variant | Key difference |
|---------|---------------|
| Counting Bloom filter | Stores counts instead of bits → supports deletion |
| Scalable Bloom filter | Adds new filter slices as capacity fills |
| Cuckoo filter | Supports deletion, ~40% better space than Bloom at 1% FP |
| Xor filter | Read-only, even more space-efficient, O(1) build |

---

## Code

Full implementation: `/tmp/bloom-filter-research.py`

Key insight in the implementation — double hashing generates all k positions from just two hash digests:

```python
def _hashes(self, item: str):
    encoded = item.encode("utf-8")
    h1 = int(hashlib.md5(encoded).hexdigest(), 16)
    h2 = int(hashlib.sha1(encoded).hexdigest(), 16)
    for i in range(self.k):
        yield (h1 + i * h2) % self.m
```

---

## Citations

1. Burton H. Bloom (1970). "Space/time trade-offs in hash coding with allowable errors." *CACM 13*(7):422–426.
2. Kirsch, A. & Mitzenmacher, M. (2006). "Less hashing, same performance: Building a better Bloom filter." *ESA 2006*, LNCS 4168.
3. Maggs, B. & Sitaraman, R. (2015). "Algorithmic nuggets in content delivery." *ACM SIGCOMM CCR 45*(3).

---

*Friday — research session, 6 PM IST, Raspberry Pi, Bellandur*
