#!/usr/bin/env python3
"""Week 3 · Task 3 — Find the same pairs without comparing everything.

Textbook §3.4.

`BruteForce` compares every pair. On 3,000 documents that is 4.5 million
comparisons and it is completely correct. On 3 million documents it is 4.5
trillion and it is completely useless.

Beat it. Find the same near-duplicate pairs while making far fewer comparisons.

    python3 bench.py
    python3 bench.py --yours

The harness counts every call you make to `similarity()`. That is your score.
It also checks **recall** - which of the truly similar pairs you found. Skipping
comparisons is easy; skipping comparisons without losing the pairs is the task.
"""


import random
from collections import defaultdict


class BruteForce:
    """Correct, and quadratic."""

    def __init__(self, threshold):
        self.threshold = threshold

    def find(self, docs, similarity):
        """docs is [set_of_shingles, ...]. Return {(i, j), ...} with i < j."""
        out = set()
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                if similarity(docs[i], docs[j]) >= self.threshold:
                    out.add((i, j))
        return out


class YourFinder:
    """Your near-duplicate finder.

        __init__(threshold)
        find(docs, similarity) -> {(i, j), ...}

    `similarity(a, b)` is the only way to compare two documents, and every call
    is counted. Everything else - signatures, banding, bucketing - is free, in
    the sense that the harness does not charge you for it. That is deliberate:
    it is also roughly true at scale, where the comparison is the expensive
    part and the hashing is linear.

    Two knobs decide everything:

        the number of hashes in a signature
        how many bands you split it into

    §3.4.2 gives you the relationship between those and the probability that a
    pair at similarity s becomes a candidate. It is an S-curve, and where its
    step sits is something you choose. Choose it on purpose and be able to say
    why in observation.md - a threshold of 0.8 does not mean bands should be
    anything in particular until you have done the arithmetic.

    You may reuse your Task 1 code.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        # n = 120 hashes, b = 30 bands -> r = 4 rows per band.
        # S-curve threshold step: t ~= (1/b)^(1/r) = (1/30)^(1/4) ~= 0.427
        # For s = 0.6: P(candidate) = 1 - (1 - 0.6^4)^30 ~= 98.5%
        self.num_hashes = 120
        self.bands = 30
        self.rows_per_band = self.num_hashes // self.bands
        self.prime = 1000003

        rng = random.Random(42)
        self.hash_params = [
            (rng.randint(1, self.prime - 1), rng.randint(0, self.prime - 1))
            for _ in range(self.num_hashes)
        ]

    def find(self, docs, similarity):
        # 1. Compute MinHash signatures for all documents
        signatures = []
        for doc in docs:
            if not doc:
                signatures.append([0] * self.num_hashes)
                continue
            sig = []
            for a, b in self.hash_params:
                min_val = min((a * shingle + b) % self.prime for shingle in doc)
                sig.append(min_val)
            signatures.append(sig)

        # 2. Banding & Bucketing (LSH)
        candidates = set()
        for b_idx in range(self.bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            buckets = defaultdict(list)
            for doc_idx, sig in enumerate(signatures):
                chunk = tuple(sig[start:end])
                buckets[chunk].append(doc_idx)

            for group in buckets.values():
                if len(group) > 1:
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            u, v = min(group[i], group[j]), max(group[i], group[j])
                            candidates.add((u, v))

        # 3. Only evaluate similarity on candidate pairs
        out = set()
        for i, j in candidates:
            if similarity(docs[i], docs[j]) >= self.threshold:
                out.add((i, j))

        return out
