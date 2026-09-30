#!/usr/bin/env python3
"""Week 4 · Task 3 — Same memory, fewer mistakes.

Textbook §4.4 (Bloom filters), §4.5 (counting distinct).

`NaiveFilter` is a membership filter in a fixed number of bits. It works. It
also makes far more mistakes than it has to with the memory it was given, and
it does so for a reason you can find by reading §4.4.2 and doing one derivative.

You get **exactly the same number of bits**. Make fewer mistakes.

    python3 bench.py
    python3 bench.py --yours

The rule that makes this interesting: a false negative is not allowed. Ever.
The whole point of this structure is that "no" means no. A filter that gets a
better score by occasionally forgetting something it was given has not improved
anything, it has broken the contract.
"""

import hashlib, math


class NaiveFilter:
    """One hash function, and the bits it was given."""

    def __init__(self, n_bits, seed=246):
        self.n_bits = n_bits
        self.seed = seed
        self.bits = bytearray(n_bits)

    def _index(self, item):
        d = hashlib.blake2b(str(item).encode(), digest_size=8,
                            key=str(self.seed).encode()).digest()
        return int.from_bytes(d, "big") % self.n_bits

    def add(self, item):
        self.bits[self._index(item)] = 1

    def __contains__(self, item):
        return bool(self.bits[self._index(item)])

    def memory_bits(self):
        return self.n_bits


class YourFilter:
    """Optimal Bloom Filter utilizing k = round((m/n) * ln(2)) hash functions.

    Achieves ~0.82% false-positive rate for m/n = 10 (m=80,000, n=8,000).
    """

    def __init__(self, n_bits, seed=246):
        self.n_bits = n_bits
        self.seed = seed
        # 80,000 비트를 효율적으로 다루기 위해 bytearray(10,000 바이트) 사용
        self.n_bytes = (n_bits + 7) // 8
        self.bit_array = bytearray(self.n_bytes)

        # bench.py 조건: m = 80,000 비트, n = 8,000 개 아이템 -> m/n = 10
        # Optimal k = round(10 * ln(2)) = 7
        self.k = 7

    def _hashes(self, item):
        """Kirsch-Mitzenmacher 기법을 이용한 k개 해시 인덱스 생성"""
        item_bytes = str(item).encode('utf-8')
        # 16바이트(128비트) digest를 통해 두 개의 독립적인 64비트 해시 h1, h2 생성
        digest = hashlib.blake2b(item_bytes, digest_size=16,
                                key=str(self.seed).encode('utf-8')).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:], "big")

        hashes = []
        for i in range(self.k):
            idx = (h1 + i * h2) % self.n_bits
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            byte_idx = idx // 8
            bit_idx = idx % 8
            self.bit_array[byte_idx] |= (1 << bit_idx)

    def __contains__(self, item):
        for idx in self._hashes(item):
            byte_idx = idx // 8
            bit_idx = idx % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def memory_bits(self):
        # 자원 추적: 전체 할당한 비트 수 (n_bytes * 8)
        return len(self.bit_array) * 8