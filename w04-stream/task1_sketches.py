#!/usr/bin/env python3
#!/usr/bin/env python3
"""Week 4 · Task 1 — Answer questions about a stream you cannot store.

Textbook §4.3 (sampling), §4.4 (Bloom filter), §4.5 (Flajolet-Martin).
"""
import argparse, random, hashlib, math


class BloomFilter:
    """Membership, with one-sided error.

    `m` bits, `k` hash functions.
    """

    def __init__(self, m, k, seed=246):
        self.m = m
        self.k = k
        self.bit_array = [0] * m
        self.seed = seed

    def _hashes(self, item):
        """해시 함수 k개를 시드 기반 sha256을 활용해 비트 위치 인덱스로 생성"""
        hashes = []
        item_bytes = str(item).encode('utf-8')
        for i in range(self.k):
            # 시드와 해시 인덱스를 조합하여 k개의 서로 다른 해시 생성
            h = hashlib.sha256(item_bytes + f"-{self.seed}-{i}".encode('utf-8')).hexdigest()
            idx = int(h, 16) % self.m
            hashes.append(idx)
        return hashes

    def add(self, item):
        for idx in self._hashes(item):
            self.bit_array[idx] = 1

    def __contains__(self, item):
        # 모든 해시 비트가 1이어야만 포함 가능성이 있음 (거짓 음성 불가능)
        return all(self.bit_array[idx] == 1 for idx in self._hashes(item))

    def expected_fp_rate(self, n_inserted):
        """교재 §4.4.2의 이론적 거짓 양성(False Positive) 확률 계산 공식

        P(FP) = (1 - exp(-k * n / m))^k
        """
        if self.m == 0:
            return 1.0
        prob_bit_zero = math.exp(-self.k * n_inserted / self.m)
        prob_bit_one = 1.0 - prob_bit_zero
        return prob_bit_one ** self.k


def _count_trailing_zeros(val):
    """정수의 이진 표현에서 하위 0의 개수(R) 반환"""
    if val == 0:
        return 64  # 64비트 기준
    count = 0
    while (val & 1) == 0:
        count += 1
        val >>= 1
    return count


def flajolet_martin(stream, n_hashes=64, seed=246):
    """Estimate how many DISTINCT items went past, in almost no memory.

    교재 §4.5.3의 Grouping & Combining 기법 적용 (Mean of Medians)
    """
    # 각 해시 함수별 R (최대 trailing zero 수) 저장
    max_r = [0] * n_hashes

    for item in stream:
        item_bytes = str(item).encode('utf-8')
        for i in range(n_hashes):
            h = hashlib.sha256(item_bytes + f"-{seed}-{i}".encode('utf-8')).hexdigest()
            val = int(h, 16)
            r = _count_trailing_zeros(val)
            if r > max_r[i]:
                max_r[i] = r

    # 각 해시 함수별 추정치 2^R 계산
    estimates = [2 ** r for r in max_r]

    # §4.5.3: 해시들을 소그룹으로 묶어 그룹 내 중앙값(Median)을 구한 뒤, 그룹 간 평균(Mean) 계산
    # n_hashes = 64개를 8개 그룹(그룹당 8개)으로 나눔
    num_groups = 8
    group_size = n_hashes // num_groups
    group_medians = []

    for g in range(num_groups):
        group_estimates = sorted(estimates[g * group_size : (g + 1) * group_size])
        # 홀수/짝수 요소 개수에 따른 중앙값 구하기
        mid = group_size // 2
        if group_size % 2 == 1:
            med = float(group_estimates[mid])
        else:
            med = (group_estimates[mid - 1] + group_estimates[mid]) / 2.0
        group_medians.append(med)

    # 그룹별 중앙값들의 평균 계산
    final_estimate = sum(group_medians) / len(group_medians)
    
    # Flajolet-Martin 보정 계수 phi ≈ 0.77351 적용
    phi = 0.77351
    return final_estimate / phi


def reservoir_sample(stream, k, seed=246):
    """Keep k items uniformly at random from a stream of unknown length."""
    rng = random.Random(seed)
    reservoir = []

    for i, item in enumerate(stream):
        if i < k:
            # 처음 k개 요소는 표본에 그대로 채움
            reservoir.append(item)
        else:
            # i번째 요소(0-indexed이므로 전체 본 개수는 i+1개)에 대해
            # k / (i + 1) 확률로 표본에 편입 결정
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item

    return reservoir


# ------------------------------------------------------------------- harness
def verify():
    fails = 0
    rng = random.Random(246)

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<46} {detail}")
        fails += not ok

    # --- Bloom: no false negatives, ever
    try:
        bf = BloomFilter(m=8192, k=5)
    except NotImplementedError:
        print("  BloomFilter is still a stub"); return 1
    inserted = [f"item-{i}" for i in range(800)]
    for x in inserted:
        bf.add(x)
    check("no false negatives", all(x in bf for x in inserted))

    absent = [f"other-{i}" for i in range(20_000)]
    fp = sum(1 for x in absent if x in bf) / len(absent)
    predicted = bf.expected_fp_rate(len(inserted))
    close = abs(fp - predicted) < max(0.02, predicted * 0.5)
    check("measured false-positive rate matches theory", close,
          f"measured {fp:.3%}, predicted {predicted:.3%}")

    # --- Flajolet-Martin: a factor of two is what this method gives you
    try:
        distinct = 20_000
        stream = [f"k{rng.randrange(distinct)}" for _ in range(120_000)]
        est = flajolet_martin(stream)
    except NotImplementedError:
        print("  flajolet_martin is still a stub"); return 1
    true_distinct = len(set(stream))
    ratio = est / true_distinct
    check("distinct estimate within a factor of 2", 0.5 <= ratio <= 2.0,
          f"estimated {est:,.0f}, true {true_distinct:,} ({ratio:.2f}x)")

    # --- Reservoir: uniform over many trials
    try:
        counts = [0] * 20
        trials = 4000
        for t in range(trials):
            s = reservoir_sample(range(20), 5, seed=t)
            for i in s:
                counts[i] += 1
    except NotImplementedError:
        print("  reservoir_sample is still a stub"); return 1
    expected = trials * 5 / 20
    spread = (max(counts) - min(counts)) / expected
    check("reservoir is uniform across items", spread < 0.15,
          f"spread {spread:.1%} around {expected:.0f}")

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())