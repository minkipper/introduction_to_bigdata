# Task 3 · Find the Same Pairs, Comparing Far Less

**Files** — `task3_scale.py` · harness `bench.py` (**do not edit the harness**)
**Theory** — §3.4
**Kind** — improvement · synthetic data, runs anywhere

---

## What you are given

`BruteForce` compares every pair. On the harness's 2,120 documents that is
2.2 million comparisons, and it is completely correct. On 3 million documents
it is 4.5 trillion, and it is completely useless.

```bash
python3 bench.py            # baseline, about 6 seconds
python3 bench.py --yours
```

The documents are synthetic with a fixed seed and **120 near-duplicate pairs
planted in them**, so the ground truth is known and everyone's numbers compare.

**Where the baseline lands:**

```
  baseline     comparisons  2,246,140   recall 100.0%   precision 100.0%     5.62s
```

## How you are scored

The harness counts every call to `similarity()`. That is your score. Signatures,
banding and bucketing are **not** charged — which is deliberate, and roughly true
at scale, where comparison is the expensive part and hashing is linear.

It also measures **recall**: how many of the truly similar pairs you found.
Skipping comparisons is trivial. Skipping comparisons without losing pairs is
the task.

## Requirements

| # | Requirement |
|---|---|
| R1 | `YourFinder(threshold)` with `find(docs, similarity)` returning `{(i, j), ...}` |
| R2 | `bench.py` unmodified |
| R3 | **Recall at least 90%.** Below that nothing else counts |
| R4 | Fewer comparisons than the baseline |
| R5 | In `observation.md`: your choice of hash count and band count, **with the arithmetic from §3.4.2 that justifies it** |

## Grading

| | Requirement |
|---|---|
| pass | R1–R4, more than 50% avoided |
| good | ≥ **95%** avoided |
| **strong** | ≥ **99%** avoided **and** recall ≥ 95% |

All three are reachable — a reasonable banding gets above 99.9% with full recall.
The failure mode is not being too slow, it is choosing bands badly and quietly
losing half the pairs. The harness will tell you; `recall` is the first number
to read.

## R5 · the arithmetic, not the guess

For a signature of `n` hashes split into `b` bands of `r = n/b` rows, a pair at
similarity `s` becomes a candidate with probability

```
    1 - (1 - s^r)^b
```

That is an S-curve, and its step sits near `(1/b)^(1/r)`. The threshold here is
**0.6**. Pick `n` and `b` so the step lands where you want it, then say in
`observation.md` what step you were aiming for and why — above the threshold, or
below it, and what each choice costs you.

Answering "I tried 120 hashes and 30 bands and it worked" is a pass. Answering
with the step you computed is the point of the task.

## What to write in `observation.md`

- Your `n` and `b`, the step they put the S-curve at, and why there
- What happened to recall when you moved the step the wrong way — try it
- The harness does not charge you for hashing. At what scale would that stop
  being a fair simplification?
