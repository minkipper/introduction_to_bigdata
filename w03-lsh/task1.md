# Task 1 · Minhash and LSH From the Matrix Up

**File** — `task1_minhash.py`
**Theory** — §3.2 shingling, §3.3 minhashing, §3.4 LSH
**Kind** — implementation · runs anywhere, no data needed

---

## What you are building

Two ideas stacked:

- **minhash** — replace a set with a short signature, such that the probability
  two signatures agree in a position **equals** their Jaccard similarity
- **LSH** — hash bands of those signatures so similar pairs collide, and only
  compare the ones that did

The textbook's §3.3.5 example is five rows and four columns, small enough to
check by hand, and the harness checks you against it.

## Requirements

| # | Requirement |
|---|---|
| R1 | `jaccard(a, b)`; an empty union is 0, not an error |
| R2 | `minhash_signatures` walks **each row once**, updating every column that has a 1 in it |
| R3 | Signatures match Figure 3.4: S1 `[1,0]`, S2 `[3,2]`, S3 `[0,0]`, S4 `[1,0]` |
| R4 | `lsh_candidates` returns pairs `(i, j)` with `i < j`, colliding in **at least one** band |
| R5 | Decide what happens when the signature length does not divide by `bands`, and say what you decided |

**R2 is the requirement that matters.** Sorting the rows, or re-scanning the
matrix once per column, gives the right answer and does not survive a matrix
that will not fit in memory. Not fitting in memory is the entire subject of this
course, so write the one-pass version even though it is more awkward.

## Pass condition

```bash
python3 task1_minhash.py --verify
```

```
  all ok
```

Then read what the harness prints afterwards. S1 and S4 agree in **both**
signature positions, which estimates their similarity as 1.0 when it is really
2/3. Two hash functions is not many, and that gap is the whole reason §3.4
exists.

## What to write in `observation.md`

- Why one pass over the rows, rather than one pass per column?
- Your R5 decision on the leftover rows
- The estimate for S1–S4 was 1.0 and the truth is 2/3. What would you change to
  narrow that, and what would it cost?
