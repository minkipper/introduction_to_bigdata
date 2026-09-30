# Task 2 · Find the Crossover on Your Own Machine

**File** — `task2_crossover.py`
**Theory** — §3.4
**Kind** — **measurement. The number is about your hardware, and nobody else's.**

---

## Why this task exists

Everybody knows brute force is quadratic. That is not the interesting part.

The interesting part is **where, on the laptop in front of you, it stops being
usable** — and that answer depends on your CPU, your memory, and how large your
shingle sets are. You cannot look it up and you cannot ask an agent for it.
You have to run it until it hurts.

## What to do

```bash
python3 task2_crossover.py --sizes 250,500,1000,2000
python3 task2_crossover.py --sizes 4000,8000
python3 task2_crossover.py --sizes 16000          # keep going
```

Each run records wall time, comparison count and peak memory for both methods
at each size, and appends to `out/crossover.json` along with your machine's
description.

| # | Requirement |
|---|---|
| A1 | At least **five** sizes, spanning at least a 16× range |
| A2 | Keep going until something is genuinely unpleasant — a minute of waiting, or memory pressure. **Record that size** |
| A3 | Plot or tabulate time against n for both methods → `out/curve.md` |
| A4 | Confirm the brute-force curve is quadratic. Doubling n should roughly **quadruple** the time — check it against your own numbers rather than asserting it |
| A5 | Report peak memory for both at your largest n |
| A6 | State your machine: CPU, RAM, and whether anything else was running |

A4 is the one people skip. You have four or five points; do the arithmetic and
say whether they actually fit. If they do not, something in your measurement is
wrong, and finding out what is more valuable than the curve.

## The crossover

| # | Requirement |
|---|---|
| A7 | At small n, brute force is **faster** than your LSH. Find the n where they cross |
| A8 | Explain why LSH loses at small n. Be specific about what it is paying for |

A8 matters. LSH does work before it compares anything — hashing every document,
several times over — and that cost is linear but not free. Below some size you
are paying for machinery you did not need.

## Pass condition

`out/crossover.json` has five or more sizes with your machine recorded, and
`out/curve.md` answers A3–A8.

```bash
python3 test_tasks.py --task 2
```

## If your machine is small

That is a finding, not a problem. A laptop that runs out of memory at n=8,000
tells you something precise about the constant factors, and "I could not get
past 8,000 because of X" with the evidence is a complete answer. Say what X was.

## What to write in `observation.md`

- Your crossover n, and your machine
- Whether the quadratic check in A4 held
- The size where it became unpleasant, and what ran out first — time or memory
