#!/usr/bin/env python3
"""Week 7 · Task 3 — The same clustering, with far fewer distances.

Textbook §7.3, and §7.1.3 for why distance is the expensive part.

`NaiveKMeans` does what the definition says: every iteration, compare every
point with every centroid. That is n x k distance computations per pass, and on
a real dataset the distance computation is almost the entire cost.

Most of that work is wasted. After the first couple of iterations, most points
do not change cluster, and most centroids barely move. The triangle inequality
lets you prove a point cannot have changed cluster **without measuring it**.

    python3 bench.py
    python3 bench.py --yours

Correctness first: your final SSE must match the baseline's. Fewer distances
with a worse clustering is not an improvement, it is a different algorithm.
"""


class NaiveKMeans:
    """Every point against every centroid, every iteration."""

    def __init__(self, k, max_iter=50):
        self.k, self.max_iter = k, max_iter

    def run(self, points, initial_centroids, distance):
        centroids = list(initial_centroids)
        labels = [None] * len(points)
        for self.iterations in range(1, self.max_iter + 1):
            changed = False
            for i, p in enumerate(points):
                best, bi = None, 0
                for j, c in enumerate(centroids):
                    d = distance(p, c)
                    if best is None or d < best - 1e-12:
                        best, bi = d, j
                if labels[i] != bi:
                    labels[i] = bi
                    changed = True
            centroids = _means(points, labels, self.k, centroids)
            if not changed:
                break
        return labels, centroids


def _means(points, labels, k, previous):
    out = []
    for j in range(k):
        group = [p for p, l in zip(points, labels) if l == j]
        if group:
            out.append(tuple(sum(v) / len(group) for v in zip(*group)))
        else:
            out.append(previous[j])
    return out


class YourKMeans:
    """Your k-means.

        __init__(k, max_iter=50)
        run(points, initial_centroids, distance) -> (labels, centroids)

    Same starting centroids, same `distance` function - which is counted. That
    counter is your score.

    The idea, from §7.3 and the triangle inequality:

        if a point is much closer to its current centroid than that centroid
        moved, and than any other centroid could have come, then it cannot have
        changed cluster, and you do not have to measure anything

    Keeping enough bookkeeping to make that test cheap, without keeping so much
    that the bookkeeping costs more than the distances, is the task.

    You may call `distance` on two centroids, and those calls are counted too -
    which is the correct accounting, since that is real work. There are k of
    them and n points, and k is much smaller than n. That asymmetry is the
    whole opening.
    """

    def __init__(self, k, max_iter=50):
        self.k, self.max_iter = k, max_iter
        self.iterations = 0
        self.first_iter_calls = 0

    def run(self, points, initial_centroids, distance):
        n = len(points)
        k = self.k
        centroids = list(initial_centroids)
        labels = [0] * n
        # Per-point bounds (Hamerly):
        # u[i]: upper bound on distance(points[i], centroids[labels[i]])
        # l[i]: lower bound on min_{j != labels[i]} distance(points[i], centroids[j])
        u = [0.0] * n
        l = [0.0] * n

        # Iteration 1: full pass to initialize exact labels, u[i], and l[i]
        self.iterations = 1
        for i, p in enumerate(points):
            best = float("inf")
            second = float("inf")
            bi = 0
            for j, c in enumerate(centroids):
                d = distance(p, c)
                if d < best - 1e-12:
                    second = best
                    best = d
                    bi = j
                elif d < second:
                    second = d
            labels[i] = bi
            u[i] = best
            l[i] = second

        self.first_iter_calls = getattr(distance, "calls", n * k)
        new_centroids = _means(points, labels, k, centroids)

        for it in range(2, self.max_iter + 1):
            self.iterations = it
            # Compute how far each centroid moved in the previous update step (k calls)
            delta = [distance(centroids[j], new_centroids[j]) for j in range(k)]
            centroids = new_centroids

            # Find largest and second-largest centroid shifts for O(1) lower-bound updates
            max1, max2, argmax1 = -1.0, -1.0, 0
            for j, dj in enumerate(delta):
                if dj > max1:
                    max2 = max1
                    max1 = dj
                    argmax1 = j
                elif dj > max2:
                    max2 = dj

            # Update per-point upper and lower bounds via triangle inequality
            for i in range(n):
                a = labels[i]
                u[i] += delta[a]
                l[i] -= (max2 if a == argmax1 else max1)

            # Pairwise centroid distances cdist[a][b] and half-distance to nearest rival s[a]
            cdist = [[0.0] * k for _ in range(k)]
            s = [float("inf")] * k
            for a in range(k):
                ca = centroids[a]
                for b in range(a + 1, k):
                    d = distance(ca, centroids[b])
                    cdist[a][b] = d
                    cdist[b][a] = d
                    if d < s[a]:
                        s[a] = d
                    if d < s[b]:
                        s[b] = d
            for a in range(k):
                s[a] *= 0.5

            changed = False
            for i, p in enumerate(points):
                a = labels[i]
                # Bound 1 (Hamerly + inter-centroid): if u[i] <= max(l[i], s[a]), cannot change
                m = l[i] if l[i] > s[a] else s[a]
                if u[i] <= m - 1e-12:
                    continue

                # Tighten upper bound u[i] with exact distance to current centroid
                ua = distance(p, centroids[a])
                u[i] = ua
                if ua <= m - 1e-12:
                    continue

                # Point still ambiguous: check rival centroids, pruning with 0.5 * cdist[bi][j]
                best = ua
                second = float("inf")
                bi = a
                for j in range(k):
                    if j == a:
                        continue
                    # Elkan pairwise triangle inequality: if best <= 0.5 * d(c_bi, c_j), skip
                    # (Wait: to maintain a valid second-closest lower bound l[i] for next iter,
                    #  if 0.5 * cdist[bi][j] > second, j cannot even affect `second`!)
                    half_cc = 0.5 * cdist[bi][j]
                    if best <= half_cc - 1e-12 and second <= (cdist[bi][j] - best) - 1e-12:
                        continue
                    d = distance(p, centroids[j])
                    if d < best - 1e-12:
                        second = best
                        best = d
                        bi = j
                    elif d < second:
                        second = d

                u[i] = best
                l[i] = second
                if bi != a:
                    labels[i] = bi
                    changed = True

            new_centroids = _means(points, labels, k, centroids)
            if not changed:
                centroids = new_centroids
                break

        return labels, centroids
