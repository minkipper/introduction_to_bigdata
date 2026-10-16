#!/usr/bin/env python3
"""Week 5 · Task 3 — PageRank on a graph that will not fit as a matrix.

Textbook §5.2 (efficient PageRank), §5.2.1 - §5.2.3.

`DenseMatrix` is PageRank written the way the equations are written: build the
transition matrix M, multiply. It is correct, it is easy to read, and it stores
n^2 numbers for a graph with almost no edges.

The web's matrix is about 99.9999% zeros. Storing them is the problem, and
§5.2 is the chapter about not doing that.

    python3 bench.py
    python3 bench.py --yours

Correctness first: the harness compares your ranks against the dense version
element by element. A fast PageRank that ranks pages differently is a different
algorithm, not a faster one.
"""


class DenseMatrix:
    """PageRank as written in the equations. Stores n^2 floats."""

    def __init__(self, beta=0.85, tol=1e-10, max_iter=100):
        self.beta, self.tol, self.max_iter = beta, tol, max_iter

    def run(self, graph):
        nodes = list(graph)
        n = len(nodes)
        index = {v: i for i, v in enumerate(nodes)}

        # the full transition matrix, zeros and all
        M = [[0.0] * n for _ in range(n)]
        for v, outs in graph.items():
            if outs:
                share = 1.0 / len(outs)
                for w in outs:
                    M[index[w]][index[v]] = share
            else:
                for i in range(n):           # dead end: spread it everywhere
                    M[i][index[v]] = 1.0 / n

        r = [1.0 / n] * n
        for self.iterations in range(1, self.max_iter + 1):
            nr = [0.0] * n
            for i in range(n):
                row = M[i]
                s = 0.0
                for j in range(n):
                    if row[j]:
                        s += row[j] * r[j]
                nr[i] = self.beta * s + (1 - self.beta) / n
            delta = sum(abs(a - b) for a, b in zip(nr, r))
            r = nr
            if delta < self.tol:
                break
        return {v: r[index[v]] for v in nodes}

    def memory_floats(self):
        return getattr(self, "_n", 0) ** 2


class YourPageRank:
    """Your PageRank.

        __init__(beta=0.85, tol=1e-10, max_iter=100)
        run(graph) -> {node: rank}
        memory_floats() -> the largest number of floats you held at once

    Same ranks, to within 1e-9 per node. Far less memory.

    `graph` is {node: [out-neighbours]}. Note what that already is: an adjacency
    list, which is the sparse representation. The dense version throws that
    structure away and then pays to get it back.

    Two things to be careful about, and they are the same two as Task 1:

      * dead ends, whose rank has to go somewhere
      * the teleport term, which touches every node and is therefore the one
        part that looks like it needs a dense operation - it does not, and
        working out why is the point of §5.2.3

    `memory_floats()` is on your honour and the harness reads it. Count the
    numbers you actually hold at once.
    """

    def __init__(self, beta=0.85, tol=1e-10, max_iter=100):
        # 계산 조건과 최대 실수 저장 개수를 기록합니다.
        self.beta, self.tol, self.max_iter = beta, tol, max_iter
        self.iterations = 0
        self._peak_floats = 0

    def run(self, graph):
        # 인접 리스트를 정수 인덱스로 바꾸며 n×n 행렬은 만들지 않습니다.
        nodes = list(graph)
        n = len(nodes)
        self.iterations = 0
        if n == 0:
            self._peak_floats = 2
            return {}
        index = {node: i for i, node in enumerate(nodes)}
        # 기준 행렬의 중복 링크 덮어쓰기와 동일한 의미를 유지합니다.
        adjacency = [list(dict.fromkeys(index[w] for w in graph[v])) for v in nodes]
        degrees = [len(graph[v]) for v in nodes]
        dangling = [i for i, degree in enumerate(degrees) if degree == 0]
        r = [1.0 / n] * n

        # 두 벡터(2n), beta/tol과 임시 실수의 보수적인 상한을 집계합니다.
        # 링크 인덱스와 차수는 정수입니다. 전체 바이트 사용량과는 다릅니다.
        # 반환 딕셔너리는 최종 벡터의 실수 객체를 참조하므로 복제하지 않습니다.
        self._peak_floats = 2 * n + 16
        for step in range(self.max_iter):
            # dead end 점수를 한 번 합산해 전체 노드에 균등 분배합니다.
            dead_mass = sum(r[i] for i in dangling)
            base = (1.0 - self.beta) / n + self.beta * dead_mass / n
            nr = [base] * n
            # 존재하는 링크만 방문하므로 반복당 O(n + E)입니다.
            for j, targets in enumerate(adjacency):
                if degrees[j]:
                    share = self.beta * r[j] / degrees[j]
                    for i in targets:
                        nr[i] += share
            # teleport는 base에 포함되므로 밀집 행렬 연산이 필요 없습니다.
            delta = sum(abs(a - b) for a, b in zip(nr, r))
            r = nr
            del nr  # 다음 반복 전에 벡터의 별도 참조를 제거합니다.
            self.iterations = step + 1
            if delta < self.tol:
                break
        return {node: r[i] for i, node in enumerate(nodes)}

    def memory_floats(self):
        # 동시에 보관하는 실수 개수의 보수적 상한을 반환합니다.
        return self._peak_floats
