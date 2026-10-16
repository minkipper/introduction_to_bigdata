#!/usr/bin/env python3
"""Week 5 · Task 1 — PageRank, and the two ways it breaks.

Textbook §5.1 (PageRank), §5.1.3 (dead ends), §5.1.4 (spider traps).

PageRank is a random surfer following links forever, and the rank of a page is
how often the surfer is there. As one line of linear algebra it is
`r = M r`, and as written it does not work on the actual web.

Two structures break it, and the fix for both is the same one line. Build the
broken version first so you can see each failure, then fix it.

    python3 task1_pagerank.py --verify
"""
import argparse

# A: -> B, C      B: -> C      C: -> A
SIMPLE = {"A": ["B", "C"], "B": ["C"], "C": ["A"]}

# C has no out-links at all. Rank leaks out of the graph and everything -> 0.
DEAD_END = {"A": ["B", "C"], "B": ["C"], "C": []}

# C and D only point at each other. They absorb everything.
SPIDER_TRAP = {"A": ["B"], "B": ["C"], "C": ["D"], "D": ["C"]}


def pagerank(graph, beta=0.85, iterations=100, tol=1e-10):
    """Rank every node. Return {node: rank}, summing to 1.

    `beta` is the probability the surfer follows a link. With probability
    1 - beta they teleport to a node chosen uniformly.

    You have to handle both of these, and the textbook handles them the same way:

      dead ends    a node with no out-links. Where does its rank go, and where
                   should it go instead?
      spider traps a group of nodes that only link to each other. Without
                   teleporting, they end up with all of it

    Stop early when the ranks stop moving - `tol` is the L1 change below which
    you should call it converged. Return the ranks, and set `pagerank.iterations`
    to how many you actually used, because Task 2 measures that.
    """
    # 모든 노드에 같은 초기 확률을 부여합니다.
    nodes = list(graph)
    n = len(nodes)
    pagerank.iterations = 0
    if n == 0:
        return {}
    ranks = {node: 1.0 / n for node in nodes}

    for step in range(iterations):
        # 나가는 링크가 없는 노드의 점수를 모아 전체 노드에 재분배합니다.
        dangling_mass = sum(ranks[node] for node in nodes if not graph[node])
        # 무작위 이동 점수와 dead end에서 재분배되는 점수를 먼저 넣습니다.
        base = (1.0 - beta) / n + beta * dangling_mass / n
        new_ranks = {node: base for node in nodes}

        # 링크를 따르는 확률 beta만큼의 점수를 연결된 노드에 나눠 줍니다.
        for node in nodes:
            links = graph[node]
            if links:
                share = beta * ranks[node] / len(links)
                for target in links:
                    new_ranks[target] += share

        # 이전 점수와의 L1 차이를 계산하고 실제 수행 횟수를 기록합니다.
        change = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks
        pagerank.iterations = step + 1
        if change < tol:
            break

    return ranks


def pagerank_no_teleport(graph, iterations=100):
    """The broken version: beta = 1, no teleporting. Build this too.

    It exists so you can watch both failures happen rather than take them on
    trust. The harness checks that it really does fail.
    """
    # 정상 버전과 동일하게 균등한 점수에서 시작합니다.
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return {}
    ranks = {node: 1.0 / n for node in nodes}

    for _ in range(iterations):
        # teleport와 dead end 재분배 없이 링크로 전달된 점수만 누적합니다.
        new_ranks = {node: 0.0 for node in nodes}
        for node in nodes:
            links = graph[node]
            if links:
                share = ranks[node] / len(links)
                for target in links:
                    new_ranks[target] += share
            # 링크가 없으면 점수가 사라지도록 그대로 둡니다.
        ranks = new_ranks

    # 총합을 정규화하지 않아 점수 유실과 trap 집중을 관찰할 수 있습니다.
    return ranks


# ------------------------------------------------------------------- harness
def verify():
    fails = 0

    def check(label, ok, detail=""):
        nonlocal fails
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:<48} {detail}")
        fails += not ok

    try:
        r = pagerank(SIMPLE)
    except NotImplementedError:
        print("  pagerank is still a stub"); return 1

    check("ranks sum to 1", abs(sum(r.values()) - 1) < 1e-6, f"{sum(r.values()):.6f}")
    check("every node ranked", set(r) == set(SIMPLE), sorted(r))
    # C is pointed at by both A and B, so it has to come first.
    check("C outranks A and B", r["C"] > r["A"] and r["C"] > r["B"],
          {k: round(v, 4) for k, v in sorted(r.items())})

    try:
        broken = pagerank_no_teleport(DEAD_END)
    except NotImplementedError:
        print("  pagerank_no_teleport is still a stub"); return 1
    check("without the fix, a dead end drains the graph",
          sum(broken.values()) < 0.5, f"total rank {sum(broken.values()):.4f}")

    fixed = pagerank(DEAD_END)
    check("with the fix, rank is conserved",
          abs(sum(fixed.values()) - 1) < 1e-6, f"{sum(fixed.values()):.6f}")

    trapped = pagerank_no_teleport(SPIDER_TRAP)
    check("without the fix, a spider trap takes everything",
          trapped["C"] + trapped["D"] > 0.95,
          f"C+D = {trapped['C'] + trapped['D']:.4f}")

    untrapped = pagerank(SPIDER_TRAP)
    check("with the fix, A and B keep some rank",
          untrapped["A"] > 0.01 and untrapped["B"] > 0.01,
          {k: round(v, 4) for k, v in sorted(untrapped.items())})

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())
