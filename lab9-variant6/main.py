"""
Лабораторная работа № 9. Вариант 6.

Способы представления графов и алгоритмы обхода.

Параметры варианта 6 (табл. 2):
  - тип графа: Р — разреженный (m ≈ 3n)
  - ориентированный
  - размеры n: 1000; 2000; 5000
  - задача: достижимые вершины из заданной
  - доп. И6: время обхода vs плотность при фиксированном n

Дисциплина: Основы алгоритмизации и программирование
Студент: Голдаева Валерия Валентиновна, группа БИ 2-1
"""

import random
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

# ===========================================================================
# 1. ПРЕДСТАВЛЕНИЯ ГРАФА (одинаковый интерфейс)
# ===========================================================================

class MatrixGraph:
    """
    Ориентированный граф на матрице смежности.
    Память: Θ(n²). Проверка ребра: Θ(1). Соседи: Θ(n).
    """

    def __init__(self, n: int) -> None:
        self.n = n
        # matrix[u][v] = 1 ⇔ есть дуга u → v
        self.matrix: List[List[int]] = [[0] * n for _ in range(n)]

    def add_edge(self, u: int, v: int) -> None:
        """Добавить дугу u → v. Θ(1)."""
        self.matrix[u][v] = 1

    def has_edge(self, u: int, v: int) -> bool:
        """Есть ли дуга u → v. Θ(1)."""
        return self.matrix[u][v] == 1

    def neighbors(self, v: int) -> List[int]:
        """Исходящие соседи вершины v. Θ(n)."""
        row = self.matrix[v]
        return [u for u in range(self.n) if row[u]]


class ListGraph:
    """
    Ориентированный граф на списках смежности.
    Память: Θ(n + m). Соседи: Θ(deg⁺ v).
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.adj: List[List[int]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int) -> None:
        """Добавить дугу u → v. Амортизированно Θ(1)."""
        self.adj[u].append(v)

    def has_edge(self, u: int, v: int) -> bool:
        """Есть ли дуга u → v. Θ(deg⁺ u)."""
        return v in self.adj[u]

    def neighbors(self, v: int) -> List[int]:
        """Исходящие соседи. Θ(deg⁺ v)."""
        return self.adj[v]


# ===========================================================================
# 2. ОБХОДЫ
# ===========================================================================

def bfs(graph, start: int):
    """
    Обход в ширину из start.

    Возвращает:
      order  — порядок посещения
      dist   — расстояние (число дуг) от start; −1 если недостижима
      parent — предок в дереве BFS (−1 у корня / недостижимых)
    Трудоёмкость: Θ(n + m) для списков, Θ(n²) для матрицы.
    """
    n = graph.n
    dist = [-1] * n
    parent = [-1] * n
    order: List[int] = []
    dist[start] = 0
    queue: deque = deque([start])         # FIFO
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in graph.neighbors(v):      # все исходящие
            if dist[u] == -1:             # ещё не посещали
                dist[u] = dist[v] + 1
                parent[u] = v
                queue.append(u)
    return order, dist, parent


def dfs_iter(graph, start: int) -> List[int]:
    """
    Итеративный DFS (явный стек) — без риска RecursionError.
    """
    visited = [False] * graph.n
    order: List[int] = []
    stack = [start]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        # reversed — чтобы порядок был ближе к рекурсивному
        for u in reversed(graph.neighbors(v)):
            if not visited[u]:
                stack.append(u)
    return order


def reachable_from(graph, start: int) -> Tuple[List[int], List[int]]:
    """
    Задача варианта: все вершины, достижимые из start.

    Возвращает (список достижимых, массив dist).
    """
    order, dist, _ = bfs(graph, start)
    return order, dist


# ===========================================================================
# 3. ГЕНЕРАЦИЯ ГРАФОВ
# ===========================================================================

def generate_sparse(cls, n: int, m_factor: int = 3, seed: int = 42):
    """
    Разреженный ориентированный граф: m ≈ m_factor · n случайных дуг.
    """
    rng = random.Random(seed)
    g = cls(n)
    target = min(m_factor * n, n * (n - 1))
    added = set()
    while len(added) < target:
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        if (u, v) not in added:
            added.add((u, v))
            g.add_edge(u, v)
    return g, len(added)


def generate_density(cls, n: int, density: float, seed: int = 42):
    """
    Ориентированный граф с заданной долей дуг от максимума n(n−1).
    """
    rng = random.Random(seed)
    g = cls(n)
    max_e = n * (n - 1)
    target = int(max_e * density)
    added = set()
    while len(added) < target:
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        if (u, v) not in added:
            added.add((u, v))
            g.add_edge(u, v)
    return g, len(added)


def deep_sizeof(obj) -> int:
    """Приблизительный расход памяти вложенных списков, байт."""
    if isinstance(obj, list):
        return sys.getsizeof(obj) + sum(deep_sizeof(x) for x in obj)
    return sys.getsizeof(obj)


# ===========================================================================
# 4. ДЕМО НА МАЛОМ ГРАФЕ
# ===========================================================================

def demo_small() -> None:
    print('=' * 70)
    print('Проверка на малом ориентированном графе (6 вершин)')
    print('=' * 70)
    g = ListGraph(6)
    # 0→1, 0→2, 1→3, 2→3, 3→4; 5→4 (5 недостижима из 0)
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (5, 4)]:
        g.add_edge(u, v)
    order, dist, parent = bfs(g, 0)
    reach = [v for v, d in enumerate(dist) if d >= 0]
    print(f'BFS от 0: порядок {order}')
    print(f'Расстояния: {dist}')
    print(f'Достижимые из 0: {reach}')
    print(f'Недостижимые: {[v for v, d in enumerate(dist) if d < 0]}')
    print(f'DFS от 0: {dfs_iter(g, 0)}')
    # путь 0 → 4
    path = []
    cur = 4
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    print(f'Кратчайший путь 0→4: {path} (длина {dist[4]})')
    print()


# ===========================================================================
# 5. ОСНОВНЫЕ ИЗМЕРЕНИЯ
# ===========================================================================

def run_main() -> None:
    print('=' * 70)
    print('Разреженный ориентированный граф (m ≈ 3n): время и память')
    print('=' * 70)
    print(f"{'n':>6} {'m':>8} {'матрица, с':>12} {'списки, с':>12}"
          f" {'матрица, КБ':>12} {'списки, КБ':>12} {'достижимо':>10}")
    for n in [1000, 2000, 5000]:
        gm, m = generate_sparse(MatrixGraph, n)
        gl, _ = generate_sparse(ListGraph, n)
        t0 = time.perf_counter()
        order_m, dist_m, _ = bfs(gm, 0)
        t_m = time.perf_counter() - t0
        t0 = time.perf_counter()
        order_l, dist_l, _ = bfs(gl, 0)
        t_l = time.perf_counter() - t0
        mem_m = deep_sizeof(gm.matrix) / 1024
        mem_l = deep_sizeof(gl.adj) / 1024
        reach = sum(1 for d in dist_l if d >= 0)
        print(f'{n:6d} {m:8d} {t_m:12.4f} {t_l:12.4f}'
              f' {mem_m:12.1f} {mem_l:12.1f} {reach:10d}')
    print()


def run_i6() -> None:
    """
    И6: время BFS vs плотность при фиксированном n.
    Ищем, когда матрица становится выгоднее списков.
    """
    print('=' * 70)
    print('И6: время обхода vs плотность (n = 400, ориентированный)')
    print('=' * 70)
    n = 400
    print(f"{'плотность':>10} {'m':>8} {'матрица, с':>12} {'списки, с':>12} {'выгоднее':>12}")
    for dens in [0.01, 0.05, 0.20, 0.50, 1.00]:
        gm, m = generate_density(MatrixGraph, n, dens)
        gl, _ = generate_density(ListGraph, n, dens)
        t0 = time.perf_counter(); bfs(gm, 0); t_m = time.perf_counter() - t0
        t0 = time.perf_counter(); bfs(gl, 0); t_l = time.perf_counter() - t0
        winner = 'матрица' if t_m < t_l else 'списки'
        print(f'{dens*100:9.0f}% {m:8d} {t_m:12.4f} {t_l:12.4f} {winner:>12}')
    print('В Python матрица из объектов list редко обгоняет списки даже при 100%')
    print('плотности из-за накладных расходов на каждую ячейку.')
    print()


if __name__ == '__main__':
    demo_small()
    run_main()
    run_i6()
