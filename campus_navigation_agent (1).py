"""
╔══════════════════════════════════════════════════════════════╗
║   🎓  INTELLIGENT CAMPUS NAVIGATION AI AGENT  🎓            ║
║   Project 01 — Graph Search & Heuristic Algorithms          ║
╚══════════════════════════════════════════════════════════════╝
"""

import heapq
import math
import time
from collections import deque

# ─────────────────────────────────────────────
#  🎨  COLORFUL TERMINAL COLORS
# ─────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    ORANGE = "\033[38;5;208m"
    PURPLE = "\033[38;5;135m"
    LIME   = "\033[38;5;118m"
    PINK   = "\033[38;5;213m"

def color(text, *codes):
    return "".join(codes) + str(text) + C.RESET


# ─────────────────────────────────────────────
#  🏛️  CAMPUS GRAPH DEFINITION
# ─────────────────────────────────────────────
class CampusGraph:
    """
    University campus modelled as a weighted undirected graph.

    State Representation:
        - Each node = campus location (integer ID)
        - Each edge = walking path with cost (distance in units)
        - Node attributes: coordinates (x, y), category, accessibility

    Cost Model:
        - Base cost: edge weight (walking distance)
        - Accessibility penalty: +2 if wheelchair route restricted
        - Time estimate: cost * 2 minutes
    """

    LOCATIONS = {
        0:  {"name": "Main Gate",       "x": 60,  "y": 100, "type": "entry",      "accessible": True},
        1:  {"name": "Library",         "x": 200, "y": 40,  "type": "academic",   "accessible": True},
        2:  {"name": "CS Department",   "x": 300, "y": 80,  "type": "academic",   "accessible": True},
        3:  {"name": "Cafeteria",       "x": 200, "y": 140, "type": "amenity",    "accessible": True},
        4:  {"name": "Sports Complex",  "x": 450, "y": 160, "type": "recreation", "accessible": False},
        5:  {"name": "Hostel A",        "x": 550, "y": 40,  "type": "housing",    "accessible": True},
        6:  {"name": "Admin Block",     "x": 120, "y": 160, "type": "academic",   "accessible": True},
        7:  {"name": "Labs",            "x": 380, "y": 40,  "type": "academic",   "accessible": True},
        8:  {"name": "Auditorium",      "x": 500, "y": 90,  "type": "amenity",    "accessible": True},
        9:  {"name": "Parking Lot",     "x": 600, "y": 150, "type": "entry",      "accessible": True},
        10: {"name": "Medical Center",  "x": 580, "y": 100, "type": "amenity",    "accessible": True},
        11: {"name": "Garden",          "x": 350, "y": 180, "type": "recreation", "accessible": True},
    }

    # (from, to, weight)
    EDGES = [
        (0,  1,  4),
        (0,  6,  5),
        (0,  9,  8),
        (1,  2,  3),
        (1,  3,  4),
        (1,  6,  3),
        (2,  7,  2),
        (2,  3,  4),
        (3,  4,  6),
        (3,  11, 3),
        (4,  11, 4),
        (4,  5,  7),
        (5,  8,  5),
        (6,  10, 4),
        (7,  8,  3),
        (8,  9,  5),
        (9,  10, 2),
        (10, 11, 6),
    ]

    def __init__(self, accessibility_mode: str = "normal"):
        """
        Build adjacency list.

        Args:
            accessibility_mode: 'normal' | 'wheelchair' | 'elevator'
        """
        self.accessibility_mode = accessibility_mode
        self.graph: dict[int, list[tuple[int, float]]] = {i: [] for i in self.LOCATIONS}
        self._build_graph()

    def _build_graph(self):
        for a, b, w in self.EDGES:
            cost = self._edge_cost(a, b, w)
            if cost is not None:
                self.graph[a].append((b, cost))
                self.graph[b].append((a, cost))

    def _edge_cost(self, a: int, b: int, w: float) -> float | None:
        """Apply accessibility constraints to edge cost."""
        if self.accessibility_mode == "wheelchair":
            if not self.LOCATIONS[a]["accessible"] or not self.LOCATIONS[b]["accessible"]:
                return None           # blocked
            return w + 1             # slightly longer accessible route
        return float(w)

    def heuristic(self, node: int, goal: int) -> float:
        """Euclidean distance heuristic (admissible)."""
        a = self.LOCATIONS[node]
        g = self.LOCATIONS[goal]
        return math.sqrt((a["x"] - g["x"]) ** 2 + (a["y"] - g["y"]) ** 2) / 20

    def node_name(self, n: int) -> str:
        return self.LOCATIONS[n]["name"]

    def node_type(self, n: int) -> str:
        return self.LOCATIONS[n]["type"]

    def path_cost(self, path: list[int]) -> float:
        total = 0.0
        for i in range(len(path) - 1):
            neighbors = dict(self.graph[path[i]])
            total += neighbors.get(path[i + 1], 0)
        return round(total, 2)


# ─────────────────────────────────────────────
#  🤖  AI AGENT — SEARCH ALGORITHMS
# ─────────────────────────────────────────────
class NavigationAgent:
    """
    AI Agent that uses various search algorithms to find
    optimal routes on the campus graph.

    Algorithms supported:
        1. A* Search        — optimal with admissible heuristic
        2. Dijkstra's       — optimal, no heuristic
        3. BFS              — optimal for unweighted (fewest hops)
        4. DFS              — not optimal, explores deeply
        5. Greedy Best-First— fast but not guaranteed optimal
        6. UCS              — Uniform Cost Search (Dijkstra variant)
    """

    def __init__(self, campus: CampusGraph):
        self.campus = campus
        self.stats: dict = {}

    # ── 1. A* Search ──────────────────────────────────────────
    def astar(self, start: int, goal: int) -> list[int] | None:
        """
        A* Search: f(n) = g(n) + h(n)
        g(n) = actual cost from start
        h(n) = heuristic estimate to goal
        """
        open_heap = [(0 + self.campus.heuristic(start, goal), 0.0, start, [start])]
        visited = set()
        nodes_expanded = 0

        while open_heap:
            f, g, current, path = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1

            if current == goal:
                self.stats = {"nodes_expanded": nodes_expanded, "algo": "A* Search"}
                return path

            for neighbor, cost in self.campus.graph[current]:
                if neighbor not in visited:
                    new_g = g + cost
                    new_f = new_g + self.campus.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (new_f, new_g, neighbor, path + [neighbor]))

        return None

    # ── 2. Dijkstra's Algorithm ───────────────────────────────
    def dijkstra(self, start: int, goal: int) -> list[int] | None:
        """
        Dijkstra's: explores by minimum cumulative cost.
        Guaranteed to find shortest path in non-negative weighted graphs.
        """
        dist = {i: float("inf") for i in self.campus.LOCATIONS}
        prev = {i: -1 for i in self.campus.LOCATIONS}
        dist[start] = 0
        heap = [(0.0, start)]
        visited = set()
        nodes_expanded = 0

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            nodes_expanded += 1

            if u == goal:
                break

            for v, w in self.campus.graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    heapq.heappush(heap, (dist[v], v))

        # Reconstruct path
        path, node = [], goal
        while node != -1:
            path.insert(0, node)
            node = prev[node]

        self.stats = {"nodes_expanded": nodes_expanded, "algo": "Dijkstra's"}
        return path if path[0] == start else None

    # ── 3. BFS ────────────────────────────────────────────────
    def bfs(self, start: int, goal: int) -> list[int] | None:
        """
        Breadth-First Search: explores level by level.
        Finds path with fewest edges (hops), not minimum cost.
        """
        queue = deque([(start, [start])])
        visited = {start}
        nodes_expanded = 0

        while queue:
            current, path = queue.popleft()
            nodes_expanded += 1

            if current == goal:
                self.stats = {"nodes_expanded": nodes_expanded, "algo": "BFS"}
                return path

            for neighbor, _ in self.campus.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    # ── 4. DFS ────────────────────────────────────────────────
    def dfs(self, start: int, goal: int) -> list[int] | None:
        """
        Depth-First Search: explores as deep as possible first.
        Not guaranteed to find optimal path.
        """
        stack = [(start, [start])]
        visited = set()
        nodes_expanded = 0

        while stack:
            current, path = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1

            if current == goal:
                self.stats = {"nodes_expanded": nodes_expanded, "algo": "DFS"}
                return path

            for neighbor, _ in self.campus.graph[current]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return None

    # ── 5. Greedy Best-First Search ───────────────────────────
    def greedy_best_first(self, start: int, goal: int) -> list[int] | None:
        """
        Greedy Best-First: always expands node closest to goal by heuristic.
        Fast but NOT guaranteed optimal.
        """
        heap = [(self.campus.heuristic(start, goal), start, [start])]
        visited = set()
        nodes_expanded = 0

        while heap:
            h, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1

            if current == goal:
                self.stats = {"nodes_expanded": nodes_expanded, "algo": "Greedy Best-First"}
                return path

            for neighbor, _ in self.campus.graph[current]:
                if neighbor not in visited:
                    heapq.heappush(heap, (self.campus.heuristic(neighbor, goal), neighbor, path + [neighbor]))

        return None

    # ── 6. Uniform Cost Search ────────────────────────────────
    def ucs(self, start: int, goal: int) -> list[int] | None:
        """
        Uniform Cost Search: expands lowest cumulative cost node.
        Equivalent to Dijkstra's; guaranteed optimal.
        """
        return self.dijkstra(start, goal)  # Same logic

    def find_route(self, start: int, goal: int, algorithm: str = "astar") -> list[int] | None:
        """Dispatch to the selected search algorithm."""
        algos = {
            "astar":   self.astar,
            "dijkstra":self.dijkstra,
            "bfs":     self.bfs,
            "dfs":     self.dfs,
            "greedy":  self.greedy_best_first,
            "ucs":     self.ucs,
        }
        fn = algos.get(algorithm)
        if fn is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return fn(start, goal)


# ─────────────────────────────────────────────
#  🖨️  COLORFUL OUTPUT PRINTER
# ─────────────────────────────────────────────
def print_header():
    lines = [
        color("╔══════════════════════════════════════════════════════════╗", C.CYAN, C.BOLD),
        color("║   🎓  INTELLIGENT CAMPUS NAVIGATION AI AGENT  🎓        ║", C.CYAN, C.BOLD),
        color("║   Graph Search · Heuristics · Optimal Routing           ║", C.CYAN),
        color("╚══════════════════════════════════════════════════════════╝", C.CYAN, C.BOLD),
    ]
    for l in lines:
        print(l)
    print()


TYPE_COLORS = {
    "academic":   C.BLUE,
    "amenity":    C.PINK,
    "recreation": C.LIME,
    "housing":    C.ORANGE,
    "entry":      C.PURPLE,
}

ALGO_LABELS = {
    "astar":    ("⭐", "A* Search",         "Heuristic + Cost"),
    "dijkstra": ("🔵", "Dijkstra's",         "Shortest Path"),
    "bfs":      ("🌊", "BFS",                "Fewest Hops"),
    "dfs":      ("🌿", "DFS",                "Deep Explore"),
    "greedy":   ("🎯", "Greedy Best-First",  "Heuristic Fast"),
    "ucs":      ("⚖️",  "UCS",               "Uniform Cost"),
}


def print_graph_summary(campus: CampusGraph):
    print(color("  📍 CAMPUS LOCATIONS", C.YELLOW, C.BOLD))
    print(color("  " + "─" * 54, C.YELLOW))
    for idx, info in campus.LOCATIONS.items():
        tc = TYPE_COLORS.get(info["type"], C.WHITE)
        acc = color("♿", C.GREEN) if info["accessible"] else color("✗", C.RED)
        node_label = color(f"[{idx:02d}]", C.CYAN)
        name_label = color(f"{info['name']:<20}", tc)
        type_label = color(f"{info['type']:<12}", C.WHITE)
        print(f"  {node_label} {name_label}  {type_label}  {acc}")
    print()


def print_result(campus: CampusGraph, agent: NavigationAgent,
                 path: list[int] | None, algo: str,
                 start: int, goal: int, time_limit: int):

    icon, alabel, adesc = ALGO_LABELS[algo]
    print(color(f"\n  {icon} Algorithm  : ", C.MAGENTA) + color(f"{alabel} ({adesc})", C.WHITE, C.BOLD))

    if path is None or len(path) < 2:
        print(color("  ✗ No path found!", C.RED, C.BOLD))
        return

    cost = campus.path_cost(path)
    hops = len(path) - 1
    est_time = cost * 2
    nodes_exp = agent.stats.get("nodes_expanded", "?")

    # Route string
    route_parts = []
    for i, n in enumerate(path):
        tc = TYPE_COLORS.get(campus.node_type(n), C.WHITE)
        label = color(campus.node_name(n), tc, C.BOLD)
        if i == 0:
            label = "🟢 " + label
        elif i == len(path) - 1:
            label = "🔴 " + label
        route_parts.append(label)

    print(color("  ─" * 28, C.CYAN))
    print(color("  ✓ Route   : ", C.GREEN) + color(" → ", C.CYAN).join(route_parts))
    print(color("  ✓ Cost    : ", C.GREEN) + color(f"{cost} units", C.WHITE, C.BOLD))
    print(color("  ✓ Hops    : ", C.GREEN) + color(str(hops), C.WHITE, C.BOLD))
    print(color("  ✓ Nodes Expanded: ", C.GREEN) + color(str(nodes_exp), C.WHITE, C.BOLD))
    if est_time <= time_limit:
        print(color(f"  ✓ Est. Time: ~{est_time} min ", C.GREEN) + color(f"(within {time_limit} min limit ✓)", C.LIME))
    else:
        print(color(f"  ✗ Est. Time: ~{est_time} min ", C.RED) + color(f"(exceeds {time_limit} min limit)", C.RED))


def benchmark_all(campus: CampusGraph, start: int, goal: int, time_limit: int):
    """Run all algorithms and compare their performance."""
    agent = NavigationAgent(campus)
    algos = ["astar", "dijkstra", "bfs", "dfs", "greedy", "ucs"]

    print(color("\n  📊 ALGORITHM BENCHMARK COMPARISON", C.YELLOW, C.BOLD))
    print(color("  " + "─" * 60, C.YELLOW))
    print(f"  {color('Algorithm', C.CYAN):<33}"
          f"  {color('Cost', C.CYAN):<14}"
          f"  {color('Hops', C.CYAN):<14}"
          f"  {color('Nodes Exp.', C.CYAN):<20}"
          f"  {color('Time (ms)', C.CYAN)}")
    print(color("  " + "─" * 60, C.WHITE))

    results = []
    for algo in algos:
        t0 = time.perf_counter()
        path = agent.find_route(start, goal, algo)
        elapsed = (time.perf_counter() - t0) * 1000

        cost = campus.path_cost(path) if path else float("inf")
        hops = len(path) - 1 if path else 0
        nexp = agent.stats.get("nodes_expanded", 0)
        results.append((algo, cost, hops, nexp, elapsed, path))

    # Sort by cost for display
    results.sort(key=lambda r: r[1])

    for i, (algo, cost, hops, nexp, elapsed, path) in enumerate(results):
        icon, alabel, _ = ALGO_LABELS[algo]
        star = color("★ BEST", C.LIME, C.BOLD) if i == 0 else ""
        row_algo = f"{icon} {color(alabel, C.WHITE)}"
        row_cost = color(str(cost), C.YELLOW)
        row_hops = color(str(hops), C.BLUE)
        row_nexp = color(str(nexp), C.MAGENTA)
        row_ms   = color(f"{elapsed:.3f}ms", C.CYAN)
        print(f"  {row_algo:<35}  {row_cost:<14}  {row_hops:<14}  {row_nexp:<20}  {row_ms:<12}  {star}")

    print()
    print(color("  💡 Insight: A* & Dijkstra/UCS find optimal (lowest cost) paths.", C.PINK))
    print(color("             BFS finds fewest hops, not lowest cost.", C.PINK))
    print(color("             Greedy is fastest but may miss optimal path.", C.PINK))


# ─────────────────────────────────────────────
#  🚀  MAIN DEMO
# ─────────────────────────────────────────────
def main():
    print_header()

    campus = CampusGraph(accessibility_mode="normal")
    agent  = NavigationAgent(campus)

    # ── Show campus layout ───────────────────
    print_graph_summary(campus)

    # ── Demo Route: Main Gate → CS Department ─
    start, goal = 0, 7       # Main Gate → Labs
    time_limit  = 20         # minutes

    print(color("  🗺️  ROUTE PLANNING DEMO", C.CYAN, C.BOLD))
    print(color(f"  From  : {campus.node_name(start)}", C.WHITE))
    print(color(f"  To    : {campus.node_name(goal)}", C.WHITE))
    print(color(f"  Limit : {time_limit} minutes", C.WHITE))

    # Run all algorithms individually
    for algo in ["astar", "dijkstra", "bfs", "dfs", "greedy", "ucs"]:
        t0   = time.perf_counter()
        path = agent.find_route(start, goal, algo)
        ms   = (time.perf_counter() - t0) * 1000
        print_result(campus, agent, path, algo, start, goal, time_limit)
        print(color(f"  ⚡ Execution: {ms:.4f} ms\n", C.YELLOW))

    # Benchmark comparison table
    benchmark_all(campus, start, goal, time_limit)

    # ── Wheelchair mode demo ─────────────────
    print(color("\n  ♿  ACCESSIBILITY MODE: Wheelchair Friendly", C.LIME, C.BOLD))
    print(color("  " + "─" * 50, C.LIME))
    campus_wc = CampusGraph(accessibility_mode="wheelchair")
    agent_wc  = NavigationAgent(campus_wc)
    path_wc   = agent_wc.find_route(3, 9, "astar")   # Cafeteria → Parking
    print_result(campus_wc, agent_wc, path_wc, "astar", 3, 9, 30)

    print(color("\n  ✅ Campus Navigation Agent complete!\n", C.GREEN, C.BOLD))


if __name__ == "__main__":
    main()
