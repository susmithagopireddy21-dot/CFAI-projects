# ============================================================
# Intelligent Campus Navigation Assistant
# Colab Console Code
# Covers CO1, CO2, CO3, CO4, CO5, CO6
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Set, Any
from collections import deque, defaultdict
import heapq
import time
import tracemalloc
import math
import random

# ============================================================
# CO1: Agent Model, PEAS, Environment, Problem Formulation
# ============================================================

print("\n================ CO1: AGENT MODEL =================")

PEAS = {
    "Performance": [
        "Shortest route",
        "Minimum travel time",
        "Accessible path if required",
        "Avoid crowded/blocked paths",
        "Explainable route"
    ],
    "Environment": [
        "University campus",
        "Buildings",
        "Walkways",
        "Stairs",
        "Ramps",
        "Crowded paths",
        "Blocked paths"
    ],
    "Actuators": [
        "Display route",
        "Recommend next step",
        "Warn about constraints"
    ],
    "Sensors": [
        "Campus map",
        "User location",
        "Crowd level",
        "Accessibility requirement",
        "Time constraint"
    ]
}

for k, v in PEAS.items():
    print(f"{k}: {v}")

ENVIRONMENT_TYPES = {
    "Observable": "Partially observable because crowd/block status may be uncertain",
    "Deterministic": "Mostly deterministic, but travel time can vary",
    "Sequential": "Each movement affects next state",
    "Dynamic": "Crowds and blockages can change",
    "Discrete": "Locations and paths represented as graph nodes and edges",
    "Multi-agent": "Other students also affect congestion"
}

print("\nEnvironment Types:")
for k, v in ENVIRONMENT_TYPES.items():
    print(f"{k}: {v}")


# ============================================================
# Knowledge Representation: Graphs, Rules, Constraints
# ============================================================

@dataclass(frozen=True)
class CampusState:
    location: str
    time_elapsed: float = 0.0


@dataclass
class Edge:
    target: str
    distance: float
    time: float
    accessible: bool = True
    crowded: bool = False
    blocked: bool = False


@dataclass
class SearchResult:
    path: List[str]
    cost: float
    expanded: int
    runtime: float
    peak_memory_kb: float
    trace: List[str] = field(default_factory=list)


class CampusGraph:
    def __init__(self):
        self.graph: Dict[str, List[Edge]] = defaultdict(list)
        self.coords: Dict[str, Tuple[float, float]] = {}

    def add_location(self, name: str, x: float, y: float):
        self.coords[name] = (x, y)

    def add_path(
        self,
        a: str,
        b: str,
        distance: float,
        time: float,
        accessible: bool = True,
        crowded: bool = False,
        blocked: bool = False
    ):
        self.graph[a].append(Edge(b, distance, time, accessible, crowded, blocked))
        self.graph[b].append(Edge(a, distance, time, accessible, crowded, blocked))

    def neighbors(self, node: str) -> List[Edge]:
        return self.graph[node]

    def heuristic_distance(self, a: str, b: str) -> float:
        ax, ay = self.coords[a]
        bx, by = self.coords[b]
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def build_sample_campus() -> CampusGraph:
    campus = CampusGraph()

    locations = {
        "Gate": (0, 0),
        "Library": (2, 2),
        "Cafeteria": (4, 1),
        "Admin": (1, 5),
        "Lab": (5, 5),
        "Auditorium": (7, 3),
        "Hostel": (8, 0),
        "Sports": (6, -2),
        "Medical": (3, -2)
    }

    for name, (x, y) in locations.items():
        campus.add_location(name, x, y)

    campus.add_path("Gate", "Library", 3, 5, accessible=True)
    campus.add_path("Gate", "Medical", 4, 6, accessible=True)
    campus.add_path("Library", "Admin", 4, 7, accessible=True)
    campus.add_path("Library", "Cafeteria", 3, 5, accessible=True, crowded=True)
    campus.add_path("Cafeteria", "Lab", 5, 8, accessible=False)
    campus.add_path("Admin", "Lab", 4, 6, accessible=True)
    campus.add_path("Lab", "Auditorium", 3, 4, accessible=True)
    campus.add_path("Auditorium", "Hostel", 4, 6, accessible=True)
    campus.add_path("Medical", "Sports", 4, 5, accessible=True)
    campus.add_path("Sports", "Hostel", 3, 4, accessible=True)
    campus.add_path("Cafeteria", "Auditorium", 4, 7, accessible=True)
    campus.add_path("Library", "Medical", 3, 4, accessible=True)

    return campus


campus = build_sample_campus()

RULES = [
    "If accessibility_required=True, avoid inaccessible edges",
    "If edge is blocked, do not use it",
    "If user is late, optimize time more than distance",
    "If path is crowded, add congestion penalty",
    "If medical emergency, prefer route through Medical if reasonable"
]

print("\nKnowledge Representation Rules:")
for rule in RULES:
    print("-", rule)


# ============================================================
# Cost Model
# ============================================================

def route_cost(
    edge: Edge,
    optimize: str = "distance",
    accessibility_required: bool = False,
    avoid_crowds: bool = True
) -> float:
    if edge.blocked:
        return float("inf")

    if accessibility_required and not edge.accessible:
        return float("inf")

    if optimize == "distance":
        base = edge.distance
    elif optimize == "time":
        base = edge.time
    else:
        base = 0.5 * edge.distance + 0.5 * edge.time

    if avoid_crowds and edge.crowded:
        base += 4

    return base


# ============================================================
# Profiling Decorator
# ============================================================

def profile_search(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.runtime = runtime
        result.peak_memory_kb = peak / 1024
        return result
    return wrapper


def reconstruct_path(parent: Dict[str, Optional[str]], goal: str) -> List[str]:
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ============================================================
# CO2: BFS, DFS, UCS, Greedy, A*, IDA* Concept
# ============================================================

@profile_search
def bfs(campus: CampusGraph, start: str, goal: str) -> SearchResult:
    queue = deque([start])
    parent = {start: None}
    expanded = 0
    trace = []

    while queue:
        node = queue.popleft()
        expanded += 1
        trace.append(f"BFS expands {node}")

        if node == goal:
            return SearchResult(reconstruct_path(parent, goal), len(reconstruct_path(parent, goal)) - 1, expanded, 0, 0, trace)

        for edge in campus.neighbors(node):
            if edge.target not in parent and not edge.blocked:
                parent[edge.target] = node
                queue.append(edge.target)

    return SearchResult([], float("inf"), expanded, 0, 0, trace)


@profile_search
def dfs(campus: CampusGraph, start: str, goal: str) -> SearchResult:
    stack = [start]
    parent = {start: None}
    expanded = 0
    trace = []

    while stack:
        node = stack.pop()
        expanded += 1
        trace.append(f"DFS expands {node}")

        if node == goal:
            return SearchResult(reconstruct_path(parent, goal), len(reconstruct_path(parent, goal)) - 1, expanded, 0, 0, trace)

        for edge in campus.neighbors(node):
            if edge.target not in parent and not edge.blocked:
                parent[edge.target] = node
                stack.append(edge.target)

    return SearchResult([], float("inf"), expanded, 0, 0, trace)


@profile_search
def ucs(
    campus: CampusGraph,
    start: str,
    goal: str,
    optimize: str = "distance",
    accessibility_required: bool = False
) -> SearchResult:
    pq = [(0, 0, start)]
    parent = {start: None}
    best_cost = {start: 0}
    expanded = 0
    counter = 0
    trace = []

    while pq:
        cost, _, node = heapq.heappop(pq)

        if cost > best_cost[node]:
            continue

        expanded += 1
        trace.append(f"UCS expands {node} with cost {cost:.2f}")

        if node == goal:
            return SearchResult(reconstruct_path(parent, goal), cost, expanded, 0, 0, trace)

        for edge in campus.neighbors(node):
            step = route_cost(edge, optimize, accessibility_required)
            new_cost = cost + step

            if new_cost < best_cost.get(edge.target, float("inf")):
                best_cost[edge.target] = new_cost
                parent[edge.target] = node
                counter += 1
                heapq.heappush(pq, (new_cost, counter, edge.target))

    return SearchResult([], float("inf"), expanded, 0, 0, trace)


@profile_search
def greedy_best_first(
    campus: CampusGraph,
    start: str,
    goal: str,
    accessibility_required: bool = False
) -> SearchResult:
    pq = [(campus.heuristic_distance(start, goal), 0, start)]
    parent = {start: None}
    visited = set()
    expanded = 0
    counter = 0
    trace = []

    while pq:
        h, _, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        expanded += 1
        trace.append(f"Greedy expands {node}, h={h:.2f}")

        if node == goal:
            path = reconstruct_path(parent, goal)
            return SearchResult(path, len(path) - 1, expanded, 0, 0, trace)

        for edge in campus.neighbors(node):
            if edge.target not in visited:
                if route_cost(edge, "distance", accessibility_required) < float("inf"):
                    parent[edge.target] = node
                    counter += 1
                    heapq.heappush(pq, (campus.heuristic_distance(edge.target, goal), counter, edge.target))

    return SearchResult([], float("inf"), expanded, 0, 0, trace)


@profile_search
def astar(
    campus: CampusGraph,
    start: str,
    goal: str,
    optimize: str = "distance",
    accessibility_required: bool = False
) -> SearchResult:
    pq = [(campus.heuristic_distance(start, goal), 0, 0, start)]
    parent = {start: None}
    g_score = {start: 0}
    expanded = 0
    counter = 0
    trace = []

    while pq:
        f, _, g, node = heapq.heappop(pq)

        if g > g_score[node]:
            continue

        expanded += 1
        trace.append(f"A* expands {node}: g={g:.2f}, h={campus.heuristic_distance(node, goal):.2f}, f={f:.2f}")

        if node == goal:
            return SearchResult(reconstruct_path(parent, goal), g, expanded, 0, 0, trace)

        for edge in campus.neighbors(node):
            step = route_cost(edge, optimize, accessibility_required)
            tentative_g = g + step

            if tentative_g < g_score.get(edge.target, float("inf")):
                parent[edge.target] = node
                g_score[edge.target] = tentative_g
                h = campus.heuristic_distance(edge.target, goal)
                counter += 1
                heapq.heappush(pq, (tentative_g + h, counter, tentative_g, edge.target))

    return SearchResult([], float("inf"), expanded, 0, 0, trace)


def ida_star_concept():
    print("\nIDA* Concept:")
    print("IDA* performs A* search using depth-first memory behavior.")
    print("It repeatedly increases an f-cost threshold.")
    print("Useful when memory is limited, but may re-expand nodes.")


def print_result(name: str, result: SearchResult, show_trace: bool = True):
    print(f"\n{name}")
    print("Path:", " -> ".join(result.path) if result.path else "No path")
    print("Cost:", round(result.cost, 2))
    print("Expanded nodes:", result.expanded)
    print("Runtime:", round(result.runtime, 6), "seconds")
    print("Peak memory:", round(result.peak_memory_kb, 2), "KB")
    if show_trace:
        print("Trace:")
        for line in result.trace[:8]:
            print(" ", line)
        if len(result.trace) > 8:
            print("  ...")


print("\n================ CO2: SEARCH ALGORITHMS =================")

start, goal = "Gate", "Hostel"

results = {
    "BFS": bfs(campus, start, goal),
    "DFS": dfs(campus, start, goal),
    "UCS Distance": ucs(campus, start, goal, optimize="distance"),
    "UCS Time": ucs(campus, start, goal, optimize="time"),
    "Greedy": greedy_best_first(campus, start, goal),
    "A* Distance": astar(campus, start, goal, optimize="distance"),
    "A* Accessible": astar(campus, start, goal, optimize="distance", accessibility_required=True)
}

for name, result in results.items():
    print_result(name, result)

ida_star_concept()

print("\nHeuristic Evaluation:")
print("The straight-line distance heuristic is usually admissible if actual walking distance is never shorter than straight-line distance.")
print("It is consistent if h(n) <= cost(n, n') + h(n') for every edge.")


# ============================================================
# Testing Small Algorithm Units
# ============================================================

print("\n================ UNIT TESTS =================")

def test_heuristic_zero():
    assert campus.heuristic_distance("Gate", "Gate") == 0

def test_bfs_finds_path():
    assert bfs(campus, "Gate", "Hostel").path[0] == "Gate"

def test_ucs_valid_goal():
    assert ucs(campus, "Gate", "Hostel").path[-1] == "Hostel"

def test_accessibility_cost():
    inaccessible_edge = Edge("X", 1, 1, accessible=False)
    assert route_cost(inaccessible_edge, accessibility_required=True) == float("inf")

tests = [
    test_heuristic_zero,
    test_bfs_finds_path,
    test_ucs_valid_goal,
    test_accessibility_cost
]

for t in tests:
    t()
    print(t.__name__, "passed")


# ============================================================
# CO3: CSP Modeling, Backtracking, MRV, Degree, LCV, Forward Checking
# Scheduling / Timetabling Example
# ============================================================

print("\n================ CO3: CSP SCHEDULING =================")

courses = ["AI", "DBMS", "Networks", "Math"]
rooms = ["R1", "R2"]
slots = ["9AM", "10AM", "11AM"]

variables = courses
domains = {course: [(room, slot) for room in rooms for slot in slots] for course in courses}

teachers = {
    "AI": "T1",
    "DBMS": "T2",
    "Networks": "T1",
    "Math": "T3"
}

def constraint_ok(course1, value1, course2, value2) -> Tuple[bool, str]:
    room1, slot1 = value1
    room2, slot2 = value2

    if slot1 == slot2 and room1 == room2:
        return False, f"{course1} and {course2} conflict: same room {room1} at {slot1}"

    if slot1 == slot2 and teachers[course1] == teachers[course2]:
        return False, f"{course1} and {course2} conflict: teacher {teachers[course1]} at {slot1}"

    return True, "OK"


def is_consistent(var, value, assignment):
    for other, other_value in assignment.items():
        ok, reason = constraint_ok(var, value, other, other_value)
        if not ok:
            return False, reason
    return True, "OK"


def mrv_select_unassigned(assignment, domains):
    unassigned = [v for v in variables if v not in assignment]
    return min(unassigned, key=lambda var: len(domains[var]))


def lcv_order_values(var, domains, assignment):
    def conflicts_count(value):
        count = 0
        for other in variables:
            if other != var and other not in assignment:
                for other_value in domains[other]:
                    ok, _ = constraint_ok(var, value, other, other_value)
                    if not ok:
                        count += 1
        return count

    return sorted(domains[var], key=conflicts_count)


def forward_check(var, value, domains, assignment):
    new_domains = {v: list(vals) for v, vals in domains.items()}

    for other in variables:
        if other != var and other not in assignment:
            filtered = []
            for other_value in new_domains[other]:
                ok, _ = constraint_ok(var, value, other, other_value)
                if ok:
                    filtered.append(other_value)
            new_domains[other] = filtered

            if not filtered:
                return None

    return new_domains


def backtracking_csp(assignment, domains, trace):
    if len(assignment) == len(variables):
        return assignment

    var = mrv_select_unassigned(assignment, domains)
    trace.append(f"Select {var} using MRV")

    for value in lcv_order_values(var, domains, assignment):
        ok, reason = is_consistent(var, value, assignment)

        if ok:
            trace.append(f"Try {var}={value}")
            assignment[var] = value
            new_domains = forward_check(var, value, domains, assignment)

            if new_domains is not None:
                result = backtracking_csp(assignment, new_domains, trace)
                if result:
                    return result

            trace.append(f"Backtrack from {var}={value}")
            del assignment[var]
        else:
            trace.append(f"Reject {var}={value}: {reason}")

    return None


csp_trace = []
schedule = backtracking_csp({}, domains, csp_trace)

print("Schedule Solution:")
print(schedule)

print("\nCSP Explanation Trace:")
for line in csp_trace[:15]:
    print(" ", line)

print("\nArc Consistency Intuition:")
print("Arc consistency removes domain values that have no compatible value in a neighboring variable.")

print("\nSAT Intuition:")
print("A timetable can be encoded as Boolean variables such as AI_R1_9AM = True/False.")


# Local Search for CSP: Min-Conflicts

def count_conflicts(assignment):
    conflicts = 0
    conflict_reasons = []

    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            c1, c2 = courses[i], courses[j]
            ok, reason = constraint_ok(c1, assignment[c1], c2, assignment[c2])
            if not ok:
                conflicts += 1
                conflict_reasons.append(reason)

    return conflicts, conflict_reasons


def min_conflicts(max_steps=100):
    assignment = {c: random.choice(domains[c]) for c in courses}

    for step in range(max_steps):
        conflicts, reasons = count_conflicts(assignment)
        if conflicts == 0:
            return assignment, step, []

        conflicted_vars = []
        for c in courses:
            temp = assignment.copy()
            local_conflicts = 0
            for other in courses:
                if other != c:
                    ok, _ = constraint_ok(c, assignment[c], other, assignment[other])
                    if not ok:
                        local_conflicts += 1
            if local_conflicts > 0:
                conflicted_vars.append(c)

        var = random.choice(conflicted_vars)

        best_value = min(
            domains[var],
            key=lambda val: count_conflicts({**assignment, var: val})[0]
        )
        assignment[var] = best_value

    return assignment, max_steps, count_conflicts(assignment)[1]


mc_solution, mc_steps, mc_failures = min_conflicts()
print("\nMin-Conflicts Solution:")
print(mc_solution)
print("Steps:", mc_steps)
print("Failures:", mc_failures)


# ============================================================
# CO4: Utility, Minimax, Alpha-Beta, Evaluation, Decision Logic
# ============================================================

print("\n================ CO4: DECISION MAKING =================")

def route_utility(
    path_cost: float,
    expanded: int,
    accessible: bool,
    crowd_risk: float
) -> float:
    utility = 100
    utility -= path_cost * 5
    utility -= expanded * 0.5
    utility -= crowd_risk * 20

    if accessible:
        utility += 10

    return utility


candidate_routes = [
    ("A* Distance", results["A* Distance"]),
    ("UCS Time", results["UCS Time"]),
    ("A* Accessible", results["A* Accessible"])
]

print("Route Utility Scores:")
best_policy = None
best_score = -float("inf")

for name, result in candidate_routes:
    accessible = name == "A* Accessible"
    crowd_risk = 0.2 if "Library" in result.path else 0.4
    score = route_utility(result.cost, result.expanded, accessible, crowd_risk)
    print(name, "utility =", round(score, 2))

    if score > best_score:
        best_score = score
        best_policy = name

print("Selected Policy:", best_policy)


# Multi-agent reasoning example:
# Student wants shortest route, crowd wants popular cafeteria route.

game_tree = {
    "Start": ["Route_Library", "Route_Medical"],
    "Route_Library": [70, 40],
    "Route_Medical": [60, 55]
}

def minimax(node, maximizing=True):
    if isinstance(node, int):
        return node

    children = game_tree[node]

    if maximizing:
        return max(minimax(child, False) for child in children)
    else:
        return min(minimax(child, True) for child in children)


def alpha_beta(node, alpha=-float("inf"), beta=float("inf"), maximizing=True):
    if isinstance(node, int):
        return node

    children = game_tree[node]

    if maximizing:
        value = -float("inf")
        for child in children:
            value = max(value, alpha_beta(child, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for child in children:
            value = min(value, alpha_beta(child, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


print("\nMinimax Value:", minimax("Start"))
print("Alpha-Beta Value:", alpha_beta("Start"))
print("Iterative Deepening Concept: run depth-limited search repeatedly with increasing depth.")
print("Expectimax Concept: replace opponent min nodes with expected-value chance nodes.")
print("Bounded Rationality: choose a good enough route under time and memory limits.")


# ============================================================
# CO5: Probability, Bayes Rule, Bayesian Network, Inference
# ============================================================

print("\n================ CO5: PROBABILISTIC REASONING =================")

def bayes_rule(prior_blocked, p_sensor_given_blocked, p_sensor_given_not_blocked):
    numerator = p_sensor_given_blocked * prior_blocked
    denominator = numerator + p_sensor_given_not_blocked * (1 - prior_blocked)
    return numerator / denominator


prior_blocked = 0.20
p_alert_given_blocked = 0.90
p_alert_given_not_blocked = 0.15

posterior = bayes_rule(prior_blocked, p_alert_given_blocked, p_alert_given_not_blocked)

print("Bayes Rule Example:")
print("P(Blocked | Sensor Alert) =", round(posterior, 3))


# Bayesian network style CPT:
# Rain -> Crowd
# Event -> Crowd
# Crowd -> TravelDelay

bayes_net = {
    "Rain": {"P(True)": 0.3},
    "Event": {"P(True)": 0.4},
    "Crowd": {
        (True, True): 0.9,
        (True, False): 0.7,
        (False, True): 0.8,
        (False, False): 0.2
    },
    "Delay": {
        True: 0.85,
        False: 0.25
    }
}

def variable_elimination_delay():
    total = 0

    for rain in [True, False]:
        for event in [True, False]:
            p_rain = bayes_net["Rain"]["P(True)"] if rain else 1 - bayes_net["Rain"]["P(True)"]
            p_event = bayes_net["Event"]["P(True)"] if event else 1 - bayes_net["Event"]["P(True)"]

            p_crowd_true = bayes_net["Crowd"][(rain, event)]
            p_crowd_false = 1 - p_crowd_true

            p_delay = (
                bayes_net["Delay"][True] * p_crowd_true +
                bayes_net["Delay"][False] * p_crowd_false
            )

            total += p_rain * p_event * p_delay

    return total


print("\nVariable Elimination Worked Example:")
print("P(Delay=True) =", round(variable_elimination_delay(), 3))

print("\nBelief Propagation Intuition:")
print("Nodes pass probability messages to update beliefs about crowd and delay.")

print("\nSampling Inference Intuition:")
print("Rejection sampling keeps samples matching evidence.")
print("Likelihood weighting weights samples by evidence likelihood.")


# Markov Chain / HMM Tracking

locations = ["Gate", "Library", "Cafeteria"]
transition = {
    "Gate": {"Gate": 0.1, "Library": 0.8, "Cafeteria": 0.1},
    "Library": {"Gate": 0.2, "Library": 0.2, "Cafeteria": 0.6},
    "Cafeteria": {"Gate": 0.1, "Library": 0.3, "Cafeteria": 0.6}
}

sensor_model = {
    "Gate": {"near_gate": 0.8, "near_library": 0.1, "near_cafe": 0.1},
    "Library": {"near_gate": 0.1, "near_library": 0.8, "near_cafe": 0.1},
    "Cafeteria": {"near_gate": 0.1, "near_library": 0.2, "near_cafe": 0.7}
}

def hmm_update(belief, observation):
    predicted = {loc: 0 for loc in locations}

    for old in locations:
        for new in locations:
            predicted[new] += belief[old] * transition[old][new]

    updated = {
        loc: predicted[loc] * sensor_model[loc][observation]
        for loc in locations
    }

    norm = sum(updated.values())

    return {loc: updated[loc] / norm for loc in locations}


belief = {"Gate": 1 / 3, "Library": 1 / 3, "Cafeteria": 1 / 3}
observations = ["near_gate", "near_library", "near_cafe"]

print("\nHMM Tracking:")
for obs in observations:
    belief = hmm_update(belief, obs)
    print("After observation", obs, "belief =", {k: round(v, 3) for k, v in belief.items()})


def expected_utility(success_prob, utility_success, utility_failure):
    return success_prob * utility_success + (1 - success_prob) * utility_failure


print("\nUncertainty-Aware Decision:")
safe_route_eu = expected_utility(0.9, 80, 20)
fast_route_eu = expected_utility(0.6, 100, 10)
print("Safe route expected utility:", safe_route_eu)
print("Fast route expected utility:", fast_route_eu)
print("Choose:", "Safe route" if safe_route_eu > fast_route_eu else "Fast route")


# ============================================================
# CO6: Hybrid Architecture
# Search + CSP + Probabilistic + Decision Logic
# ============================================================

print("\n================ CO6: HYBRID CAMPUS AI AGENT =================")

@dataclass
class UserRequest:
    start: str
    goal: str
    optimize: str = "distance"
    accessibility_required: bool = False
    deadline_minutes: Optional[float] = None


class IntelligentCampusAgent:
    def __init__(self, campus: CampusGraph):
        self.campus = campus

    def explain_limitations(self):
        return [
            "Heuristic bias: straight-line distance may ignore stairs or construction.",
            "Uncertainty miscalibration: crowd estimates may be wrong.",
            "Ethical issue: accessibility preferences must be respected.",
            "Privacy issue: location tracking should use consent."
        ]

    def decide(self, request: UserRequest):
        trace = []

        trace.append("Step 1: Apply rules and constraints.")
        if request.accessibility_required:
            trace.append("Accessibility is required, inaccessible edges are forbidden.")

        trace.append("Step 2: Run A* search.")
        route = astar(
            self.campus,
            request.start,
            request.goal,
            optimize=request.optimize,
            accessibility_required=request.accessibility_required
        )

        trace.append("Step 3: Estimate uncertainty using Bayesian delay model.")
        delay_probability = variable_elimination_delay()
        trace.append(f"Estimated probability of delay: {delay_probability:.3f}")

        trace.append("Step 4: Compute expected utility.")
        success_prob = max(0.1, 1 - delay_probability)
        eu = expected_utility(success_prob, 90 - route.cost * 3, 20)

        trace.append(f"Expected utility of selected route: {eu:.2f}")

        if request.deadline_minutes is not None and route.cost > request.deadline_minutes:
            trace.append("Warning: route may exceed deadline.")

        trace.append("Step 5: Return explainable recommendation.")

        return {
            "path": route.path,
            "cost": route.cost,
            "expanded": route.expanded,
            "expected_utility": eu,
            "trace": trace,
            "limitations": self.explain_limitations()
        }


agent = IntelligentCampusAgent(campus)

request = UserRequest(
    start="Gate",
    goal="Hostel",
    optimize="time",
    accessibility_required=True,
    deadline_minutes=15
)

recommendation = agent.decide(request)

print("Recommended Path:")
print(" -> ".join(recommendation["path"]))

print("Cost:", round(recommendation["cost"], 2))
print("Expanded:", recommendation["expanded"])
print("Expected Utility:", round(recommendation["expected_utility"], 2))

print("\nExplainable Reasoning Trace:")
for line in recommendation["trace"]:
    print("-", line)

print("\nEthics and Limitations:")
for item in recommendation["limitations"]:
    print("-", item)




