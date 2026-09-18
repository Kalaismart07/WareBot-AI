# optimization.py
# WareBot AI - Warehouse Route & Task Optimization

import heapq
import math


# ============================================================
# 1. WAREHOUSE GRID
# ============================================================

WAREHOUSE_ROWS = 10
WAREHOUSE_COLS = 10

# 0 = free path
# 1 = obstacle
WAREHOUSE_GRID = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


# ============================================================
# 2. WAREHOUSE STATIONS
# ============================================================

STATIONS = {
    "P1": (0, 9),
    "P2": (5, 0),
    "P3": (9, 9),
    "PACK": (9, 5),
}


# ============================================================
# 3. NEIGHBOURS
# ============================================================

def get_neighbors(position):

    row, col = position

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    neighbors = []

    for dr, dc in directions:

        new_row = row + dr
        new_col = col + dc

        if (
            0 <= new_row < WAREHOUSE_ROWS
            and 0 <= new_col < WAREHOUSE_COLS
            and WAREHOUSE_GRID[new_row][new_col] == 0
        ):
            neighbors.append((new_row, new_col))

    return neighbors


# ============================================================
# 4. DIJKSTRA SHORTEST PATH
# ============================================================

def shortest_path(start, goal):

    queue = [(0, start)]

    distances = {start: 0}

    previous = {}

    while queue:

        current_distance, current = heapq.heappop(queue)

        if current == goal:
            break

        if current_distance > distances.get(current, math.inf):
            continue

        for neighbor in get_neighbors(current):

            new_distance = current_distance + 1

            if new_distance < distances.get(neighbor, math.inf):

                distances[neighbor] = new_distance

                previous[neighbor] = current

                heapq.heappush(
                    queue,
                    (new_distance, neighbor)
                )

    if goal not in distances:
        return []

    path = []

    current = goal

    while current != start:

        path.append(current)

        current = previous[current]

    path.append(start)

    path.reverse()

    return path


# ============================================================
# 5. ROBOT TASK ALLOCATION
# ============================================================

def allocate_task(robot_data, station):

    """
    Select the best robot for a warehouse task.

    Higher score = better candidate.
    """

    station_position = STATIONS[station]

    best_robot = None
    best_score = -math.inf

    for robot in robot_data:

        robot_position = robot["position"]

        path = shortest_path(
            robot_position,
            station_position
        )

        if not path:
            continue

        distance = len(path) - 1

        battery = robot.get("battery", 50)

        maintenance_risk = robot.get(
            "maintenance_risk",
            0
        )

        # Scoring logic
        score = (
            (battery * 0.6)
            - (distance * 2)
            - (maintenance_risk * 25)
        )

        if score > best_score:

            best_score = score

            best_robot = {
                "robot_id": robot["robot_id"],
                "station": station,
                "distance": distance,
                "score": round(score, 2),
                "path": path,
            }

    return best_robot


# ============================================================
# 6. DEMO
# ============================================================

if __name__ == "__main__":

    robots = [
        {
            "robot_id": "R01",
            "position": (0, 0),
            "battery": 85,
            "maintenance_risk": 0,
        },
        {
            "robot_id": "R02",
            "position": (4, 2),
            "battery": 72,
            "maintenance_risk": 0,
        },
        {
            "robot_id": "R03",
            "position": (8, 4),
            "battery": 55,
            "maintenance_risk": 1,
        },
        {
            "robot_id": "R04",
            "position": (9, 0),
            "battery": 91,
            "maintenance_risk": 0,
        },
    ]

    selected_robot = allocate_task(
        robots,
        "P3"
    )

    print("\n===================================")
    print("     WAREBOT ROUTE OPTIMIZER")
    print("===================================")

    if selected_robot:

        print(
            f"\nSelected Robot : "
            f"{selected_robot['robot_id']}"
        )

        print(
            f"Target Station : "
            f"{selected_robot['station']}"
        )

        print(
            f"Distance       : "
            f"{selected_robot['distance']} steps"
        )

        print(
            f"Optimization Score : "
            f"{selected_robot['score']}"
        )

        print("\nOptimized Path:")

        print(selected_robot["path"])

    else:

        print("\nNo valid route found.")