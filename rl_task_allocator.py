import random
import numpy as np


class WarehouseTaskAllocator:
    """
    Q-Learning based dynamic task allocator for warehouse robots.

    State:
        robot health
        battery level
        maintenance risk
        distance to station

    Action:
        Select a robot for the current warehouse task.
    """

    def __init__(self, robots, stations):
        self.robots = robots
        self.stations = stations

        self.q_table = {}

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2

    def get_state(self, robot):
        battery = float(robot.get("battery", 100))
        health = float(robot.get("health_score", 100))
        maintenance_risk = int(robot.get("maintenance_risk", 0))
        distance = float(robot.get("distance", 10))

        battery_level = min(int(battery // 20), 5)
        health_level = min(int(health // 20), 5)
        distance_level = min(int(distance // 5), 5)

        return (
            battery_level,
            health_level,
            maintenance_risk,
            distance_level
        )

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.robots))

        return self.q_table[state]

    def choose_action(self, state):
        q_values = self.get_q_values(state)

        if random.random() < self.epsilon:
            return random.randrange(len(self.robots))

        return int(np.argmax(q_values))

    def calculate_reward(self, robot):
        battery = float(robot.get("battery", 100))
        health = float(robot.get("health_score", 100))
        maintenance_risk = int(robot.get("maintenance_risk", 0))
        distance = float(robot.get("distance", 10))

        reward = 0

        # Higher battery is better
        reward += battery * 0.25

        # Higher health is better
        reward += health * 0.30

        # Shorter distance is better
        reward += max(0, 50 - distance) * 0.35

        # Avoid robots requiring maintenance
        if maintenance_risk == 1:
            reward -= 40

        return reward

    def train(self, episodes=500):
        for _ in range(episodes):

            for robot_index, robot in enumerate(self.robots):

                state = self.get_state(robot)

                action = self.choose_action(state)

                reward = self.calculate_reward(robot)

                next_state = state

                old_q = self.get_q_values(state)[action]
                next_q = np.max(self.get_q_values(next_state))

                new_q = old_q + self.learning_rate * (
                    reward +
                    self.discount_factor * next_q -
                    old_q
                )

                self.q_table[state][action] = new_q

    def allocate_task(self, robots):
        if not robots:
            return None

        best_robot = None
        best_score = float("-inf")

        for robot in robots:

            state = self.get_state(robot)

            q_values = self.get_q_values(state)

            robot_index = robots.index(robot)

            if robot_index < len(q_values):
                rl_score = q_values[robot_index]
            else:
                rl_score = 0

            # Small operational scoring component
            battery = float(robot.get("battery", 100))
            health = float(robot.get("health_score", 100))
            distance = float(robot.get("distance", 10))
            maintenance_risk = int(
                robot.get("maintenance_risk", 0)
            )

            operational_score = (
                battery * 0.25 +
                health * 0.30 +
                max(0, 50 - distance) * 0.35
            )

            if maintenance_risk == 1:
                operational_score -= 40

            final_score = (
                rl_score * 0.50 +
                operational_score * 0.50
            )

            if final_score > best_score:
                best_score = final_score
                best_robot = robot

        return {
            "robot_id": best_robot.get("robot_id"),
            "rl_score": round(float(best_score), 2),
            "battery": round(
                float(best_robot.get("battery", 0)), 2
            ),
            "health_score": round(
                float(best_robot.get("health_score", 0)), 2
            ),
            "maintenance_risk": int(
                best_robot.get("maintenance_risk", 0)
            ),
            "distance": round(
                float(best_robot.get("distance", 0)), 2
            )
        }


if __name__ == "__main__":

    robots = [
        {
            "robot_id": "R01",
            "battery": 92,
            "health_score": 94,
            "maintenance_risk": 0,
            "distance": 12
        },
        {
            "robot_id": "R02",
            "battery": 65,
            "health_score": 72,
            "maintenance_risk": 0,
            "distance": 8
        },
        {
            "robot_id": "R03",
            "battery": 88,
            "health_score": 90,
            "maintenance_risk": 1,
            "distance": 6
        },
        {
            "robot_id": "R04",
            "battery": 78,
            "health_score": 86,
            "maintenance_risk": 0,
            "distance": 10
        }
    ]

    stations = ["P1", "P2", "P3"]

    allocator = WarehouseTaskAllocator(
        robots,
        stations
    )

    allocator.train(episodes=500)

    result = allocator.allocate_task(robots)

    print("\n======================================")
    print("WAREBOT AI - RL TASK ALLOCATOR")
    print("======================================")

    print("Algorithm       : Q-Learning")
    print("Training        : 500 episodes")
    print("Selected Robot  :", result["robot_id"])
    print("RL Score        :", result["rl_score"])
    print("Battery         :", result["battery"])
    print("Health Score    :", result["health_score"])
    print("Maintenance Risk:", result["maintenance_risk"])
    print("Distance        :", result["distance"])

    print("======================================")