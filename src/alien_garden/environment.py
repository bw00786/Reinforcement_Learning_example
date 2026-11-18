"""Grid-based cooperative alien garden environment."""

from dataclasses import dataclass

import numpy as np


EMPTY = 0
SEEDLING = 1
MATURE = 2

WAIT = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
PLANT = 5
HARVEST = 6
ACTION_NAMES = ("wait", "up", "down", "left", "right", "plant", "harvest")


@dataclass
class StepResult:
    observations: np.ndarray
    cooperative_reward: float
    agent_rewards: list[float]
    done: bool
    info: dict


class AlienGardenEnv:
    """A deterministic, cooperative grid-world garden."""

    def __init__(self, grid_size: int = 8, num_agents: int = 3, max_steps: int = 100, seed: int = 42):
        if grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.plant_age = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.positions = np.zeros((num_agents, 2), dtype=np.int32)
        self.steps = 0
        self.harvested = 0
        self.planted = 0

    def reset(self) -> np.ndarray:
        self.rng = np.random.default_rng(self.seed)
        self.grid.fill(EMPTY)
        self.plant_age.fill(0)
        self.positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        for agent in range(self.num_agents):
            self.positions[agent] = np.array(
                [agent % self.grid_size, (agent * 2) % self.grid_size], dtype=np.int32
            )

        initial_cells = self.rng.choice(
            self.grid_size * self.grid_size,
            size=min(4 * self.num_agents, self.grid_size * self.grid_size),
            replace=False,
        )
        for index in initial_cells:
            row, col = divmod(int(index), self.grid_size)
            self.grid[row, col] = MATURE
            self.plant_age[row, col] = 3

        self.steps = 0
        self.harvested = 0
        self.planted = 0
        return self._observations()

    def _observations(self) -> np.ndarray:
        observations = []
        radius = 1
        for agent, (row, col) in enumerate(self.positions):
            padded = np.pad(self.grid, radius, mode="constant", constant_values=-1)
            prow, pcol = int(row) + radius, int(col) + radius
            local = padded[prow - radius : prow + radius + 1, pcol - radius : pcol + radius + 1]
            observations.append(
                np.concatenate(
                    (
                        [row / self.grid_size, col / self.grid_size],
                        local.astype(np.float32).ravel(),
                        [agent / max(1, self.num_agents - 1)],
                    )
                )
            )
        return np.asarray(observations, dtype=np.float32)

    def global_state(self) -> np.ndarray:
        """Return the full state for centralized value estimation."""
        return np.concatenate(
            (
            self.grid.astype(np.float32).ravel() / MATURE,
            np.minimum(self.plant_age, 2).astype(np.float32).ravel() / 2,
            self.positions.astype(np.float32).ravel() / self.grid_size,
            np.array([self.steps / self.max_steps], dtype=np.float32),
            )
        )

    def step(self, actions: list[int]) -> StepResult:
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        for plant_row, plant_col in np.ndindex(self.grid.shape):
            if self.grid[plant_row, plant_col] == SEEDLING:
                self.plant_age[plant_row, plant_col] += 1
                if self.plant_age[plant_row, plant_col] >= 2:
                    self.grid[plant_row, plant_col] = MATURE

        personal = [0.0] * self.num_agents
        cooperative = 0.0

        for agent, action in enumerate(actions):
            row, col = self.positions[agent]
            if action in (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT):
                if action == MOVE_UP:
                    row = max(0, row - 1)
                elif action == MOVE_DOWN:
                    row = min(self.grid_size - 1, row + 1)
                elif action == MOVE_LEFT:
                    col = max(0, col - 1)
                else:
                    col = min(self.grid_size - 1, col + 1)
                self.positions[agent] = (row, col)
            elif action == PLANT:
                if self.grid[row, col] == EMPTY:
                    self.grid[row, col] = SEEDLING
                    self.plant_age[row, col] = 0
                    self.planted += 1
                    cooperative += 0.1
                    personal[agent] += 0.1
            elif action == HARVEST:
                if self.grid[row, col] == MATURE:
                    self.grid[row, col] = EMPTY
                    self.plant_age[row, col] = 0
                    self.harvested += 1
                    cooperative += 1.0
                    personal[agent] += 1.0
            elif action != WAIT:
                raise ValueError(f"Invalid action {action}")

        self.steps += 1
        mature_count = int(np.count_nonzero(self.grid == MATURE))
        done = self.steps >= self.max_steps or mature_count == 0 and np.count_nonzero(self.grid) == 0
        progress = (self.harvested + 0.5 * self.planted) / max(1, self.max_steps)
        cooperative += progress

        info = {
            "harvested": self.harvested,
            "planted": self.planted,
            "mature": mature_count,
            "steps": self.steps,
        }
        return StepResult(self._observations(), cooperative, personal, done, info)

    def progress_fraction(self) -> float:
        occupied = int(np.count_nonzero(self.grid))
        capacity = self.grid_size * self.grid_size
        return min(1.0, (self.harvested + occupied * 0.25) / max(1, capacity))
