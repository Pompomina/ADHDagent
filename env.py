"""Custom Gymnasium environment: ADHDSchedulingEnv."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    NUM_SLOTS, MIN_TASKS, MAX_TASKS, OBS_DIM, NUM_ACTIONS, REST_ACTION,
    ENERGY_DECAY_BASE, ENERGY_DECAY_PER_DIFF,
    FOCUS_DECAY_BASE, FOCUS_DECAY_PER_DIFF,
    CRASH_PROB, CRASH_FOCUS_PENALTY,
    REST_ENERGY_RECOVERY, REST_FOCUS_RECOVERY,
    REWARD_TASK_COMPLETE_MULT, REWARD_MISSED_DEADLINE,
    REWARD_STREAK_PENALTY, REWARD_STREAK_THRESHOLD,
    REWARD_REST_LOW_ENERGY, LOW_ENERGY_THRESHOLD,
    REWARD_INVALID_ACTION, DEFAULT_SEED,
)
from utils import sigmoid, make_rng
from task_generator import generate_task_set


class ADHDSchedulingEnv(gym.Env):
    """One episode = one simulated day with NUM_SLOTS time slots.

    Observation (dim=34):
        [slot/NUM_SLOTS, energy, focus, streak/NUM_SLOTS,
         (remaining/3, difficulty, priority/3, deadline/(NUM_SLOTS-1), done) × MAX_TASKS]

    Action space: Discrete(MAX_TASKS + 1)
        0 … MAX_TASKS-1 : work on task i for one slot
        MAX_TASKS        : REST
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._init_seed = seed
        self._rng = make_rng(seed)

        # Internal state — populated by reset()
        self.slot: int = 0
        self.energy: float = 1.0
        self.focus: float = 1.0
        self.work_streak: int = 0
        self.tasks: list[dict] = []
        self._deadline_penalized: set[int] = set()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = make_rng(seed)

        self.slot = 0
        self.energy = 1.0
        self.focus = 1.0
        self.work_streak = 0
        self._deadline_penalized = set()

        num_tasks = int(self._rng.integers(MIN_TASKS, MAX_TASKS + 1))
        self.tasks = generate_task_set(num_tasks, NUM_SLOTS, self._rng)

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        action = int(action)

        if action == REST_ACTION:
            reward += self._apply_rest_action()
        elif action < len(self.tasks) and not self.tasks[action]["done"]:
            reward += self._apply_task_action(action)
        else:
            # Invalid: task index out of range or already done
            reward += REWARD_INVALID_ACTION
            reward += self._apply_rest_action()

        reward += self._check_deadlines()
        self.slot += 1

        terminated = self._all_tasks_done() or self.slot >= NUM_SLOTS
        truncated = False
        return self._get_obs(), float(reward), terminated, truncated, {}

    def render(self) -> None:
        print(
            f"Slot {self.slot:02d}/{NUM_SLOTS} | "
            f"E={self.energy:.2f} F={self.focus:.2f} streak={self.work_streak}"
        )
        for t in self.tasks:
            status = "DONE" if t["done"] else f"rem={t['remaining']}"
            missed = "*" if (t["id"] in self._deadline_penalized) else " "
            print(
                f"  [{missed}] Task {t['id']}: p={t['priority']} "
                f"d={t['difficulty']:.2f} dl={t['deadline']} {status}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = self.slot / NUM_SLOTS
        obs[1] = self.energy
        obs[2] = self.focus
        obs[3] = self.work_streak / NUM_SLOTS
        for i in range(MAX_TASKS):
            base = 4 + i * 5
            if i < len(self.tasks):
                t = self.tasks[i]
                obs[base + 0] = t["remaining"] / 3.0
                obs[base + 1] = t["difficulty"]
                obs[base + 2] = t["priority"] / 3.0
                obs[base + 3] = t["deadline"] / (NUM_SLOTS - 1)
                obs[base + 4] = float(t["done"])
            # Padding slots remain 0.0
        return obs

    def _apply_task_action(self, task_idx: int) -> float:
        reward = 0.0
        t = self.tasks[task_idx]
        d = t["difficulty"]

        # Decay energy and focus
        self.energy = float(np.clip(
            self.energy - ENERGY_DECAY_BASE - ENERGY_DECAY_PER_DIFF * d, 0.0, 1.0
        ))
        self.focus = float(np.clip(
            self.focus - FOCUS_DECAY_BASE - FOCUS_DECAY_PER_DIFF * d, 0.0, 1.0
        ))

        # Random focus crash
        if self._rng.random() < CRASH_PROB:
            self.focus = float(np.clip(self.focus - CRASH_FOCUS_PENALTY, 0.0, 1.0))

        # Work streak and overwork penalty
        self.work_streak += 1
        if self.work_streak > REWARD_STREAK_THRESHOLD:
            reward += REWARD_STREAK_PENALTY

        # Stochastic progress
        p_success = sigmoid(3.0 * (self.energy + self.focus - d - 0.8))
        if self._rng.random() < p_success:
            t["remaining"] -= 1
            if t["remaining"] <= 0:
                t["done"] = True
                reward += REWARD_TASK_COMPLETE_MULT * t["priority"]

        return reward

    def _apply_rest_action(self) -> float:
        # Check low-energy condition BEFORE recovery
        rest_reward = REWARD_REST_LOW_ENERGY if self.energy < LOW_ENERGY_THRESHOLD else 0.0

        self.energy = float(np.clip(self.energy + REST_ENERGY_RECOVERY, 0.0, 1.0))
        self.focus = float(np.clip(self.focus + REST_FOCUS_RECOVERY, 0.0, 1.0))
        self.work_streak = 0

        return rest_reward

    def _check_deadlines(self) -> float:
        """Apply missed-deadline penalty once per task when deadline slot passes."""
        reward = 0.0
        for t in self.tasks:
            if not t["done"] and t["id"] not in self._deadline_penalized:
                # self.slot is the slot that just finished (action was taken)
                if self.slot >= t["deadline"]:
                    reward += REWARD_MISSED_DEADLINE
                    self._deadline_penalized.add(t["id"])
        return reward

    def _all_tasks_done(self) -> bool:
        return all(t["done"] for t in self.tasks)
