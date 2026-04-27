"""Custom Gymnasium environment: ADHDSchedulingEnv."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    NUM_SLOTS, MIN_TASKS, MAX_TASKS, OBS_DIM, NUM_ACTIONS, REST_ACTION,
    MAX_TASK_DURATION, MAX_PRIORITY,
    INITIAL_ENERGY_RANGE, INITIAL_FOCUS_RANGE,
    ENERGY_DECAY_BASE, ENERGY_DECAY_PER_DIFF,
    FOCUS_DECAY_BASE, FOCUS_DECAY_PER_DIFF,
    CRASH_PROB, CRASH_FOCUS_PENALTY,
    SUCCESS_SIGMOID_SCALE, SUCCESS_DIFFICULTY_OFFSET,
    REST_ENERGY_RECOVERY, REST_FOCUS_RECOVERY,
    REWARD_TASK_COMPLETE_MULT, REWARD_MISSED_DEADLINE,
    REWARD_STREAK_PENALTY, REWARD_STREAK_THRESHOLD,
    REWARD_REST_LOW_ENERGY, LOW_ENERGY_THRESHOLD,
    REWARD_INVALID_ACTION, STEP_PENALTY, DEFAULT_SEED,
)
from utils import make_rng
from task_generator import generate_task_set


class ADHDSchedulingEnv(gym.Env):
    """One episode = one simulated day with NUM_SLOTS time slots.

    Observation (dim=34):
        [slot/NUM_SLOTS, energy, focus, streak/NUM_SLOTS,
         (remaining/3, difficulty, priority/3, deadline/(NUM_SLOTS-1), done) × MAX_TASKS]

    Padding slots use [0, 0, 0, 1, 1] so the agent can distinguish them from
    real undone tasks (done=0).

    Action space: Discrete(MAX_TASKS + 1)
        0 … MAX_TASKS-1 : work on task i for one slot
        MAX_TASKS        : REST
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        tasks: list[dict] | None = None,
    ) -> None:
        """Configure spaces and optional fixed task set.

        Args:
            seed: Random seed for reproducible episode generation.
            tasks: Optional fixed task list reused on every reset.
                   When provided, only energy/focus are randomised per episode.
        """
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._init_seed = seed
        self._rng = make_rng(seed)
        self.rest_action = REST_ACTION      # instance attr for baseline compat
        self.num_slots = NUM_SLOTS          # instance attr for evaluate.py/demo.py

        self.fixed_tasks = deepcopy(tasks) if tasks is not None else None

        # Internal state — populated by reset()
        self.slot: int = 0
        self.energy: float = 1.0
        self.focus: float = 1.0
        self.work_streak: int = 0
        self.missed_deadline_count: int = 0
        self.tasks: list[dict] = []

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
        self.energy = float(self._rng.uniform(*INITIAL_ENERGY_RANGE))
        self.focus  = float(self._rng.uniform(*INITIAL_FOCUS_RANGE))
        self.work_streak = 0
        self.missed_deadline_count = 0

        if self.fixed_tasks is not None:
            self.tasks = deepcopy(self.fixed_tasks)
            for t in self.tasks:
                t["remaining"] = t["duration"]
                t["done"] = False
                t["_deadline_penalized"] = False
        else:
            num_tasks = int(self._rng.integers(MIN_TASKS, MAX_TASKS + 1))
            self.tasks = generate_task_set(num_tasks, NUM_SLOTS, self._rng)
            for t in self.tasks:
                t["_deadline_penalized"] = False

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        reward = STEP_PENALTY   # mild per-step time cost

        info: dict[str, Any] = {
            "invalid_action": False,
            "action_type": "rest" if action == REST_ACTION else "work",
            "task_success": False,
            "completed_task_id": None,
            "missed_deadlines": [],
        }

        if action == REST_ACTION:
            reward += self._apply_rest_action()
        elif self._is_valid_task_action(action):
            reward += self._apply_task_action(action, info)
        else:
            info["invalid_action"] = True
            info["action_type"] = "invalid_as_rest"
            reward += REWARD_INVALID_ACTION
            reward += self._apply_rest_action()

        self.slot += 1
        deadline_penalty, missed_ids = self._check_deadlines()
        reward += deadline_penalty
        info["missed_deadlines"] = missed_ids
        info.update(self._get_info())

        terminated = self._all_tasks_done() or self.slot >= NUM_SLOTS
        truncated = False
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> str | None:
        task_bits = []
        for t in self.tasks:
            status = "done" if t["done"] else f"rem={t['remaining']}"
            missed = "*" if t.get("_deadline_penalized") else " "
            task_bits.append(
                f"[{missed}]T{t['id']}(p={t['priority']} d={t['difficulty']:.2f} "
                f"dl={t['deadline']} {status})"
            )
        text = (
            f"slot={self.slot}/{NUM_SLOTS} E={self.energy:.2f} F={self.focus:.2f} "
            f"streak={self.work_streak} | " + "  ".join(task_bits)
        )
        if mode == "ansi":
            return text
        print(text)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = self.slot / NUM_SLOTS
        obs[1] = self.energy
        obs[2] = self.focus
        obs[3] = min(self.work_streak / NUM_SLOTS, 1.0)
        for i in range(MAX_TASKS):
            base = 4 + i * 5
            if i < len(self.tasks):
                t = self.tasks[i]
                obs[base + 0] = t["remaining"] / MAX_TASK_DURATION
                obs[base + 1] = t["difficulty"]
                obs[base + 2] = t["priority"] / MAX_PRIORITY
                obs[base + 3] = t["deadline"] / max(1, NUM_SLOTS - 1)
                obs[base + 4] = 1.0 if t["done"] else 0.0
            else:
                # Padded slot: mark done=1 so agent knows it's not a real task
                obs[base + 3] = 1.0
                obs[base + 4] = 1.0
        return obs

    def _get_info(self) -> dict:
        completed = sum(1 for t in self.tasks if t["done"])
        completed_value = sum(t["priority"] for t in self.tasks if t["done"])
        return {
            "slot": self.slot,
            "energy": self.energy,
            "focus": self.focus,
            "work_streak": self.work_streak,
            "completed_tasks": completed,
            "completed_value": completed_value,
            "missed_deadline_count": self.missed_deadline_count,
            "tasks": deepcopy(self.tasks),
        }

    def _is_valid_task_action(self, action: int) -> bool:
        return 0 <= action < len(self.tasks) and not self.tasks[action]["done"]

    def _apply_task_action(self, task_idx: int, info: dict) -> float:
        reward = 0.0
        t = self.tasks[task_idx]
        d = t["difficulty"]

        self.work_streak += 1
        self.energy = float(np.clip(
            self.energy - ENERGY_DECAY_BASE - ENERGY_DECAY_PER_DIFF * d, 0.0, 1.0
        ))
        self.focus = float(np.clip(
            self.focus - FOCUS_DECAY_BASE - FOCUS_DECAY_PER_DIFF * d, 0.0, 1.0
        ))

        if self._rng.random() < CRASH_PROB:
            self.focus = float(np.clip(self.focus - CRASH_FOCUS_PENALTY, 0.0, 1.0))
            info["focus_crash"] = True
        else:
            info["focus_crash"] = False

        p_success = self._success_probability(d)
        info["p_success"] = p_success
        info["task_id"] = t["id"]

        if self._rng.random() < p_success:
            t["remaining"] = max(0, t["remaining"] - 1)
            info["task_success"] = True
            if t["remaining"] == 0:
                t["done"] = True
                info["completed_task_id"] = t["id"]
                reward += REWARD_TASK_COMPLETE_MULT * t["priority"]

        if self.work_streak > REWARD_STREAK_THRESHOLD:
            reward += REWARD_STREAK_PENALTY

        return reward

    def _apply_rest_action(self) -> float:
        # Check low-energy condition BEFORE recovery
        rest_reward = REWARD_REST_LOW_ENERGY if self.energy < LOW_ENERGY_THRESHOLD else 0.0
        self.energy = float(np.clip(self.energy + REST_ENERGY_RECOVERY, 0.0, 1.0))
        self.focus  = float(np.clip(self.focus  + REST_FOCUS_RECOVERY,  0.0, 1.0))
        self.work_streak = 0
        return rest_reward

    def _check_deadlines(self) -> tuple[float, list[int]]:
        """Apply one-time penalty per task when its deadline slot passes."""
        penalty = 0.0
        missed: list[int] = []
        for t in self.tasks:
            if (
                not t["done"]
                and not t.get("_deadline_penalized", False)
                and self.slot > t["deadline"]
            ):
                t["_deadline_penalized"] = True
                penalty += REWARD_MISSED_DEADLINE
                self.missed_deadline_count += 1
                missed.append(t["id"])
        return penalty, missed

    def _success_probability(self, difficulty: float) -> float:
        value = SUCCESS_SIGMOID_SCALE * (
            self.energy + self.focus - difficulty - SUCCESS_DIFFICULTY_OFFSET
        )
        return float(1.0 / (1.0 + np.exp(-value)))

    def _all_tasks_done(self) -> bool:
        return all(t["done"] for t in self.tasks)
