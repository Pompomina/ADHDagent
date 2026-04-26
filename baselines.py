"""Heuristic baseline policies for the scheduling environment.

Each policy exposes a select_action(obs, env) -> int method so it can be
used interchangeably with learned agents during rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import (
    REST_ACTION,
    EDF_REST_ENERGY_THRESHOLD,
    EA_ENERGY_THRESHOLD, EA_FOCUS_THRESHOLD, EA_DIFFICULTY_MARGIN,
    EA_PRIORITY_WEIGHT, EA_DIFFICULTY_WEIGHT, EA_DEADLINE_WEIGHT,
    MAX_TASKS,
)


class BaselinePolicy(Protocol):
    """Structural type for policies compatible with ADHDSchedulingEnv rollouts."""

    name: str

    def select_action(self, obs: np.ndarray, env) -> int: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def unfinished_task_indices(env) -> list[int]:
    """Indices of tasks that are real, unfinished, and have remaining work."""
    return [
        idx
        for idx, task in enumerate(env.tasks[:MAX_TASKS])
        if not bool(task.get("done", False)) and int(task.get("remaining", 0)) > 0
    ]


def not_yet_missed_task_indices(env, candidates: list[int]) -> list[int]:
    """Filter candidates to tasks whose deadlines have not yet passed."""
    return [
        idx for idx in candidates
        if int(env.tasks[idx]["deadline"]) >= int(env.slot)
    ]


def make_baseline_policies() -> list[BaselinePolicy]:
    """Create all three baseline policies in reporting order."""
    return [RandomPolicy(), EDFPolicy(), EnergyAwarePolicy()]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@dataclass
class RandomPolicy:
    """Random baseline: samples uniformly from unfinished tasks + REST.

    When possible, tasks whose deadlines have already passed are excluded so
    the random baseline does not waste slots on hopeless work.
    """

    name: str = "random"

    def select_action(self, obs: np.ndarray, env) -> int:
        candidates = unfinished_task_indices(env)
        candidates = not_yet_missed_task_indices(env, candidates) or candidates
        candidates.append(REST_ACTION)
        return int(env._rng.choice(candidates))


@dataclass
class EDFPolicy:
    """Earliest-deadline-first with energy guard and tiebreaking.

    Rests when energy is critically low. Among tied deadlines prefers higher
    priority, then easier tasks. Avoids overdue tasks when alternatives exist.
    """

    rest_threshold: float = EDF_REST_ENERGY_THRESHOLD
    name: str = "edf"

    def select_action(self, obs: np.ndarray, env) -> int:
        candidates = unfinished_task_indices(env)
        if not candidates or env.energy < self.rest_threshold:
            return REST_ACTION
        candidates = not_yet_missed_task_indices(env, candidates) or candidates
        return min(
            candidates,
            key=lambda idx: (
                int(env.tasks[idx]["deadline"]),
                -int(env.tasks[idx]["priority"]),
                float(env.tasks[idx]["difficulty"]),
                idx,
            ),
        )


@dataclass
class EnergyAwarePolicy:
    """Energy-aware heuristic balancing priority, difficulty, and urgency.

    Rests when energy or focus is below threshold. Among unfinished tasks,
    prefers those that are feasible given current state (difficulty within
    energy+focus capacity), then scores by the guide formula.
    """

    energy_threshold: float = EA_ENERGY_THRESHOLD
    focus_threshold:  float = EA_FOCUS_THRESHOLD
    difficulty_margin: float = EA_DIFFICULTY_MARGIN
    name: str = "energy_aware"

    def select_action(self, obs: np.ndarray, env) -> int:
        candidates = unfinished_task_indices(env)
        if not candidates or env.energy < self.energy_threshold or env.focus < self.focus_threshold:
            return REST_ACTION
        candidates = not_yet_missed_task_indices(env, candidates) or candidates
        capacity = min(1.0, (env.energy + env.focus) / 2.0 + self.difficulty_margin)
        feasible = [i for i in candidates if float(env.tasks[i]["difficulty"]) <= capacity]
        pool = feasible if feasible else candidates
        return max(
            pool,
            key=lambda idx: (self._score(env, idx), -int(env.tasks[idx]["deadline"]), -idx),
        )

    def _score(self, env, idx: int) -> float:
        t = env.tasks[idx]
        time_to_deadline = max(0, int(t["deadline"]) - int(env.slot))
        return (
            EA_PRIORITY_WEIGHT * int(t["priority"])
            - EA_DIFFICULTY_WEIGHT * float(t["difficulty"])
            - EA_DEADLINE_WEIGHT * time_to_deadline
        )


# ---------------------------------------------------------------------------
# Rollout helper
# ---------------------------------------------------------------------------

def rollout_policy(env, policy, seed: int | None = None) -> dict:
    """Run one full episode and return a compact result dict.

    Args:
        env: An ADHDSchedulingEnv instance.
        policy: Any object with select_action(obs, env) -> int.
        seed: Optional reset seed.

    Returns:
        Dict with total_reward, completed_tasks, completed_value,
        missed_deadline_count, rest_actions, invalid_actions, steps,
        trajectory (list of per-slot dicts), final_tasks.
    """
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    trajectory: list[dict] = []

    while not done:
        action = int(policy.select_action(obs, env))
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        invalid = bool(info.get("invalid_action", False))
        if invalid:
            action_label = f"T{action}->REST"
        elif action == REST_ACTION:
            action_label = "REST"
        else:
            action_label = f"T{action}"

        trajectory.append({
            "slot": int(info["slot"]) - 1,
            "action": action,
            "action_label": action_label,
            "action_type": str(info.get("action_type", "")),
            "reward": float(reward),
            "energy": float(info["energy"]),
            "focus": float(info["focus"]),
            "completed_tasks": int(info["completed_tasks"]),
            "completed_value": int(info["completed_value"]),
            "missed_deadline_count": int(info["missed_deadline_count"]),
            "invalid_action": invalid,
            "task_success": bool(info.get("task_success", False)),
            "missed_deadlines": list(info.get("missed_deadlines", [])),
        })
        obs = next_obs
        done = bool(terminated or truncated)

    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "total_reward": total_reward,
        "completed_tasks": int(info["completed_tasks"]),
        "completed_value": int(info["completed_value"]),
        "missed_deadline_count": int(info["missed_deadline_count"]),
        "rest_actions": sum(1 for row in trajectory if row["action"] == REST_ACTION),
        "invalid_actions": sum(1 for row in trajectory if row["invalid_action"]),
        "steps": len(trajectory),
        "trajectory": trajectory,
        "final_tasks": info["tasks"],
    }
