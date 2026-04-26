"""Synthetic daily task generation for the scheduling prototype.

Generated tasks are small dictionaries consumed directly by ADHDSchedulingEnv.
The generator intentionally mixes urgent, high-value, short tasks with broader
random tasks so training episodes have varied deadline pressure and reward
trade-offs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import (
    MAX_PRIORITY, MAX_TASKS, MAX_TASK_DURATION,
    MIN_TASKS, MIN_TASK_DURATION, NUM_SLOTS,
)


def generate_task_set(
    num_tasks: int | None = None,
    num_slots: int = NUM_SLOTS,
    rng: Any | None = None,
) -> list[dict]:
    """Generate one synthetic day of tasks.

    The first task is biased toward being moderately urgent and meaningful so
    each day contains at least one clear scheduling target. Remaining tasks are
    sampled from broader distributions, then sorted by deadline, priority, and
    id to produce a stable order for the environment's task-index actions.

    Args:
        num_tasks: Number of tasks to generate. If None, samples MIN_TASKS–MAX_TASKS.
        num_slots: Number of available scheduling slots in the day.
        rng: Numpy random generator. If None, a new default generator is used.

    Returns:
        List of task dicts: {id, difficulty, priority, duration, deadline,
        remaining, done}.

    Raises:
        ValueError: If num_tasks is outside configured bounds or num_slots <= 0.
    """
    rng = np.random.default_rng() if rng is None else rng
    if num_tasks is None:
        num_tasks = int(rng.integers(MIN_TASKS, MAX_TASKS + 1))
    if not MIN_TASKS <= num_tasks <= MAX_TASKS:
        raise ValueError(f"num_tasks must be between {MIN_TASKS} and {MAX_TASKS}")
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")

    tasks: list[dict] = []
    for task_id in range(num_tasks):
        if task_id == 0:
            # Forced "anchor" task: moderately urgent, meaningful, completable early.
            difficulty = float(rng.uniform(0.35, 0.70))
            priority = int(rng.choice([2, MAX_PRIORITY], p=[0.65, 0.35]))
            duration = int(rng.integers(1, min(MAX_TASK_DURATION, 2) + 1))
            deadline = int(rng.integers(
                max(2, num_slots // 4),
                max(3, num_slots // 2) + 1,
            ))
        else:
            # Remaining tasks: bell-shaped difficulty, mixed urgency spread.
            difficulty = float(rng.beta(2.0, 2.0))
            priority = int(rng.choice([1, 2, MAX_PRIORITY], p=[0.35, 0.40, 0.25]))
            duration = int(rng.integers(MIN_TASK_DURATION, MAX_TASK_DURATION + 1))
            if task_id % 2 == 0:
                # Late deadline
                deadline = int(rng.integers(max(1, num_slots // 3), num_slots))
            else:
                # Early deadline
                deadline = int(rng.integers(0, max(1, (2 * num_slots) // 3)))

        tasks.append({
            "id": task_id,
            "difficulty": round(float(np.clip(difficulty, 0.0, 1.0)), 3),
            "priority": priority,
            "duration": duration,
            "deadline": int(np.clip(deadline, 0, num_slots - 1)),
            "remaining": duration,
            "done": False,
        })

    # Sort for stable ordering: earliest deadline first, then highest priority.
    tasks.sort(key=lambda t: (t["deadline"], -t["priority"], t["id"]))
    for new_id, task in enumerate(tasks):
        task["id"] = new_id

    return tasks
