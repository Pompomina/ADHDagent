"""Synthetic task generator for one simulated day."""

from __future__ import annotations

import numpy as np


def generate_task_set(
    num_tasks: int,
    num_slots: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate a list of num_tasks dicts for one episode.

    Each task:
        id, difficulty∈[0,1], priority∈{1,2,3}, duration∈{1,2,3},
        deadline∈[duration, num_slots-1], remaining=duration, done=False

    Diversity guarantees (post-sampling corrections):
        - At least one task with deadline <= num_slots // 2  (early deadline)
        - At least one task with priority == 3               (urgent)
    """
    tasks = []
    for i in range(num_tasks):
        difficulty = float(rng.uniform(0.0, 1.0))
        priority = int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
        duration = int(rng.choice([1, 2, 3]))
        # Deadline must be reachable: deadline >= duration (so task can complete)
        deadline = int(rng.integers(duration, num_slots))
        tasks.append({
            "id": i,
            "difficulty": round(difficulty, 4),
            "priority": priority,
            "duration": duration,
            "deadline": deadline,
            "remaining": duration,
            "done": False,
        })

    # --- Diversity corrections ---
    # Ensure at least one early deadline
    half = num_slots // 2
    if not any(t["deadline"] <= half for t in tasks):
        idx = int(rng.integers(0, num_tasks))
        t = tasks[idx]
        t["deadline"] = int(rng.integers(t["duration"], half + 1))

    # Ensure at least one urgent (priority 3) task
    if not any(t["priority"] == 3 for t in tasks):
        idx = int(rng.integers(0, num_tasks))
        tasks[idx]["priority"] = 3

    return tasks
