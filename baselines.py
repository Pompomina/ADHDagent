"""Three heuristic baseline policies.

Common interface: select_action(obs, env) -> int

Policies access env.tasks / env.energy / env.focus / env.slot directly
for semantic decision-making (obs decoding would introduce rounding noise).
"""

from __future__ import annotations

import numpy as np

from config import REST_ACTION


class RandomPolicy:
    """Uniform random over valid actions: undone tasks + REST."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, env) -> int:
        valid = [i for i, t in enumerate(env.tasks) if not t["done"]]
        valid.append(REST_ACTION)
        return int(self._rng.choice(valid))


class EDFPolicy:
    """Earliest-Deadline-First.

    Picks the undone task with the minimum deadline, preferring tasks whose
    deadline has not yet passed. Falls back to REST if energy is critically low.

    Fix over naive EDF: tasks past their deadline are deprioritised (sorted
    after still-reachable ones) so the policy doesn't grind a hopeless task
    at the cost of every remaining slot.
    """

    REST_ENERGY_THRESHOLD: float = 0.25  # raised from 0.15 — collapse happens fast

    def select_action(self, obs: np.ndarray, env) -> int:
        if env.energy < self.REST_ENERGY_THRESHOLD:
            return REST_ACTION
        undone = [(i, t) for i, t in enumerate(env.tasks) if not t["done"]]
        if not undone:
            return REST_ACTION
        # Prefer tasks with deadlines still in the future; break ties by deadline
        def edf_key(item: tuple[int, dict]) -> tuple[int, int]:
            _, t = item
            overdue = 1 if t["deadline"] < env.slot else 0
            return (overdue, t["deadline"])
        return min(undone, key=edf_key)[0]


class EnergyAwarePolicy:
    """Energy-aware heuristic.

    REST if energy < 0.3 or focus < 0.3.
    Otherwise pick the undone task with the highest score:
        score = 2 * priority - difficulty - 0.1 * time_to_deadline
    """

    ENERGY_THRESHOLD: float = 0.3
    FOCUS_THRESHOLD: float = 0.3

    def select_action(self, obs: np.ndarray, env) -> int:
        if env.energy < self.ENERGY_THRESHOLD or env.focus < self.FOCUS_THRESHOLD:
            return REST_ACTION
        undone = [(i, t) for i, t in enumerate(env.tasks) if not t["done"]]
        if not undone:
            return REST_ACTION

        def score(item: tuple[int, dict]) -> float:
            _, t = item
            time_to_deadline = max(0, t["deadline"] - env.slot)
            return 2.0 * t["priority"] - t["difficulty"] - 0.1 * time_to_deadline

        return max(undone, key=score)[0]
