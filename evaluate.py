"""Compare all policies over many sampled days and produce plots."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from config import (
    EVAL_EPISODES, MODEL_PATH, FIGURES_DIR, DEFAULT_SEED, REST_ACTION,
    REWARD_TASK_COMPLETE_MULT,
)
from env import ADHDSchedulingEnv
from baselines import RandomPolicy, EDFPolicy, EnergyAwarePolicy
from utils import ensure_dir, plot_comparison_bars


def run_policy_episodes(
    policy,
    env: ADHDSchedulingEnv,
    n_episodes: int,
    seed_offset: int = 0,
) -> dict:
    """Run policy for n_episodes and return averaged metrics dict.

    All policies share the same seed_offset so they face identical task draws,
    making the comparison statistically fair.
    """
    is_ppo = hasattr(policy, "predict")

    accum: dict[str, list[float]] = {
        "priority_weighted_value": [],
        "tasks_completed": [],
        "urgent_completion_rate": [],
        "rest_count": [],
        "missed_deadlines": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False

        ep_value = 0.0
        ep_tasks_done = 0
        ep_urgent_done = 0
        ep_urgent_total = sum(1 for t in env.tasks if t["priority"] == 3)
        ep_rest = 0
        completed_ids: set[int] = set()

        while not done:
            if is_ppo:
                action, _ = policy.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = policy.select_action(obs, env)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if action == REST_ACTION:
                ep_rest += 1

            for i, t in enumerate(env.tasks):
                if t["done"] and i not in completed_ids:
                    completed_ids.add(i)
                    ep_value += REWARD_TASK_COMPLETE_MULT * t["priority"]
                    ep_tasks_done += 1
                    if t["priority"] == 3:
                        ep_urgent_done += 1

        ep_missed = sum(
            1 for t in env.tasks
            if not t["done"] and env.slot > t["deadline"]
        )

        accum["priority_weighted_value"].append(ep_value)
        accum["tasks_completed"].append(ep_tasks_done)
        accum["urgent_completion_rate"].append(
            ep_urgent_done / max(ep_urgent_total, 1)
        )
        accum["rest_count"].append(ep_rest)
        accum["missed_deadlines"].append(ep_missed)

    return {k: float(np.mean(v)) for k, v in accum.items()}


def evaluate_all(
    n_episodes: int = EVAL_EPISODES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict]:
    """Evaluate all 4 policies, print table, save bar plots."""
    env = ADHDSchedulingEnv(seed=seed)
    ppo_model = PPO.load(MODEL_PATH)

    policies: dict[str, object] = {
        "Random": RandomPolicy(seed=seed),
        "EDF": EDFPolicy(),
        "EnergyAware": EnergyAwarePolicy(),
        "PPO": ppo_model,
    }

    results: dict[str, dict] = {}
    for name, pol in policies.items():
        print(f"Evaluating {name} over {n_episodes} episodes...")
        results[name] = run_policy_episodes(
            pol, env, n_episodes, seed_offset=seed * 10_000
        )

    # Console table
    df = pd.DataFrame(results).T
    print("\n=== Evaluation Results ===")
    print(df.to_string(float_format="{:.3f}".format))

    # Bar plots
    ensure_dir(FIGURES_DIR)
    for metric in df.columns:
        plot_comparison_bars(
            results,
            metric,
            save_path=os.path.join(FIGURES_DIR, f"{metric}.png"),
        )
    print(f"\nBar plots saved to {FIGURES_DIR}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    evaluate_all(n_episodes=args.episodes, seed=args.seed)
