"""Recursive train → evaluate → check → retrain pipeline.

Runs locally. Each trial trains PPO for N steps, evaluates all 4 policies,
checks success criteria, and scales up training budget if not met.

Logs every trial as JSON to experiments/run_<TIMESTAMP>/trial_NN.json.

Usage:
    python pipeline.py [--steps 100000] [--seed 42] [--episodes 100]
"""

from __future__ import annotations

import argparse
import json
import os
import time

from config import (
    PPO_TRAIN_STEPS, DEFAULT_SEED, EVAL_EPISODES,
    EXPERIMENTS_DIR,
)
from train import train
from evaluate import evaluate_all
from utils import ensure_dir

MAX_TRAIN_STEPS: int = 300_000
STEP_SCALE_FACTOR: float = 1.5
MAX_TRIALS: int = 6


def success_criteria(results: dict[str, dict]) -> bool:
    """PPO must beat Random AND be ≥90% of EDF on priority-weighted value."""
    ppo_val = results["PPO"]["priority_weighted_value"]
    rnd_val = results["Random"]["priority_weighted_value"]
    edf_val = results["EDF"]["priority_weighted_value"]
    beats_random = ppo_val > rnd_val
    competitive_edf = ppo_val >= 0.9 * edf_val
    return beats_random and competitive_edf


def run_pipeline(
    initial_steps: int = PPO_TRAIN_STEPS,
    seed: int = DEFAULT_SEED,
    eval_episodes: int = EVAL_EPISODES,
) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EXPERIMENTS_DIR, f"run_{timestamp}")
    ensure_dir(run_dir)
    print(f"Pipeline run directory: {run_dir}")

    steps = initial_steps
    last_log: dict = {}

    for trial in range(1, MAX_TRIALS + 1):
        print(f"\n{'='*60}")
        print(f"  Pipeline Trial {trial}/{MAX_TRIALS} | Steps: {steps:,}")
        print(f"{'='*60}")

        # --- Train ---
        train_metrics = train(steps=steps, seed=seed, verbose=0)

        # --- Evaluate ---
        results = evaluate_all(n_episodes=eval_episodes, seed=seed)

        # --- Check ---
        success = success_criteria(results)

        # --- Log ---
        last_log = {
            "trial": trial,
            "steps": steps,
            "seed": seed,
            "train_metrics": train_metrics,
            "eval_results": results,
            "success": success,
        }
        log_path = os.path.join(run_dir, f"trial_{trial:02d}.json")
        with open(log_path, "w") as f:
            json.dump(last_log, f, indent=2)
        print(f"Trial {trial} logged → {log_path}")

        ppo_val = results["PPO"]["priority_weighted_value"]
        rnd_val = results["Random"]["priority_weighted_value"]
        edf_val = results["EDF"]["priority_weighted_value"]
        print(
            f"  PPO={ppo_val:.2f}  Random={rnd_val:.2f}  EDF={edf_val:.2f}  "
            f"success={'YES ✓' if success else 'no'}"
        )

        if success:
            print(f"\nSuccess criteria met at trial {trial}!")
            break

        # Scale up budget
        next_steps = min(int(steps * STEP_SCALE_FACTOR), MAX_TRAIN_STEPS)
        if next_steps == steps:
            print("Reached max training budget. Stopping.")
            break
        steps = next_steps

    # Summary
    summary = {
        "run_dir": run_dir,
        "final_trial": last_log.get("trial"),
        "final_steps": last_log.get("steps"),
        "success": last_log.get("success", False),
        "final_results": last_log.get("eval_results", {}),
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPipeline complete. Summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive train/eval pipeline")
    parser.add_argument("--steps", type=int, default=PPO_TRAIN_STEPS,
                        help="Initial training steps (default: 100000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES,
                        help="Episodes per evaluation (default: 200)")
    args = parser.parse_args()
    run_pipeline(
        initial_steps=args.steps,
        seed=args.seed,
        eval_episodes=args.episodes,
    )
