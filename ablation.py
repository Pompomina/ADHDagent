"""Parameter ablation studies: crash probability and missed-deadline penalty.

Each ablation trains a fresh PPO model under a modified config value, then
evaluates all four policies on the same task sets. Results are plotted as line
charts showing how policy performance varies with the parameter.

Usage:
    python ablation.py                  # run both ablations
    python ablation.py --crash-only     # crash probability only
    python ablation.py --reward-only    # missed deadline penalty only
    python ablation.py --steps 30000 --episodes 50   # quick smoke run
"""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import config as _config
from config import DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR
from utils import ensure_dir, generate_eval_task_sets, get_pyplot, rollout_metrics, load_policy_suite
from baselines import rollout_policy
from env import ADHDSchedulingEnv

# ---------------------------------------------------------------------------
# Ablation parameter grids
# ---------------------------------------------------------------------------

CRASH_PROB_VALUES: list[float] = [0.0, 0.04, 0.08, 0.15]
MISSED_DEADLINE_VALUES: list[float] = [-0.5, -1.0, -1.5, -2.0]

ABLATION_TRAIN_STEPS: int = 50_000
ABLATION_EVAL_EPISODES: int = 100

_POLICY_COLORS = {"random": "#BAB0AC", "edf": "#F58518", "energy_aware": "#54A24B", "ppo": "#4C78A8"}


# ---------------------------------------------------------------------------
# Config override context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def override_config(**kwargs):
    """Temporarily monkeypatch module-level config constants.

    Example:
        with override_config(CRASH_PROB=0.0, REWARD_MISSED_DEADLINE=-2.0):
            ...  # config values patched inside this block
    """
    old_values: dict[str, Any] = {}
    try:
        for key, value in kwargs.items():
            old_values[key] = getattr(_config, key)
            setattr(_config, key, value)
            # Also patch into already-imported env module so it picks up the change.
            import env as _env_module
            if hasattr(_env_module, key):
                setattr(_env_module, key, value)
        yield
    finally:
        for key, value in old_values.items():
            setattr(_config, key, value)
            import env as _env_module
            if hasattr(_env_module, key):
                setattr(_env_module, key, value)


# ---------------------------------------------------------------------------
# Core: train + evaluate under a given config override
# ---------------------------------------------------------------------------

def _train_and_evaluate(
    override: dict[str, Any],
    model_path: str | Path,
    train_steps: int,
    eval_episodes: int,
    seed: int,
) -> dict[str, float]:
    """Train PPO and evaluate all policies under a config override.

    Returns:
        Dict mapping policy name to average total_reward.
    """
    from train import train, load_trained_policy
    from stable_baselines3.common.monitor import Monitor

    with override_config(**override):
        # Train PPO
        env = Monitor(ADHDSchedulingEnv(seed=seed))
        from stable_baselines3 import PPO
        model = PPO(
            _config.PPO_POLICY,
            env,
            learning_rate=_config.PPO_LEARNING_RATE,
            n_steps=_config.PPO_N_STEPS,
            batch_size=64,
            n_epochs=10,
            gamma=_config.PPO_GAMMA,
            verbose=0,
            seed=seed,
        )
        model.learn(total_timesteps=train_steps)
        model.save(model_path)
        env.close()

        # Evaluate
        task_sets = generate_eval_task_sets(num_episodes=eval_episodes, seed=seed)
        ppo_policy = load_trained_policy(model_path=model_path, deterministic=True)
        policies, _ = load_policy_suite(include_ppo=False)
        policies.append(ppo_policy)

        policy_rewards: dict[str, list[float]] = {p.name: [] for p in policies}
        for episode_idx, tasks in enumerate(task_sets):
            for policy_idx, policy in enumerate(policies):
                env_seed = seed + 1000 * episode_idx + policy_idx
                env = ADHDSchedulingEnv(seed=env_seed, tasks=deepcopy(tasks))
                result = rollout_policy(env, policy, seed=env_seed)
                m = rollout_metrics(result, num_slots=env.num_slots)
                policy_rewards[policy.name].append(float(m["total_reward"]))

    return {name: float(np.mean(rewards)) for name, rewards in policy_rewards.items()}


# ---------------------------------------------------------------------------
# Crash probability ablation
# ---------------------------------------------------------------------------

def run_crash_prob_ablation(
    seed: int = DEFAULT_SEED,
    train_steps: int = ABLATION_TRAIN_STEPS,
    eval_episodes: int = ABLATION_EVAL_EPISODES,
) -> list[dict]:
    """Train + evaluate over CRASH_PROB_VALUES. Returns list of result dicts."""
    results = []
    tmp_model = Path(RESULTS_DIR) / "ablation_tmp_model.zip"
    ensure_dir(tmp_model.parent)

    for crash_prob in CRASH_PROB_VALUES:
        print(f"  CRASH_PROB={crash_prob:.2f} — training {train_steps} steps ...")
        rewards = _train_and_evaluate(
            override={"CRASH_PROB": crash_prob},
            model_path=tmp_model,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            seed=seed,
        )
        row = {"crash_prob": crash_prob}
        row.update(rewards)
        results.append(row)
        print(f"    PPO={rewards.get('ppo', float('nan')):.2f}  "
              f"EDF={rewards.get('edf', float('nan')):.2f}  "
              f"EA={rewards.get('energy_aware', float('nan')):.2f}")

    if tmp_model.exists():
        tmp_model.unlink()
    return results


# ---------------------------------------------------------------------------
# Missed-deadline penalty ablation
# ---------------------------------------------------------------------------

def run_reward_ablation(
    seed: int = DEFAULT_SEED,
    train_steps: int = ABLATION_TRAIN_STEPS,
    eval_episodes: int = ABLATION_EVAL_EPISODES,
) -> list[dict]:
    """Train + evaluate over MISSED_DEADLINE_VALUES. Returns list of result dicts."""
    results = []
    tmp_model = Path(RESULTS_DIR) / "ablation_tmp_model.zip"
    ensure_dir(tmp_model.parent)

    for penalty in MISSED_DEADLINE_VALUES:
        print(f"  REWARD_MISSED_DEADLINE={penalty:.1f} — training {train_steps} steps ...")
        rewards = _train_and_evaluate(
            override={"REWARD_MISSED_DEADLINE": penalty},
            model_path=tmp_model,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            seed=seed,
        )
        row = {"missed_deadline_penalty": penalty}
        row.update(rewards)
        results.append(row)
        print(f"    PPO={rewards.get('ppo', float('nan')):.2f}  "
              f"EDF={rewards.get('edf', float('nan')):.2f}  "
              f"EA={rewards.get('energy_aware', float('nan')):.2f}")

    if tmp_model.exists():
        tmp_model.unlink()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_ablation(
    results: list[dict],
    x_key: str,
    xlabel: str,
    title: str,
    output_path: str | Path,
) -> Path:
    plt = get_pyplot()
    x_vals = [float(row[x_key]) for row in results]
    policy_names = [k for k in results[0] if k != x_key]

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in policy_names:
        y_vals = [float(row[name]) for row in results]
        color = _POLICY_COLORS.get(name, "#999999")
        label = {"random": "Random", "edf": "EDF",
                 "energy_aware": "Energy-Aware", "ppo": "PPO"}.get(name, name)
        ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=label, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average total reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_ablation_plots(
    crash_results: list[dict] | None,
    reward_results: list[dict] | None,
    figure_dir: str | Path = FIGURES_DIR,
) -> list[Path]:
    """Save ablation line charts. Pass None to skip a panel."""
    paths = []
    if crash_results:
        paths.append(_plot_ablation(
            crash_results, "crash_prob", "Focus crash probability",
            "PPO vs Baselines: Focus Crash Probability Ablation",
            Path(figure_dir) / "fig_ablation_crash_prob.png",
        ))
    if reward_results:
        paths.append(_plot_ablation(
            reward_results, "missed_deadline_penalty", "Missed deadline penalty",
            "PPO vs Baselines: Missed Deadline Penalty Ablation",
            Path(figure_dir) / "fig_ablation_missed_penalty.png",
        ))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run parameter ablation studies.")
    parser.add_argument("--crash-only", action="store_true", help="run crash prob ablation only")
    parser.add_argument("--reward-only", action="store_true", help="run reward ablation only")
    parser.add_argument("--steps", type=int, default=ABLATION_TRAIN_STEPS,
                        help=f"training steps per config (default {ABLATION_TRAIN_STEPS})")
    parser.add_argument("--episodes", type=int, default=ABLATION_EVAL_EPISODES,
                        help=f"eval episodes per config (default {ABLATION_EVAL_EPISODES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--figure-dir", type=str, default=FIGURES_DIR)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_crash = not args.reward_only
    run_reward = not args.crash_only

    crash_results: list[dict] | None = None
    reward_results: list[dict] | None = None

    if run_crash:
        print("=== Crash probability ablation ===")
        crash_results = run_crash_prob_ablation(
            seed=args.seed, train_steps=args.steps, eval_episodes=args.episodes
        )

    if run_reward:
        print("=== Missed deadline penalty ablation ===")
        reward_results = run_reward_ablation(
            seed=args.seed, train_steps=args.steps, eval_episodes=args.episodes
        )

    paths = save_ablation_plots(crash_results, reward_results, figure_dir=args.figure_dir)
    print("\nSaved ablation plots:")
    for p in paths:
        print(f"  {p}")
