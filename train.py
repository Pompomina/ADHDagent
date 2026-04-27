"""Train a PPO agent on ADHDSchedulingEnv and save the model."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from config import (
    PPO_TRAIN_STEPS, PPO_LEARNING_RATE, PPO_GAMMA,
    PPO_POLICY, PPO_N_STEPS, PPO_VERBOSE,
    MODEL_PATH, DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
)
from env import ADHDSchedulingEnv
from utils import ensure_dir, plot_training_curve


# ---------------------------------------------------------------------------
# Training reward logger (CSV-based, richer than Monitor episode_rewards)
# ---------------------------------------------------------------------------

class TrainingRewardLogger:
    """SB3 callback that records completed episode rewards to CSV during training."""

    def __init__(self, log_path: str | Path, rolling_window: int = 20) -> None:
        self.log_path = Path(log_path)
        self.rolling_window = rolling_window
        self.rows: list[dict[str, float]] = []
        self._callback: Any = None

    def make_callback(self):
        """Create and return the SB3 BaseCallback instance for model.learn."""
        from stable_baselines3.common.callbacks import BaseCallback

        logger = self

        class _Callback(BaseCallback):
            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    episode = info.get("episode")
                    if episode is None:
                        continue
                    reward = float(episode["r"])
                    recent = [r["episode_reward"] for r in logger.rows]
                    recent.append(reward)
                    window = recent[-logger.rolling_window:]
                    logger.rows.append({
                        "training_step": float(self.num_timesteps),
                        "episode_reward": reward,
                        "mean_episode_reward": float(sum(window) / len(window)),
                    })
                return True

        self._callback = _Callback()
        return self._callback

    def save(self) -> Path | None:
        """Write collected data to CSV; returns path or None if no data."""
        if not self.rows:
            return None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["training_step", "episode_reward", "mean_episode_reward"],
            )
            writer.writeheader()
            writer.writerows(self.rows)
        return self.log_path


def save_training_curve(
    log_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot a PPO training curve from a TrainingRewardLogger CSV file."""
    rows: list[dict[str, float]] = []
    with Path(log_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "training_step": float(row["training_step"]),
                "episode_reward": float(row["episode_reward"]),
                "mean_episode_reward": float(row["mean_episode_reward"]),
            })
    if not rows:
        raise ValueError(f"training log has no rows: {log_path}")

    from utils import ensure_dir, get_pyplot
    plt = get_pyplot()
    steps = [r["training_step"] for r in rows]
    rewards = [r["episode_reward"] for r in rows]
    means = [r["mean_episode_reward"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rewards, color="#BAB0AC", alpha=0.35, linewidth=1, label="Episode reward")
    ax.plot(steps, means, color="#4C78A8", linewidth=2.4, label="Rolling mean reward")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode reward")
    ax.set_title("PPO Training Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Helper factories (used by test.py and pipeline.py)
# ---------------------------------------------------------------------------

def make_training_env(seed: int = DEFAULT_SEED, use_monitor: bool = True):
    """Create a fresh training env, optionally wrapped in SB3 Monitor."""
    env = ADHDSchedulingEnv(seed=seed)
    return Monitor(env) if use_monitor else env


def validate_training_env(seed: int = DEFAULT_SEED) -> None:
    """Run SB3's env checker and close the env."""
    env = ADHDSchedulingEnv(seed=seed)
    check_env(env, warn=True)
    env.close()
    print("validate_training_env: PASS")


def load_trained_model(model_path: str | Path = MODEL_PATH, env=None):
    """Load a saved PPO model. Raises FileNotFoundError if missing."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"trained PPO model not found: {path}")
    return PPO.load(path, env=env)


class TrainedPPOPolicy:
    """Adapter exposing a saved SB3 PPO model as a rollout-compatible policy."""

    name: str = "ppo"

    def __init__(self, model, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def select_action(self, obs, env) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)


def load_trained_policy(
    model_path: str | Path = MODEL_PATH,
    env=None,
    deterministic: bool = True,
) -> TrainedPPOPolicy:
    """Load a saved model and wrap it as a rollout-compatible policy."""
    model = load_trained_model(model_path=model_path, env=env)
    return TrainedPPOPolicy(model=model, deterministic=deterministic)


# ---------------------------------------------------------------------------
# Main training entry point (kept for pipeline.py / mcp_server.py compat)
# ---------------------------------------------------------------------------

def train(
    steps: int = PPO_TRAIN_STEPS,
    learning_rate: float = PPO_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    verbose: int = 1,
) -> dict:
    """Train PPO, save model, return metrics dict."""
    ensure_dir(os.path.dirname(MODEL_PATH))
    env = make_training_env(seed=seed, use_monitor=True)

    # PPO_N_STEPS (512) << SB3 default 2048: with ~12-step episodes, smaller
    # rollout buffers update more frequently → better credit assignment.
    model = PPO(
        PPO_POLICY,
        env,
        learning_rate=learning_rate,
        n_steps=PPO_N_STEPS,
        batch_size=64,
        n_epochs=10,
        gamma=PPO_GAMMA,
        verbose=verbose,
        seed=seed,
    )
    log_path = Path(RESULTS_DIR) / "ppo_training_log.csv"
    reward_logger = TrainingRewardLogger(log_path=log_path)
    model.learn(total_timesteps=steps, callback=reward_logger.make_callback())
    model.save(MODEL_PATH)

    ep_rewards = env.get_episode_rewards()
    last50 = ep_rewards[-50:] if len(ep_rewards) >= 50 else ep_rewards
    mean_reward = float(sum(last50) / max(len(last50), 1))
    print(f"\nTraining done — mean reward (last 50 eps): {mean_reward:.3f}")
    print(f"Model saved to: {MODEL_PATH}")

    # Save training curve — prefer CSV-based logger; fall back to Monitor rewards.
    ensure_dir(FIGURES_DIR)
    curve_path = f"{FIGURES_DIR}/fig_ppo_training_curve.png"
    saved_log = reward_logger.save()
    if saved_log is not None:
        save_training_curve(log_path=saved_log, output_path=curve_path)
    else:
        plot_training_curve(ep_rewards, save_path=curve_path)
    print(f"Training curve saved to: {curve_path}")

    return {
        "steps": steps,
        "mean_reward_last50": mean_reward,
        "total_episodes": len(ep_rewards),
        "model_path": MODEL_PATH,
        "training_curve": curve_path,
        "training_log": str(log_path) if saved_log else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on ADHDSchedulingEnv")
    parser.add_argument("--steps", type=int, default=PPO_TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--check-env", action="store_true",
                        help="run SB3 env checker before training")
    args = parser.parse_args()
    if args.check_env:
        validate_training_env(seed=args.seed)
    train(steps=args.steps, learning_rate=args.lr, seed=args.seed)
