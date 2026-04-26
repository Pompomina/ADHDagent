"""Train a PPO agent on ADHDSchedulingEnv and save the model."""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from config import (
    PPO_TRAIN_STEPS, PPO_LEARNING_RATE, MODEL_PATH, DEFAULT_SEED, FIGURES_DIR,
)
from env import ADHDSchedulingEnv
from utils import ensure_dir, plot_training_curve


def train(
    steps: int = PPO_TRAIN_STEPS,
    learning_rate: float = PPO_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    verbose: int = 1,
) -> dict:
    """Train PPO, save model, return basic metrics dict."""
    ensure_dir(os.path.dirname(MODEL_PATH))
    env = Monitor(ADHDSchedulingEnv(seed=seed))

    # n_steps=512 instead of SB3 default 2048:
    # With 16-slot episodes (~12 steps avg), 2048 spans ~170 episodes before
    # any gradient update. 512 = ~42 episodes per update — much better
    # credit assignment for this short-horizon task.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        verbose=verbose,
        seed=seed,
    )
    model.learn(total_timesteps=steps)
    model.save(MODEL_PATH)

    ep_rewards = env.get_episode_rewards()
    last50 = ep_rewards[-50:] if len(ep_rewards) >= 50 else ep_rewards
    mean_reward = float(sum(last50) / max(len(last50), 1))
    print(f"\nTraining done — mean reward (last 50 eps): {mean_reward:.3f}")
    print(f"Model saved to: {MODEL_PATH}")

    # Save training curve
    ensure_dir(FIGURES_DIR)
    curve_path = f"{FIGURES_DIR}/training_curve.png"
    plot_training_curve(ep_rewards, save_path=curve_path)
    print(f"Training curve saved to: {curve_path}")

    return {
        "steps": steps,
        "mean_reward_last50": mean_reward,
        "total_episodes": len(ep_rewards),
        "model_path": MODEL_PATH,
        "training_curve": curve_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on ADHDSchedulingEnv")
    parser.add_argument("--steps", type=int, default=PPO_TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    train(steps=args.steps, learning_rate=args.lr, seed=args.seed)
