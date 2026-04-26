"""Train a PPO agent on ADHDSchedulingEnv and save the model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from config import (
    PPO_TRAIN_STEPS, PPO_LEARNING_RATE, PPO_GAMMA,
    MODEL_PATH, DEFAULT_SEED, FIGURES_DIR,
)
from env import ADHDSchedulingEnv
from utils import ensure_dir, plot_training_curve


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

    # n_steps=512 (not SB3 default 2048): with ~12-step episodes, smaller
    # rollout buffers update more frequently → better credit assignment.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=PPO_GAMMA,
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
    parser.add_argument("--check-env", action="store_true",
                        help="run SB3 env checker before training")
    args = parser.parse_args()
    if args.check_env:
        validate_training_env(seed=args.seed)
    train(steps=args.steps, learning_rate=args.lr, seed=args.seed)
