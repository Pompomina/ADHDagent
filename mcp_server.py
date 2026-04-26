"""Minimal MCP server exposing train/evaluate/results as Claude Code tools.

Install: pip install mcp
Run:     python mcp_server.py

Wire into Claude Code via .mcp.json:
{
  "mcpServers": {
    "adhd-scheduler": {
      "command": ".venv/bin/python",
      "args": ["mcp_server.py"],
      "cwd": "/Users/mina/dev/ADHDagent"
    }
  }
}

Once wired, Claude Code can call train_agent / evaluate_policies /
get_latest_results as native MCP tool calls for interactive
hyperparameter tuning without leaving the conversation.
"""

from __future__ import annotations

import json
import os

from config import (
    PPO_TRAIN_STEPS, PPO_LEARNING_RATE, DEFAULT_SEED,
    EVAL_EPISODES, EXPERIMENTS_DIR,
)
from train import train as _train
from evaluate import evaluate_all as _evaluate

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise SystemExit(
        "mcp package not installed.\n"
        "Run: pip install mcp\n"
        "Then restart the server."
    )

mcp = FastMCP("ADHD Scheduler RL")


@mcp.tool()
def train_agent(
    steps: int = PPO_TRAIN_STEPS,
    learning_rate: float = PPO_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Train the PPO agent for the specified number of timesteps.

    Args:
        steps: Total environment steps (default 100 000).
        learning_rate: PPO learning rate (default 3e-4).
        seed: Random seed for reproducibility.

    Returns:
        Dict with steps, mean_reward_last50, total_episodes, model_path.
    """
    return _train(steps=steps, learning_rate=learning_rate, seed=seed, verbose=0)


@mcp.tool()
def evaluate_policies(
    n_episodes: int = EVAL_EPISODES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Evaluate all 4 policies (Random, EDF, EnergyAware, PPO) over n_episodes.

    Args:
        n_episodes: Number of evaluation episodes (default 200).
        seed: Seed for reproducible episode generation.

    Returns:
        Nested dict: {policy_name: {metric: value}}.
        Metrics: priority_weighted_value, tasks_completed,
                 urgent_completion_rate, rest_count, missed_deadlines.
    """
    results = _evaluate(n_episodes=n_episodes, seed=seed)
    # Ensure JSON-serialisable (numpy floats → Python floats)
    return {
        policy: {k: float(v) for k, v in metrics.items()}
        for policy, metrics in results.items()
    }


@mcp.tool()
def get_latest_results() -> dict:
    """Return the most recent pipeline trial JSON from experiments/.

    Returns:
        The full contents of the latest trial_NN.json, or an error dict.
    """
    if not os.path.exists(EXPERIMENTS_DIR):
        return {"error": f"No experiments directory found at {EXPERIMENTS_DIR}"}

    runs = sorted(
        d for d in os.listdir(EXPERIMENTS_DIR)
        if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))
    )
    if not runs:
        return {"error": "No runs found in experiments/"}

    latest_run = os.path.join(EXPERIMENTS_DIR, runs[-1])
    trials = sorted(
        f for f in os.listdir(latest_run)
        if f.startswith("trial_") and f.endswith(".json")
    )
    if not trials:
        return {"error": f"No trial files in {latest_run}"}

    with open(os.path.join(latest_run, trials[-1])) as f:
        return json.load(f)


if __name__ == "__main__":
    mcp.run()
