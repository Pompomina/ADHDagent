"""Shared utilities: math helpers, seeding, recording, plotting, eval helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def pretty_print_tasks(tasks: list[dict]) -> None:
    header = f"{'ID':>3} {'pri':>3} {'diff':>6} {'dur':>4} {'dl':>3} {'rem':>4} {'done':>5}"
    print(header)
    print("-" * len(header))
    for t in tasks:
        print(
            f"{t['id']:>3} {t['priority']:>3} {t['difficulty']:>6.2f} "
            f"{t['duration']:>4} {t['deadline']:>3} {t['remaining']:>4} "
            f"{'yes' if t['done'] else 'no':>5}"
        )


# ---------------------------------------------------------------------------
# Rollout recording
# ---------------------------------------------------------------------------

@dataclass
class _SlotRecord:
    slot: int
    action: int
    reward: float
    energy: float
    focus: float
    tasks_snapshot: list[dict]


class RolloutRecorder:
    """Accumulates per-slot data during a single episode rollout."""

    def __init__(self) -> None:
        self._records: list[_SlotRecord] = []

    def record(
        self,
        slot: int,
        action: int,
        reward: float,
        energy: float,
        focus: float,
        tasks_snapshot: list[dict],
    ) -> None:
        self._records.append(
            _SlotRecord(slot, action, reward, energy, focus, tasks_snapshot)
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "slot": r.slot,
                "action": r.action,
                "reward": r.reward,
                "energy": r.energy,
                "focus": r.focus,
            }
            for r in self._records
        ]
        return pd.DataFrame(rows)

    @property
    def records(self) -> list[_SlotRecord]:
        return self._records


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_energy_focus(
    recorder: RolloutRecorder,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    df = recorder.to_dataframe()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["slot"], df["energy"], label="energy", color="steelblue")
    ax.plot(df["slot"], df["focus"], label="focus", color="darkorange")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Level")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_action_timeline(
    recorder: RolloutRecorder,
    title: str,
    num_actions: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """Color-coded bar chart: each slot colored by action (REST vs task id)."""
    from config import REST_ACTION, NUM_ACTIONS as _NA
    if num_actions is None:
        num_actions = _NA

    df = recorder.to_dataframe()
    cmap = plt.get_cmap("tab10")
    colors = [cmap(a % 10) for a in df["action"]]

    fig, ax = plt.subplots(figsize=(10, 2))
    for _, row in df.iterrows():
        c = "lightgray" if row["action"] == REST_ACTION else cmap(int(row["action"]) % 10)
        ax.barh(0, 1, left=row["slot"], color=c, edgecolor="white", height=0.6)

    # Legend
    from config import MAX_TASKS
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i % 10), label=f"Task {i}")
               for i in range(MAX_TASKS)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="lightgray", label="REST"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=4)
    ax.set_xlim(0, len(df))
    ax.set_yticks([])
    ax.set_xlabel("Slot")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_comparison_bars(
    results: dict[str, dict],
    metric: str,
    save_path: Optional[str] = None,
) -> None:
    """Bar plot comparing policies on a single metric."""
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=["#4878cf", "#6acc65", "#d65f5f", "#b47cc7"])
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylim(0, max(values) * 1.2 + 0.1)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_training_curve(
    episode_rewards: list[float],
    window: int = 50,
    title: str = "PPO Training Curve",
    save_path: Optional[str] = None,
) -> None:
    """Plot raw episode rewards and a rolling-mean smoothing line."""
    rewards = np.array(episode_rewards)
    episodes = np.arange(1, len(rewards) + 1)

    # Rolling mean (valid region only)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        smooth_eps = episodes[window - 1:]
    else:
        smoothed = rewards
        smooth_eps = episodes

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, rewards, alpha=0.25, color="steelblue", linewidth=0.8,
            label="Episode reward")
    ax.plot(smooth_eps, smoothed, color="steelblue", linewidth=2,
            label=f"{window}-ep rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_task_completion_summary(
    policy_results: dict[str, list[dict]],
    title: str = "Task Completion Summary",
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar chart: completed / overdue-incomplete / not-done per policy.

    policy_results: {policy_name: [task_dict, ...]} — task list at episode end.
    """
    names = list(policy_results.keys())
    completed = []
    missed_deadline = []
    not_done = []

    for name in names:
        tasks = policy_results[name]
        c = sum(1 for t in tasks if t["done"])
        # Not done AND deadline was passed (overdue)
        md = sum(1 for t in tasks if not t["done"] and t.get("overdue", False))
        nd = sum(1 for t in tasks if not t["done"] and not t.get("overdue", False))
        completed.append(c)
        missed_deadline.append(md)
        not_done.append(nd)

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, completed, width, label="Completed", color="#6acc65")
    ax.bar(x, missed_deadline, width, label="Deadline missed", color="#d65f5f")
    ax.bar(x + width, not_done, width, label="Unfinished (no deadline miss)", color="#f0c66a")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Task count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Evaluation helpers (ported from hanzhewu/adhd-scheduling util.py)
# ---------------------------------------------------------------------------

_MPL_CACHE_DIR = Path("/tmp") / "adhd_scheduling_matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

POLICY_DISPLAY_NAMES: dict[str, str] = {
    "random": "Random",
    "edf": "EDF",
    "energy_aware": "Energy-Aware",
    "ppo": "PPO",
}


def display_policy_name(policy_name: str) -> str:
    """Return a presentation-friendly policy label."""
    return POLICY_DISPLAY_NAMES.get(policy_name, policy_name.replace("_", " ").title())


def get_pyplot():
    """Import Matplotlib in headless Agg mode with a writable cache directory."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt
    return _plt


def generate_eval_task_sets(
    num_episodes: int,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    """Generate reproducible task sets for multi-episode evaluation.

    Args:
        num_episodes: Number of simulated days to sample.
        seed: Random seed for task generation.

    Returns:
        List of one task set per episode.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    from task_generator import generate_task_set
    rng = np.random.default_rng(seed)
    return [generate_task_set(rng=rng) for _ in range(num_episodes)]


def is_urgent_task(task: dict[str, Any], num_slots: int = 16) -> bool:
    """Return True if a task's deadline falls in the first third of the day."""
    urgent_cutoff = max(1, num_slots // 3)
    return int(task["deadline"]) <= urgent_cutoff


def rollout_metrics(result: dict, num_slots: int = 16) -> dict[str, float | str]:
    """Compute evaluation metrics from one rollout result dict.

    Requires 'completed_task_id' to be present in trajectory rows for accurate
    urgent_completion_rate; otherwise urgent rate will be 0.

    Returns:
        Flat metric dict: policy, total_reward, completed_value, completed_tasks,
        urgent_completion_rate, rest_actions, missed_deadlines, invalid_actions,
        work_actions, total_actions.
    """
    final_tasks = result["final_tasks"]
    completion_slots: dict[int, int] = {
        int(row["completed_task_id"]): int(row["slot"])
        for row in result["trajectory"]
        if row.get("completed_task_id") is not None
    }
    urgent_tasks = [t for t in final_tasks if is_urgent_task(t, num_slots=num_slots)]
    urgent_completed = sum(
        1
        for t in urgent_tasks
        if int(t["id"]) in completion_slots
        and completion_slots[int(t["id"])] <= int(t["deadline"])
    )
    urgent_rate = urgent_completed / len(urgent_tasks) if urgent_tasks else 0.0

    trajectory = result["trajectory"]
    invalid_actions = sum(1 for row in trajectory if row.get("invalid_action", False))
    rest_actions = sum(1 for row in trajectory if row["action_label"] == "REST")
    work_actions = len(trajectory) - rest_actions - invalid_actions

    return {
        "policy": str(result["policy"]),
        "total_reward": float(result["total_reward"]),
        "completed_value": float(result["completed_value"]),
        "completed_tasks": float(result["completed_tasks"]),
        "urgent_completion_rate": float(urgent_rate),
        "rest_actions": float(result["rest_actions"]),
        "missed_deadlines": float(result["missed_deadline_count"]),
        "invalid_actions": float(invalid_actions),
        "work_actions": float(work_actions),
        "total_actions": float(len(trajectory)),
    }


def load_policy_suite(
    include_ppo: bool = True,
    model_path: str | Path = "models/ppo_scheduler.zip",
) -> tuple[list[Any], str | None]:
    """Load heuristic baselines and optionally append trained PPO.

    Returns:
        (policies, skipped_message). skipped_message is None when PPO was
        loaded or not requested.
    """
    from baselines import make_baseline_policies
    from train import load_trained_policy

    policies: list[Any] = list(make_baseline_policies())
    if not include_ppo:
        return policies, None
    try:
        policies.append(load_trained_policy(model_path=model_path, deterministic=True))
    except Exception as exc:
        return policies, f"PPO skipped: {exc}"
    return policies, None


def print_task_set(tasks: list[dict[str, Any]]) -> None:
    """Print a compact human-readable task list."""
    for task in tasks:
        urgency = "urgent" if is_urgent_task(task) else "normal"
        print(
            f"  T{task['id']}: priority={task['priority']} "
            f"difficulty={task['difficulty']:.2f} duration={task['duration']} "
            f"deadline={task['deadline']} {urgency}"
        )
