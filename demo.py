"""Generate a human-readable sample-day demo for scheduling policies."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from baselines import rollout_policy
from config import DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR
from env import ADHDSchedulingEnv
from task_generator import generate_task_set
from utils import (
    display_policy_name,
    ensure_dir,
    get_pyplot,
    load_policy_suite,
    print_task_set,
)


def run_sample_day(
    seed: int = DEFAULT_SEED,
    include_ppo: bool = True,
) -> tuple[list[dict[str, Any]], list[dict], str | None]:
    """Run all available policies on one shared sampled day.

    Returns:
        (tasks, rollout_results, skipped_ppo_message)
    """
    rng = np.random.default_rng(seed)
    tasks = generate_task_set(rng=rng)
    policies, skipped = load_policy_suite(include_ppo=include_ppo)
    results: list[dict] = []
    for idx, policy in enumerate(policies):
        env = ADHDSchedulingEnv(seed=seed + idx, tasks=deepcopy(tasks))
        results.append(rollout_policy(env, policy, seed=seed + idx))
    return tasks, results, skipped


def save_sample_day_rollouts_csv(results: list[dict], output_path: str | Path) -> Path:
    """Save slot-level rollout records for all policies to CSV."""
    path = Path(output_path)
    ensure_dir(path.parent)
    fieldnames = [
        "policy", "slot", "action", "action_label", "energy", "focus",
        "reward", "cumulative_reward", "completed_tasks", "completed_value",
        "missed_deadline_count", "invalid_action", "completed_task_id",
    ]
    rows = []
    for result in results:
        cumulative_reward = 0.0
        for row in result["trajectory"]:
            cumulative_reward += float(row["reward"])
            rows.append({
                "policy": result["policy"],
                "slot": row["slot"],
                "action": row["action"],
                "action_label": row["action_label"],
                "energy": row["energy"],
                "focus": row["focus"],
                "reward": row["reward"],
                "cumulative_reward": cumulative_reward,
                "completed_tasks": row["completed_tasks"],
                "completed_value": row["completed_value"],
                "missed_deadline_count": row["missed_deadline_count"],
                "invalid_action": row["invalid_action"],
                "completed_task_id": row.get("completed_task_id"),
            })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_demo_summary(
    tasks: list[dict[str, Any]],
    results: list[dict],
    skipped: str | None,
) -> None:
    """Print the task set and compact policy outcomes."""
    print("\nSample-day tasks:")
    print_task_set(tasks)
    print("\nSample-day policy summary")
    print(f"{'Policy':<16} {'Reward':>8} {'Value':>8} {'Tasks':>8} {'REST':>8} {'Missed':>8}")
    print("-" * 64)
    for result in results:
        print(
            f"{display_policy_name(result['policy']):<16} "
            f"{result['total_reward']:8.2f} "
            f"{result['completed_value']:8d} "
            f"{result['completed_tasks']:8d} "
            f"{result['rest_actions']:8d} "
            f"{result['missed_deadline_count']:8d}"
        )
    if skipped:
        print(skipped)


def save_action_timeline(results: list[dict], output_path: str | Path) -> Path:
    """Save a horizontal timeline showing actions by slot and policy."""
    plt = get_pyplot()
    max_steps = max(result["steps"] for result in results)
    fig_height = max(2.5, 0.65 * len(results) + 1.2)
    fig, ax = plt.subplots(figsize=(max(8, max_steps * 0.55), fig_height))

    color_map = {"REST": "#72B7B2", "invalid": "#BAB0AC", "work": "#4C78A8"}
    for y, result in enumerate(results):
        for row in result["trajectory"]:
            label = row["action_label"]
            if row["invalid_action"]:
                color = color_map["invalid"]
            elif label == "REST":
                color = color_map["REST"]
            else:
                color = color_map["work"]
            ax.barh(y, 1, left=row["slot"], height=0.62, color=color, edgecolor="white")
            ax.text(row["slot"] + 0.5, y, label, ha="center", va="center", fontsize=8)

    ax.set_yticks(range(len(results)), [display_policy_name(r["policy"]) for r in results])
    ax.set_xticks(range(max_steps))
    ax.set_xlabel("Time slot")
    ax.set_title("Sample-Day Action Timeline")
    ax.set_xlim(0, max_steps)
    ax.invert_yaxis()
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_state_trace_plot(results: list[dict], state_key: str, output_path: str | Path) -> Path:
    """Save a line plot comparing energy or focus over the sample day."""
    plt = get_pyplot()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    title_key = state_key.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, result in enumerate(results):
        trajectory = result["trajectory"]
        slots = [row["slot"] for row in trajectory]
        values = [row[state_key] for row in trajectory]
        ax.plot(slots, values, marker="o", linewidth=2,
                label=display_policy_name(result["policy"]),
                color=colors[idx % len(colors)])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time slot")
    ax.set_ylabel(title_key)
    ax.set_title(f"{title_key} by Policy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_completion_summary(results: list[dict], output_path: str | Path) -> Path:
    """Save a grouped bar chart of completed value and missed deadlines."""
    plt = get_pyplot()
    labels = [display_policy_name(r["policy"]) for r in results]
    values = [r["completed_value"] for r in results]
    missed = [r["missed_deadline_count"] for r in results]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, values, width, label="Completed value", color="#54A24B")
    ax.bar(x + width / 2, missed, width, label="Missed deadlines", color="#E45756")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count / value")
    ax.set_title("Sample-Day Completion Summary")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_demo_plots(results: list[dict], figure_dir: str | Path = FIGURES_DIR) -> list[Path]:
    """Save all four sample-day demo figures."""
    fd = ensure_dir(figure_dir)
    return [
        save_action_timeline(results, fd / "fig_sample_day_timeline.png"),
        save_state_trace_plot(results, "energy", fd / "fig_energy_by_policy.png"),
        save_state_trace_plot(results, "focus", fd / "fig_focus_by_policy.png"),
        save_completion_summary(results, fd / "fig_sample_day_completion.png"),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a sample-day scheduling demo.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--figure-dir", type=str, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-ppo", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    tasks, results, skipped = run_sample_day(seed=args.seed, include_ppo=not args.no_ppo)
    print_demo_summary(tasks, results, skipped)

    if not args.no_csv:
        path = save_sample_day_rollouts_csv(
            results, Path(args.results_dir) / "sample_day_rollouts.csv"
        )
        print(f"\nSaved demo CSV:\n  {path}")

    if not args.no_plots:
        paths = save_demo_plots(results, figure_dir=args.figure_dir)
        print("\nSaved demo plots:")
        for p in paths:
            print(f"  {p}")
