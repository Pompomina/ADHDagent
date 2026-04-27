"""Evaluate baseline and PPO policies over many simulated scheduling days."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from statistics import mean

import numpy as np

from baselines import rollout_policy
from config import DEFAULT_SEED, EVAL_EPISODES, FIGURES_DIR, RESULTS_DIR
from env import ADHDSchedulingEnv
from utils import (
    display_policy_name,
    ensure_dir,
    generate_eval_task_sets,
    get_pyplot,
    load_policy_suite,
    rollout_metrics,
)


def evaluate_policies(
    num_episodes: int = EVAL_EPISODES,
    seed: int = DEFAULT_SEED,
    include_ppo: bool = True,
) -> tuple[list[dict], str | None]:
    """Run each policy on a reproducible set of sampled days.

    Returns:
        (per_episode_rows, skipped_ppo_message)
    """
    task_sets = generate_eval_task_sets(num_episodes=num_episodes, seed=seed)
    policies, skipped = load_policy_suite(include_ppo=include_ppo)
    rows: list[dict] = []

    for episode_idx, tasks in enumerate(task_sets):
        for policy_idx, policy in enumerate(policies):
            env_seed = seed + 1000 * episode_idx + policy_idx
            env = ADHDSchedulingEnv(seed=env_seed, tasks=deepcopy(tasks))
            result = rollout_policy(env, policy, seed=env_seed)
            metrics = rollout_metrics(result, num_slots=env.num_slots)
            metrics["episode"] = float(episode_idx)
            rows.append(metrics)

    return rows, skipped


def summarize_metrics(rows: list[dict]) -> list[dict]:
    """Aggregate per-episode rows by policy: mean, std, se."""
    if not rows:
        return []
    policy_order = list(dict.fromkeys(str(row["policy"]) for row in rows))
    metric_names = [
        "total_reward", "completed_value", "completed_tasks",
        "urgent_completion_rate", "rest_actions", "missed_deadlines",
        "invalid_actions", "work_actions", "total_actions",
    ]
    summaries = []
    for policy in policy_order:
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary: dict = {"policy": policy}
        for m in metric_names:
            values = [float(row[m]) for row in policy_rows]
            summary[f"avg_{m}"] = float(mean(values))
            summary[f"std_{m}"] = float(np.std(values))
            summary[f"se_{m}"] = float(np.std(values) / np.sqrt(len(values)))
        summaries.append(summary)
    return summaries


def print_summary_table(summaries: list[dict]) -> None:
    """Print main metrics as a console table."""
    headers = ["Policy", "Reward", "Value", "Tasks", "Urgent", "REST", "Missed", "Invalid"]
    print("\nEvaluation summary")
    print(
        f"{headers[0]:<16} {headers[1]:>8} {headers[2]:>8} {headers[3]:>8} "
        f"{headers[4]:>8} {headers[5]:>8} {headers[6]:>8} {headers[7]:>8}"
    )
    print("-" * 82)
    for row in summaries:
        print(
            f"{display_policy_name(str(row['policy'])):<16} "
            f"{float(row['avg_total_reward']):8.2f} "
            f"{float(row['avg_completed_value']):8.2f} "
            f"{float(row['avg_completed_tasks']):8.2f} "
            f"{float(row['avg_urgent_completion_rate']):8.2f} "
            f"{float(row['avg_rest_actions']):8.2f} "
            f"{float(row['avg_missed_deadlines']):8.2f} "
            f"{float(row['avg_invalid_actions']):8.2f}"
        )


def _metric_ylim(summaries: list[dict], metric: str, hard_max: float | None = None) -> tuple[float, float]:
    values = [float(row[metric]) for row in summaries]
    max_val = max(values, default=0.0)
    upper = 0.1 if max_val <= 0 else max_val * 1.2
    if hard_max is not None:
        upper = min(hard_max, upper)
    return 0.0, max(0.05, upper)


def save_bar_plot(
    summaries: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
    ylim: tuple[float, float] | None = None,
    show_values: bool = False,
    error_metric: str | None = None,
) -> Path:
    plt = get_pyplot()
    labels = [display_policy_name(str(row["policy"])) for row in summaries]
    values = [float(row[metric]) for row in summaries]
    errors = [float(row[error_metric]) for row in summaries] if error_metric else None

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        labels, values,
        yerr=errors, capsize=4 if errors else 0,
        color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(labels)],
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_values:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_box_plot(
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> Path:
    plt = get_pyplot()
    policy_order = list(dict.fromkeys(str(row["policy"]) for row in rows))
    labels = [display_policy_name(p) for p in policy_order]
    values = [
        [float(row[metric]) for row in rows if row["policy"] == policy]
        for policy in policy_order
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    boxes = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(labels)]
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_rest_missed_plot(summaries: list[dict], output_path: str | Path) -> Path:
    plt = get_pyplot()
    labels = [display_policy_name(str(row["policy"])) for row in summaries]
    rests = [float(row["avg_rest_actions"]) for row in summaries]
    missed = [float(row["avg_missed_deadlines"]) for row in summaries]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, rests, width, label="REST actions", color="#4C78A8")
    ax.bar(x + width / 2, missed, width, label="Missed deadlines", color="#E45756")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Average count per episode")
    ax.set_title("REST Actions and Missed Deadlines")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_action_distribution_plot(summaries: list[dict], output_path: str | Path) -> Path:
    plt = get_pyplot()
    labels = [display_policy_name(str(row["policy"])) for row in summaries]
    work, rest, invalid = [], [], []
    for row in summaries:
        total = max(float(row["avg_total_actions"]), 1.0)
        work.append(float(row["avg_work_actions"]) / total)
        rest.append(float(row["avg_rest_actions"]) / total)
        invalid.append(float(row["avg_invalid_actions"]) / total)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, work, label="Work", color="#4C78A8")
    ax.bar(x, rest, bottom=work, label="REST", color="#72B7B2")
    ax.bar(x, invalid, bottom=np.array(work) + np.array(rest), label="Invalid", color="#BAB0AC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of actions")
    ax.set_title("Action Distribution by Policy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_csv(rows: list[dict], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {output_path}")
    path = Path(output_path)
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_evaluation_csvs(
    rows: list[dict],
    summaries: list[dict],
    results_dir: str | Path = RESULTS_DIR,
) -> list[Path]:
    d = ensure_dir(results_dir)
    return [
        write_csv(rows, d / "evaluation_episodes.csv"),
        write_csv(summaries, d / "evaluation_summary.csv"),
    ]


def save_evaluation_plots(
    rows: list[dict],
    summaries: list[dict],
    figure_dir: str | Path = FIGURES_DIR,
) -> list[Path]:
    fd = ensure_dir(figure_dir)
    return [
        save_bar_plot(summaries, "avg_completed_value", "Average Completed Task Value",
                      "Average Priority-Weighted Completed Value", fd / "fig_avg_completed_value.png"),
        save_bar_plot(summaries, "avg_urgent_completion_rate", "Urgent Task Completion Rate",
                      "Urgent Task Completion Rate by Policy", fd / "fig_urgent_completion_rate.png",
                      ylim=_metric_ylim(summaries, "avg_urgent_completion_rate", hard_max=1.0),
                      show_values=True),
        save_bar_plot(summaries, "avg_completed_tasks", "Average Completed Tasks",
                      "Average Completed Tasks by Policy", fd / "fig_completed_tasks.png"),
        save_bar_plot(summaries, "avg_total_reward", "Average Final Reward",
                      "Average Final Reward by Policy", fd / "fig_avg_total_reward.png",
                      show_values=True),
        save_bar_plot(summaries, "avg_invalid_actions", "Average Invalid Actions",
                      "Invalid Actions by Policy", fd / "fig_invalid_actions.png", show_values=True),
        save_rest_missed_plot(summaries, fd / "fig_rest_and_missed.png"),
        save_box_plot(rows, "total_reward", "Final reward",
                      "Final Reward Distribution by Policy", fd / "fig_reward_distribution.png"),
        save_box_plot(rows, "completed_value", "Completed task value",
                      "Completed Value Distribution by Policy",
                      fd / "fig_completed_value_distribution.png"),
        save_action_distribution_plot(summaries, fd / "fig_action_distribution.png"),
    ]


# ---------------------------------------------------------------------------
# Backward-compatible wrapper used by pipeline.py and mcp_server.py
# ---------------------------------------------------------------------------

def evaluate_all(
    n_episodes: int = EVAL_EPISODES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict]:
    """Evaluate all policies and return a dict keyed by policy name.

    Kept for backward compatibility with pipeline.py and mcp_server.py.
    Keys: priority_weighted_value, tasks_completed, urgent_completion_rate,
          rest_count, missed_deadlines.
    """
    rows, _ = evaluate_policies(num_episodes=n_episodes, seed=seed)
    summaries = summarize_metrics(rows)
    print_summary_table(summaries)

    result: dict[str, dict] = {}
    for s in summaries:
        name = str(s["policy"])
        result[name] = {
            "priority_weighted_value": float(s["avg_completed_value"]),
            "tasks_completed": float(s["avg_completed_tasks"]),
            "urgent_completion_rate": float(s["avg_urgent_completion_rate"]),
            "rest_count": float(s["avg_rest_actions"]),
            "missed_deadlines": float(s["avg_missed_deadlines"]),
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate scheduling policies.")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--figure-dir", type=str, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-ppo", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    _rows, _skipped = evaluate_policies(
        num_episodes=args.episodes,
        seed=args.seed,
        include_ppo=not args.no_ppo,
    )
    _summaries = summarize_metrics(_rows)
    print_summary_table(_summaries)
    if _skipped:
        print(_skipped)

    if not args.no_csv:
        paths = save_evaluation_csvs(_rows, _summaries, results_dir=args.results_dir)
        print("\nSaved CSVs:")
        for p in paths:
            print(f"  {p}")

    if not args.no_plots:
        paths = save_evaluation_plots(_rows, _summaries, figure_dir=args.figure_dir)
        print("\nSaved plots:")
        for p in paths:
            print(f"  {p}")
