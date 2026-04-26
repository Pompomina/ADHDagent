"""Human-readable rollout demo: run all policies on the same day."""

from __future__ import annotations

import os

from stable_baselines3 import PPO

from config import MODEL_PATH, FIGURES_DIR, DEFAULT_SEED, REST_ACTION
from env import ADHDSchedulingEnv
from baselines import RandomPolicy, EDFPolicy, EnergyAwarePolicy
from utils import (
    RolloutRecorder, plot_energy_focus, plot_action_timeline,
    plot_task_completion_summary, pretty_print_tasks, ensure_dir,
)


def run_rollout(policy, env: ADHDSchedulingEnv) -> RolloutRecorder:
    """Single deterministic episode. Returns populated RolloutRecorder."""
    is_ppo = hasattr(policy, "predict")
    obs, _ = env.reset(seed=DEFAULT_SEED)
    recorder = RolloutRecorder()
    done = False

    while not done:
        slot = env.slot
        energy = env.energy
        focus = env.focus

        if is_ppo:
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = policy.select_action(obs, env)

        obs, reward, terminated, truncated, _ = env.step(action)
        recorder.record(slot, action, reward, energy, focus,
                        [dict(t) for t in env.tasks])
        done = terminated or truncated

    return recorder


def print_rollout_table(recorder: RolloutRecorder, policy_name: str) -> None:
    df = recorder.to_dataframe()
    total = df["reward"].sum()
    print(f"\n{'='*60}")
    print(f"  {policy_name}  —  total reward: {total:.2f}")
    print(f"{'='*60}")
    print(f"{'Slot':>5} {'Action':>8} {'Reward':>8} {'Energy':>8} {'Focus':>8}")
    print("-" * 45)
    for _, row in df.iterrows():
        action_label = "REST" if row["action"] == REST_ACTION else f"Task {int(row['action'])}"
        print(
            f"{int(row['slot']):>5} {action_label:>8} "
            f"{row['reward']:>8.2f} {row['energy']:>8.2f} {row['focus']:>8.2f}"
        )


def _task_completion_snapshot(env: ADHDSchedulingEnv) -> list[dict]:
    """Return task list annotated with overdue flag for summary plot."""
    return [
        {**t, "overdue": not t["done"] and env.slot > t["deadline"]}
        for t in env.tasks
    ]


def run_demo() -> None:
    ensure_dir(FIGURES_DIR)

    # All policies share the same env instance (same reset seed = same day)
    env = ADHDSchedulingEnv(seed=DEFAULT_SEED)
    ppo_model = PPO.load(MODEL_PATH)

    policies: dict[str, object] = {
        "Random": RandomPolicy(seed=DEFAULT_SEED),
        "EDF": EDFPolicy(),
        "EnergyAware": EnergyAwarePolicy(),
        "PPO": ppo_model,
    }

    # Print the day's task set (same for all policies due to fixed seed)
    env.reset(seed=DEFAULT_SEED)
    print("\n=== Today's Tasks ===")
    pretty_print_tasks(env.tasks)

    completion_snapshots: dict[str, list[dict]] = {}

    for name, pol in policies.items():
        recorder = run_rollout(pol, env)
        print_rollout_table(recorder, name)
        completion_snapshots[name] = _task_completion_snapshot(env)

        plot_energy_focus(
            recorder,
            title=f"{name} — Energy & Focus over Time",
            save_path=os.path.join(FIGURES_DIR, f"demo_{name}_energy.png"),
        )
        plot_action_timeline(
            recorder,
            title=f"{name} — Action Timeline",
            save_path=os.path.join(FIGURES_DIR, f"demo_{name}_actions.png"),
        )

    # Task completion summary across all policies
    summary_path = os.path.join(FIGURES_DIR, "demo_task_completion_summary.png")
    plot_task_completion_summary(
        completion_snapshots,
        title="Task Completion Summary (same day, all policies)",
        save_path=summary_path,
    )
    print(f"\nDemo figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    run_demo()
