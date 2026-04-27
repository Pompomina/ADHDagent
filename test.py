"""Manual smoke tests for Phase 1, 2, and 3 implementation pieces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines import make_baseline_policies, rollout_policy
from config import DEFAULT_SEED, MAX_TASKS, MAX_TASK_DURATION, MIN_TASKS, NUM_SLOTS, MODEL_PATH
from env import ADHDSchedulingEnv
from task_generator import generate_task_set
from train import load_trained_model, load_trained_policy, make_training_env
from evaluate import evaluate_policies, summarize_metrics
from demo import run_sample_day


def validate_phase1_contract(seed: int = DEFAULT_SEED) -> None:
    """Run small assertions for the Phase 1 API and data contract."""
    rng = np.random.default_rng(seed)
    tasks = generate_task_set(rng=rng)
    assert MIN_TASKS <= len(tasks) <= MAX_TASKS
    for task in tasks:
        assert {"id", "difficulty", "priority", "duration", "deadline", "remaining", "done"} <= set(task)
        assert 0.0 <= task["difficulty"] <= 1.0
        assert task["priority"] in {1, 2, 3}
        assert 1 <= task["duration"] <= MAX_TASK_DURATION
        assert 0 <= task["deadline"] < NUM_SLOTS
        assert task["remaining"] == task["duration"]
        assert task["done"] is False

    env = ADHDSchedulingEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)

    padded_task_action = len(env.tasks)
    if padded_task_action < env.rest_action:
        _, _, _, _, info = env.step(padded_task_action)
        assert info["invalid_action"] is True
        assert info["action_type"] == "invalid_as_rest"

    print("validate_phase1_contract: PASS")


def run_random_rollout(seed: int = DEFAULT_SEED) -> None:
    """Run one random episode and print a compact trajectory."""
    env = ADHDSchedulingEnv(seed=seed)
    obs, info = env.reset(seed=seed)
    print("Initial tasks:")
    for task in info["tasks"]:
        print(
            f"  T{task['id']}: priority={task['priority']} difficulty={task['difficulty']:.2f} "
            f"duration={task['duration']} deadline={task['deadline']}"
        )
    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: 0-{env.rest_action - 1} task actions, {env.rest_action}=REST\n")

    done = False
    total_reward = 0.0
    step = 0
    while not done:
        action = int(env._rng.integers(0, env.action_space.n))
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_label = "REST" if action == env.rest_action else f"T{action}"
        print(
            f"slot={step:02d} action={action_label:<4} reward={reward:5.2f} "
            f"energy={info['energy']:.2f} focus={info['focus']:.2f} "
            f"completed={info['completed_tasks']} missed={info['missed_deadline_count']}"
        )
        done = terminated or truncated
        step += 1

    print(f"\nFinal total reward: {total_reward:.2f}")
    env.render()


def validate_phase2_baselines(seed: int = DEFAULT_SEED) -> None:
    """Validate that each Phase 2 baseline can complete one episode.

    The policies are run on the same fixed task set so differences come from
    policy choices rather than task sampling. Assertions check that rollouts
    stay inside the action space, avoid invalid padded/completed task actions,
    and keep reported state values within expected bounds.
    """
    rng = np.random.default_rng(seed)
    fixed_tasks = generate_task_set(rng=rng)
    policies = make_baseline_policies()

    for policy in policies:
        env = ADHDSchedulingEnv(seed=seed, tasks=fixed_tasks)
        result = rollout_policy(env, policy, seed=seed)

        assert result["policy"] == policy.name
        assert 1 <= result["steps"] <= NUM_SLOTS
        assert 0 <= result["completed_tasks"] <= len(fixed_tasks)
        assert 0 <= result["completed_value"] <= sum(t["priority"] for t in fixed_tasks)
        assert 0 <= result["rest_actions"] <= result["steps"]
        for row in result["trajectory"]:
            assert 0 <= row["slot"] < NUM_SLOTS
            assert env.action_space.contains(row["action"])
            assert row["invalid_action"] is False
            assert 0.0 <= row["energy"] <= 1.0
            assert 0.0 <= row["focus"] <= 1.0

    print("validate_phase2_baselines: PASS")


def validate_phase3_training_interface(seed: int = DEFAULT_SEED) -> None:
    """Validate Phase 3 training helpers without requiring a long PPO run.

    Checks that the training environment factory returns a usable env, and that
    load_trained_model works when a model is available.
    """
    env = make_training_env(seed=seed, use_monitor=False)
    obs, _ = env.reset(seed=seed)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert Path(MODEL_PATH).name == "ppo_scheduler.zip"

    if Path(MODEL_PATH).exists():
        try:
            model = load_trained_model(MODEL_PATH, env=env)
        except ModuleNotFoundError:
            model = None
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            assert env.action_space.contains(int(action))

    print("validate_phase3_training_interface: PASS")


def print_policy_rollout(result: dict) -> None:
    """Print one rollout using the shared baseline/PPO table format."""
    print(
        f"\nPolicy={result['policy']} total_reward={result['total_reward']:.2f} "
        f"completed={result['completed_tasks']} value={result['completed_value']} "
        f"rests={result['rest_actions']} missed={result['missed_deadline_count']}"
        f" invalid={result['invalid_actions']}"
    )
    for row in result["trajectory"]:
        missed = f" missed={row['missed_deadlines']}" if row["missed_deadlines"] else ""
        success = "*" if row["task_success"] else " "
        print(
            f"  slot={row['slot']:02d} action={row['action_label']:<4}{success} "
            f"reward={row['reward']:5.2f} energy={row['energy']:.2f} "
            f"focus={row['focus']:.2f} completed={row['completed_tasks']}{missed}"
        )


def run_policy_rollouts(seed: int = DEFAULT_SEED) -> None:
    """Print human-readable trajectories for baselines and trained PPO.

    Shows the shared task set, each policy's aggregate outcome, and per-slot
    details. If the trained PPO model is unavailable, the PPO row is skipped.
    """
    rng = np.random.default_rng(seed)
    fixed_tasks = generate_task_set(rng=rng)

    print("\nShared task set:")
    for task in fixed_tasks:
        print(
            f"  T{task['id']}: priority={task['priority']} difficulty={task['difficulty']:.2f} "
            f"duration={task['duration']} deadline={task['deadline']}"
        )

    for policy in make_baseline_policies():
        env = ADHDSchedulingEnv(seed=seed, tasks=fixed_tasks)
        result = rollout_policy(env, policy, seed=seed)
        print_policy_rollout(result)

    ppo_env = ADHDSchedulingEnv(seed=seed, tasks=fixed_tasks)
    try:
        ppo_policy = load_trained_policy(MODEL_PATH, env=ppo_env, deterministic=True)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"\nPolicy=ppo skipped: {exc}")
    else:
        print_policy_rollout(rollout_policy(ppo_env, ppo_policy, seed=seed))


def validate_phase4_evaluation_and_demo(seed: int = DEFAULT_SEED) -> None:
    """Fast 2-episode evaluation and demo smoke check."""
    rows, _ = evaluate_policies(num_episodes=2, seed=seed, include_ppo=False)
    assert len(rows) >= 2 * 3, f"expected ≥6 rows, got {len(rows)}"
    summaries = summarize_metrics(rows)
    assert len(summaries) == 3
    for s in summaries:
        assert "avg_total_reward" in s
        assert "avg_urgent_completion_rate" in s
        assert float(s["avg_urgent_completion_rate"]) >= 0.0

    tasks, results, _ = run_sample_day(seed=seed, include_ppo=False)
    assert len(tasks) >= MIN_TASKS
    assert len(results) == 3

    print("validate_phase4_evaluation_and_demo: PASS")


if __name__ == "__main__":
    validate_phase1_contract()
    validate_phase2_baselines()
    validate_phase3_training_interface()
    validate_phase4_evaluation_and_demo()
    run_random_rollout()
    run_policy_rollouts()
