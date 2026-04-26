# Central configuration — do not scatter magic numbers across files.

# Scheduling dimensions
NUM_SLOTS: int = 16
MIN_TASKS: int = 4
MAX_TASKS: int = 6
OBS_DIM: int = 4 + MAX_TASKS * 5   # = 34

# Action space
NUM_ACTIONS: int = MAX_TASKS + 1   # 0-5 = work on task i, 6 = REST
REST_ACTION: int = MAX_TASKS       # = 6

# Energy / focus dynamics (per work slot)
ENERGY_DECAY_BASE: float = 0.08
ENERGY_DECAY_PER_DIFF: float = 0.05
FOCUS_DECAY_BASE: float = 0.06
FOCUS_DECAY_PER_DIFF: float = 0.08

# Focus crash event
CRASH_PROB: float = 0.05
CRASH_FOCUS_PENALTY: float = 0.30

# REST recovery
REST_ENERGY_RECOVERY: float = 0.15
REST_FOCUS_RECOVERY: float = 0.18

# Reward coefficients
REWARD_TASK_COMPLETE_MULT: float = 2.0   # multiplied by priority
REWARD_MISSED_DEADLINE: float = -1.0    # once per task
REWARD_STREAK_PENALTY: float = -0.5
REWARD_STREAK_THRESHOLD: int = 3
REWARD_REST_LOW_ENERGY: float = 0.2
REWARD_INVALID_ACTION: float = -0.1
LOW_ENERGY_THRESHOLD: float = 0.2       # energy level that triggers rest reward

# PPO training
PPO_TRAIN_STEPS: int = 100_000
PPO_LEARNING_RATE: float = 3e-4

# Evaluation
EVAL_EPISODES: int = 200

# Paths
MODEL_PATH: str = "models/ppo_scheduler.zip"
FIGURES_DIR: str = "report_figures"
EXPERIMENTS_DIR: str = "experiments"

# Seeds
DEFAULT_SEED: int = 42
