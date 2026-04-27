# Central configuration — do not scatter magic numbers across files.

# Scheduling dimensions
NUM_SLOTS: int = 16
MIN_TASKS: int = 4
MAX_TASKS: int = 6
OBS_DIM: int = 4 + MAX_TASKS * 5   # = 34

# Task attribute ranges
MIN_TASK_DURATION: int = 1
MAX_TASK_DURATION: int = 3
MIN_PRIORITY: int = 1
MAX_PRIORITY: int = 3

# Action space
NUM_ACTIONS: int = MAX_TASKS + 1   # 0-5 = work on task i, 6 = REST
REST_ACTION: int = MAX_TASKS       # = 6

# Initial state — randomised per episode for richer training distribution
INITIAL_ENERGY_RANGE: tuple = (0.65, 0.95)
INITIAL_FOCUS_RANGE:  tuple = (0.60, 0.90)

# Energy / focus dynamics (per work slot)
ENERGY_DECAY_BASE: float = 0.08
ENERGY_DECAY_PER_DIFF: float = 0.05
FOCUS_DECAY_BASE: float = 0.06
FOCUS_DECAY_PER_DIFF: float = 0.08

# Focus crash event
CRASH_PROB: float = 0.08
CRASH_FOCUS_PENALTY: float = 0.30

# Success probability formula: sigmoid(SCALE * (energy + focus - difficulty - OFFSET))
SUCCESS_SIGMOID_SCALE: float = 3.0
SUCCESS_DIFFICULTY_OFFSET: float = 0.8

# REST recovery
REST_ENERGY_RECOVERY: float = 0.15
REST_FOCUS_RECOVERY: float = 0.18

# Reward coefficients
REWARD_TASK_COMPLETE_MULT: float = 2.0   # multiplied by priority
REWARD_MISSED_DEADLINE: float = -1.5    # once per task
REWARD_STREAK_PENALTY: float = -0.5
REWARD_STREAK_THRESHOLD: int = 3
REWARD_REST_LOW_ENERGY: float = 0.2
REWARD_INVALID_ACTION: float = -0.25
LOW_ENERGY_THRESHOLD: float = 0.30      # energy level that triggers rest reward
STEP_PENALTY: float = -0.01            # per-step time cost

# Baseline-specific thresholds (derived from above so they stay consistent)
EDF_REST_ENERGY_THRESHOLD: float = LOW_ENERGY_THRESHOLD * 0.8  # ~0.24
EA_ENERGY_THRESHOLD: float = LOW_ENERGY_THRESHOLD               # 0.30
EA_FOCUS_THRESHOLD: float = 0.30
EA_DIFFICULTY_MARGIN: float = 0.20
EA_PRIORITY_WEIGHT: float = 2.0
EA_DIFFICULTY_WEIGHT: float = 1.0
EA_DEADLINE_WEIGHT: float = 0.1

# PPO training
PPO_TRAIN_STEPS: int = 100_000
PPO_LEARNING_RATE: float = 3e-4
PPO_GAMMA: float = 0.98
PPO_POLICY: str = "MlpPolicy"
PPO_N_STEPS: int = 512        # smaller than SB3 default 2048; better credit assignment for ~12-step episodes
PPO_VERBOSE: int = 1

# Evaluation
EVAL_EPISODES: int = 300

# Paths
MODEL_PATH: str = "models/ppo_scheduler.zip"
FIGURES_DIR: str = "report_figures"
RESULTS_DIR: str = "results"
EXPERIMENTS_DIR: str = "experiments"

# Seeds
DEFAULT_SEED: int = 42
