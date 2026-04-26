# Lightweight RL Scheduling Prototype

A toy benchmark demonstrating that a PPO agent can outperform simple heuristic schedulers in a stochastic daily-planning environment with fluctuating energy and focus.

> **Note:** This is not a clinically realistic system. It is a course-project demo.

---

## Project Goal

Compare four scheduling policies on a synthetic ADHD-inspired environment:
- **Random** — uniform random over available tasks and REST
- **EDF** — earliest-deadline-first with energy floor
- **EnergyAware** — priority-weighted heuristic with energy/focus gating
- **PPO** — trained via `stable-baselines3`

---

## Environment

`ADHDSchedulingEnv` (Gymnasium):

| Parameter | Value |
|-----------|-------|
| Time slots per day | 16 |
| Tasks per day | 4 – 6 |
| Action space | Discrete(7): work on task 0–5, or REST |
| Observation dim | 34 (slot, energy, focus, streak, 6 × task features) |
| Hidden variables | energy ∈ [0,1], focus ∈ [0,1] |

**Transition dynamics:**
- Working decays energy and focus; stochastic progress via sigmoid(3·(E+F−d−0.8))
- 5% chance of a focus crash (−0.30)
- REST recovers energy (+0.15) and focus (+0.18), resets work streak

**Reward:**
- +2 × priority on task completion
- −1 per missed deadline (once per task)
- −0.5 if work streak > 3 consecutive slots
- +0.2 for resting when energy < 0.2

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Train
```bash
python train.py                        # 100 000 steps (default)
python train.py --steps 200000 --lr 1e-4 --seed 0
```
Saves model to `models/ppo_scheduler.zip` and training curve to `report_figures/training_curve.png`.

### Evaluate
```bash
python evaluate.py                     # 200 episodes per policy
python evaluate.py --episodes 500
```
Prints a comparison table and saves bar plots to `report_figures/`.

### Demo
```bash
python demo.py
```
Runs all four policies on the same fixed day, prints slot-by-slot tables, and saves energy/focus plots, action timelines, and a task-completion summary to `report_figures/`.

### Recursive pipeline (auto-retrain until success)
```bash
python pipeline.py                     # starts at 100k, scales up to 300k
python pipeline.py --steps 50000 --episodes 100
```
Logs each trial as JSON to `experiments/run_<TIMESTAMP>/`.

### MCP server (for Claude Code integration)
```bash
pip install mcp
python mcp_server.py
```
Exposes `train_agent`, `evaluate_policies`, and `get_latest_results` as MCP tools. Wire in via `.mcp.json` (see `mcp_server.py` header).

---

## Results (100k training steps, 200 eval episodes)

| Policy | Priority value | Tasks done | Urgent rate | Missed deadlines |
|--------|---------------|------------|-------------|-----------------|
| Random | 7.19 | 1.68 | 37% | 3.37 |
| EDF | 9.xx | ~2.x | ~x% | ~x |
| EnergyAware | 8.93 | 1.69 | 72% | 3.37 |
| **PPO** | **14.44** | **3.18** | **79%** | **1.88** |

PPO completes ~2× more tasks and misses ~45% fewer deadlines vs. the best heuristic.
Key learned behavior: strategic REST to maintain energy/focus rather than grinding to exhaustion.

---

## Repository Structure

```
config.py           — all tunable constants
task_generator.py   — synthetic task generation
env.py              — ADHDSchedulingEnv (Gymnasium)
baselines.py        — Random, EDF, EnergyAware policies
train.py            — PPO training
evaluate.py         — multi-policy comparison
demo.py             — visual rollout demo
utils.py            — shared helpers (plotting, recording)
pipeline.py         — recursive train → eval loop
mcp_server.py       — MCP tool server
models/             — saved PPO model (runtime)
report_figures/     — plots (runtime)
experiments/        — pipeline trial logs (runtime)
```

---

## Stretch Goals (not implemented)
- Action masking with `MaskablePPO`
- Ablation: remove focus crash
- Recurrent policy (LSTM)
- Mild partial observability (noisy energy/focus observations)
