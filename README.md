# bianca_rl_summative — KigaliCare

Reinforcement learning for **care navigation in a digital mental-health support platform** (Kimironko pilot,
Gasabo District, Kigali). A coordinator agent looks after 5 pilot users for 90 days and must decide, one action
per day, whom to nudge, assess, book with a counsellor, or refer to specialist care, while keeping users engaged
and avoiding crisis events. Four algorithms are trained with **TorchRL** on the *same* environment:
**DQN** (value-based) and **REINFORCE, PPO, A2C** (policy gradient), 10 hyper-parameter configurations each.

> The environment is a simplified simulation for studying RL; its coefficients are modelling assumptions, not clinical estimates.

## Quick start (uv only)

```bash
git clone <your-repo-url> bianca_rl_summative && cd bianca_rl_summative
uv sync                 # creates .venv and installs everything from pyproject.toml / uv.lock
uv run main.py          # opens the GUI and replays the best trained agent, printing verbose output
```

Other commands:

```bash
uv run main.py play --algo ppo          # best run of one algorithm (dqn | reinforce | ppo | a2c)
uv run main.py play --random            # random baseline
uv run main.py play --headless          # no window, saves assets/demo.gif
uv run main.py train                    # re-run all 40 sweep runs (~1 h on 1 CPU core)
uv run main.py train ppo a2c            # or only some algorithms
uv run main.py report                   # rebuild tables (logs/) and figures (assets/)
uv run pytest tests                     # environment unit tests 
```

On Linux without a display use `uv run main.py play --headless`. Python 3.11-3.13 is selected automatically by uv
(`.python-version`).

## Project layout

```
main.py                 single entry point (play / train / report)
play.py                 shortcut for `main.py play`
environment/custom_env.py   KigaliCareEnv (Gymnasium): actions, observations, rewards, terminal rules
environment/rendering.py    pygame 2D visualisation (human window or rgb_array)
training/common.py      TorchRL env factory, networks, evaluation, logging, model loading
training/dqn_training.py    DQN (QValueModule, EGreedyModule, DQNLoss, replay buffer, soft/hard target updates)
training/pg_training.py     REINFORCE, PPO (ClipPPOLoss + GAE), A2C (A2CLoss + GAE)
training/sweeps.py      the 10 configurations per algorithm
training/run_sweeps.py  sweep runner      training/analysis.py  tables + figures
models/dqn, models/pg   saved checkpoints (+ models/best.json = best run per algorithm)
logs/                   per-run JSON logs, sweep_summary.csv, best_test_results.json
assets/                 figures and environment screenshots used in the report
```

## Environment summary

| Item | Definition |
|---|---|
| Actions (26, `Discrete`) | 5 users × {self-help content, check-in, self-assessment, book counsellor, refer to KMHRC} + no-op |
| Observation (33 floats in [0,1]) | per user: last-assessed severity, engagement, trust, days since contact, appointment pending, active; plus counsellor slots left, referral places left, day progress |
| Hidden state | true severity (only seen through noisy self-assessments), per-user responsiveness |
| Reward | +5 × severity reduction, −0.05 × total severity, −1 per dropout, −10 crisis, −0.2 invalid action, −0.5 needless referral, +1 per retained healthy user at day 90 |
| Start state | 5 active users, severity U(0.2, 0.8), engagement U(0.5, 0.9), trust U(0.4, 0.7), 3 slots, 2 referral places |
| Terminal | crisis (severity ≥ 0.95), all users dropped out; truncation at day 90 |

## Results (seed 0, 61,440 steps per run; best run per algorithm re-tested on 100 unseen seeds)

| Agent | Return | Users retained /5 | Crisis rate |
|---|---|---|---|
| PPO (run 10) | 5.65 | 3.5 | 0.02 |
| DQN (run 2) | 0.73 | 1.2 | 0.00 |
| Rule-based reference | -0.17 | 5.0 | 0.00 |
| A2C (run 1) | -3.99 | 1.1 | 0.04 |
| REINFORCE (run 4) | -7.11 | 1.1 | 0.03 |

One seed per configuration, so small differences may be noise. See the report for the full 4 × 10 sweep tables.
