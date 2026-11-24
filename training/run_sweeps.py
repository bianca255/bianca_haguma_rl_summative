"""Run all (or selected) sweeps:  uv run python -m training.run_sweeps [dqn reinforce ppo a2c]"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
from training.sweeps import SWEEPS


def run(algos=None, start=1):
    algos = algos or list(SWEEPS)
    for algo in algos:
        if algo == "dqn":
            from training.dqn_training import train
            fn = lambda i, kw: train(i, **kw)
        else:
            from training.pg_training import train
            fn = lambda i, kw, a=algo: train(a, i, **kw)
        for i, kw in enumerate(SWEEPS[algo], start=1):
            if i < start:
                continue
            d = fn(i, kw)
            print(f"[{algo}] run {i:02d} {kw} -> eval {d['final_eval']['mean']:.2f} "
                  f"(retained {d['final_eval']['retained']:.1f}, crisis {d['final_eval']['crisis_rate']:.2f}) "
                  f"{d['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or None)
