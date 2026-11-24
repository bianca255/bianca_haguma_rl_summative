"""KigaliCare RL summative - single entry point.

    uv run main.py                      # watch the BEST trained agent (GUI + verbose terminal output)
    uv run main.py play --algo ppo      # best run of a specific algorithm
    uv run main.py play --random        # random baseline
    uv run main.py play --headless      # no window; saves assets/demo.gif
    uv run main.py train [dqn reinforce ppo a2c]   # run the 10-config sweeps (slow)
    uv run main.py report               # regenerate tables + figures in assets/ and logs/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def best_checkpoint(algo=None):
    best = json.load(open(os.path.join(ROOT, "models", "best.json")))
    if algo:
        return os.path.join(ROOT, best[algo]["path"]), algo
    # overall best = highest independent test return
    test = json.load(open(os.path.join(ROOT, "logs", "best_test_results.json")))
    a = max(best, key=lambda k: test[k]["test"]["mean"])
    return os.path.join(ROOT, best[a]["path"]), a


def play(args):
    import numpy as np
    import torch
    from environment.custom_env import KigaliCareEnv, N_ACTIONS, decode_action, INTERVENTION_NAMES
    headless = args.headless
    env = KigaliCareEnv(render_mode="rgb_array" if headless else "human")
    if args.random:
        label, act = "random", lambda o: int(env.action_space.sample())
    else:
        from training.common import load_agent
        path, algo = (args.model, args.algo or "model") if args.model else best_checkpoint(args.algo)
        algo, act_fn, ck = load_agent(path)
        label = f"{algo.upper()} ({os.path.relpath(path, ROOT)})"
        act = lambda o: act_fn(torch.as_tensor(o))
    print(f"\n=== Agent: {label} ===")
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, total, t = False, 0.0, 0
        print(f"\n--- Episode {ep + 1} (seed {args.seed + ep}) ---")
        while not done:
            a = act(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r; t += 1; done = term or trunc
            print(f"day {t:02d} | action {a:02d} {env.last_action_text:<34} | reward {r:+6.2f} | return {total:+7.2f} | "
                  f"retained {info['retained']}/5 | mean severity {info['mean_sev']:.2f}")
            if headless and (t % 2 == 0 or done):
                frames.append(env.render())
        end = "CRISIS (terminated)" if info["crisis"] else "all dropped out" if info["retained"] == 0 else "90-day horizon reached"
        print(f"Episode ended: {end} | return {total:+.2f} | users retained {info['retained']}/5")
    if headless and frames:
        import imageio
        os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
        out = os.path.join(ROOT, "assets", "demo.gif")
        imageio.mimsave(out, frames[::2], duration=0.25)
        print(f"Saved {out}")
    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    pl = sub.add_parser("play")
    for q in (p, pl):
        q.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"])
        q.add_argument("--model")
        q.add_argument("--random", action="store_true")
        q.add_argument("--headless", action="store_true")
        q.add_argument("--episodes", type=int, default=1)
        q.add_argument("--seed", type=int, default=123)
    tr = sub.add_parser("train"); tr.add_argument("algos", nargs="*")
    sub.add_parser("report")
    args = p.parse_args()
    if args.cmd == "train":
        from training.run_sweeps import run
        run(args.algos or None)
    elif args.cmd == "report":
        from training.analysis import main as rep
        rep()
    else:
        play(args)


if __name__ == "__main__":
    main()
