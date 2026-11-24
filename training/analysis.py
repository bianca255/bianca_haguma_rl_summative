"""Aggregate logs -> tables (CSV/JSON) and report figures in assets/.  uv run main.py report"""
from __future__ import annotations

import glob
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
from training.common import ROOT, LOG_DIR, MODEL_DIR, evaluate, load_agent
from training.sweeps import SWEEPS
from environment.custom_env import KigaliCareEnv, N_ACTIONS, NOOP

ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COL = {"dqn": "#3470be", "reinforce": "#c84040", "ppo": "#2ea06e", "a2c": "#e8a828"}


def load_runs():
    runs = {}
    for a in ALGOS:
        runs[a] = [json.load(open(f)) for f in sorted(glob.glob(str(LOG_DIR / a / "run_*.json")))]
    return runs


def smooth(y, k=15):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(k - 1, c[0]), c])


def curve(run, n=60):
    """Training return vs frames, binned."""
    fr = np.array([f for f, _ in run["ep_returns"]]); v = np.array([r for _, r in run["ep_returns"]])
    total = run["cfg"]["total_frames"]
    edges = np.linspace(0, total, n + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fr > lo) & (fr <= hi)
        if m.any():
            xs.append((lo + hi) / 2); ys.append(v[m].mean())
    return np.array(xs), np.array(ys)


def heuristic(o):
    sev = [o[i * 6] for i in range(5)]; eng = [o[i * 6 + 1] for i in range(5)]
    act = [o[i * 6 + 5] > 0 for i in range(5)]; dsc = [o[i * 6 + 3] for i in range(5)]
    appt = [o[i * 6 + 4] for i in range(5)]; slots, refs = o[-3], o[-2]
    cand = [i for i in range(5) if act[i]]
    if not cand:
        return NOOP
    i = max(cand, key=lambda j: sev[j])
    if sev[i] > 0.7 and refs > 0:
        return i * 5 + 4
    j = min(cand, key=lambda j: eng[j])
    if eng[j] < 0.45:
        return j * 5 + 1
    if slots > 0 and not appt[i] and sev[i] > 0.45:
        return i * 5 + 3
    k = max(cand, key=lambda j: dsc[j])
    if dsc[k] > 0.4:
        return k * 5 + (2 if sev[k] > 0.5 else 1)
    return i * 5


def main():
    ASSETS.mkdir(exist_ok=True)
    runs = load_runs()
    rows, best = [], {}
    for a in ALGOS:
        for r in runs[a]:
            tail = [v for _, v in r["ep_returns"]][-30:]
            rows.append(dict(algo=NAMES[a], run=r["run"], cfg=json.dumps({k: v for k, v in r["cfg"].items() if k not in ("total_frames", "seed")}),
                             train_last30=np.mean(tail), eval_mean=r["final_eval"]["mean"], eval_std=r["final_eval"]["std"],
                             retained=r["final_eval"]["retained"], crisis=r["final_eval"]["crisis_rate"],
                             gen_mean=r["gen_eval"]["mean"], seconds=r["seconds"]))
        if runs[a]:
            best[a] = max(runs[a], key=lambda r: r["final_eval"]["mean"])
    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "sweep_summary.csv", index=False)

    # ---- independent test of best runs (100 fresh seeds) + baselines
    test = {}
    for a, r in best.items():
        _, fn, _ = load_agent(MODEL_DIR / ("dqn" if a == "dqn" else "pg") / f"{a}_run_{r['run']:02d}.pt")
        test[a] = dict(test=evaluate(fn, 100, seed0=90_000), gen=evaluate(fn, 100, seed0=95_000, volatility=1.6), run=r["run"])
    rng = np.random.default_rng(0)
    base = {
        "random": lambda o: int(rng.integers(N_ACTIONS)),
        "no-op": lambda o: NOOP,
        "rule-based": lambda o: heuristic(o.numpy()),
    }
    for n, f in base.items():
        test[n] = dict(test=evaluate(f, 100, seed0=90_000), gen=evaluate(f, 100, seed0=95_000, volatility=1.6), run=None)
    json.dump(test, open(LOG_DIR / "best_test_results.json", "w"), indent=1)
    json.dump({a: {"run": r["run"], "path": f"models/{'dqn' if a == 'dqn' else 'pg'}/{a}_run_{r['run']:02d}.pt"} for a, r in best.items()},
              open(MODEL_DIR / "best.json", "w"), indent=1)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    # Fig 1: comparative training reward (best run per algo + sweep mean band)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a in ALGOS:
        if not runs[a]:
            continue
        x, y = curve(best[a]); ax[0].plot(x, smooth(y, 5), color=COL[a], label=NAMES[a])
        M = np.array([np.interp(x, *curve(r)) for r in runs[a]])
        ax[1].plot(x, np.median(M, 0), color=COL[a], label=NAMES[a])
        ax[1].fill_between(x, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=COL[a], alpha=0.15)
    for ax_, t in zip(ax, ["Best run per algorithm", "Median and IQR across all 10 runs"]):
        ax_.axhline(test["rule-based"]["test"]["mean"], color="grey", ls="--", lw=1, label="rule-based")
        ax_.axhline(test["random"]["test"]["mean"], color="black", ls=":", lw=1, label="random")
        ax_.set_title(t); ax_.set_xlabel("Environment steps"); ax_.set_ylabel("Training episode return")
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(ASSETS / "fig_rewards.png", dpi=170); plt.close()

    # Fig 2: DQN loss
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for r in runs["dqn"]:
        f, l = zip(*r["loss"]); ax.plot(f, smooth(l, 10), color="#3470be", alpha=0.18)
    f, l = zip(*best["dqn"]["loss"]); ax.plot(f, smooth(l, 10), color="#3470be", lw=2, label=f"best (run {best['dqn']['run']:02d})")
    ax.set_yscale("log"); ax.set_xlabel("Environment steps"); ax.set_ylabel("TD (Huber) loss"); ax.set_title("DQN objective")
    ax.legend(); plt.tight_layout(); plt.savefig(ASSETS / "fig_dqn_loss.png", dpi=170); plt.close()

    # Fig 3: PG entropy
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for a in ["reinforce", "ppo", "a2c"]:
        f, e = zip(*best[a]["entropy"]); ax.plot(f, smooth(e, 5), color=COL[a], label=f"{NAMES[a]} (best)")
    ax.axhline(np.log(N_ACTIONS), color="grey", ls="--", lw=1, label="uniform policy")
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Policy entropy (nats)"); ax.set_title("Policy entropy"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(ASSETS / "fig_entropy.png", dpi=170); plt.close()

    # Fig 4: convergence (greedy eval during training)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for a in ALGOS:
        f, v = zip(*best[a]["eval_curve"]); ax.plot(f, v, "o-", ms=3, color=COL[a], label=NAMES[a])
    ax.axhline(test["rule-based"]["test"]["mean"], color="grey", ls="--", lw=1, label="rule-based")
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Greedy eval return (10 eps)"); ax.set_title("Convergence"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(ASSETS / "fig_convergence.png", dpi=170); plt.close()

    # Fig 5: generalisation
    names = [n for n in ALGOS] + ["rule-based", "random"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4)); w = 0.38; x = np.arange(len(names))
    ax.bar(x - w / 2, [test[n]["test"]["mean"] for n in names], w, yerr=[test[n]["test"]["std"] / 10 for n in names], label="unseen seeds (volatility 1.0)", color="#3470be")
    ax.bar(x + w / 2, [test[n]["gen"]["mean"] for n in names], w, yerr=[test[n]["gen"]["std"] / 10 for n in names], label="unseen seeds, harder dynamics (volatility 1.6)", color="#e8a828")
    ax.set_xticks(x); ax.set_xticklabels([NAMES.get(n, n) for n in names], fontsize=8); ax.set_ylabel("Mean return (100 eps)")
    ax.set_title("Generalisation"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(ASSETS / "fig_generalisation.png", dpi=170); plt.close()

    print(df.groupby("algo")[["eval_mean", "gen_mean"]].agg(["max", "mean"]).round(2))
    print(json.dumps({k: (round(v["test"]["mean"], 2), round(v["gen"]["mean"], 2), v["run"]) for k, v in test.items()}))


if __name__ == "__main__":
    main()
