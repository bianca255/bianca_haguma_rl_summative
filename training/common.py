"""Shared helpers: TorchRL env factory, networks, evaluation, logging, model loading."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment.custom_env import KigaliCareEnv, OBS_DIM, N_ACTIONS, EPISODE_DAYS

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

torch.set_num_threads(1)


def make_env(volatility: float = 1.0):
    """The SAME environment is used for every algorithm (TorchRL-wrapped Gymnasium env)."""
    from torchrl.envs import GymWrapper, TransformedEnv, RewardSum
    env = GymWrapper(KigaliCareEnv(volatility=volatility), categorical_action_encoding=True)
    return TransformedEnv(env, RewardSum())


def mlp(inp: int, out: int, hidden: int, layers: int = 2) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


@torch.no_grad()
def evaluate(act_fn, n_episodes: int = 20, seed0: int = 10_000, volatility: float = 1.0) -> dict:
    """Greedy evaluation of `act_fn(obs_tensor)->int` on fixed, unseen seeds."""
    rets, retained, crises = [], [], 0
    for k in range(n_episodes):
        env = KigaliCareEnv(volatility=volatility)
        obs, _ = env.reset(seed=seed0 + k)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act_fn(torch.as_tensor(obs)))
            R += r
            done = term or trunc
        rets.append(R)
        retained.append(info["retained"])
        crises += int(info["crisis"])
    return {"mean": float(np.mean(rets)), "std": float(np.std(rets)),
            "retained": float(np.mean(retained)), "crisis_rate": crises / n_episodes}


def greedy_fn(net: nn.Module):
    return lambda obs: int(net(obs.float()).argmax().item())


class RunLog:
    def __init__(self, algo: str, run_id: int, cfg: dict):
        self.d = {"algo": algo, "run": run_id, "cfg": cfg, "ep_returns": [], "loss": [], "entropy": [],
                  "eval_curve": [], "final_eval": None, "gen_eval": None, "seconds": None}
        self.t0 = time.time()

    def finish(self, net, algo, run_id, cfg):
        fn = greedy_fn(net)
        self.d["final_eval"] = evaluate(fn, 30)
        self.d["gen_eval"] = evaluate(fn, 30, seed0=50_000, volatility=1.6)  # harder, unseen dynamics
        self.d["seconds"] = time.time() - self.t0
        out = LOG_DIR / algo
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"run_{run_id:02d}.json", "w") as f:
            json.dump(self.d, f)
        sub = "dqn" if algo == "dqn" else "pg"
        (MODEL_DIR / sub).mkdir(parents=True, exist_ok=True)
        torch.save({"algo": algo, "cfg": cfg, "state_dict": net.state_dict()},
                   MODEL_DIR / sub / f"{algo}_run_{run_id:02d}.pt")
        return self.d


def load_agent(path):
    """Rebuild a trained agent from a saved checkpoint -> callable(obs)->action."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    out = N_ACTIONS
    net = mlp(OBS_DIM, out, cfg["hidden"], cfg.get("layers", 2))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return ck["algo"], greedy_fn(net), ck
