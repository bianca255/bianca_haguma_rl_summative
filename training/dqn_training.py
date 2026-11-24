"""DQN (value-based) trained with TorchRL."""
from __future__ import annotations

import torch
from torch import nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import QValueModule, EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate, HardUpdate

from environment.custom_env import OBS_DIM, N_ACTIONS
from training.common import make_env, mlp, evaluate, greedy_fn, RunLog

DEFAULTS = dict(lr=1e-3, gamma=0.99, hidden=128, layers=2, batch_size=64, buffer_size=20_000,
                eps_start=1.0, eps_end=0.05, eps_anneal_frac=0.5, target_update="soft", tau=0.01,
                hard_every=250, double_dqn=False, updates_per_batch=16, frames_per_batch=32,
                init_random_frames=512, total_frames=61_440, seed=0, loss_function="smooth_l1")


def train(run_id: int, **overrides) -> dict:
    cfg = {**DEFAULTS, **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env()
    env.set_seed(cfg["seed"])
    qnet = mlp(OBS_DIM, N_ACTIONS, cfg["hidden"], cfg["layers"])
    value = TensorDictModule(qnet, in_keys=["observation"], out_keys=["action_value"])
    policy = TensorDictSequential(value, QValueModule(spec=env.action_spec, action_space="categorical"))
    egreedy = EGreedyModule(env.action_spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                            annealing_num_steps=int(cfg["total_frames"] * cfg["eps_anneal_frac"]))
    explore = TensorDictSequential(policy, egreedy)
    collector = Collector(env, explore, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=cfg["total_frames"], init_random_frames=cfg["init_random_frames"])
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])
    loss_mod = DQNLoss(policy, action_space="categorical", loss_function=cfg["loss_function"],
                       double_dqn=cfg["double_dqn"], delay_value=True)
    loss_mod.make_value_estimator(gamma=cfg["gamma"])
    updater = SoftUpdate(loss_mod, tau=cfg["tau"]) if cfg["target_update"] == "soft" else None
    hard = HardUpdate(loss_mod, value_network_update_interval=cfg["hard_every"]) if cfg["target_update"] == "hard" else None
    opt = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])

    log = RunLog("dqn", run_id, cfg)
    frames, next_eval = 0, cfg["total_frames"] // 10
    for batch in collector:
        frames += batch.numel()
        rb.extend(batch.reshape(-1).cpu())
        done = batch["next", "done"].squeeze(-1)
        if done.any():
            for v in batch["next", "episode_reward"][done].flatten().tolist():
                log.d["ep_returns"].append((frames, v))
        if len(rb) >= cfg["batch_size"] and frames > cfg["init_random_frames"]:
            losses = []
            for _ in range(cfg["updates_per_batch"]):
                loss = loss_mod(rb.sample())["loss"]
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(loss_mod.parameters(), 10.0)
                opt.step()
                (updater or hard).step()
                losses.append(loss.item())
            log.d["loss"].append((frames, sum(losses) / len(losses)))
        egreedy.step(batch.numel())
        if frames >= next_eval:
            log.d["eval_curve"].append((frames, evaluate(greedy_fn(qnet), 10)["mean"]))
            next_eval += cfg["total_frames"] // 10
    collector.shutdown()
    return log.finish(qnet, "dqn", run_id, cfg)
