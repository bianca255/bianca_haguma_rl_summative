"""Policy-gradient methods with TorchRL: REINFORCE, PPO and A2C (same env, same frame budget)."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss, A2CLoss
from torchrl.objectives.value import GAE

from environment.custom_env import OBS_DIM, N_ACTIONS, EPISODE_DAYS
from training.common import make_env, mlp, RunLog

COMMON = dict(hidden=128, layers=2, gamma=0.99, total_frames=61_440, seed=0)
DEFAULTS = {
    "reinforce": dict(COMMON, lr=3e-3, entropy_coef=0.01, episodes_per_batch=8, normalize_returns=True),
    "ppo": dict(COMMON, lr=3e-4, lmbda=0.95, clip_eps=0.2, entropy_coef=0.01, epochs=8, minibatch=128,
                frames_per_batch=1024, critic_coef=1.0, max_grad_norm=0.5),
    "a2c": dict(COMMON, lr=1e-3, lmbda=0.95, entropy_coef=0.01, critic_coef=0.5, frames_per_batch=256,
                max_grad_norm=0.5),
}


def _build(env, cfg):
    logits_net = mlp(OBS_DIM, N_ACTIONS, cfg["hidden"], cfg["layers"])
    actor = ProbabilisticActor(
        TensorDictModule(logits_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], spec=env.action_spec, distribution_class=Categorical, return_log_prob=True)
    value_net = mlp(OBS_DIM, 1, cfg["hidden"], cfg["layers"])
    critic = ValueOperator(value_net, in_keys=["observation"])
    return logits_net, actor, critic


def _log_returns(log, frames, batch):
    done = batch["next", "done"].squeeze(-1)
    if done.any():
        for v in batch["next", "episode_reward"][done].flatten().tolist():
            log.d["ep_returns"].append((frames, v))


def train_actor_critic(algo: str, run_id: int, **overrides) -> dict:
    cfg = {**DEFAULTS[algo], **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env(); env.set_seed(cfg["seed"])
    logits_net, actor, critic = _build(env, cfg)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic)
    if algo == "ppo":
        loss_mod = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_eps"], entropy_coeff=cfg["entropy_coef"],
                               critic_coeff=cfg["critic_coef"], normalize_advantage=True)
    else:
        loss_mod = A2CLoss(actor, critic, entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["critic_coef"])
    opt = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])
    collector = Collector(env, actor, frames_per_batch=cfg["frames_per_batch"], total_frames=cfg["total_frames"])
    log = RunLog(algo, run_id, cfg)
    frames, next_eval = 0, cfg["total_frames"] // 10
    for batch in collector:
        frames += batch.numel()
        _log_returns(log, frames, batch)
        with torch.no_grad():
            gae(batch)
        data = batch.reshape(-1)
        ents, losses = [], []
        if algo == "ppo":
            for _ in range(cfg["epochs"]):
                perm = torch.randperm(data.shape[0])
                for s in range(0, data.shape[0], cfg["minibatch"]):
                    mb = data[perm[s:s + cfg["minibatch"]]]
                    out = loss_mod(mb)
                    loss = out["loss_objective"] + out["loss_critic"] + (out["loss_entropy"] if "loss_entropy" in out.keys() else 0.0)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg["max_grad_norm"]); opt.step()
                    ents.append(out["entropy"].item() if "entropy" in out.keys() else float("nan")); losses.append(loss.item())
        else:
            out = loss_mod(data)
            loss = out["loss_objective"] + out["loss_critic"] + (out["loss_entropy"] if "loss_entropy" in out.keys() else 0.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg["max_grad_norm"]); opt.step()
            ents.append(out["entropy"].item() if "entropy" in out.keys() else float("nan")); losses.append(loss.item())
        log.d["entropy"].append((frames, sum(ents) / len(ents)))
        log.d["loss"].append((frames, sum(losses) / len(losses)))
        if frames >= next_eval:
            from training.common import evaluate, greedy_fn
            log.d["eval_curve"].append((frames, evaluate(greedy_fn(logits_net), 10)["mean"]))
            next_eval += cfg["total_frames"] // 10
    collector.shutdown()
    return log.finish(logits_net, algo, run_id, cfg)


def train_reinforce(run_id: int, **overrides) -> dict:
    """Monte-Carlo policy gradient (Williams, 1992). TorchRL provides env, rollouts and the actor;
    the REINFORCE objective itself is written explicitly because TorchRL ships no REINFORCE loss."""
    cfg = {**DEFAULTS["reinforce"], **overrides}
    torch.manual_seed(cfg["seed"])
    env = make_env(); env.set_seed(cfg["seed"])
    logits_net, actor, _ = _build(env, cfg)
    opt = torch.optim.Adam(logits_net.parameters(), lr=cfg["lr"])
    log = RunLog("reinforce", run_id, cfg)
    frames, next_eval = 0, cfg["total_frames"] // 10
    from training.common import evaluate, greedy_fn
    while frames < cfg["total_frames"]:
        obs_l, act_l, ret_l = [], [], []
        for _ in range(cfg["episodes_per_batch"]):
            with torch.no_grad():
                ro = env.rollout(max_steps=EPISODE_DAYS, policy=actor, break_when_any_done=True)
            r = ro["next", "reward"].squeeze(-1)
            G, acc = torch.zeros_like(r), 0.0
            for t in reversed(range(r.shape[0])):
                acc = r[t] + cfg["gamma"] * acc
                G[t] = acc
            obs_l.append(ro["observation"]); act_l.append(ro["action"]); ret_l.append(G)
            frames += r.shape[0]
            log.d["ep_returns"].append((frames, float(r.sum())))
        obs, act, G = torch.cat(obs_l), torch.cat(act_l), torch.cat(ret_l)
        if cfg["normalize_returns"]:
            G = (G - G.mean()) / (G.std() + 1e-8)
        dist = Categorical(logits=logits_net(obs))
        loss = -(dist.log_prob(act) * G).mean() - cfg["entropy_coef"] * dist.entropy().mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(logits_net.parameters(), 0.5); opt.step()
        log.d["entropy"].append((frames, dist.entropy().mean().item()))
        log.d["loss"].append((frames, loss.item()))
        if frames >= next_eval:
            log.d["eval_curve"].append((frames, evaluate(greedy_fn(logits_net), 10)["mean"]))
            next_eval += cfg["total_frames"] // 10
    return log.finish(logits_net, "reinforce", run_id, cfg)


def train(algo: str, run_id: int, **overrides) -> dict:
    return train_reinforce(run_id, **overrides) if algo == "reinforce" else train_actor_critic(algo, run_id, **overrides)
