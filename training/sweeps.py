"""Hyperparameter sweeps: 10 different configurations per algorithm (same env, same frame budget).

Run 01 of each algorithm is the baseline default; runs 02-10 vary one or two controls each
(algorithm-specific controls listed in the report tables).
"""
from __future__ import annotations

SWEEPS = {
    "dqn": [  # lr, gamma, batch, buffer, eps_end, eps_anneal_frac, target update, double DQN, hidden
        dict(),
        dict(lr=3e-4),
        dict(lr=3e-3),
        dict(gamma=0.95),
        dict(gamma=0.999),
        dict(eps_end=0.2, eps_anneal_frac=0.8),
        dict(eps_end=0.01, eps_anneal_frac=0.2),
        dict(target_update="hard", hard_every=500),
        dict(double_dqn=True, buffer_size=5_000, batch_size=128),
        dict(hidden=256, tau=0.005, double_dqn=True),
    ],
    "reinforce": [  # lr, gamma, entropy, episodes per batch, return normalisation, hidden
        dict(),
        dict(lr=1e-3),
        dict(lr=1e-2),
        dict(gamma=0.95),
        dict(gamma=0.999),
        dict(entropy_coef=0.0),
        dict(entropy_coef=0.05),
        dict(episodes_per_batch=4),
        dict(episodes_per_batch=16),
        dict(normalize_returns=False, hidden=256),
    ],
    "ppo": [  # lr, gamma, GAE lambda, clip, entropy, epochs, minibatch, rollout size
        dict(),
        dict(lr=1e-3),
        dict(lr=1e-4),
        dict(gamma=0.95),
        dict(lmbda=0.8),
        dict(clip_eps=0.1),
        dict(clip_eps=0.4),
        dict(entropy_coef=0.05),
        dict(epochs=4, minibatch=256, frames_per_batch=512),
        dict(hidden=256, epochs=16, entropy_coef=0.0),
    ],
    "a2c": [  # lr, gamma, GAE lambda, entropy, critic coeff, rollout size
        dict(),
        dict(lr=3e-4),
        dict(lr=3e-3),
        dict(gamma=0.95),
        dict(lmbda=0.8),
        dict(lmbda=1.0),
        dict(entropy_coef=0.05),
        dict(critic_coef=1.0),
        dict(frames_per_batch=1024),
        dict(hidden=256, frames_per_batch=128, entropy_coef=0.0),
    ],
}
