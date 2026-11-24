"""KigaliCare: a care-navigation environment for a digital mental-health support platform.

Scenario (Kimironko pilot, Gasabo District, Kigali)
---------------------------------------------------
A platform coordinator looks after a cohort of 5 pilot users for 90 days. Each day the
coordinator can take ONE action: nudge/assess/book/refer for a single user, or do nothing.
Actions have delayed, persistent effects (engagement, trust, severity) and consume scarce
resources (weekly counsellor slots, monthly referral places at the Kigali Mental Health
Referral Centre). Severity is only observed through the last self-assessment, so the agent
must trade off information gathering, retention, and costly clinical escalation.

NOTE: All dynamics are a simplified *simulation* designed for studying RL algorithms. The
coefficients are modelling assumptions, not clinical estimates.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

N_USERS = 5
EPISODE_DAYS = 90
N_INTERVENTIONS = 5  # SELF_HELP, CHECK_IN, ASSESS, BOOK_COUNSELLOR, REFER
NOOP = N_USERS * N_INTERVENTIONS  # 25
N_ACTIONS = NOOP + 1  # 26
INTERVENTION_NAMES = ["self-help", "check-in", "assess", "book counsellor", "refer (KMHRC)"]
SLOTS_PER_WEEK = 3
REFERRALS_PER_MONTH = 2
USER_FEATURES = 6  # obs severity, engagement, trust, days since contact, appt pending, active
OBS_DIM = N_USERS * USER_FEATURES + 3


def decode_action(a: int):
    """Return (user, intervention) or None for no-op."""
    if a == NOOP:
        return None
    return a // N_INTERVENTIONS, a % N_INTERVENTIONS


class KigaliCareEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode=None, volatility: float = 1.0, shock_prob: float = 0.012):
        super().__init__()
        self.render_mode = render_mode
        self.volatility = volatility
        self.shock_prob = shock_prob
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._renderer = None
        self.last_action_text = "-"
        self.last_reward = 0.0
        self.total_reward = 0.0

    # ------------------------------------------------------------------ helpers
    def _obs(self):
        o = np.zeros(OBS_DIM, dtype=np.float32)
        for i in range(N_USERS):
            o[i * USER_FEATURES: (i + 1) * USER_FEATURES] = [
                self.obs_sev[i], self.eng[i], self.trust[i],
                min(self.dsc[i], 14) / 14.0, float(self.appt[i]), float(self.active[i]),
            ]
        o[-3] = self.slots / SLOTS_PER_WEEK
        o[-2] = self.referrals / REFERRALS_PER_MONTH
        o[-1] = self.day / EPISODE_DAYS
        return np.clip(o, 0.0, 1.0)

    def _measure(self, i):
        self.obs_sev[i] = float(np.clip(self.sev[i] + self.np_random.normal(0, 0.04), 0, 1))

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        self.day = 0
        self.sev = r.uniform(0.2, 0.8, N_USERS)
        self.eng = r.uniform(0.5, 0.9, N_USERS)
        self.trust = r.uniform(0.4, 0.7, N_USERS)
        self.dsc = r.integers(0, 6, N_USERS).astype(float)
        self.appt = np.zeros(N_USERS, dtype=bool)
        self.active = np.ones(N_USERS, dtype=bool)
        self.resp = r.uniform(0.7, 1.3, N_USERS)  # hidden per-user responsiveness
        self.obs_sev = np.clip(self.sev + r.normal(0, 0.1, N_USERS), 0, 1)
        self.slots = SLOTS_PER_WEEK
        self.referrals = REFERRALS_PER_MONTH
        self.last_action_text = "episode start"
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.crisis_user = -1
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        r = self.np_random
        prev_sev = self.sev.copy()
        was_active = self.active.copy()
        reward = 0.0
        text = "no-op"
        decoded = decode_action(action)

        # 1. pending appointments from yesterday take place
        for i in range(N_USERS):
            if self.active[i] and self.appt[i]:
                gain = 0.10 * self.resp[i] * (0.6 + 0.4 * self.trust[i])
                self.sev[i] -= gain
                self.trust[i] += 0.08
                self.eng[i] += 0.10
                self.dsc[i] = 0
                self.appt[i] = False
                self._measure(i)

        # 2. coordinator action
        if decoded is not None:
            u, k = decoded
            text = f"{INTERVENTION_NAMES[k]} -> user {u + 1}"
            if not self.active[u]:
                reward -= 0.2
                text += " (dropped out)"
            elif k == 0:  # self-help content: cheap, effect scales with engagement
                self.sev[u] -= 0.04 * self.resp[u] * self.eng[u]
                self.eng[u] += 0.02
                self.dsc[u] = 0
            elif k == 1:  # check-in message: retention
                self.eng[u] += 0.12
                self.trust[u] += 0.04
                self.dsc[u] = 0
            elif k == 2:  # self-assessment: information, with survey fatigue
                self._measure(u)
                self.eng[u] -= 0.03
                self.dsc[u] = 0
            elif k == 3:  # book counsellor (scarce weekly slots)
                if self.slots > 0 and not self.appt[u]:
                    self.slots -= 1
                    self.appt[u] = True
                    self.dsc[u] = 0
                else:
                    reward -= 0.2
                    text += " (no slot / already booked)"
            elif k == 4:  # referral to specialist care (scarce monthly places)
                if self.referrals > 0:
                    self.referrals -= 1
                    if self.sev[u] > 0.6:
                        self.sev[u] -= 0.25 * self.resp[u]
                        self.trust[u] += 0.10
                        self.eng[u] += 0.10
                    else:
                        self.trust[u] -= 0.20  # unnecessary escalation erodes trust
                        reward -= 0.5
                    self.dsc[u] = 0
                    self._measure(u)
                else:
                    reward -= 0.2
                    text += " (no places left)"

        # 3. daily dynamics for active users
        newly_dropped = 0
        for i in range(N_USERS):
            if not self.active[i]:
                continue
            self.dsc[i] += 1
            self.eng[i] -= 0.015 + (0.015 if self.dsc[i] > 7 else 0.0) + 0.01 * (1 - self.trust[i])
            drift = self.volatility * (0.008 * (1 - self.eng[i]) + r.normal(0, 0.008)) - 0.004 * self.eng[i]
            self.sev[i] += drift
            if r.random() < self.shock_prob * self.volatility:
                self.sev[i] += r.uniform(0.06, 0.15)  # life shock
            self.sev[i] = float(np.clip(self.sev[i], 0, 1))
            self.eng[i] = float(np.clip(self.eng[i], 0, 1))
            self.trust[i] = float(np.clip(self.trust[i], 0, 1))
            if self.eng[i] < 0.08:
                self.active[i] = False
                self.appt[i] = False
                newly_dropped += 1

        # 4. calendar: weekly slots, monthly referral places
        self.day += 1
        if self.day % 7 == 0:
            self.slots = SLOTS_PER_WEEK
        if self.day % 30 == 0:
            self.referrals = REFERRALS_PER_MONTH

        # 5. reward
        act = self.active
        improvement = float(np.sum((prev_sev - self.sev)[was_active & act]))
        reward += 5.0 * improvement
        reward -= 0.05 * float(np.sum(self.sev[act]))  # ongoing symptom burden
        reward -= 1.0 * newly_dropped
        crisis = bool(np.any(self.sev[act] >= 0.95))
        if crisis:
            reward -= 10.0
            self.crisis_user = int(np.argmax(np.where(act, self.sev, -1)))

        terminated = crisis or not bool(act.any())
        truncated = self.day >= EPISODE_DAYS and not terminated
        # on a natural finish, small bonus for every retained user in good health
        if truncated:
            reward += 1.0 * float(np.sum(act & (self.sev < 0.4)))

        self.last_action_text = text
        self.last_reward = reward
        self.total_reward += reward
        info = {"crisis": crisis, "retained": int(act.sum()), "mean_sev": float(self.sev[act].mean()) if act.any() else 1.0}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import Renderer
            self._renderer = Renderer(human=self.render_mode == "human")
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
