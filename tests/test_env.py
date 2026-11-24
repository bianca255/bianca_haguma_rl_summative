import numpy as np
from environment.custom_env import KigaliCareEnv, OBS_DIM, N_ACTIONS, NOOP, EPISODE_DAYS


def test_reset_shape_and_determinism():
    a, b = KigaliCareEnv(), KigaliCareEnv()
    oa, _ = a.reset(seed=1); ob, _ = b.reset(seed=1)
    assert oa.shape == (OBS_DIM,) and np.allclose(oa, ob)


def test_horizon_truncation():
    env = KigaliCareEnv(); env.reset(seed=0)
    for _ in range(EPISODE_DAYS):
        _, _, term, trunc, _ = env.step(NOOP)
        if term or trunc:
            break
    assert term or trunc


def test_actions_change_state():
    env = KigaliCareEnv(); env.reset(seed=0)
    eng0 = env.eng[0]
    env.step(1)  # check-in user 1
    assert env.eng[0] > eng0 - 0.05
    assert env.dsc[0] == 1  # reset to 0 by the action, then +1 by the daily update


def test_resource_limits():
    env = KigaliCareEnv(); env.reset(seed=0)
    for u in range(4):
        env.step(u * 5 + 3)  # book counsellor for four users; only 3 weekly slots
    assert env.slots == 0
    assert env.last_reward < 0.5  # the 4th booking was invalid and penalised in its text
    assert "no slot" in env.last_action_text


def test_observation_range():
    env = KigaliCareEnv(); env.reset(seed=2)
    for _ in range(30):
        o, *_ = env.step(env.action_space.sample())
        assert o.min() >= 0 and o.max() <= 1 and o.shape == (OBS_DIM,)
