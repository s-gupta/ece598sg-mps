import numpy as np
from gym.envs.registration import register
from .cartpole import CartPoleEnv

register(
    id='CartPole-v2',
    entry_point='envs:CartPoleEnv',
)
