import numpy as np
from absl import flags
from gym.envs.registration import register
from .pendulum import PendulumEnv 
from .double_integrator import DoubleIntegratorEnv 
FLAGS = flags.FLAGS

register(
    id='DoubleIntegrator-v1',
    entry_point='envs:DoubleIntegratorEnv',
    max_episode_steps=200,
    kwargs={'max_acc': np.inf, 'init_y': 5., 'init_v': 3.},
)

register(
    id='PendulumBalance-v1',
    entry_point='envs:PendulumEnv',
    max_episode_steps=200,
    kwargs={'init_theta': 0.2, 'init_thetadot': 0.0, 'max_torque': 1.0, 'noise': FLAGS.pendulum_noise}
)

register(
    id='PendulumInvert-v1',
    entry_point='envs:PendulumEnv',
    max_episode_steps=200,
    kwargs={'init_theta': np.pi, 'init_thetadot': 0.0, 'max_torque': 1.0, 'noise': 0.0}
)
