import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, max_speed=np.inf, max_torque=np.inf,
                 init_theta=np.pi, init_thetadot=1, noise=0.):
        logging.info('PendulumEnv.max_torque: %f', max_torque)
        logging.info('PendulumEnv.max_speed: %f', max_speed)
        logging.info('PendulumEnv.init_theta: %f', init_theta)
        logging.info('PendulumEnv.init_thetadot: %f', init_thetadot)
        logging.info('PendulumEnv.noise: %f', noise)

        self.init_theta = init_theta
        self.init_thetadot = init_thetadot
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.noise = noise
        
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.states = []
        self.controls = []

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        high = np.array([np.inf, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        self.total_time = 0
        self.total_time_upright = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + thdot ** 2 + (u ** 2)

        thnoise = self.np_random.randn()*self.noise
        thdotnoise = self.np_random.randn()*self.noise
        
        newth = th + thdot * dt 
        newth = angle_normalize(newth + thnoise)
        
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. * u / (m * l ** 2)) * dt
        newthdot = newthdot + thdotnoise
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        self.states.append(self.state)
        self.controls.append(u)
        self.total_time += 1
        self.total_time_upright += np.abs(th) < 0.1
        metric = {'fraction_upright': self.total_time_upright / self.total_time}
        return self._get_obs(), -costs, False, {'metric': metric}

    def reset(self):
        high = np.array([self.init_theta, self.init_thetadot])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.states.append(self.state)
        self.last_u = None
        self.total_time = 0
        self.total_time_upright = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1.3, 1.3, -1.3, 1.3)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
