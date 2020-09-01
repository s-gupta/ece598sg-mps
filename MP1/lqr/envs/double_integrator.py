import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class DoubleIntegratorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, max_acc=1., init_y=3, init_v=4):
        self.dt = .05
        self.max_acc = max_acc 
        self.action_space = spaces.Box(
            low=-max_acc,
            high=max_acc, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, shape=(2,),
            dtype=np.float32
        )
        self.init_y = init_y 
        self.init_v = init_v

        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        u = np.clip(u, -self.max_acc, self.max_acc)[0]
        y, v = self.state
        newv = v + u*self.dt
        newy = y + v*self.dt
        self.state = np.array([newy, newv])
        costs = newv**2 + newy**2 + u**2
        self.last_u = u
        success = np.abs(y) < 0.1 and np.abs(v) < 0.1
        return self._get_obs(), -costs, False, {'metric': {'success': success}}

    def reset(self):
        high = np.array([self.init_y, self.init_v])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        y, v = self.state
        return np.array([y, v])

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 200)
            self.viewer.set_bounds(-5.0, 5.0, -2.0, 2.0)
            w = 0.6; h = 0.4
            l, r, t, b = -w/2, w/2, h/2, -h/2
            box = rendering.make_polygon([(l,b), (l,t), (r,t), (r,b)])
            box.set_color(.8, .3, .3)
            self.box_transform = rendering.Transform()
            box.add_attr(self.box_transform)
            self.viewer.add_geom(box)
            
            line = rendering.make_polyline([(-5,-h/2), (5,-h/2)])
            line.set_color(0, 0, 0)
            self.viewer.add_geom(line)

        self.box_transform.set_translation(self.state[0], 0)
        if self.last_u:
            arrow = rendering.make_polyline([(self.state[0], 0), 
                                             (self.state[0]+self.last_u, 0)])
            arrow.set_linewidth(2)
            self.viewer.add_onetime(arrow)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
