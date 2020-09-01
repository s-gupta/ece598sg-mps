import gym
import numpy as np
import logging
import time
from PIL import Image
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 1, 'Number of episodes to evaluate.')
flags.DEFINE_string('env_name', None, 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'To save visualization or not')
flags.DEFINE_float('pendulum_noise', 0.0, 'Standard deviation for additive gaussian noise for balancing a pendulum.')

class DummyController(object):
    def __init__(self, env, state):
        # You can get the name of the environment from env.unwrapped.spec.id.
        # You can use this to select the appropriate dynamics for controller
        # design. You can pre-compute the control matrices in the __init__
        # function, and compute control values in the act function.

        # If you have a time-varying controller, then you can keep track of the
        # step count for use in the act function.
        self.step = 0
        self.action_space = env.action_space

    def act(self, state):
        # Samples a random action in the action space. See OpenAI gym
        # documentation.
        u = self.action_space.sample()
        self.step += 1
        return u

def main(_):
    # This import is needed here to make sure that pendulum_noise can be passed
    # in through a command line argument.
    import envs
    env = gym.make(FLAGS.env_name)
    env.seed(0)
    total_rewards, total_metrics = [], []
    for i in range(FLAGS.num_episodes):
        reward_i = 0
        state = env.reset()
        states, controls = [state], []
        controller = DummyController(env, state)
        done = False
        gif = []
        for j in range(200):
            action = controller.act(state)
            state, reward, done, info = env.step(action)
            if FLAGS.vis:
                img = env.render()
            if FLAGS.vis_save:
                img = env.render(mode="rgb_array")
                gif.append(Image.fromarray(img))
            states.append(state)
            controls.append(action)
            reward_i += reward
        if FLAGS.vis_save:
            gif[0].save(fp=f'vis-{env.unwrapped.spec.id}-{i}.gif',
                        format='GIF', append_images=gif,
                        save_all=True, duration=50, loop=0)
        metric_name = list(info['metric'].keys())[0]
        metric_value = info['metric'][metric_name]
        total_metrics += [metric_value]
        logging.error('Final State: %7.3f, %7.3f. Episode Cost: %9.3f, %s: %7.3f.',
                      state[0], state[1], -reward_i, metric_name, metric_value)
        total_rewards += [reward_i]
    logging.error('Average Cost: %7.3f', -np.mean(total_rewards))
    logging.error('%s: %7.3f', metric_name, np.mean(total_metrics))
    env.close()

if __name__ == '__main__':
    app.run(main)
