import gym
import numpy as np
from pathlib import Path

import envs
import logging
import time
import torch
from absl import app
from absl import flags
from policies import DQNPolicy, ActorCriticPolicy
from trainer_ac import train_model_ac
from trainer_dqn import train_model_dqn
from evaluation import val, test_model_in_env

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to evaluate.')
flags.DEFINE_integer('episode_len', 200, 'Length of each episode at test time.')
flags.DEFINE_string('env_name', 'CartPole-v2', 'Name of environment.')
flags.DEFINE_boolean('vis', False, 'To visualize or not.')
flags.DEFINE_boolean('vis_save', False, 'To save visualization or not')
flags.DEFINE_integer('num_train_envs', 1, '')
flags.DEFINE_integer('seed', 0, 'Seed for randomly initializing policies.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor gamma.')
flags.DEFINE_enum('algo', None, ['dqn', 'ac'], 'which algo to use, dqn or ac')
flags.DEFINE_string('logdir', None, 'Directory to store loss plots, etc.')
flags.mark_flag_as_required('logdir')
flags.mark_flag_as_required('algo')


def get_dims(env_name):
    if env_name == 'CartPole-v2':
        return 4, 2

def main(_):
    torch.manual_seed(FLAGS.seed)
    logdir = Path(FLAGS.logdir) / f'seed{FLAGS.seed}' 
    logdir.mkdir(parents=True, exist_ok=True)
    
    # Setup training environments.
    train_envs = [gym.make(FLAGS.env_name) for _ in range(FLAGS.num_train_envs)]
    [env.seed(i+FLAGS.seed) for i, env in enumerate(train_envs)]
    
    # Setting up validation environments.
    val_envs = [gym.make(FLAGS.env_name) for _ in range(FLAGS.num_episodes)]
    [env.seed(i+1000) for i, env in enumerate(val_envs)]
    val_fn = lambda model, device: val(model, device, val_envs, FLAGS.episode_len)

    torch.set_num_threads(1)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    state_dim, action_dim = get_dims(FLAGS.env_name)
    
    if FLAGS.algo == 'dqn':
        n_models = 1
        models, targets = [], []
        for i in range(n_models):
            models.append(DQNPolicy(state_dim, [16, 32, 64], action_dim, device))
            models[-1].to(device)
        
        for i in range(n_models):
            targets.append(DQNPolicy(state_dim, [16, 32, 64], action_dim, device))
            targets[-1].to(device)
        
        train_model_dqn(models, targets, state_dim, action_dim, train_envs,
                        FLAGS.gamma, device, logdir, val_fn)
        model = models[0]

    elif FLAGS.algo == 'ac':
        model = ActorCriticPolicy(state_dim, [16, 32, 64], action_dim)
        train_model_ac(model, train_envs, FLAGS.gamma, device, logdir, val_fn)
    
    [env.close() for env in train_envs]
    [env.close() for env in val_envs]
    
    if FLAGS.vis or FLAGS.vis_save:
        env_vis = gym.make(FLAGS.env_name)
        state, g, gif, info = test_model_in_env(
            model, env_vis, FLAGS.episode_len, device, vis=FLAGS.vis, 
            vis_save=FLAGS.vis_save)
        if FLAGS.vis_save:  
            gif[0].save(fp=f'{logdir}/vis-{env_vis.unwrapped.spec.id}.gif',
                        format='GIF', append_images=gif,
                        save_all=True, duration=50, loop=0)
        env_vis.close()

if __name__ == '__main__':
    app.run(main)
