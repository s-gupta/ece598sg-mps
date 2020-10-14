import gym
import numpy as np
from pathlib import Path

import io
from tqdm import tqdm
import utils
import envs
import logging
import time
import torch
from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', None, 'Name of environment.')
flags.DEFINE_string('datadir', 'data/', 'Directory to store loss plots, etc.')
flags.mark_flag_as_required('env_name')

def main(_):
    datadir = Path(FLAGS.datadir)
    in_file_name = datadir / (FLAGS.env_name + '.pkl')
    dt = utils.load_variables(str(in_file_name))
    all_states = dt['states']
    all_actions = dt['actions']

    envs = gym.make('Visual'+FLAGS.env_name)
    all_obss = [[None for j in range(all_states.shape[1])] 
                      for i in range(all_states.shape[0])]
    all_obss = np.array(all_obss)
    for i in tqdm(range(all_states.shape[0])):
      for j in tqdm(range(all_states.shape[1])):
        obs = envs.reset_to_state(all_states[i,j,:])
        obs_bytes = io.BytesIO()
        obs.save(obs_bytes, format='PNG')
        all_obss[i,j] = obs_bytes
    data_file_name = datadir / ('Visual' + FLAGS.env_name + '.pkl')
    utils.save_variables(str(data_file_name), [all_obss, all_actions], 
                         ['states', 'actions'], overwrite=True)

if __name__ == '__main__':
    app.run(main)
