import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

num_steps_per_rollout = 5 
num_updates = 10000
reset_every = 200
val_every = 10000

replay_buffer_size = 1000000
q_target_update_every = 50
q_batch_size = 256
q_num_steps = 1
    
def log(writer, iteration, name, value, print_every=10, log_every=10):
    # A simple function to let you log progress into console and tensorboard.
    if np.mod(iteration, print_every) == 0:
        if name is not None:
            print('{:8d}{:>30s}: {:0.3f}'.format(iteration, name, value))
        else:
            print('')
    if name is not None and np.mod(iteration, log_every) == 0:
        writer.add_scalar(name, value, iteration)

# You may want to make a replay buffer class that a) stores rollouts as they
# come along, overwriting older rollouts as needed, and b) allows random
# sampling of transition quadruples for training of the Q-networks.
class ReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim):
        # TODO
        None

    def insert(self, rollouts):
        # TODO
        None

    def sample_batch(self, batch_size):
        samples = None
        # TODO
        return samples

# Starting off from states in envs, rolls out num_steps_per_rollout for each
# environment envs using the policy in model. Returns rollouts in the form of
# states, actions, rewards and new states. Also returns the state the
# environments end up in after num_steps_per_rollout time steps.
def collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device):
    rollouts = []
    # TODO
    return rollouts, states
   
# Function to train the Q function. Samples q_num_steps batches of size
# q_batch_size from the replay buffer, runs them through the target network to
# obtain target values for the model to regress to. Takes optimization steps to
# do so. Returns the bellman_error for plotting.
def update_model(replay_buffer, models, targets, optim, gamma, action_dim,
                 q_batch_size, q_num_steps):
    total_bellman_error = 0.
    # TODO
    return total_bellman_error

def train_model_dqn(models, targets, state_dim, action_dim, envs, gamma, device, logdir, val_fn):
    train_writer = SummaryWriter(logdir / 'train')
    val_writer = SummaryWriter(logdir / 'val')
    
    # Set model into training mode
    [m.train() for m in models]
    
    # You may want to setup an optimizer, loss functions for training.
    optim = None

    # Set up the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, 1)

    # Resetting all environments to initialize the state.
    num_steps, total_samples = 0, 0
    states = [e.reset() for e in envs]
    
    for updates_i in range(num_updates):
        # Come up with a schedule for epsilon
        epsilon = 0.5 
        # TODO  

        # Put model in training mode.

        if np.mod(updates_i, q_target_update_every) == 0:
            # If you are using a target network, every few updates you may want
            # to copy over the model to the target network.
            # TODO
            None
        
        # Collect rollouts using the policy.
        rollouts, states = collect_rollouts(models, envs, states, num_steps_per_rollout, epsilon, device)
        num_steps += num_steps_per_rollout
        total_samples += num_steps_per_rollout*len(envs)
        
        # Push rollouts into the replay buffer.
        replay_buffer.insert(rollouts)


        # Use replay buffer to update the policy and take gradient steps.
        bellman_error = update_model(replay_buffer, models, targets, optim,
                                     gamma, action_dim, q_batch_size,
                                     q_num_steps)
        log(train_writer, updates_i, 'train-samples', total_samples, 100, 10)
        log(train_writer, updates_i, 'train-bellman-error', bellman_error, 100, 10)
        log(train_writer, updates_i, 'train-epsilon', epsilon, 100, 10)
        log(train_writer, updates_i, None, None, 100, 10)


        # We are solving a continuing MDP which never returns a done signal. We
        # are going to manully reset the environment every few time steps. To
        # track progress on the training envirnments you can maintain the
        # returns on the training environments, and log or print it out when
        # you reset the environments.
        if num_steps >= reset_every:
            states = [e.reset() for e in envs]
            num_steps = 0
        
        # Every once in a while run the policy on the environment in the
        # validation set. We will use this to plot the learning curve as a
        # function of the number of samples.
        cross_boundary = total_samples // val_every > \
            (total_samples - len(envs)*num_steps_per_rollout) // val_every
        if cross_boundary:
            models[0].eval()
            mean_reward = val_fn(models[0], device)
            log(val_writer, total_samples, 'val-mean_reward', mean_reward, 1, 1)
            log(val_writer, total_samples, None, None, 1, 1)
            models[0].train()
