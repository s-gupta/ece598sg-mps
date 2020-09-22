from pathlib import Path
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from absl import flags, app
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist('logdirs', [], 
    'Space separated list of directories to plot results from.')
flags.DEFINE_string('output_file_name', 'out.pdf', 
    'Output file to generate plot.')
flags.DEFINE_integer('seeds', 5,
    'Number of seeds per run')

def main(_):
    sns.color_palette()
    fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    print(FLAGS.logdirs)
    for logdir in FLAGS.logdirs:
        print(logdir)
        samples = []
        rewards = []
        for seed in range(FLAGS.seeds):
            logdir_ = Path(logdir) / f'seed{seed}'
            logdir_ = logdir_ / 'val'
            event_acc = EventAccumulator(str(logdir_))
            event_acc.Reload()
            _, step_nums, vals = zip(*event_acc.Scalars('val-mean_reward'))
            samples.append(step_nums)
            rewards.append(vals)
        samples = np.array(samples)
        assert(np.all(samples == samples[:1,:]))
        rewards = np.array(rewards)
        mean_rewards = np.mean(rewards, 0)
        std_rewards = np.std(rewards, 0)
        ax.plot(samples[0,:], mean_rewards, label=logdir)
        ax.fill_between(samples[0,:], 
                        mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2)

    ax.legend(loc=4)
    ax.set_ylim([0, 210])
    ax.grid('major')
    fig.savefig(FLAGS.output_file_name, bbox_inches='tight')


if __name__ == '__main__':
    app.run(main)
