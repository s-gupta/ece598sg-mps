import torch
from tqdm import tqdm
import logging
import numpy as np
from PIL import Image
from torchvision import transforms

def test_model_in_env(model, env, episode_len, device, 
                      vis=False, vis_save=False, visual=False):
    g = 0
    state = env.reset()
    init_state = state
    gif = []
    model.reset()    
    with torch.no_grad():
        for t in range(episode_len):
            if not visual:
                state = torch.from_numpy(state).float().to(device).unsqueeze(0)
            else:
                state = transforms.ToTensor()(state).to(device).unsqueeze(0)
            action = model.act(state).detach().cpu().numpy()
            state, reward, done, info = env.step(action[0])
            g += reward
            if vis: env.render()
            if vis_save: gif.append(Image.fromarray(env.render(mode='rgb_array')))
            if done: break
    return state, g, gif, info

def val(model, device, envs, episode_len, visual=False):
    states = [e.reset() for e in envs]
    all_rewards = []
    model.reset()
    for i in tqdm(range(episode_len)):
        with torch.no_grad():
            if not visual:
                states = np.array(states)
                _states = torch.from_numpy(states).float().to(device)
            else:
                _states = [transforms.ToTensor()(state).to(device).unsqueeze(0) for state in states]
                _states = torch.cat(_states, 0)
            _actions = model.act(_states)
        actions = _actions.cpu().numpy()
        step_data = [env.step(actions[i]) for i, env in enumerate(envs)]
        new_states, rewards, dones, infos = list(zip(*step_data))
        states = new_states
        all_rewards.append(rewards)
    
    metric_name = list(infos[0]['metric'].keys())[0]
    all_metrics = [info['metric'][metric_name] for info in infos]
    all_metrics = np.array(all_metrics)
    all_rewards = np.array(all_rewards).T
    avg_reward = np.mean(np.sum(all_rewards, 1), 0)
    avg_metric = np.mean(all_metrics)
    logging.error('Episode Reward: %9.3f, %s: %7.3f.', avg_reward, 
                  metric_name, avg_metric)
