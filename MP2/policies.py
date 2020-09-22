from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical 

class DQNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, device):
        super(DQNPolicy, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i, l in enumerate(hidden_layers[:-1]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.layers = nn.Sequential(*layers)
        self.device = device

    def forward(self, x):
        qvals = self.layers(x)
        return qvals 
  
    def act(self, x, epsilon=0.):
        qvals = self.forward(x)
        act = torch.argmax(qvals, 1, keepdim=True)
        if epsilon > 0:
            act_random = torch.multinomial(torch.ones(qvals.shape[1],), 
                                           act.shape[0], replacement=True)
            act_random = act_random.reshape(-1,1).to(self.device)
            combine = torch.rand(qvals.shape[0], 1) > epsilon
            combine = combine.float().to(self.device)
            act = act * combine + (1-combine) * act_random
            act = act.long()
        return act

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(ActorCriticPolicy, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i, l in enumerate(hidden_layers[:-1]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        
        self.actor = nn.Linear(hidden_layers[-1], output_dim)
        self.critic = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        x = self.layers(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic
  
    def actor_to_distribution(self, actor):
        action_distribution = Categorical(logits=actor.unsqueeze(-2))
        return action_distribution
        
    def act(self, x, sample=False):
        actor, critic = self.forward(x)
        action_distribution = self.actor_to_distribution(actor)
        if sample:
            action = action_distribution.sample()
        else:
            action = action_distribution.probs.argmax(-1)
        return action


