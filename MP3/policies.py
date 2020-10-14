from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class NNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, discrete):
        super(NNPolicy, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i, l in enumerate(hidden_layers[:-1]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        self.discrete = discrete
        self.actor = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        x = self.layers(x)
        actor = self.actor(x)
        return actor
  
    def act(self, x, sample=False):
        actor = self.forward(x)
        if self.discrete:
            action = actor.argmax(-1, keepdims=True)
        else:
            action = actor
        return action
    
    def reset(self):
        None

class CNNPolicy(nn.Module):
    def __init__(self, stack_states, input_dim, hidden_layers, output_dim, discrete):
        super(CNNPolicy, self).__init__()
        self.discrete = discrete 
        c, h, w = input_dim
        self.convs = nn.ModuleList() 
        neurons = c*stack_states
        for n in hidden_layers:
            conv = nn.Conv2d(in_channels=neurons, out_channels=n, kernel_size=5,
                             stride=[2,2], padding=1)
            neurons = n
            self.convs.append(conv)
            self.convs.append(nn.ReLU())
        self.encoder = nn.Sequential(*self.convs)
        
        out = self.encoder(torch.zeros(1, c*stack_states, h, w))
        b, c, h, w = out.shape
        self.actor = nn.Sequential(
            nn.Linear(c*h*w, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim))
        self.stack_states = stack_states

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        actor = self.actor(x)
        return actor
  
    def act(self, x, sample=False):
        if len(self.history) == 0:
            for i in range(self.stack_states):
                self.history.append(x.unsqueeze(1))
        self.history.insert(0, x.unsqueeze(1))
        self.history.pop()
        x = torch.cat(self.history, 1)
        b, t, c, h, w = x.shape
        x = x.reshape(b, c*t, h, w)
        actor = self.forward(x)
        if self.discrete:
            action = actor.argmax(-1, keepdims=True)
        else: 
            action = actor
        return action
    
    def reset(self):
        self.history = []
