import random
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

class ResBlock(nn.Module):
    def __init__(self, channels=128):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=channels//8, num_channels=channels)
        self.gn2 = nn.GroupNorm(num_groups=channels//8, num_channels=channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels=13, n_blocks=10, n_channels=128, value_layers=64, n_actions=8*8*4, board_size=8*8):
        super(ResNet, self).__init__()
        self.conv_init = nn.Conv2d(input_channels, n_channels, kernel_size=3, padding=1, bias=False)
        self.gn_init = nn.GroupNorm(num_groups=n_channels//8, num_channels=n_channels)
        
        self.res_blocks = nn.ModuleList([ResBlock(n_channels) for _ in range(n_blocks-1)])
        
        self.policy_conv = nn.Conv2d(n_channels, 2, kernel_size=1, bias=False)
        self.policy_gn = nn.GroupNorm(num_groups=1, num_channels=2)
        self.policy = nn.Linear(2*board_size, n_actions)
        
        self.value_conv = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
        self.value_gn = nn.GroupNorm(num_groups=1, num_channels=1)
        self.value_linear = nn.Linear(1*board_size, value_layers, bias=False)
        self.value = nn.Linear(value_layers, 1)
        
    def forward(self, x):
        x = F.relu(self.gn_init(self.conv_init(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        policy = F.relu(self.policy_gn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy(policy)
        
        value = F.relu(self.value_gn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_linear(value))
        value = self.value(value)
        value = torch.tanh(value)
        return policy, value
    
class ReplayBuffer():
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        
    def add(self, states, action_probs, values):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((states, action_probs, values))

    # should return a batch size of the appropriate states, action_probs, and values
    @abstractmethod
    def sample(self, batch_size):
        pass
    
class Trainer():
    def __init__(self, model, device, replay_buffer, lr=0.001, weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.replay_buffer = replay_buffer
        
    def train(self, batch_size):
        self.model.train()
        states, t_action_probs, t_values = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        t_action_probs = torch.FloatTensor(np.array(t_action_probs)).to(self.device)
        t_values = torch.FloatTensor(np.array(t_values)).to(self.device)
        
        self.optimizer.zero_grad()
        p_action_probs, p_values = self.model(states)
        loss = F.cross_entropy(p_action_probs, t_action_probs) + F.mse_loss(p_values.view(-1), t_values.view(-1))
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
                