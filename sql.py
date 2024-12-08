import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from env import *
from torch.distributions.categorical import Categorical

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, alpha):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, num_actions)
        self.alpha = alpha

    def get_Q(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_V(self, q):
        # print(q)
        # print(q.shape)
        v = self.alpha * torch.log(torch.mean(torch.exp(q / self.alpha), dim=-1))
        return v