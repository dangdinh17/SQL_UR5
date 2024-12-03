import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from sql import *
from env import *

class SoftQ(object):
    def __init__(self, state_dim, action_dim):
        self.alpha = 2
        self.soft_q_net = SoftQNetwork(state_dim, action_dim, self.alpha).to(device)
        self.v_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=1e-3)
        self.gamma = 0.9

    def get_action(self, state):
        py_state = torch.from_numpy(state).float()
        temp_q = self.soft_q_net.get_Q(py_state)
        dist = torch.exp((temp_q-self.soft_q_net.get_V(temp_q))/self.alpha)
        dist = dist / torch.sum(dist)
        m = Categorical(dist.squeeze(0))
        a = m.sample()
        return a.item()

    def train(self, batch):
        state = batch[0]  # array [64 1 2]
        action = batch[1]  # array [64, ]
        reward = batch[2]  # array [64, ]
        next_state = batch[3]
        state = torch.from_numpy(state).float().to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        reward = torch.FloatTensor(reward).float().to(device)

        q = self.soft_q_net.get_Q(state).squeeze(1)
        est_q = q.clone()
        next_q = self.soft_q_net.get_Q(next_state).squeeze(1)
        next_v = self.soft_q_net.get_V(next_q)
        for i in range(len(action)):
            est_q[i][action[i]] = reward[i] + self.gamma * next_v[i]
        q_loss = F.mse_loss(q, est_q.detach())
        self.soft_q_optimizer.zero_grad()
        q_loss.backward()
        self.soft_q_optimizer.step()

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    env = ur5GymEnv()
    action_dim = 4
    state_dim = 6
    agent = SoftQ(state_dim = state_dim, action_dim = action_dim)
    max_MC_iter = 200
    max_epi_iter = 500
    batch_size = 64
    replay_buffer = ReplayBuffer(10000)
    train_curve = []
    for epi in range(max_epi_iter):
        # env.reset()
        state = env.reset()
        acc_reward = 0
        for MC_iter in range(max_MC_iter):
            # print("MC= ", MC_iter)
            # state = np.array(env.agt1_pos).reshape((1, 2))
            action1 = agent.get_action(state)
            # print(action1)
            state, reward, done, _ = env.step(action1)
            acc_reward = acc_reward + reward
            next_state = np.array(env.agt1_pos).reshape((1, 2))
            replay_buffer.push(state, action1, reward_list[0], next_state, done)
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer.sample(batch_size))
            if done:
                break
        print('Episode', epi, 'reward', acc_reward / MC_iter)
        train_curve.append(acc_reward)
    plt.plot(train_curve, linewidth=1, label='SAC')
    plt.show()