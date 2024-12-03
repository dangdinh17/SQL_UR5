import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym

# Soft Q-learning - Model

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, emb_size):
        super(QNetwork, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

class SoftQPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, emb_size, action_std):
        super(SoftQPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, action_dim),
            nn.Tanh()
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob



# Define replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.idx = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
class SoftQLearning:
    def __init__(self, state_dim, action_dim, emb_size, action_std, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device = torch.device('cpu')):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.emb_size = emb_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.q_network = QNetwork(state_dim, action_dim, emb_size).to(self.device)
        self.target_q_network = QNetwork(state_dim, action_dim, emb_size).to(self.device)
        self.policy = SoftQPolicy(state_dim, action_dim, emb_size, action_std).to(self.device)
        
        # Initialize target Q network with the same weights
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(list(self.q_network.parameters()) + list(self.target_q_network.parameters()), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Loss function
        self.mse_loss = nn.MSELoss()

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, done_flags = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        done_flags = torch.FloatTensor(done_flags).to(self.device)

        # Get Q-values for current state-action pairs
        q1, q2 = self.q_network(states, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_action, _ = self.policy.act(next_states)
            target_q1, target_q2 = self.target_q_network(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_action.log_prob(next_action).sum(-1, keepdim=True)
            target_q = rewards + (1 - done_flags) * self.gamma * target_q

        # Compute loss for Q-networks
        q1_loss = self.mse_loss(q1, target_q)
        q2_loss = self.mse_loss(q2, target_q)

        q_loss = q1_loss + q2_loss

        # Optimize Q-networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy (maximize Q-value + entropy)
        action, log_prob = self.policy.act(states)
        q1, q2 = self.q_network(states, action)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_prob - q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target Q-network using soft update
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return q_loss.item(), policy_loss.item()
