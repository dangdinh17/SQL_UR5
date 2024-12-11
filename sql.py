import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from collections import deque
from env import ur5GymEnv
from matplotlib import pyplot as plt
from tqdm import tqdm


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 0)
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(-1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(-1)
        )

# Soft Q-Learning implementation
class SoftQLearning:
    def __init__(self, state_dim, action_dim, hidden_dim=256, gamma=0.99, tau=0.005, alpha=1, k=1e-4, buffer_size=10000):
        self.gamma = gamma
        self.tau = tau
        self.alpha_0 = alpha
        self.alpha = alpha
        self.k = k  # Decay rate for alpha

        # Add CUDA device support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.q_losses = []
        self.policy_losses = []
        self.episode_rewards = []

        self.update_step = 0

    def update(self, batch_size):
        # Decay alpha
        self.alpha = self.alpha_0 * np.exp(-self.k * self.update_step)
        self.update_step += 1

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device),
        )

        # Update Q-function
        with torch.no_grad():
            next_actions, next_log_probs = self.policy_network.sample_action(next_states)
            target_q = self.target_q_network(next_states, next_actions)
            target_value = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_probs)

        q_values = self.q_network(states, actions)
        q_loss = ((q_values - target_value) ** 2).mean()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        new_actions, log_probs = self.policy_network.sample_action(states)
        q_new_actions = self.q_network(states, new_actions)
        policy_loss = (self.alpha * log_probs - q_new_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.q_losses.append(q_loss.item())

        # Update target Q-network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    def normalize_state(self, state):
        return (state - np.mean(state)) / (np.std(state) + 1e-6)

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'policy_network_state_dict': self.policy_network.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")